"""Atlas VFX LTX Prep: plate + a rendered layer (Card Project folder) -> the inputs of an LTX-2.5 inpaint
(in/outpainting IC-LoRA) integration pass: display-referred reference frames at work size and half size, masks at both
sizes, a start image with the layer comped in, and an ATLAS_VFX_LTX_PREP record for LTX Return."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl

from . import card_project as cp
from . import ltx_integration as li


class AtlasVFXLTXPrep:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "ATLAS_VFX_LTX_PREP", "IMAGE")
    RETURN_NAMES = ("reference", "mask", "reference_half", "mask_half", "start_image", "ltx_prep", "keyframes_half")
    DESCRIPTION = ("Builds an LTX inpaint pass for one layer: the plate through the named ACES 2.0 SDR view (reference "
                   "video), the layer's projected alpha as the mask (smeared down for a contact shadow, dilated), and "
                   "the first frame with the layer comped in (start image). Length must be 8k+1.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "layer_folder": ("STRING", {"default": ""}),
                "first_frame": ("INT", {"default": 1130, "min": 0, "max": 1_000_000}),
                "length": ("INT", {"default": 121, "min": 9, "max": 1001, "step": 8}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "mask_thresh": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_px": ("INT", {"default": 24, "min": 0, "max": 512}),
                "dilate_px": ("INT", {"default": 12, "min": 0, "max": 512}),
                "keyframe_every": ("INT", {"default": 24, "min": 1, "max": 1000,
                                           "tooltip": "the layer comped over the plate every N frames (0, N, 2N.. and "
                                                      "the last), half size, for LTXVAddGuide keyframes"}),
                "mode": (["full", "crop"], {"default": "full",
                                            "tooltip": "full: whole plate scaled to the work size. crop: a window "
                                                       "that follows the element, 1:1 with the plate when it fits "
                                                       "(native detail), smoothed; Return pastes it back"}),
                "crop_margin": ("INT", {"default": 96, "min": 0, "max": 1024}),
                "extra_layers": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "more Card Project / LTX Return folders comped over the plate "
                                                       "after layer_folder, back to front (one per line)"}),
                "reference": (["plate", "comp"], {"default": "plate",
                                                  "tooltip": "plate: the IC-LoRA reference is the bare plate (inpaint "
                                                             "paints the layer back). comp: plate + all layers (relight "
                                                             "passes such as Day-to-Night)"}),
                "mask_mode": (["layer", "none"], {"default": "layer",
                                                  "tooltip": "none: mask = whole frame (Return then delivers a full "
                                                             "replacement plate, alpha 1)"}),
            },
            "optional": {
                "sensor_rs_folder": ("STRING", {"default": "", "tooltip": "Sensor RS Maps folder: the plate is "
                                                "de-rolling-shuttered before LTX (clean space). LTX Return puts it back"}),
                "sensor_lens_folder": ("STRING", {"default": "", "tooltip": "Sensor Lens Maps folder: the plate is "
                                                  "undistorted / devignetted before LTX. LTX Return puts it back"}),
            },
        }

    def run(self, plate_ref, layer_folder, first_frame, length, width, height, mask_thresh, shadow_px, dilate_px,
            keyframe_every=24, mode="full", crop_margin=96, extra_layers="", reference="plate", mask_mode="layer",
            sensor_rs_folder="", sensor_lens_folder=""):
        import torch

        if (length - 1) % 8:
            raise ValueError(f"length {length} is not 8k+1")
        sensor = li.sensor_maps(sensor_rs_folder, sensor_lens_folder)
        def load(folder):
            folder = Path(pl.clean_path(folder))
            rj = json.loads(next(folder.glob("*.render.json")).read_text(encoding="utf-8"))
            return {"name": rj["card"], "pattern": str(folder / rj["pattern"]), "range": rj["frames"]}

        main = load(layer_folder)
        layers = [main] + [load(f) for f in extra_layers.splitlines() if f.strip()]
        rj = {"card": main["name"]}
        pattern = main["pattern"]
        frames = list(range(first_frame, first_frame + length))
        if mode == "crop" and (frames[0] < main["range"][0] or frames[-1] > main["range"][1]):
            raise ValueError(f"crop mode: layer covers {main['range']}, asked {frames[0]}-{frames[-1]}")
        H0, W0 = plate_ref["height"], plate_ref["width"]

        def read(L, n):
            """A layer frame, or nothing (zeros) outside its range."""
            if L["range"][0] <= n <= L["range"][1]:
                return exr_io.read_rgba(L["pattern"] % n)[0]
            return np.zeros((H0, W0, 4), np.float32)
        fwd, inv = li.view_processors(plate_ref["ocio_config"])
        off = plate_ref["frame_offset"]
        size, half = (width, height), (width // 2, height // 2)
        plate_size = (plate_ref["width"], plate_ref["height"])
        windows = None
        if mode == "crop":
            boxes = [li.bbox(exr_io.read_rgba(pattern % n)[0][..., 3], mask_thresh) for n in frames]
            windows = li.crop_windows(boxes, size, plate_size, crop_margin)
        view = (lambda im, i: li.crop(im, windows[i], size)) if windows else (lambda im, i: li.resize(im, size))
        ref_full, ref_half, m_full, m_half = [], [], [], []
        start, rt = None, None
        key_idx = sorted(set(list(range(0, length, keyframe_every)) + [length - 1]))
        keys = []
        for (i, img), n in zip(pl.decode_rgb(plate_ref, frames[0] - off, frames[-1] - off), frames):
            lin = pl.to_working(img, plate_ref)
            if sensor is not None:
                lin = li.remove_camera(lin, sensor, n)      # clean space: matches the pinhole card layers
            stack = [read(L, n) for L in layers]
            layer = stack[0]
            comp = lin
            for L in stack:
                comp = cp.over(L, comp)
            j = n - frames[0]
            if j == 0:
                start = view(li.to_view(comp, fwd), j)
                rt = li.roundtrip_report(lin, fwd, inv)
            if j in key_idx:
                keys.append(li.resize(view(li.to_view(comp, fwd), j), half))
            r = view(li.to_view(comp if reference == "comp" else lin, fwd), j)
            ref_full.append(r)
            ref_half.append(li.resize(r, half))
            a = layer[..., 3]
            if windows:
                x0, y0, w, h = windows[j]
                a = a[y0:y0 + h, x0:x0 + w]
            if mask_mode == "layer":
                m = li.make_mask(a, size, mask_thresh, shadow_px, dilate_px)
            else:
                m = np.ones((size[1], size[0]), np.float32)
            m_full.append(m)
            m_half.append((li.resize(m, half) > 0.5).astype(np.float32))
        prep = {"plate_ref": plate_ref, "layer": rj["card"], "layer_pattern": pattern, "frames": [frames[0], frames[-1]],
                "work_size": list(size), "plate_size": list(plate_size), "mode": mode, "windows": windows,
                "mask": {"thresh": mask_thresh, "shadow_px": shadow_px, "dilate_px": dilate_px},
                "masks_work": None, "view": [pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW], "roundtrip": rt,
                "mask_coverage": float(np.mean(m_full)), "keyframes": key_idx,
                "layers": [L["name"] for L in layers], "reference": reference, "mask_mode": mask_mode,
                "sensor": li.sensor_record(sensor)}
        masks = np.stack(m_full).astype(np.float32)
        prep["masks_work"] = masks                       # kept in memory for Return (not JSON)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return (t(np.stack(ref_full)), t(masks), t(np.stack(ref_half)), t(np.stack(m_half)), t(start[None]), prep, t(np.stack(keys)))


NODE_CLASS_MAPPINGS = {"AtlasVFXLTXPrep": AtlasVFXLTXPrep}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLTXPrep": "Atlas VFX LTX Prep"}
