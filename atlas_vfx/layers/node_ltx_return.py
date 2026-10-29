"""Atlas VFX LTX Return: LTX's integrated frames -> a premultiplied ACEScg RGBA layer at plate resolution (alpha = the
inpaint mask, feathered), named like a Card Project folder so the Nuke Builder can use it in place of the original
layer. Writes a comp preview and a sidecar with the colour round-trip report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl

from . import card_project as cp
from . import ltx_integration as li


class AtlasVFXLTXReturn:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("layer_folder", "preview")
    DESCRIPTION = ("Takes the LTX output (display-referred, work size), inverts the named view back to ACEScg, upscales "
                   "it to plate size and keeps it only inside the inpaint mask: <name>.<frame>.exr, RGBA half, "
                   "premultiplied ACEScg. Outside the mask alpha is 0, so the plate (and its lock) is untouched.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "ltx_prep": ("ATLAS_VFX_LTX_PREP",),
                "feather_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "name": ("STRING", {"default": "element_ltx"}),
                "source_model": ("STRING", {"default": "ltx-2.5-distilled + ltx-2.3-ic-lora-in-outpainting"}),
                "preview_every": ("INT", {"default": 24, "min": 1, "max": 1000}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, images, ltx_prep, feather_px, name, source_model, preview_every, out_dir):
        import cv2
        import torch

        ref = ltx_prep["plate_ref"]
        f0, f1 = ltx_prep["frames"]
        frames = list(range(f0, f1 + 1))
        ltx = images.cpu().numpy()[..., :3]
        masks = ltx_prep["masks_work"]
        if len(ltx) < len(frames):
            raise ValueError(f"LTX gave {len(ltx)} frames for {len(frames)}")
        W, H = ltx_prep["plate_size"]
        ww, wh = ltx_prep["work_size"]
        if ltx.shape[1:3] != (wh, ww):
            ltx = np.stack([li.resize(f, (ww, wh)) for f in ltx])
        _, inv = li.view_processors(ref["ocio_config"])
        fwd, _ = li.view_processors(ref["ocio_config"])
        stem = Path(ref["path"]).stem
        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / stem / "layers" / name)
        off = ref["frame_offset"]
        sens = ltx_prep.get("sensor") or {}
        sensor = li.sensor_maps(sens.get("rs_folder") or "", sens.get("lens_folder") or "")
        previews, cover = [], []
        for i, n in enumerate(frames):
            if ltx_prep.get("windows"):
                rgba = li.return_window(ltx[i], masks[i], ltx_prep["windows"][i], (W, H), inv, feather_px)
            else:
                rgba = li.return_frame(ltx[i], masks[i], (W, H), inv, feather_px)
            if sensor is not None:
                rgba = li.apply_camera(rgba, sensor, n)     # LTX worked in clean space: put the plate camera back
            exr_io.write_rgba(out / f"{name}.{n:04d}.exr", rgba,
                              attrs={"layer": name, "frame": n, "source": "LTX integration pass",
                                     "replaces": ltx_prep["layer"]})
            cover.append(float((rgba[..., 3] > 0.5).mean()))
            if i % preview_every == 0:
                _, img = next(pl.decode_rgb(ref, n - off, n - off, width=1024))
                lin = pl.to_working(img, ref)
                small = cv2.resize(rgba, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_AREA)
                previews.append(li.to_view(cp.over(small, lin), fwd))
        report = {"card": name, "frames": [f0, f1], "pattern": f"{name}.%04d.exr", "colorspace": "ACEScg (premultiplied)",
                  "replaces_layer": ltx_prep["layer"], "work_size": [ww, wh], "plate_size": [W, H],
                  "roundtrip": ltx_prep["roundtrip"], "mask": ltx_prep["mask"], "feather_px": feather_px,
                  "coverage_max": max(cover), "source_model": source_model, "mode": ltx_prep.get("mode", "full"),
                  "window_scale": ([round(w[3] / wh, 3) for w in ltx_prep["windows"]] if ltx_prep.get("windows") else None),
                  "sensor": ({**sens, "applied": "vignette -> CA + distortion -> RS"} if sensor is not None else None),
                  "note": "inside the mask: LTX output upscaled from work size; outside: alpha 0 (plate untouched)"}
        (out / f"{name}.render.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXLTXReturn",
                         inputs={k: v for k, v in ltx_prep.items() if k != "masks_work"}, outputs=report)
        return {"ui": {"text": [json.dumps({k: report[k] for k in ("frames", "roundtrip", "coverage_max")})]},
                "result": (str(out), torch.from_numpy(np.stack(previews)))}


NODE_CLASS_MAPPINGS = {"AtlasVFXLTXReturn": AtlasVFXLTXReturn}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLTXReturn": "Atlas VFX LTX Return"}
