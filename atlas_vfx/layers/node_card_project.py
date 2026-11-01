"""Atlas VFX Card Project (catalogue 14): a ATLAS_VFX_CARD rendered through the shot camera for every plate frame ->
premultiplied ACEScg RGBA EXR sequence (Nuke frame numbers) + a comp-over-plate preview."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import camera_checks as cc
from atlas_vfx.camera_side import clean_space as cs
from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import card_project as cp
from . import ltx_integration as li
from . import holdout as ho


class AtlasVFXCardProject:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("exr_folder", "preview")
    DESCRIPTION = ("Renders a card (Foreground Element, or any ATLAS_VFX_CARD) through the shot camera for the plate's frames: "
                   "<name>.<frame>.exr, RGBA half, premultiplied ACEScg, plate resolution x scale. Preview = card over "
                   "the plate through the ACES 2.0 SDR view.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "card": ("ATLAS_VFX_CARD",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 100}),
                "preview_every": ("INT", {"default": 24, "min": 1, "max": 1000}),
                "preview_width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 2}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "holdout_depth": ("STRING", {"default": "", "tooltip": "Plate Depth folder (depth.<frame>.exr): hide the "
                                                                       "card wherever the plate is nearer the camera"}),
                "holdout_margin_m": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.05,
                                               "tooltip": "plate must be this much nearer before the card is hidden"}),
                "holdout_soft_m": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.05,
                                             "tooltip": "depth range over which the holdout ramps in"}),
                "sensor_lens_folder": ("STRING", {"default": "", "tooltip": "Sensor Lens Maps of this camera: also "
                                                  "write the card redistorted onto the plate as shot (plate/), next "
                                                  "to the clean (pinhole) frames. Nuke Builder picks one"}),
            },
        }

    def run(self, plate_ref, shot_camera, card, scale, frame_step, preview_every, preview_width, out_dir,
            holdout_depth="", holdout_margin_m=0.6, holdout_soft_m=0.1, sensor_lens_folder=""):
        import cv2
        import torch

        cam = sc.ShotCamera(shot_camera)
        cc.check_frame_offset(cam, plate_ref, "Card Project")
        if pl.clean_path(holdout_depth):
            cc.require_metric(cam, "Card Project", "holdout margin and softness (metres)")
        lensed = cc.has_lens(cam)
        sensor = li.sensor_maps("", sensor_lens_folder) if pl.clean_path(sensor_lens_folder) else None
        if sensor is not None and scale != 1:
            raise ValueError("Card Project: the redistorted (plate/) version needs scale 1 (the lens maps are plate "
                             "size)")
        if sensor is not None and pl.clean_path(holdout_depth):
            raise ValueError("Card Project: a depth holdout (Plate Depth is of the distorted plate) with a lens folder "
                             "is not supported yet; run the holdout without sensor_lens_folder")
        # pinhole layers are rendered through the clean camera: no lens, on the clean frame (plate + overscan)
        rcam = cs.clean_camera(cam, sensor.lens_folder) if sensor is not None else cam
        cache = {}

        def texture(path):
            if path not in cache:
                cache.clear()                     # sequences: one frame at a time
                cache[path] = exr_io.read_rgba(path)[0]
            return cache[path]
        off = plate_ref["frame_offset"]
        f0 = max(plate_ref["nuke_first"], cam.frames[0])
        f1 = min(plate_ref["nuke_last"], cam.frames[-1])
        name = pl.safe_name(card["name"], "card name")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / Path(plate_ref["path"]).stem / "layers" / f"{cam.name}_{name}")
        written, coverage, previews = [], [], []
        hd = pl.clean_path(holdout_depth)
        depth = ho.depth_frames(hd) if hd else {}
        held = {}
        disp_proc = pl._cpu(plate_ref["ocio_config"], "ACEScg", None, pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW)
        for n in range(f0, f1 + 1, frame_step):
            corners, tex_path = cp.card_at(card, n)
            img = cp.render(rcam, n, corners, texture(tex_path), scale=scale)
            if depth:
                if n in depth:
                    img, held[n] = ho.apply(img, cam, n, corners, depth[n], holdout_margin_m, holdout_soft_m)
                else:
                    held[n] = None                      # no depth for this frame: card left as is
            p = exr_io.write_rgba(out / f"{name}.{n:04d}.exr", img,
                                  attrs={"card": name, "camera": cam.name, "frame": n, "scale": scale,
                                         "space": "clean"})
            plate_img = img
            if sensor is not None:
                plate_img = li.apply_camera(img, sensor, n)
                exr_io.write_rgba(out / "plate" / f"{name}.{n:04d}.exr", plate_img,
                                  attrs={"card": name, "camera": cam.name, "frame": n, "space": "plate",
                                         "lens_folder": str(sensor.lens_folder)})
            written.append(p.name)
            coverage.append(float((img[..., 3] > 0.5).mean()))
            if (n - f0) % preview_every == 0:
                i, frame = next(pl.decode_rgb(plate_ref, n - off, n - off, width=preview_width))
                lin = pl.to_working(frame, plate_ref)
                small = cv2.resize(plate_img, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_AREA)
                comp = np.ascontiguousarray(cp.over(small, lin))
                disp_proc.applyRGB(comp)
                previews.append(np.clip(comp, 0, 1).astype(np.float32))
        report = {"card": name, "camera": cam.name, "frames": [f0, f1], "frame_step": frame_step, "scale": scale,
                  "files": len(written), "pattern": f"{name}.%04d.exr", "corners_m": card["corners"],
                  "animated": bool(card.get("corners_by_frame") or card.get("texture_pattern")),
                  "coverage_max": max(coverage), "frames_with_card": int(sum(c > 0 for c in coverage)),
                  "colorspace": "ACEScg (premultiplied)",
                  # rendered pinhole: a clean-space layer. The camera's lens says what the plate needs to match it
                  "space": "clean", "lens": cam.data.get("lens") or {"distortion_model": "none"},
                  "scale_method": (cam.data.get("scale") or {}).get("method"),
                  # clean = pinhole (this folder); plate = on the plate as shot. With no lens they are the same frames
                  "sizes": {"clean": [rcam.data["plate"]["width"], rcam.data["plate"]["height"]],
                            "plate": [cam.data["plate"]["width"], cam.data["plate"]["height"]]},
                  "versions": ({"clean": f"{name}.%04d.exr", "plate": f"plate/{name}.%04d.exr"} if sensor is not None
                               else {"clean": f"{name}.%04d.exr"} if lensed
                               else {"clean": f"{name}.%04d.exr", "plate": f"{name}.%04d.exr"})}
        if depth:
            vals = [v for v in held.values() if v is not None]
            report["holdout"] = {"depth": hd, "margin_m": holdout_margin_m, "soft_m": holdout_soft_m,
                                 "frames_without_depth": sorted(k for k, v in held.items() if v is None),
                                 "removed_pct_median": round(100 * float(np.median(vals)), 1) if vals else None,
                                 "frames_fully_hidden": int(sum(v > 0.999 for v in vals))}
        (out / f"{name}.render.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXCardProject",
                         inputs={"plate_ref": plate_ref, "camera": cam.name, "card": card, "scale": scale,
                                 "frame_step": frame_step},
                         outputs=report)
        return (str(out), torch.from_numpy(np.stack(previews)))


NODE_CLASS_MAPPINGS = {"AtlasVFXCardProject": AtlasVFXCardProject}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXCardProject": "Atlas VFX Card Project"}
