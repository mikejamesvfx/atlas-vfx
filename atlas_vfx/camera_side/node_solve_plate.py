"""Atlas VFX Solve Plate (catalogue 4): a moving plate -> shot camera with focal and lens distortion solved, plus the
3D points (a COLMAP model the points consumers read through source.path). Wraps camera_side/plate_solve.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import camera_io as cio
from . import plate as pl
from . import plate_solve as ps
from . import tracks2d
from .sensors import PRESETS, preset_for_body, sensor_block

log = logging.getLogger("atlas_vfx")
LENS_MODELS = {"k1 k2 (RADIAL)": "RADIAL", "k1 (SIMPLE_RADIAL)": "SIMPLE_RADIAL", "k1 k2 p1 p2 (OPENCV)": "OPENCV"}


class AtlasVFXSolvePlate:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_TRACKED_CAMERA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("shot_camera", "path", "report", "preview")
    DESCRIPTION = ("3D camera tracker. Tracking proxies of the plate (log luma + CLAHE), SIFT features, sequential "
                   "matching, incremental SfM (pycolmap) with the focal and the lens distortion solved, frames the "
                   "first pass dropped mapped onto the model again, points that moved (high reprojection) dropped and "
                   "the whole solve re-adjusted. Masks (Solve Mask, any MASK batch, 1 = moving) keep features off "
                   "people, cars, trees. Tracks (Track 2D) add points the artist chose. Output: unscaled camera (run "
                   "Measure & Scale for metres), the longest run of registered frames, a report with a low-parallax "
                   "warning when the focal is weakly determined (a nodal pan), and a preview: green points under "
                   "0.5 px, yellow to 2 px, red beyond. For a rolling-shutter plate chain RS Solve after it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "sensor_preset": (["auto", *PRESETS, "custom"], {"default": "auto"}),
                "sensor_width_mm": ("FLOAT", {"default": 23.5, "min": 1.0, "max": 100.0, "step": 0.01}),
                "sensor_height_mm": ("FLOAT", {"default": 12.39, "min": 1.0, "max": 100.0, "step": 0.01}),
                "focal_guess_mm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01,
                                             "tooltip": "starting focal, e.g. the zoom mark; 0 = let the solver "
                                                        "guess. Solved either way unless solve_focal is off"}),
                "solve_focal": ("BOOLEAN", {"default": True}),
                "solve_distortion": ("BOOLEAN", {"default": True}),
                "lens_model": (list(LENS_MODELS), {"default": "k1 k2 (RADIAL)"}),
                "proxy_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 2,
                                        "tooltip": "the solver's image width (the DSCF3000 solves used 1024)"}),
                "max_features": ("INT", {"default": 8192, "min": 500, "max": 65536}),
                "overlap": ("INT", {"default": 15, "min": 2, "max": 200,
                                    "tooltip": "each frame is matched to this many neighbours (quadratic beyond)"}),
                "reject_px": ("FLOAT", {"default": 2.0, "min": 0.3, "max": 20.0, "step": 0.1,
                                        "tooltip": "drop points whose worst observation is further off than this "
                                                   "(proxy px): things that moved"}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "one per plate frame, 1 = moving (no features)"}),
                "mask_dilate_px": ("INT", {"default": 4, "min": 0, "max": 64}),
                "tracks": ("ATLAS_TRACKS_2D",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
            },
        }

    def run(self, plate_ref, sensor_preset, sensor_width_mm, sensor_height_mm, focal_guess_mm, solve_focal,
            solve_distortion, lens_model, proxy_width, max_features, overlap, reject_px, name, out_dir,
            masks=None, mask_dilate_px=4, tracks=None, seed=0):
        import numpy as np
        import torch

        if sensor_preset == "auto":
            sensor_preset = preset_for_body(plate_ref.get("camera_body"), plate_ref["width"], plate_ref["height"])
            if sensor_preset is None:
                raise ValueError(f"Solve Plate: no sensor preset for {plate_ref.get('camera_body')} at "
                                 f"{plate_ref['width']}x{plate_ref['height']}; pick one or use custom")
        if lens_model not in LENS_MODELS:
            raise ValueError(f"Solve Plate: lens_model {lens_model!r}")
        sensor = sensor_block(sensor_preset, sensor_width_mm, sensor_height_mm)
        first, last = plate_ref["first"], plate_ref["last"]
        stem = Path(plate_ref["path"]).stem
        name = pl.safe_name(name.strip() or f"{stem}_solve_{first:04d}-{last:04d}")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / stem / "solve" / name, plate_path=plate_ref["path"])
        proxy_dir, _ = pl.write_tracking_proxies(plate_ref, out / "proxy", width=proxy_width)
        W, _H = pl.proxy_size(plate_ref, proxy_width)
        names = [f"plate_{i:04d}.png" for i in range(first, last + 1)]
        mask_list = None
        if masks is not None:
            m = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            if m.ndim == 2:
                m = m[None]
            if len(m) != len(names):
                raise ValueError(f"Solve Plate: {len(m)} masks for {len(names)} plate frames ({first}-{last})")
            mask_list = list(m)
        if tracks is not None:
            tracks2d.validate(tracks)
        opts = ps.SolveOptions(camera_model=LENS_MODELS[lens_model],
                               focal_px=focal_guess_mm / sensor["width_mm"] * W if focal_guess_mm > 0 else None,
                               refine_focal=solve_focal, refine_distortion=solve_distortion,
                               max_image_size=proxy_width, max_features=max_features, overlap=overlap,
                               reject_px=reject_px, seed=seed)
        res = ps.solve_folder(proxy_dir, names, out, opts, masks=mask_list, mask_dilate_px=mask_dilate_px,
                              tracks=tracks, log=lambda s: log.info("atlas-vfx Solve Plate: %s", s))
        fps = plate_ref["fps"][0] / plate_ref["fps"][1]
        off = plate_ref["frame_offset"]
        cam = cio.from_colmap(res.model_dir, plate=cio.plate_block(plate_ref), sensor=sensor, frame_offset=off,
                              name=name, frames=(res.run[0] + off, res.run[1] + off), fps=fps)
        cam.data.setdefault("extensions", {})["atlas_vfx_solve_plate"] = {
            "report": str(out / "solve_report.json"), "registered": res.report["registered"],
            "of": res.report["of"], "missing": res.report["missing"], "warnings": res.report["warnings"],
            "degeneracy": res.report["degeneracy"], "focal_px_proxy": res.report["camera"]["focal_px"],
            "proxy_dir": str(proxy_dir)}
        path = cio.sc.save(cam, out / f"{name}.shot_camera.json")
        pl.write_sidecar(out / f"{name}.shot_camera.sidecar.json", "AtlasVFXSolvePlate",
                         inputs={"plate_ref": plate_ref, "sensor": sensor, "options": res.report["options"],
                                 "masked": mask_list is not None, "mask_dilate_px": mask_dilate_px,
                                 "tracks": len((tracks or {}).get("tracks", {}))},
                         outputs={"path": str(path), "model": str(res.model_dir), "report": res.report})
        k = cam.intrinsics(cam.frames[0])
        lens = cam.data.get("lens", {})
        head = (f"{name}: {res.report['registered']}/{res.report['of']} frames, camera {cam.frames[0]}-"
                f"{cam.frames[-1]}, {res.report['points3D']} points, {res.report['mean_reproj_px']:.3f} px at "
                f"{W} wide, focal {k.focal_mm:.3f} mm, lens {lens.get('distortion_model')} "
                f"{[round(v, 5) for v in lens.get('distortion_params', [])]}")
        text = [head] + [f"WARNING: {w}" for w in res.report["warnings"]]
        prev = ps.preview(res.model_dir, proxy_dir)
        if prev:
            h = min(p.shape[0] for p in prev)
            preview = torch.from_numpy(np.stack([p[:h] for p in prev]).astype(np.float32) / 255.0)
        else:
            preview = torch.zeros((1, 64, 64, 3))
        return {"ui": {"text": text},
                "result": (cam.data, str(path), json.dumps(res.report, indent=1, default=str), preview)}


NODE_CLASS_MAPPINGS = {"AtlasVFXSolvePlate": AtlasVFXSolvePlate}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSolvePlate": "Atlas VFX Solve Plate"}
