"""Atlas VFX Plate Depth (catalogue 9): MoGe (ComfyUI core) on display proxies -> depth scaled to the shot camera
against the solve's points -> float EXR per frame (Z + A) + per-frame scale report."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import measure as ms
from . import plate as pl
from . import plate_depth as pd

log = logging.getLogger("atlas_vfx")


class AtlasVFXPlateDepth:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("depth_folder", "preview", "report")
    DESCRIPTION = ("MoGe depth per frame on the display proxy, with the shot camera's horizontal FOV, scaled per "
                   "frame to the shot camera by a robust median against the solve's points it saw. Writes "
                   "depth.<frame>.exr (Z = z depth in camera units, NOT 1/z; A = valid) and depth_scales.json.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_model": ("MOGE_MODEL",),
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "solve_points": ("STRING", {"default": "", "tooltip": "COLMAP folder; empty = the camera's solve"}),
                "proxy_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 2}),
                "resolution_level": ("INT", {"default": 9, "min": 0, "max": 9}),
                "refine_steps": ("INT", {"default": 3, "min": 0, "max": 10}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 100}),
                "preview_every": ("INT", {"default": 24, "min": 1, "max": 1000}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, moge_model, plate_ref, shot_camera, solve_points, proxy_width, resolution_level, refine_steps,
            frame_step, preview_every, out_dir):
        import torch

        cam = sc.ShotCamera(shot_camera)
        pts, note = ms.points_for_camera(cam, solve_points)
        if pts is None:
            raise ValueError("Plate Depth needs the solve's 3D points to scale MoGe depth (camera from a COLMAP "
                             "solve, or give solve_points)")
        off = plate_ref["frame_offset"]
        f0 = max(plate_ref["first"], cam.frames[0] - off)
        f1 = min(plate_ref["last"], cam.frames[-1] - off)
        if f1 < f0:
            raise ValueError("camera and plate frames do not overlap")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / Path(plate_ref["path"]).stem / "depth" / cam.name)
        units = "m" if cam.data["scale"]["method"] != "unscaled" else "solve units"

        depths, masks, stats, previews = {}, {}, [], []
        for i, img in pl.decode_rgb(plate_ref, f0, f1, width=proxy_width, step=frame_step):
            n = i + off
            disp = pl.to_display(img, plate_ref)
            bchw = torch.from_numpy(disp).movedim(-1, 0)[None].contiguous()
            with torch.no_grad():
                r = moge_model.infer(bchw, resolution_level=resolution_level, fov_x=pd.hfov_deg(cam, n),
                                     force_projection=True, apply_mask=False, refine_steps=refine_steps)
            d = r["depth"][0].float().cpu().numpy()
            m = r["mask"][0].cpu().numpy().astype(bool) & np.isfinite(d) & (d > 0)
            depths[n], masks[n] = d, m
            stats.append(pd.frame_scale(cam, n, np.where(m, d, 0.0), pts))
        stats = pd.fill_scales(stats)
        for k, s in enumerate(stats):
            n = s["frame"]
            z = depths[n] * s["scale"]
            pd.write_exr(out / f"depth.{n:04d}.exr", z, masks[n],
                         {"camera": cam.name, "frame": n, "units": units, "scale": s["scale"],
                          "kind": "z depth (not 1/z)", "model": "MoGe (ComfyUI core)"})
            if k % max(1, preview_every // frame_step) == 0:
                inv = np.where(masks[n], 1.0 / np.maximum(z, 1e-6), 0.0)
                lo, hi = np.percentile(inv[masks[n]], [1, 99]) if masks[n].any() else (0, 1)
                g = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
                previews.append(np.repeat(g[..., None], 3, -1))
        scales = np.array([s["scale"] for s in stats])
        report = {"camera": cam.name, "frames": [stats[0]["frame"], stats[-1]["frame"]], "frame_step": frame_step,
                  "units": units, "solve_points": note, "proxy_width": proxy_width,
                  "scale_median": float(np.median(scales)),
                  "scale_frame_to_frame_pct_median": float(np.median(np.abs(np.diff(scales)) / scales[1:]) * 100)
                  if len(scales) > 1 else 0.0,
                  "point_ratio_spread_pct_median": float(np.median([s.get("spread_pct", np.nan) for s in stats
                                                                     if not s.get("filled")])),
                  "filled_frames": [s["frame"] for s in stats if s.get("filled")], "per_frame": stats}
        (out / "depth_scales.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        pl.write_sidecar(out / "depth.sidecar.json", "AtlasVFXPlateDepth",
                         inputs={"plate_ref": plate_ref, "camera": cam.name, "solve_points": note,
                                 "proxy_width": proxy_width, "resolution_level": resolution_level,
                                 "refine_steps": refine_steps, "frame_step": frame_step,
                                 "model": "MOGE_MODEL from LoadMoGeModel (see the graph)"},
                         outputs={"folder": str(out), "pattern": "depth.%04d.exr (Nuke frames)",
                                  "frames": report["frames"]})
        summary = {k: report[k] for k in report if k != "per_frame"}
        return (str(out), torch.from_numpy(np.stack(previews)), json.dumps(summary, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateDepth": AtlasVFXPlateDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateDepth": "Atlas VFX Plate Depth"}
