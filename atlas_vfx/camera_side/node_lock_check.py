"""Atlas VFX Lock Check (catalogue 8): grid card + measured points through every frame on the display proxy -> mp4."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import lock_check as lc
from . import plate as pl


def _centre(cam: sc.ShotCamera, spec: str, points: dict):
    spec = spec.strip()
    if spec in ("", "auto"):
        f0 = -cam.cam_to_world(cam.frames[0])[:3, 2]
        flat = np.array([f0[0], 0.0, f0[2]])
        flat /= np.linalg.norm(flat)
        c = cam.position(cam.frames[0]) + 4.0 * flat
        return np.array([c[0], 0.0, c[2]]), "auto: ground, 4 m ahead of the first camera"
    if spec in points:
        return np.array(points[spec], float), f"measured point {spec}"
    try:
        xyz = np.array([float(v) for v in spec.split(",")])
        assert xyz.shape == (3,)
        return xyz, "explicit"
    except (ValueError, AssertionError):
        raise ValueError(f"card centre {spec!r}: use auto, a measured point label ({sorted(points)}) or x,y,z")


class AtlasVFXLockCheck:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("mp4", "preview")
    DESCRIPTION = ("Projects a gridded card (green) and the measured points (red) through every frame onto the "
                   "display proxy. If the card stays glued to its surface, the camera is right.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "centre": ("STRING", {"default": "auto", "tooltip": "auto | measured point label | x,y,z metres"}),
                "orientation": (["ground", "vertical"], {"default": "ground"}),
                "width_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "height_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "divisions": ("INT", {"default": 6, "min": 1, "max": 50}),
                "proxy_width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 2}),
                "preview_every": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "out_path": ("STRING", {"default": ""}),
            },
            "optional": {"measure_report": ("STRING", {"forceInput": True})},
        }

    def run(self, plate_ref, shot_camera, centre, orientation, width_m, height_m, divisions, proxy_width,
            preview_every, out_path, measure_report=None):
        import torch

        cam = sc.ShotCamera(shot_camera)
        points = {}
        if measure_report and measure_report.strip() not in ("", "None", "null"):
            points = json.loads(measure_report).get("points_m", {})
        elif "atlas_vfx_measure" in cam.data.get("extensions", {}):
            points = cam.data["extensions"]["atlas_vfx_measure"].get("points_m", {})
        c, how = _centre(cam, centre, points)
        lines = lc.card_lines(cam, c, orientation, width_m, height_m, divisions)
        out = pl.resolve_out_dir(out_path, pl.WORK_ROOT / Path(plate_ref["path"]).stem / "lock_check" /
                                 f"{cam.name}_{orientation}.mp4", plate_path=plate_ref["path"], what="out_path")
        path, kept = lc.render(plate_ref, cam, out, lines=lines, marks=points, width=proxy_width,
                               keep_every=preview_every)
        pl.write_sidecar(path.with_suffix(".sidecar.json"), "AtlasVFXLockCheck",
                         inputs={"plate_ref": plate_ref, "camera": cam.name, "centre": c.tolist(), "centre_from": how,
                                 "orientation": orientation, "size_m": [width_m, height_m], "divisions": divisions,
                                 "marks": points},
                         outputs={"mp4": str(path),
                                  "frames": [max(plate_ref["nuke_first"], cam.frames[0]),
                                             min(plate_ref["nuke_last"], cam.frames[-1])]})
        return (str(path), torch.from_numpy(np.stack(kept)))


NODE_CLASS_MAPPINGS = {"AtlasVFXLockCheck": AtlasVFXLockCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLockCheck": "Atlas VFX Lock Check"}
