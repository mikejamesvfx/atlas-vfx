"""Atlas VFX Depth Holdout: cut a keyed element where the background is nearer the camera than its feet."""
from __future__ import annotations

import json

import numpy as np

from . import relight as rl


class AtlasVFXDepthHoldout:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("matte", "occluder", "report")
    DESCRIPTION = ("Grass / props in front of an inserted element. Needs MoGe geometry of the BACKGROUND plate. The "
                   "element's distance is the background depth just under its feet (lowest rows of the matte), "
                   "smoothed over time; background pixels nearer than that by more than margin (fraction of the "
                   "distance) occlude, ramping over soft. matte = input matte x (1 - occluder). distance_scale moves "
                   "the element back (> 1) or forward (< 1) if the feet sit on something raised.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_geometry": ("MOGE_GEOMETRY",),
                "matte": ("MASK",),
                "margin": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.5, "step": 0.005}),
                "soft": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.005}),
                "foot_rows": ("INT", {"default": 6, "min": 1, "max": 200}),
                "smooth": ("INT", {"default": 5, "min": 0, "max": 50}),
                "distance_scale": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.01}),
                "blur_px": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    def run(self, background_geometry, matte, margin, soft, foot_rows, smooth, distance_scale, blur_px):
        import cv2
        import torch

        if "depth" not in background_geometry:
            raise ValueError("background_geometry has no depth")
        d = background_geometry["depth"].float().cpu().numpy()
        valid = background_geometry["mask"].float().cpu().numpy() if "mask" in background_geometry else None
        al = matte.float().cpu().numpy()
        if len(d) == 1 and len(al) > 1:
            d = np.repeat(d, len(al), 0)
            valid = None if valid is None else np.repeat(valid, len(al), 0)
        if len(d) != len(al):
            raise ValueError(f"background geometry has {len(d)} frames, matte {len(al)}")
        occ, dist = rl.depth_holdout(d, al, valid, margin, soft, foot_rows, smooth, distance_scale)
        if blur_px > 0:
            occ = np.stack([cv2.GaussianBlur(o, (0, 0), blur_px) for o in occ])
        out = al * (1 - occ)
        rep = {"frames": len(al), "element_distance_first_last": [dist[0], dist[-1]], "margin": margin,
               "occluded_fraction_of_matte": round(float((al * occ).sum() / max(al.sum(), 1e-6)), 4)}
        return (torch.from_numpy(out.astype(np.float32)), torch.from_numpy(occ.astype(np.float32)), json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXDepthHoldout": AtlasVFXDepthHoldout}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXDepthHoldout": "Atlas VFX Depth Holdout"}
