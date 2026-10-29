"""Atlas VFX Normals: camera-space normals from MoGe geometry, smoothed in time and inside the matte."""
from __future__ import annotations

import json

import numpy as np

from . import relight as rl


class AtlasVFXNormals:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("normals", "report")
    DESCRIPTION = ("Normals for Relight from a MOGE_GEOMETRY packet (core MoGe Inference; MoGe-3 or MoGe-2-normal). "
                   "Uses the predicted normals when present, else derives them from the point map. Converts OpenCV to "
                   "OpenGL camera space (x right, y up, z toward camera), averages +-temporal frames and blurs inside "
                   "the matte. Output is a normal map encoded 0-1 (n * 0.5 + 0.5), the same as MoGe Render normal_opengl.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_geometry": ("MOGE_GEOMETRY",),
                "temporal": ("INT", {"default": 1, "min": 0, "max": 8,
                                     "tooltip": "average over +-N frames (MoGe normals flicker per frame)"}),
                "sigma": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1,
                                    "tooltip": "spatial blur in pixels, alpha-weighted when a matte is given"}),
            },
            "optional": {"matte": ("MASK",)},
        }

    def run(self, moge_geometry, temporal, sigma, matte=None):
        import cv2
        import torch

        if "normal" in moge_geometry:
            n = moge_geometry["normal"].float().cpu().numpy()
            src = "predicted"
        elif "points" in moge_geometry:
            n = rl.normals_from_points(moge_geometry["points"].float().cpu().numpy())
            src = "from points"
        else:
            raise ValueError("moge_geometry has neither normals nor points")
        n = rl.opencv_to_opengl(n)
        al = None
        if matte is not None:
            al = matte.float().cpu().numpy()
            if al.shape[1:] != n.shape[1:3]:
                al = np.stack([cv2.resize(a, (n.shape[2], n.shape[1]), interpolation=cv2.INTER_LINEAR) for a in al])
            if len(al) != len(n):
                raise ValueError(f"matte has {len(al)} frames, geometry {len(n)}")
        n = rl.smooth_normals(n, al, temporal, sigma)
        rep = {"frames": int(n.shape[0]), "size": [int(n.shape[2]), int(n.shape[1])], "source": src,
               "temporal": temporal, "sigma": sigma, "matte_weighted": al is not None}
        return (torch.from_numpy(np.ascontiguousarray(n * 0.5 + 0.5)), json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXNormals": AtlasVFXNormals}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXNormals": "Atlas VFX Normals"}
