"""Atlas VFX Light From Plate: sun direction / colour and sky-ground ambient read off a background plate."""
from __future__ import annotations

import json

import numpy as np

from . import relight as rl


class AtlasVFXLightFromPlate:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_VFX_LIGHT_RIG", "IMAGE", "STRING")
    RETURN_NAMES = ("light_rig", "preview", "report")
    DESCRIPTION = ("Builds the light rig Relight uses from the background plate: the sun is the brightest blob of the "
                   "blurred plate (tracked per frame, median-smoothed), its colour is the hue of the brightest 0.5 %, "
                   "ambient is the mean of the top third (sky) and bottom third (ground). 'manual' places the sun at "
                   "sun_u / sun_v (normalised, may be off frame). behind > 0 = sun behind the subject (rim / back "
                   "light), < 0 = in front (key). Preview marks the sun on the plate.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "encoding": (["srgb", "linear"], {"default": "srgb",
                                                  "tooltip": "srgb = display-referred plate; linear = scene-linear"}),
                "fov_x": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 170.0, "step": 0.5,
                                    "tooltip": "horizontal field of view of the plate camera (degrees)"}),
                "sun": (["auto", "manual"], {"default": "auto"}),
                "sun_u": ("FLOAT", {"default": 0.2, "min": -5.0, "max": 6.0, "step": 0.01}),
                "sun_v": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 6.0, "step": 0.01}),
                "behind": ("FLOAT", {"default": 0.6, "min": -3.0, "max": 3.0, "step": 0.05}),
                "smooth": ("INT", {"default": 5, "min": 0, "max": 50, "tooltip": "median over +-N frames"}),
            },
            "optional": {
                "sun_colour": ("STRING", {"default": "", "tooltip": "override, linear r, g, b, e.g. 1.0, 0.62, 0.28"}),
            },
        }

    def run(self, background, encoding, fov_x, sun, sun_u, sun_v, behind, smooth, sun_colour=""):
        import torch

        bg = background[..., :3].float().cpu().numpy()
        lin = rl.srgb_to_lin(bg) if encoding == "srgb" else bg
        col = [float(x) for x in sun_colour.split(",")] if sun_colour.strip() else None
        rig = rl.light_from_plate(lin, fov_x, behind, smooth, (sun_u, sun_v) if sun == "manual" else None, col)
        prev = bg.copy()
        H, W = prev.shape[1:3]
        r = max(4, W // 200)
        for t, (u, v) in enumerate(rig["sun_uv"]):
            x, y = int(np.clip(u * W, 0, W - 1)), int(np.clip(v * H, 0, H - 1))
            prev[t, max(0, y - r):y + r, max(0, x - 1):x + 2] = (1, 0, 1)
            prev[t, max(0, y - 1):y + 2, max(0, x - r):x + r] = (1, 0, 1)
        rep = {k: rig[k] for k in ("frames", "fov_x", "behind", "source", "sun_colour")}
        rep.update(sun_uv_first=rig["sun_uv"][0], sky_first=rig["sky"][0], ground_first=rig["ground"][0])
        return (rig, torch.from_numpy(prev), json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXLightFromPlate": AtlasVFXLightFromPlate}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLightFromPlate": "Atlas VFX Light From Plate"}
