"""Atlas VFX Relight: normal-based relight of a keyed element with a light rig from the background plate."""
from __future__ import annotations

import json

import numpy as np

from . import relight as rl


def _f(d, lo, hi, step=0.05, tip=""):
    return ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step, "tooltip": tip})


class AtlasVFXRelight:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("relit", "ambient_pass", "key_pass", "report")
    DESCRIPTION = ("Relights a keyed element (flat studio light) into the background's light. Albedo = element / flat "
                   "studio key; new light = sky-ground ambient + sun. A sun behind the subject lands as a rim from the "
                   "normals (turned to the sun side, glancing) plus a matte-edge rim that catches hair. rim_max caps "
                   "the key at that many times the ambient so faces turned to the sun do not blow out. Exposure is "
                   "matched to the background around the element and smoothed over time. Outputs are straight (not "
                   "premultiplied), in the input encoding; passes are for rebalancing in comp. 2.5D: no cast shadows, "
                   "no specular, no subsurface; run the harmonise pass after it. Optional sun_rig (Atlas VFX "
                   "Physical Sky's light_rig) replaces the light rig's sun direction and colour (the plate's 'auto' "
                   "estimate) and keeps its ambient; sun_strength_from 'physical sky' then sets the key to the "
                   "physical sun-to-sky ratio times the plate sky, with sun_gain as a trim (1 = physical). Without "
                   "sun_rig nothing changes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "element": ("IMAGE",),
                "matte": ("MASK",),
                "normals": ("IMAGE", {"tooltip": "Atlas VFX Normals output (OpenGL, encoded 0-1)"}),
                "light_rig": ("ATLAS_VFX_LIGHT_RIG",),
                "encoding": (["srgb", "linear"], {"default": "srgb"}),
                "sun_gain": _f(4.0, 0.0, 50.0, 0.1),
                "ambient_gain": _f(1.0, 0.0, 10.0),
                "studio_flat": _f(0.45, 0.0, 1.0, 0.05, "how flat the source studio light is (1 = fully flat)"),
                "facing_power": _f(2.0, 0.1, 10.0, 0.1),
                "fresnel_power": _f(4.0, 0.1, 10.0, 0.1),
                "edge_rim": _f(1.5, 0.0, 5.0, 0.1, "matte-edge rim strength (hair)"),
                "edge_px": _f(8.0, 0.0, 64.0, 0.5, "matte-edge rim width"),
                "rim_max": _f(2.0, 0.0, 20.0, 0.1, "cap key at N x ambient luminance (0 = off)"),
                "exposure_match": ("BOOLEAN", {"default": True}),
                "exposure_bias": _f(0.9, 0.05, 4.0, 0.05, "element / background-ring luminance after matching"),
                "exposure_smooth": ("INT", {"default": 25, "min": 1, "max": 200,
                                            "tooltip": "frames in the exposure moving average (log)"}),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "for exposure matching (resized to the element)"}),
                "sun_rig": ("ATLAS_VFX_LIGHT_RIG", {"tooltip": "Physical Sky light_rig: replaces the sun of light_rig"}),
                "sun_strength_from": (["sun_gain", "physical sky"], {
                    "default": "sun_gain",
                    "tooltip": "physical sky: key = sun_rig sun_strength x plate sky luminance x sun_gain (trim)"}),
            },
        }

    def run(self, element, matte, normals, light_rig, encoding, sun_gain, ambient_gain, studio_flat, facing_power,
            fresnel_power, edge_rim, edge_px, rim_max, exposure_match, exposure_bias, exposure_smooth, background=None,
            sun_rig=None, sun_strength_from="sun_gain"):
        import cv2
        import torch

        sun_source = "light_rig"
        if sun_rig is not None:
            if sun_strength_from == "physical sky":
                sun_gain = rl.physical_sun_gain(light_rig, sun_rig, sun_gain)
            light_rig = rl.override_sun(light_rig, sun_rig)
            sun_source = f"sun_rig ({sun_strength_from})"
        elif sun_strength_from == "physical sky":
            raise ValueError("sun_strength_from 'physical sky' needs a sun_rig (Atlas VFX Physical Sky light_rig)")

        fg = element[..., :3].float().cpu().numpy()
        T, H, W = fg.shape[:3]

        def fit(x, interp):
            x = np.asarray(x, np.float32)
            if x.shape[1:3] == (H, W):
                return x
            return np.stack([cv2.resize(a, (W, H), interpolation=interp) for a in x])

        al = fit(matte.float().cpu().numpy(), cv2.INTER_LINEAR)
        n = fit(normals[..., :3].float().cpu().numpy(), cv2.INTER_LINEAR) * 2 - 1
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
        for name, x in (("matte", al), ("normals", n)):
            if len(x) != T:
                raise ValueError(f"{name} has {len(x)} frames, element {T}")
        to_lin = rl.srgb_to_lin if encoding == "srgb" else (lambda x: x)
        enc = rl.lin_to_srgb if encoding == "srgb" else (lambda x: x)
        bg = None if background is None else to_lin(fit(background[..., :3].float().cpu().numpy(), cv2.INTER_AREA))
        relit, passes, gains = rl.relight(to_lin(fg), al, n, light_rig, bg, sun_gain, ambient_gain,
                                          studio_flat=studio_flat, facing_power=facing_power,
                                          fresnel_power=fresnel_power, edge_rim=edge_rim, edge_px=edge_px,
                                          rim_max=rim_max, exposure_match=exposure_match and bg is not None,
                                          exposure_bias=exposure_bias, exposure_smooth=exposure_smooth)
        out = [torch.from_numpy(np.ascontiguousarray(enc(x))) for x in (relit, passes["ambient"], passes["key"])]
        rep = {"frames": T, "exposure_gain_first_last": [gains[0], gains[-1]], "encoding": encoding,
               "sun_gain": sun_gain, "rim_max": rim_max}
        if sun_rig is not None:
            rep.update(sun_source=sun_source, sun_dir_first=light_rig["sun_dir"][0],
                       sun_colour=light_rig["sun_colour"], sun_strength=sun_rig.get("sun_strength"))
        return (*out, json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXRelight": AtlasVFXRelight}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXRelight": "Atlas VFX Relight"}
