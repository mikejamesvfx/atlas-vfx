"""Atlas VFX Plate Lock Screen Key: quick colour-difference matte off the plate's blue or green screen."""
from __future__ import annotations

from . import _comfy
from .core import key


class AtlasVFXPlateLockScreenKey:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("matte",)
    DESCRIPTION = ("Foreground matte (1 = performer) from the screen. It's the fallback so the lock runs without a "
                   "pull; a proper Keylight/IBK matte from Nuke is better, load it as a MASK instead.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "screen": (["blue", "green"], {"default": "blue"}),
            "low": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                              "tooltip": "screen-difference below this is solid foreground"}),
            "high": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.005,
                               "tooltip": "screen-difference above this is clean screen"}),
        }}

    def run(self, plate, screen, low, high):
        import numpy as np
        frames = _comfy.image_np(plate)
        return (_comfy.to_mask(np.stack([key.screen_key(f, screen, low, high) for f in frames])),)


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockScreenKey": AtlasVFXPlateLockScreenKey}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockScreenKey": "Atlas VFX Plate Lock Screen Key"}
