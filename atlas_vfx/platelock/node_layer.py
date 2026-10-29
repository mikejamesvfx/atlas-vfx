"""Atlas VFX Plate Lock Layer: pull one performer's locked layer out of a lock result."""
from __future__ import annotations

from . import _comfy


class AtlasVFXPlateLockLayer:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("rgb", "alpha", "names")
    DESCRIPTION = "One performer's warped RGB and alpha from the lock (by name; empty = the first)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("ATLAS_VFX_PLATELOCK_RESULT",), "name": ("STRING", {"default": ""})}}

    def run(self, result, name):
        names = list(result.people)
        key = name.strip() or names[0]
        if key not in result.people:
            raise KeyError(f"no performer {key!r}; have {names}")
        pr = result.people[key]
        return (_comfy.to_image(pr.rgb), _comfy.to_mask(pr.alpha), ", ".join(names))


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockLayer": AtlasVFXPlateLockLayer}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockLayer": "Atlas VFX Plate Lock Layer"}
