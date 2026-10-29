"""Atlas VFX Plate Lock Person: one performer's masks (plate and AI) plus optional correspondences."""
from __future__ import annotations

import os

from . import _comfy, adapters
from .core.pipeline import PersonInput


class AtlasVFXPlateLockPerson:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_VFX_PLATELOCK_PEOPLE",)
    RETURN_NAMES = ("people",)
    DESCRIPTION = ("Adds a performer to the lock. Feed the SAM masks of the same person from the plate and from the "
                   "AI take. Chain one node per person through 'people'. Correspondences are optional: a body "
                   "solve (.npz with xy (N,V,2) and vis (N,V)) for each take, or a matcher dump. Without them "
                   "the warp is driven by the silhouette only, which moves interiors with their edges.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "person_a"}),
                "plate_masks": ("MASK",),
                "ai_masks": ("MASK",),
                "depth": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                    "tooltip": "distance from camera; nearer performers stack on top"}),
            },
            "optional": {
                "people": ("ATLAS_VFX_PLATELOCK_PEOPLE",),
                "body_plate_npz": ("STRING", {"default": ""}),
                "body_ai_npz": ("STRING", {"default": ""}),
                "correspondences_npz": ("STRING", {"default": ""}),
            },
        }

    def run(self, name, plate_masks, ai_masks, depth, people=None, body_plate_npz="", body_ai_npz="",
            correspondences_npz=""):
        fns = []
        bp, ba = _comfy.clean_path(body_plate_npz), _comfy.clean_path(body_ai_npz)
        if bp and ba:
            pxy, pvis = adapters.load_body_npz(bp)
            axy, avis = adapters.load_body_npz(ba)
            fns.append(adapters.corr_from_body(pxy, pvis, axy, avis))
        cp = _comfy.clean_path(correspondences_npz)
        if cp:
            if not os.path.exists(cp):
                raise FileNotFoundError(cp)
            fns.append(adapters.load_correspondences_npz(cp))
        corr = adapters.corr_concat(*fns) if fns else None
        p = PersonInput(name.strip() or f"person_{len(people or [])}", _comfy.mask_np(plate_masks),
                        _comfy.mask_np(ai_masks), float(depth), corr)
        out = list(people or [])
        if any(q.name == p.name for q in out):
            raise ValueError(f"two performers are both called {p.name!r}")
        out.append(p)
        return (out,)


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockPerson": AtlasVFXPlateLockPerson}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockPerson": "Atlas VFX Plate Lock Person"}
