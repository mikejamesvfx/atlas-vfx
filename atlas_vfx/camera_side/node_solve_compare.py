"""Atlas VFX Solve Compare (catalogue 6): similarity fit of two cameras of the same shot, agreement report."""
from __future__ import annotations

import json

from . import compare as cmp


class AtlasVFXSolveCompare:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "summary", "residual_pct_of_path", "view_angle_median_deg", "scale_b_per_a")
    DESCRIPTION = ("Fits B ~ s R A + t on camera centres over the frames both have; reports scale, centre residual "
                   "(% of B's path) and view-direction angle after the fit.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"camera_a": ("ATLAS_TRACKED_CAMERA",), "camera_b": ("ATLAS_TRACKED_CAMERA",)}}

    def run(self, camera_a, camera_b):
        r = cmp.compare(camera_a, camera_b)
        line = cmp.summary_line(r)
        return {"ui": {"text": [line]},
                "result": (json.dumps(r, indent=1), line, r["residual_pct_of_path"], r["view_angle_deg"]["median"],
                           r["scale_b_per_a"])}


NODE_CLASS_MAPPINGS = {"AtlasVFXSolveCompare": AtlasVFXSolveCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSolveCompare": "Atlas VFX Solve Compare"}
