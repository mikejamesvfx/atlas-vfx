"""Atlas VFX Plate Lock QC: measure the raw AI take against the plate before fixing anything."""
from __future__ import annotations

import json

from . import _comfy
from .core import qc


def summary(rep: dict) -> str:
    lines = []
    for name, r in rep["people"].items():
        lines.append(f"{name}: IoU {r['mean_iou']:.3f} (min {r['min_iou']:.3f}), edge {r['mean_edge_px']:.2f}px "
                     f"(max {r['max_edge_px']:.2f}), lag {r['lag_frames']:+.0f}f")
    lines.append("needs: " + (", ".join(rep["needs"]) or "nothing, already locked"))
    lines += ["  " + s for s in rep["reasons"]]
    return "\n".join(lines)


class AtlasVFXPlateLockQC:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "report_json")
    OUTPUT_NODE = True
    DESCRIPTION = ("Compares each performer's AI mask with their plate mask frame for frame: overlap, edge distance "
                   "in pixels, and the motion lag. Says which stages the shot needs (retime, people warp, key "
                   "snap). Run it first; the lock reports the same numbers again afterwards.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"people": ("ATLAS_VFX_PLATELOCK_PEOPLE",)}}

    def run(self, people):
        import numpy as np
        reps = {}
        for p in people:
            n = p.plate_masks.shape[0]
            am = p.ai_masks
            if am.shape[1:] != p.plate_masks.shape[1:]:
                am = _comfy.resize_stack(am, p.plate_masks.shape[2], p.plate_masks.shape[1])
            reps[p.name] = qc.person_report(p.plate_masks, am[np.clip(np.arange(n), 0, am.shape[0] - 1)])
        rep = qc.shot_report(reps)
        text = summary(rep)
        return {"ui": {"text": [text]}, "result": (text, json.dumps(rep, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockQC": AtlasVFXPlateLockQC}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockQC": "Atlas VFX Plate Lock QC"}
