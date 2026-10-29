"""Atlas VFX Plate Lock Write: STMaps, layers, shot.json and a Nuke script for a finished lock."""
from __future__ import annotations

import hashlib
import json
import os
import platform

from ..camera_side import plate as pl
from . import _comfy
from .core import deliver

DEFAULT_OUT = str(pl.WORK_ROOT / "platelock" / "shot")


class AtlasVFXPlateLockWrite:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_dir",)
    OUTPUT_NODE = True
    DESCRIPTION = ("Writes the lock for Nuke: re-shot background and per-performer RGBA EXRs, STMaps into the "
                   "original AI frames (so comp can redo or dial back any correction), shot.json with the retime "
                   "curves, gauge and QC, and platelock_lock.nk with Reads and per-performer TimeWarps. "
                   "Outputs go outside the repo.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("ATLAS_VFX_PLATELOCK_RESULT",),
            "out_dir": ("STRING", {"default": DEFAULT_OUT}),
            "first_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000}),
            "ai_first_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000}),
            "ai_path": ("STRING", {"default": "", "tooltip": "the AI take's EXR sequence, for the Nuke Reads"}),
            "write_comp": ("BOOLEAN", {"default": True}),
        }}

    def run(self, result, out_dir, first_frame, ai_first_frame, ai_path, write_comp):
        od = str(pl.resolve_out_dir(out_dir, DEFAULT_OUT))      # under the work root, never the repo
        from atlas_vfx import __version__
        extra = {"sidecar": {
            "node": "AtlasVFXPlateLockWrite", "atlas_vfx": __version__, "python": platform.python_version(),
            "inputs": getattr(result, "inputs", {}),
            "ai_path": _comfy.clean_path(ai_path),
            "core_hash": _core_hash(),
        }}
        deliver.write_lock(result, od, first_frame=first_frame, ai_first_frame=ai_first_frame,
                           ai_path=_comfy.clean_path(ai_path).replace("\\", "/") or "ai/ai.####.exr",
                           write_comp=write_comp, extra=extra)
        return {"ui": {"text": [od]}, "result": (od,)}


def _core_hash() -> str:
    h = hashlib.sha1()
    core = os.path.join(os.path.dirname(__file__), "core")
    for f in sorted(os.listdir(core)):
        if f.endswith(".py"):
            with open(os.path.join(core, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:12]


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockWrite": AtlasVFXPlateLockWrite}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockWrite": "Atlas VFX Plate Lock Write"}
