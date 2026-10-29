"""Atlas VFX Plate Lock Read EXR: load an EXR sequence (plate or AI take) plus its header metadata."""
from __future__ import annotations

import json
import os

from . import _comfy
from .core import seqio


class AtlasVFXPlateLockReadEXR:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("frames", "metadata", "first_frame")
    DESCRIPTION = ("Reads an EXR sequence as linear float (no clamping, no display transform) and reports the "
                   "header metadata the lock can use: fps, shutter, focal, sensor, timecode, and anything that "
                   "looks like gyro or readout data. Headers parse on any EXR; pixels need OpenImageIO or OpenEXR "
                   "for compressed files.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": "", "tooltip": "folder, shot.####.exr, shot.%04d.exr or a glob"}),
            "first": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "index into the sorted files"}),
            "last": ("INT", {"default": -1, "min": -1, "max": 1_000_000, "tooltip": "-1 = to the end"}),
            "every": ("INT", {"default": 1, "min": 1, "max": 100}),
        }}

    @classmethod
    def IS_CHANGED(cls, path, first, last, every):
        files = seqio.list_sequence(_comfy.clean_path(path))
        if not files:
            return float("nan")
        st = [os.stat(f) for f in (files[0], files[-1])]
        return f"{len(files)}:{st[0].st_mtime_ns}:{st[1].st_mtime_ns}:{first}:{last}:{every}"

    def run(self, path, first, last, every):
        frames, headers, files = seqio.read_sequence(_comfy.clean_path(path), first,
                                                     None if last < 0 else last, every=every)
        rep = seqio.sequence_report(headers)
        rep["files"] = [files[0], files[-1]]
        import re
        m = re.findall(r"(\d+)(?=\.[^.]+$)", os.path.basename(files[0]))
        first_no = int(m[-1]) if m else 0
        return (_comfy.to_image(frames), json.dumps(rep, indent=1, default=str), first_no)


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockReadEXR": AtlasVFXPlateLockReadEXR}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockReadEXR": "Atlas VFX Plate Lock Read EXR"}
