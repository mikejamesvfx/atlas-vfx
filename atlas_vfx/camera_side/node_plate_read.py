"""Atlas VFX Plate Read (catalogue 1): probe a movie plate -> ATLAS_PLATE_SEQ + a display-referred preview batch."""
from __future__ import annotations

import json
import os

from . import plate as pl

DEFAULT_PLATE = ""      # no plate ships with the pack; path widgets expand ~ and %VARS%


class AtlasVFXPlateRead:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_PLATE_SEQ", "IMAGE", "STRING")
    RETURN_NAMES = ("plate_ref", "preview", "info")
    DESCRIPTION = ("Reads a plate in place (never copied). Range comes from the stream (DSCF3000 is full range); "
                   "input space 'auto' reads the Fuji recording-mode tag (F-Log / F-Log2 F-Gamut), else the camera body default. Preview goes through the "
                   "named OCIO display/view.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": DEFAULT_PLATE, "tooltip": "movie plate; ~ and %VARS% expand"}),
                "first": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                  "tooltip": "plate frame index, 0-based (Nuke frame = index + frame_offset)"}),
                "last": ("INT", {"default": -1, "min": -1, "max": 1_000_000, "tooltip": "-1 = last frame"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "input_space": ("STRING", {"default": "auto",
                                           "tooltip": "OCIO colorspace of the plate; auto = camera body default"}),
                "range": (["auto", "full", "legal"], {"default": "auto",
                                                      "tooltip": "auto trusts the stream; override only broken files"}),
                "frame_offset": ("INT", {"default": pl.NUKE_FRAME_OFFSET, "min": 0, "max": 1_000_000}),
                "preview_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 2}),
                "preview_every": ("INT", {"default": 8, "min": 1, "max": 1000}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, path, **kw):
        try:
            st = os.stat(path)
            return f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return float("nan")

    def run(self, path, first, last, ocio_config, input_space, range, frame_offset, preview_width,
            preview_every):
        import numpy as np
        import torch

        ref = pl.plate_ref(pl.clean_path(path), first=first, last=last, ocio_config_path=ocio_config,
                           input_space=input_space, range_override=range, frame_offset=frame_offset)
        frames = [pl.to_display(img, ref) for _, img in
                  pl.decode_rgb(ref, width=preview_width, step=preview_every)]
        preview = torch.from_numpy(np.stack(frames).astype(np.float32))
        info = json.dumps({k: ref[k] for k in (
            "path", "width", "height", "frame_count", "fps", "range", "range_source", "colorspace",
            "colorspace_source", "camera_body", "timecode", "first", "last", "nuke_first", "nuke_last")}, indent=1)
        return (ref, preview, info)


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateRead": AtlasVFXPlateRead}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateRead": "Atlas VFX Plate Read"}
