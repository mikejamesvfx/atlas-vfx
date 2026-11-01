"""Atlas VFX Shot Setup (catalogue 70): the choice before anything else. What the shot starts from (layers/shot_mode.py):
a background plate, an actor on a green or blue screen, or nothing (create both). Gives every mode a plate_ref (the
plate, the screen plate, or a virtual plate) and an ATLAS_VFX_SHOT record that says which it is."""
from __future__ import annotations

import json
import os

from ..camera_side import plate as pl
from . import shot_mode as sm


class AtlasVFXShotSetup:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_PLATE_SEQ", "ATLAS_VFX_SHOT", "IMAGE", "STRING")
    RETURN_NAMES = ("plate_ref", "shot", "preview", "info")
    DESCRIPTION = ("Start here. plate: a real background plate (read like Plate Read). foreground: an actor shot on "
                   "a green or blue screen (read the same way; key and track it, generate the background behind). "
                   "generate: no plate; width, height, fps and frames describe the virtual plate the background "
                   "is made for. Every mode hands on a plate_ref, so the nodes after it work the same.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(sm.MODES), {"default": "plate",
                                          "tooltip": "plate = background plate, foreground = actor on a screen, "
                                                     "generate = create both (no plate)"}),
                "path": ("STRING", {"default": "", "tooltip": "plate / screen plate (movie file); blank for "
                                                             "generate. ~ and %VARS% expand"}),
                "first": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                  "tooltip": "plate frame index, 0-based (Nuke frame = index + frame_offset)"}),
                "last": ("INT", {"default": -1, "min": -1, "max": 1_000_000,
                                 "tooltip": "-1 = last frame (generate: frames - 1)"}),
                "screen": (list(sm.SCREENS), {"default": "green", "tooltip": "foreground only: the screen colour"}),
                "width": ("INT", {"default": 4096, "min": 16, "max": 16384, "step": 2,
                                  "tooltip": "generate only: the virtual plate"}),
                "height": ("INT", {"default": 2160, "min": 16, "max": 16384, "step": 2,
                                   "tooltip": "generate only: the virtual plate"}),
                "fps": ("STRING", {"default": "24", "tooltip": "generate only: 24, 25, 23.976 or 24000/1001"}),
                "frames": ("INT", {"default": 121, "min": 1, "max": 100_000,
                                   "tooltip": "generate only: frames in the shot"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "input_space": ("STRING", {"default": "auto", "tooltip": "plate modes: OCIO colorspace of the "
                                                                         "plate; auto = camera tag / body default"}),
                "frame_offset": ("INT", {"default": pl.NUKE_FRAME_OFFSET, "min": 0, "max": 1_000_000}),
                "preview_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 2}),
                "preview_every": ("INT", {"default": 8, "min": 1, "max": 1000}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, mode, path, **kw):
        if mode == "generate":
            return json.dumps(kw, sort_keys=True, default=str)
        try:
            st = os.stat(pl.clean_path(path))
            return f"{st.st_size}:{st.st_mtime_ns}:{json.dumps(kw, sort_keys=True, default=str)}"
        except OSError:
            return float("nan")

    def run(self, mode, path, first, last, screen, width, height, fps, frames, ocio_config, input_space,
            frame_offset, preview_width, preview_every):
        import numpy as np
        import torch

        mode = sm.check_mode(mode)
        if mode == "generate":
            if str(path).strip():
                raise ValueError("Shot Setup: generate has no plate (clear the path, or pick plate / foreground)")
            ref = pl.virtual_plate_ref(width, height, fps, frames, first, last, ocio_config, frame_offset)
            pw, ph = pl.proxy_size(ref, preview_width)
            preview = torch.zeros((1, ph, pw, 3), dtype=torch.float32)
        else:
            if not str(path).strip():
                raise ValueError(f"Shot Setup: {mode} needs a plate path")
            ref = pl.plate_ref(pl.clean_path(path), first=first, last=last, ocio_config_path=ocio_config,
                               input_space=input_space, frame_offset=frame_offset)
            frames_ = [pl.to_display(img, ref) for _, img in
                       pl.decode_rgb(ref, width=preview_width, step=preview_every)]
            preview = torch.from_numpy(np.stack(frames_).astype(np.float32))
        shot = sm.record(mode, ref, screen if mode == "foreground" else None)
        info = json.dumps({**shot, **{k: ref[k] for k in ("path", "colorspace", "colorspace_source", "first", "last",
                                                          "nuke_first", "nuke_last")}}, indent=1)
        return (ref, shot, preview, info)


NODE_CLASS_MAPPINGS = {"AtlasVFXShotSetup": AtlasVFXShotSetup}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXShotSetup": "Atlas VFX Shot Setup"}
