"""Atlas VFX Solve Mask (catalogue 67): masks of moving things for Solve Plate (Grounding DINO boxes + SAM 2.1
tracking, camera_side/solve_mask.py)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import plate as pl
from . import solve_mask as sm

log = logging.getLogger("atlas_vfx")


def _models_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir)
    except ImportError:
        return Path.home() / "ComfyUI_V141" / "ComfyUI" / "models"


class AtlasVFXSolveMask:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "preview", "report")
    DESCRIPTION = ("Masks of things that move, for Solve Plate. Grounding DINO finds the classes you name on the middle "
                   "frame of each window, SAM 2.1 tracks them through the window both ways; windows are unioned. "
                   "One mask per plate frame at proxy_width (Solve Plate resizes them), 1 = moving. The plate is "
                   "seen through the shipped OCIO display view. Models: tools/fetch_solve_models.py.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "classes": ("STRING", {"default": sm.DEFAULT_PROMPT,
                                       "tooltip": "what moves, comma or dot separated"}),
                "window": ("INT", {"default": 16, "min": 1, "max": 512,
                                   "tooltip": "frames per detection; smaller finds things that enter sooner"}),
                "box_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.01}),
                "dilate_px": ("INT", {"default": 6, "min": 0, "max": 64}),
                "proxy_width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 2}),
            }
        }

    def run(self, plate_ref, classes, window, box_threshold, text_threshold, dilate_px, proxy_width):
        import numpy as np
        import torch

        sm.prompt_text(classes)                        # validate before decoding anything
        frames = []
        for _i, rgb in pl.decode_rgb(plate_ref, width=proxy_width):
            frames.append((pl.to_display(rgb, plate_ref) * 255 + 0.5).astype(np.uint8))
        masker = sm.Masker(_models_root())
        try:
            masks, rep = sm.masks_for_clip(frames, masker, classes, window, box_threshold, text_threshold, dilate_px)
        finally:
            masker.free()
        rep["frames"] = [plate_ref["first"], plate_ref["last"]]
        log.info("atlas-vfx Solve Mask: %s", json.dumps({k: rep[k] for k in ("prompt", "coverage")}))
        m = torch.from_numpy(np.stack(masks))
        f = np.stack(frames).astype(np.float32) / 255.0
        f[..., 0] = np.maximum(f[..., 0], np.stack(masks) * 0.8)
        head = f"{len(masks)} masks, {100 * rep['coverage']:.1f} % of the frame on average, prompt '{rep['prompt']}'"
        return {"ui": {"text": [head]}, "result": (m, torch.from_numpy(f), json.dumps(rep, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXSolveMask": AtlasVFXSolveMask}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSolveMask": "Atlas VFX Solve Mask"}
