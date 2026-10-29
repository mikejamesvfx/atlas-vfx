"""Atlas VFX Tracking Proxy (catalogue 2): log-luma + CLAHE PNGs for solving, written outside the repo."""
from __future__ import annotations

from . import plate as pl


class AtlasVFXTrackingProxy:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("folder", "preview")
    DESCRIPTION = ("Tracking proxies, not for viewing: plate luma code values (log lifts the shadows), per-frame "
                   "stretch, CLAHE. One plate_<index>.png per frame plus track_proxy.json. Empty out_dir writes to "
                   "atlas-vfx-work/<plate>/track_<first>-<last>_w<width>.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 2}),
                "out_dir": ("STRING", {"default": ""}),
                "clip_limit": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.1}),
                "preview_every": ("INT", {"default": 8, "min": 1, "max": 1000}),
            }
        }

    def run(self, plate_ref, width, out_dir, clip_limit, preview_every):
        import numpy as np
        import torch

        out, kept = pl.write_tracking_proxies(plate_ref, out_dir, width=width,
                                              clip_limit=clip_limit, keep_every=preview_every)
        prev = np.stack(kept).astype(np.float32) / 255.0
        preview = torch.from_numpy(np.repeat(prev[..., None], 3, axis=-1))
        return (str(out), preview)


NODE_CLASS_MAPPINGS = {"AtlasVFXTrackingProxy": AtlasVFXTrackingProxy}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXTrackingProxy": "Atlas VFX Tracking Proxy"}
