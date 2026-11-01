"""Atlas VFX Track 2D (catalogue 68): artist-chosen points tracked through the plate (pyramidal LK, forward-backward
checked; camera_side/tracks2d.py), for Solve Plate's tracks input."""
from __future__ import annotations

import json

from . import plate as pl
from . import tracks2d as t2


class AtlasVFXTrack2D:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_TRACKS_2D", "IMAGE", "STRING")
    RETURN_NAMES = ("tracks", "preview", "summary")
    DESCRIPTION = ("Tracks points you pick (seeds: JSON [{\"label\": \"corner_a\", \"frame\": 120, \"xy\": [x, y]}], "
                   "frame = plate frame index, xy = plate pixels from the top-left; Solve View writes them when you "
                   "click) forwards and backwards with pyramidal Lucas-Kanade on CLAHE'd luma. A frame is kept while "
                   "the forward-backward error stays under fb_px. Feed Solve Plate's tracks input so these points "
                   "are in the solve even where SIFT found nothing.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "seeds": ("STRING", {"default": "[]", "multiline": True}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                                    "tooltip": "tracking resolution as a fraction of the plate"}),
                "span": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "frames each way; 0 = all"}),
                "fb_px": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0, "step": 0.05}),
                "window": ("INT", {"default": 41, "min": 7, "max": 151, "step": 2}),
                "levels": ("INT", {"default": 4, "min": 0, "max": 8}),
            }
        }

    def run(self, plate_ref, seeds, scale, span, fb_px, window, levels):
        import cv2
        import numpy as np
        import torch

        parsed = t2.parse_seeds(seeds)
        W = max(64, int(round(plate_ref["width"] * scale / 2)) * 2)
        s = W / plate_ref["width"]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        imgs = {i: (clahe.apply(y16) >> 8).astype(np.uint8) for i, y16 in pl.decode_luma16(plate_ref, width=W)}
        tr = t2.track_seeds(imgs, parsed, plate_ref["width"], plate_ref["height"], scale=s, fb_px=fb_px,
                            span=span or None, win=window, levels=levels)
        lines = []
        for label, t in tr["tracks"].items():
            fr = [int(f) for f in t]
            lines.append(f"{label}: {min(fr)}-{max(fr)} ({len(fr)} frames)")
        key = parsed[0]["frame"] if parsed else plate_ref["first"]
        img = cv2.cvtColor(imgs[key], cv2.COLOR_GRAY2RGB)
        for label, t in tr["tracks"].items():
            pts = [(int(round(x * s)), int(round(y * s))) for _f, (x, y) in sorted(t.items(), key=lambda kv: int(kv[0]))]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (255, 200, 0), 1, cv2.LINE_AA)
            if str(key) in t:
                x, y = t[str(key)]
                cv2.circle(img, (int(round(x * s)), int(round(y * s))), 5, (0, 255, 0), 1, cv2.LINE_AA)
        preview = torch.from_numpy(img[None].astype(np.float32) / 255.0)
        summary = "\n".join(lines) or "no seeds"
        return {"ui": {"text": [summary]}, "result": (tr, preview, json.dumps(tr)[:20000])}


NODE_CLASS_MAPPINGS = {"AtlasVFXTrack2D": AtlasVFXTrack2D}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXTrack2D": "Atlas VFX Track 2D"}
