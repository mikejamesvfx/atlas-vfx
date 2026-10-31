"""Atlas VFX Alpha Prep: any image or clip, a single still included, made ready for the LTX-2.5 Alpha-Gen IC-LoRA
(ltx-2.5-22b-ic-lora-alpha-gen-0.9): fitted inside 1920x1088, padded (never stretched) to a multiple of 32, and held
to an 8n+1 frame count. A still is held for `still_frames` frames so the model sees a short static clip; Alpha Return
averages the mattes back to one, which is cleaner than a single-frame run. Settings follow the model card: empty
prompt, LoRA 1.0, reference strength 1.0, single stage at this size, at most 145 frames."""
from __future__ import annotations

import json
import math

import numpy as np

MAX_FRAMES = 145
PAD_VALUE = 0.5          # as ImagePadForOutpaint, which the owner's tested graph used


def plan(n: int, width: int, height: int, max_w: int = 1920, max_h: int = 1088, still_frames: int = 9) -> dict:
    if n < 1:
        raise ValueError("no frames")
    if n > MAX_FRAMES:
        raise ValueError(f"{n} frames: Alpha-Gen takes at most {MAX_FRAMES} per run (longer leaks RGB into the matte); "
                         f"split the shot with Chunk Plan")
    held = still_frames if n == 1 else n
    held = -(-(held - 1) // 8) * 8 + 1
    if held > MAX_FRAMES:
        raise ValueError(f"{n} frames hold to {held}, over {MAX_FRAMES}")
    s = min(1.0, max_w / width, max_h / height)
    w, h = max(2, round(width * s / 2) * 2), max(2, round(height * s / 2) * 2)
    return {"frames": n, "held": held, "still": n == 1, "size": [width, height], "work": [w, h],
            "padded": [math.ceil(w / 32) * 32, math.ceil(h / 32) * 32]}


def prepare(frames: np.ndarray, p: dict) -> np.ndarray:
    import cv2

    n, held = p["frames"], p["held"]
    w, h = p["work"]
    W, H = p["padded"]
    out = np.full((held, H, W, 3), PAD_VALUE, np.float32)
    interp = cv2.INTER_AREA if w < frames.shape[2] else cv2.INTER_CUBIC
    for i in range(held):
        f = frames[min(i, n - 1), ..., :3]
        out[i, :h, :w] = np.clip(cv2.resize(np.ascontiguousarray(f, np.float32), (w, h), interpolation=interp), 0, 1)
    return out


class AtlasVFXAlphaPrep:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "ATLAS_VFX_ALPHA_PREP", "STRING")
    RETURN_NAMES = ("image", "length", "width", "height", "alpha_prep", "info")
    DESCRIPTION = ("Gets any image or clip ready for the LTX Alpha-Gen matte: a single still works too (held for a "
                   "few frames, averaged back by Alpha Return). image goes to the LTX guide; length / width / height "
                   "to the empty latents; alpha_prep to Alpha Return. Display-referred input (sRGB), as LTX sees it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "a still or a clip (sRGB, 0-1)"}),
                "still_frames": ("INT", {"default": 9, "min": 1, "max": 49, "step": 8,
                                         "tooltip": "a still is held this long (8n+1); its mattes are averaged"}),
                "max_width": ("INT", {"default": 1920, "min": 256, "max": 1920, "step": 32,
                                      "tooltip": "the model's limit is 1920x1088"}),
                "max_height": ("INT", {"default": 1088, "min": 256, "max": 1088, "step": 32}),
            }
        }

    def run(self, images, still_frames=9, max_width=1920, max_height=1088):
        import torch

        frames = images.float().cpu().numpy()
        p = plan(len(frames), frames.shape[2], frames.shape[1], max_width, max_height, still_frames)
        out = prepare(frames, p)
        info = json.dumps({**p, "text": f"{p['frames']} frame(s) {p['size'][0]}x{p['size'][1]} -> {p['held']} frames "
                                        f"at {p['work'][0]}x{p['work'][1]} padded to "
                                        f"{p['padded'][0]}x{p['padded'][1]}"})
        return (torch.from_numpy(out), p["held"], p["padded"][0], p["padded"][1], p, info)


NODE_CLASS_MAPPINGS = {"AtlasVFXAlphaPrep": AtlasVFXAlphaPrep}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXAlphaPrep": "Atlas VFX Alpha Prep"}
