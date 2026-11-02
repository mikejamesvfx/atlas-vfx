"""Atlas VFX Alpha Return: the decoded LTX Alpha-Gen output back to a matte at the source size: padding cropped off,
the matte channel taken, clamped, scaled back up, the held frames trimmed (or, for a still, averaged to one cleaner
matte)."""
from __future__ import annotations

import json

import numpy as np

CHANNELS = {"red": 0, "green": 1, "blue": 2, "mean": None}


def matte(decoded: np.ndarray, p: dict, channel: str = "green", still_average: bool = True) -> np.ndarray:
    """decoded (T, Hp, Wp, 3) from the VAE -> (N, H, W) matte 0-1 at the source size."""
    import cv2

    if channel not in CHANNELS:
        raise ValueError(f"channel {channel!r}: one of {tuple(CHANNELS)}")
    held, n = p["held"], p["frames"]
    if len(decoded) < held:
        raise ValueError(f"LTX gave {len(decoded)} frames, Alpha Prep held {held}")
    Wp, Hp = p["padded"]
    if decoded.shape[1:3] != (Hp, Wp):
        decoded = np.stack([cv2.resize(f, (Wp, Hp), interpolation=cv2.INTER_LINEAR) for f in decoded[:held]])
    w, h = p["work"]
    W, H = p["size"]
    d = decoded[:held, :h, :w, :3]
    c = CHANNELS[channel]
    m = d.mean(-1) if c is None else d[..., c]
    m = m.astype(np.float32)
    m = m.mean(0, keepdims=True) if (p["still"] and still_average) else m[:n]   # average first: clamping biases it
    m = np.clip(m, 0.0, 1.0)
    interp = cv2.INTER_CUBIC if W > w else cv2.INTER_AREA
    return np.clip(np.stack([cv2.resize(f, (W, H), interpolation=interp) for f in m]), 0.0, 1.0).astype(np.float32)


class AtlasVFXAlphaReturn:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "matte_image", "info")
    DESCRIPTION = ("Turns the decoded LTX Alpha-Gen output into a matte at the source size (white = foreground, soft "
                   "edges kept). For a still, the held frames' mattes are averaged into one. mask goes to EXR Write's "
                   "mask, Relight Transfer's matte or holdout, Grade's mask; matte_image is the same as an image to "
                   "look at.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "VAE Decode of the Alpha-Gen latent"}),
                "alpha_prep": ("ATLAS_VFX_ALPHA_PREP",),
                "channel": (list(CHANNELS), {"default": "green", "tooltip": "the matte channel of the decode"}),
                "still_average": ("BOOLEAN", {"default": True, "tooltip": "a still: average the held frames' mattes"}),
            }
        }

    def run(self, images, alpha_prep, channel="green", still_average=True):
        import torch

        m = matte(images.float().cpu().numpy(), alpha_prep, channel, still_average)
        info = {"frames": int(len(m)), "size": alpha_prep["size"], "channel": channel,
                "still_averaged": bool(alpha_prep["still"] and still_average),
                "coverage": round(float((m > 0.5).mean()), 4), "soft_px": round(float(((m > 0.02) & (m < 0.98)).mean()), 4)}
        img = torch.from_numpy(np.repeat(m[..., None], 3, -1))
        return (torch.from_numpy(m), img, json.dumps(info))


NODE_CLASS_MAPPINGS = {"AtlasVFXAlphaReturn": AtlasVFXAlphaReturn}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXAlphaReturn": "Atlas VFX Alpha Return"}
