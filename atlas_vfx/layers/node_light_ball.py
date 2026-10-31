"""Atlas VFX Light Ball: pastes the light-direction ball the LTX-2.3 Relight IC-LoRA reads onto every reference frame,
at the size and place it was trained with (143 px, 22 px in from the top-right corner of 1280x704, scaled to the
frame's width). The balls ship in assets/relight_balls (MIT, see atlas_components.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .node_sun_direction import DIRECTIONS, HEIGHTS

BALLS = Path(__file__).resolve().parents[2] / "assets" / "relight_balls"
SIZE, PAD, TRAIN_W = 143, 22, 1280


def ball_image(direction: str, height: str) -> np.ndarray:
    """The ball PNG as float RGB 0-1. direction / height must be from the lists (never a free path)."""
    import cv2

    if direction not in DIRECTIONS or height not in HEIGHTS:
        raise ValueError(f"no ball for {direction!r} / {height!r}")
    p = BALLS / f"{direction}__{height}.png"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"ball image missing: {p.name} (assets/relight_balls)")
    return img[..., ::-1].astype(np.float32) / 255.0


def place(width: int) -> tuple[int, int, int]:
    """(x, y, size) of the ball on a frame `width` wide."""
    s = width / TRAIN_W
    size, pad = max(8, round(SIZE * s)), round(PAD * s)
    return width - pad - size, pad, size


def paste(frames: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """frames (N, H, W, 3) 0-1; returns a copy with the ball pasted on each frame."""
    import cv2

    out = np.array(frames, np.float32, copy=True)
    x, y, size = place(out.shape[2])
    b = cv2.resize(ball, (size, size), interpolation=cv2.INTER_AREA)
    out[:, y:y + size, x:x + size, :3] = b
    return out


class AtlasVFXLightBall:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = ("Pastes the light-direction ball from Sun Direction onto every frame of the reference (LTX Prep's "
                   "reference output), top-right, at the size the Relight LoRA was trained with. Goes between LTX "
                   "Prep and the LTX guide node.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"tooltip": "LTX Prep's reference (already mirrored if Prep was)"}),
                             "sun": ("ATLAS_VFX_SUN",)}}

    def run(self, image, sun):
        import torch

        frames = image.float().cpu().numpy()
        out = paste(frames, ball_image(sun["direction"], sun["height"]))
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {"AtlasVFXLightBall": AtlasVFXLightBall}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLightBall": "Atlas VFX Light Ball"}
