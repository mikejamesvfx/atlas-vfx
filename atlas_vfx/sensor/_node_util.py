"""Small helpers shared by the Sensor nodes (no ComfyUI imports at module level)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import plate as pl

CATEGORY = "Atlas VFX/Sensor"


def progress_bar():
    try:
        import comfy.utils  # type: ignore
        bar = comfy.utils.ProgressBar(100)
        return lambda n, total: bar.update_absolute(int(100 * n / max(total, 1)))
    except ImportError:
        return None


def work_dir(cam_data: dict | None, sub: str, out_dir: str = "") -> Path:
    """The node's output folder: out_dir if given, else under the work root; either way checked by resolve_out_dir."""
    stem = Path(cam_data["plate"]["path"]).stem if cam_data else "shot"
    return pl.resolve_out_dir(out_dir, pl.WORK_ROOT / stem / "sensor" / sub)


def map_preview(st: np.ndarray, gain: float = 0.1, width: int = 1024) -> np.ndarray:
    """Displacement of an STMap as colour: R = 0.5 + dx * gain, G = 0.5 + dy * gain (px at map size), B = 0.5."""
    import cv2
    from .lens import from_st, pixel_grid
    h, w = st.shape[:2]
    X, Y = pixel_grid(w, h)
    x, y = from_st(st, w, h)
    img = np.stack([0.5 + (x - X) * gain, 0.5 + (y - Y) * gain, np.full_like(X, 0.5)], -1).astype(np.float32)
    hh = int(round(h * width / w))
    return np.clip(cv2.resize(img, (width, hh), interpolation=cv2.INTER_AREA), 0, 1)


def grid_image(w: int, h: int, step: int = 64) -> np.ndarray:
    img = np.full((h, w, 3), 0.18, np.float32)
    img[::step] = 1.0
    img[:, ::step] = 1.0
    return img
