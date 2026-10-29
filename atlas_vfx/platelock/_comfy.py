"""Tensor <-> numpy glue for the platelock nodes. Imports torch lazily so the core stays testable without it."""
from __future__ import annotations

import numpy as np

CATEGORY = "Atlas VFX/Layers/Plate Lock"


def image_np(t) -> np.ndarray:
    """ComfyUI IMAGE [B,H,W,C] -> float32 numpy, RGB only."""
    a = t.detach().cpu().numpy().astype(np.float32) if hasattr(t, "detach") else np.asarray(t, np.float32)
    if a.ndim == 3:
        a = a[None]
    return a[..., :3]


def mask_np(t) -> np.ndarray:
    """ComfyUI MASK [B,H,W] (or IMAGE) -> float32 numpy [B,H,W]."""
    a = t.detach().cpu().numpy().astype(np.float32) if hasattr(t, "detach") else np.asarray(t, np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    if a.ndim == 2:
        a = a[None]
    return a


def to_image(a: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))


def to_mask(a: np.ndarray):
    import torch
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[None]
    return torch.from_numpy(np.ascontiguousarray(a))


def resize_stack(a: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    """Resize a [B,H,W] or [B,H,W,C] stack."""
    import cv2
    if a.shape[1] == height and a.shape[2] == width:
        return a
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if a.shape[2] > width else cv2.INTER_LINEAR)
    return np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in a]).astype(np.float32)


def clean_path(p: str) -> str:
    """Quotes and whitespace stripped, ~ and %VARS% expanded (the same rule as camera_side.plate.clean_path)."""
    from ..camera_side.plate import clean_path as _clean
    return _clean(p)
