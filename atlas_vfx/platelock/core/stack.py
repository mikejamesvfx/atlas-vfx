"""Depth-ordered restack of performer layers over the re-shot background."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Layer:
    rgb: np.ndarray     # HxWx3, unpremultiplied
    alpha: np.ndarray   # HxW
    depth: float        # distance from camera; larger = further back
    name: str = ""


def over(fg_rgb, fg_a, bg_rgb):
    a = np.clip(fg_a, 0, 1)[..., None]
    return fg_rgb * a + bg_rgb * (1 - a)


def restack(bg: np.ndarray, layers: list[Layer]) -> np.ndarray:
    out = np.asarray(bg, np.float32).copy()
    for L in sorted(layers, key=lambda l: -l.depth):
        out = over(L.rgb, L.alpha, out)
    return out


def layer_depth_from_matte(depth_map: np.ndarray, matte: np.ndarray, thr: float = 0.5) -> float:
    """Median depth under a performer's matte (from a depth solve on the plate or AI frame)."""
    m = np.asarray(matte) > thr
    d = np.asarray(depth_map)[m]
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float("inf")
