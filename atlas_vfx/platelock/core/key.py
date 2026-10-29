"""Plate key and edge snap.

A clean screen gives a ground-truth silhouette on every plate frame. Edge snap
builds a backward map that slides the AI performer's silhouette onto it: each
output pixel finds its nearest plate edge point, that point's nearest AI edge
point gives the local edge displacement, and the displacement field is
smoothed so interiors move with their edges instead of tearing.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter

from . import warp


def screen_key(img: np.ndarray, screen: str = "blue", low: float = 0.04, high: float = 0.2) -> np.ndarray:
    """Simple colour-difference key. Returns foreground alpha (1 = performer).

    For production, feed a proper key (Keylight / IBK pull) in instead; this is
    the fallback so the pipeline runs without one.
    """
    im = np.asarray(img, np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if screen == "blue":
        diff = b - np.maximum(r, g)
    elif screen == "green":
        diff = g - np.maximum(r, b)
    else:
        raise ValueError("screen must be 'blue' or 'green'")
    t = np.clip((diff - low) / max(high - low, 1e-6), 0, 1)
    t = t * t * (3 - 2 * t)
    return (1 - t).astype(np.float32)


def _edge(binary: np.ndarray) -> np.ndarray:
    return binary & ~binary_erosion(binary)


def edge_snap(ai_matte: np.ndarray, plate_matte: np.ndarray, *, threshold: float = 0.5,
              max_shift: float = 12.0, sigma: float = 3.0, region: np.ndarray | None = None,
              band: float | None = None):
    """Backward map that moves the AI silhouette onto the plate silhouette.

    Use it after the coarse warp (retime + correspondence field), when the
    remaining error is a few pixels. `region` limits where the snap applies
    (e.g. one performer's dilated plate matte when two people are close).
    `band` (px) confines the snap to a strip around the plate edge, so an
    interior already locked by correspondences is left alone. None moves the
    interior with its edge (use when there are no correspondences).
    """
    a = np.asarray(ai_matte) > threshold
    p = np.asarray(plate_matte) > threshold
    h, w = p.shape
    if not a.any() or not p.any():
        return warp.identity_maps(h, w)
    ae, pe = _edge(a), _edge(p)

    pdist, (py, px) = distance_transform_edt(~pe, return_indices=True)   # nearest plate edge
    _, (ay, ax) = distance_transform_edt(~ae, return_indices=True)   # nearest AI edge

    # displacement on plate edge points: plate edge -> nearest AI edge
    ey, ex = np.nonzero(pe)
    dxe = (ax[ey, ex] - ex).astype(np.float32)
    dye = (ay[ey, ex] - ey).astype(np.float32)
    mag = np.hypot(dxe, dye)
    k = np.where(mag > max_shift, max_shift / np.maximum(mag, 1e-6), 1.0)
    dxe *= k
    dye *= k

    # every pixel inherits the displacement of its nearest plate edge point
    DX = np.zeros((h, w), np.float32)
    DY = np.zeros((h, w), np.float32)
    DX[ey, ex] = dxe
    DY[ey, ex] = dye
    DX = DX[py, px]
    DY = DY[py, px]
    if sigma > 0:
        DX = gaussian_filter(DX, sigma)
        DY = gaussian_filter(DY, sigma)
    # re-pin the edge itself: smoothing must not soften the snap where it matters
    DX[ey, ex] = dxe
    DY[ey, ex] = dye
    if sigma > 0:
        DX = gaussian_filter(DX, sigma * 0.5)
        DY = gaussian_filter(DY, sigma * 0.5)

    if band is not None:
        wb = np.clip(1.0 - pdist / max(band, 1e-6), 0, 1).astype(np.float32)
        wb = wb * wb * (3 - 2 * wb)
        DX *= wb
        DY *= wb
    maps = warp.from_displacement(DX, DY)
    if region is not None:
        maps = warp.feather_to_identity(maps, region, radius=max_shift * 2)
    return maps


def edge_snap_iterative(ai_matte, plate_matte, iters: int = 3, min_px: float = 0.0, **kw):
    """Repeat the snap on the already-snapped matte; converges in 2–3 passes.

    Stops early once the mean edge distance is under `min_px`, so a performer
    the coarse warp already nailed is not touched.
    """
    from .qc import edge_distance
    h, w = np.asarray(plate_matte).shape
    total = warp.identity_maps(h, w)
    cur = np.asarray(ai_matte, np.float32)
    for _ in range(iters):
        if min_px > 0:
            ed = edge_distance(cur, plate_matte)
            if not np.isfinite(ed) or ed < min_px:
                break
        step = edge_snap(cur, plate_matte, **kw)
        total = warp.compose(total, step)
        cur = warp.remap(np.asarray(ai_matte, np.float32), *total)
    return total
