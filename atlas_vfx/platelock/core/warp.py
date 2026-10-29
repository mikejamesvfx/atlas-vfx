"""Backward warp maps, composition, correspondence fields and STMaps.

A "map" is a pair (map_x, map_y), each HxW float32, for the OUTPUT frame:
    out[y, x] = src[map_y[y, x], map_x[y, x]]
Maps compose, invert to STMaps for Nuke, and never touch pixels until the end,
so every correction stays one resample away from the original AI frame.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

_BORDER = {"reflect": cv2.BORDER_REFLECT101, "constant": cv2.BORDER_CONSTANT,
           "replicate": cv2.BORDER_REPLICATE}


def identity_maps(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


def remap(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
          interp: str = "linear", border: str = "constant") -> np.ndarray:
    flag = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}[interp]
    src = np.asarray(img, np.float32)
    squeeze = src.ndim == 3 and src.shape[2] > 4
    if squeeze:  # opencv remap caps channels; split if needed
        return np.stack([remap(src[..., c], map_x, map_y, interp, border) for c in range(src.shape[2])], -1)
    return cv2.remap(src, map_x.astype(np.float32), map_y.astype(np.float32), flag,
                     borderMode=_BORDER[border], borderValue=0)


def compose(first, second):
    """Map for 'warp by `first`, then warp the result by `second`'.

    out(x) = src(first(second(x)))
    """
    fx, fy = first
    sx, sy = second
    return (remap(fx, sx, sy, border="replicate"), remap(fy, sx, sy, border="replicate"))


def displacement(maps):
    mx, my = maps
    ix, iy = identity_maps(*mx.shape)
    return mx - ix, my - iy


def from_displacement(dx, dy):
    ix, iy = identity_maps(*dx.shape)
    return (ix + dx).astype(np.float32), (iy + dy).astype(np.float32)


def correspondence_field(dst_pts: np.ndarray, src_pts: np.ndarray, shape, *,
                         smoothing: float = 1.0, grid_step: int = 4, max_points: int = 800,
                         kernel: str = "thin_plate_spline", seed: int = 0):
    """Dense backward map from sparse matches.

    dst_pts: (M,2) xy in the OUTPUT (plate) frame.
    src_pts: (M,2) xy of the same features in the SOURCE (AI) frame.
    Works for body-solve vertex pairs, keypoints or dense-matcher output alike.
    Evaluated on a coarse grid and upsampled, so it stays smooth and cheap.
    """
    h, w = shape[:2]
    dst = np.asarray(dst_pts, float)
    src = np.asarray(src_pts, float)
    ok = np.all(np.isfinite(dst), 1) & np.all(np.isfinite(src), 1)
    dst, src = dst[ok], src[ok]
    if dst.shape[0] < 3:
        return identity_maps(h, w)
    if dst.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(dst.shape[0], max_points, replace=False)
        dst, src = dst[keep], src[keep]
    disp = src - dst
    rbf = RBFInterpolator(dst, disp, kernel=kernel, smoothing=smoothing, degree=1)
    gy = np.arange(0, h + grid_step, grid_step, dtype=float)
    gx = np.arange(0, w + grid_step, grid_step, dtype=float)
    GX, GY = np.meshgrid(gx, gy)
    d = rbf(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GY.shape + (2,)).astype(np.float32)
    # bilinear upsample from grid to pixels
    mx = (np.arange(w, dtype=np.float32) / grid_step)[None, :].repeat(h, 0)
    my = (np.arange(h, dtype=np.float32) / grid_step)[:, None].repeat(w, 1)
    dx = cv2.remap(d[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(d[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return from_displacement(dx, dy)


def feather_to_identity(maps, mask: np.ndarray, radius: float = 12.0):
    """Keep a map inside a (dilated) region and ease to identity outside it."""
    m = np.asarray(mask) > 0.5
    if not m.any():
        return maps
    dist = distance_transform_edt(~m)
    wgt = np.clip(1 - dist / max(radius, 1e-6), 0, 1).astype(np.float32)
    dx, dy = displacement(maps)
    return from_displacement(dx * wgt, dy * wgt)


def smooth_maps(maps, sigma: float):
    if sigma <= 0:
        return maps
    dx, dy = displacement(maps)
    return from_displacement(gaussian_filter(dx, sigma), gaussian_filter(dy, sigma))


def temporal_smooth(map_seq, sigma: float = 1.0):
    """Smooth a sequence of maps over time (on displacement)."""
    if sigma <= 0 or len(map_seq) < 2:
        return map_seq
    from scipy.ndimage import gaussian_filter1d
    D = np.stack([np.stack(displacement(m), 0) for m in map_seq], 0)
    D = gaussian_filter1d(D, sigma, axis=0, mode="nearest")
    return [from_displacement(D[i, 0], D[i, 1]) for i in range(D.shape[0])]


def to_stmap(maps, src_w: int, src_h: int) -> np.ndarray:
    """Backward map -> Nuke STMap (HxWx2, s/t normalised, origin bottom-left)."""
    mx, my = maps
    s = (mx + 0.5) / src_w
    t = 1.0 - (my + 0.5) / src_h
    return np.stack([s, t], -1).astype(np.float32)


def from_stmap(st: np.ndarray, src_w: int, src_h: int):
    mx = st[..., 0] * src_w - 0.5
    my = (1.0 - st[..., 1]) * src_h - 0.5
    return mx.astype(np.float32), my.astype(np.float32)
