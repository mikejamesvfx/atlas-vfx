"""Sensor Apply core, both directions.

apply  (clean / LTX render -> plate look), in this order:
  1. vignette (multiplies LINEAR ACEScg only)
  2. lateral CA + distortion (per-channel redistort STMaps)
  3. rolling shutter (rs STMap; several averaged when per-row motion blur is on)
remove (plate -> clean space, e.g. before LTX Prep), the exact reverse:
  1. de-RS (ders STMap)  2. undistort + CA removal (per-channel undistort STMaps)  3. divide by the vignette gain

The maps are made at plate resolution for whole plate frames. An image that is a crop of the plate and/or a resize
(an LTX render from LTX Prep: full plate at the work size, or a per-frame crop window) gets the matching part of the
map: `window_map` cuts the window out of the plate-res map and re-expresses it in the image's own s/t.

Steps 2 and 3 are composed into one map per channel before sampling, so the image is filtered once. Composition: the
sensor pixel p shows the lens-applied image at rs(p), which shows the clean image at redistort_c(rs(p)).
Maps are resolution independent (Nuke s/t); they are resampled to the image size.
"""
from __future__ import annotations

import numpy as np

from .lens import from_st


def resize_map(st: np.ndarray, w: int, h: int) -> np.ndarray:
    if st.shape[1] == w and st.shape[0] == h:
        return st
    import cv2
    return cv2.resize(st.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """STMap of `inner` applied after `outer` was used to look up: result(p) = inner(outer(p)).
    i.e. outer is the map of the LAST operation (rs), inner the one before it (lens). Same size."""
    import cv2
    h, w = outer.shape[:2]
    x, y = from_st(outer, w, h)
    out = cv2.remap(inner.astype(np.float32), (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def warp(img: np.ndarray, st: np.ndarray, edge: str = "edge") -> np.ndarray:
    """Sample img (H, W, C) at an STMap (bilinear). edge: 'edge' repeats border pixels, 'black' is Nuke's default."""
    import cv2
    h, w = img.shape[:2]
    st = resize_map(st, w, h)
    x, y = from_st(st, w, h)
    border = cv2.BORDER_REPLICATE if edge == "edge" else cv2.BORDER_CONSTANT
    src = np.ascontiguousarray(img, np.float32)
    out = cv2.remap(src, (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=border, borderValue=0)
    return out.reshape(img.shape)


def window_map(st: np.ndarray, win, plate_size) -> np.ndarray:
    """A plate-res STMap (plate s/t) -> the map for an image showing plate window win = [x0, y0, w, h] (plate px),
    in that image's own s/t. Positions outside the window come out < 0 or > 1 (edge-clamped when sampling)."""
    if win is None:
        return st
    W, H = plate_size
    x0, y0, w, h = [int(v) for v in win]
    cut = st[y0:y0 + h, x0:x0 + w].astype(np.float64)
    if cut.shape[:2] != (h, w):
        raise ValueError(f"window {win} is outside the {W}x{H} map")
    x, y = cut[..., 0] * W, (1.0 - cut[..., 1]) * H
    return np.stack([(x - x0) / w, 1.0 - (y - y0) / h], -1).astype(np.float32)


def window_gain(gain: np.ndarray, win) -> np.ndarray:
    if win is None:
        return gain
    x0, y0, w, h = [int(v) for v in win]
    return gain[y0:y0 + h, x0:x0 + w]


def _resize_gain(gain, w, h):
    if gain.shape == (h, w):
        return gain
    import cv2
    return cv2.resize(gain.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def remove(img: np.ndarray, *, gain: np.ndarray | None = None, lens: dict | None = None,
           ders: np.ndarray | None = None, edge: str = "edge") -> np.ndarray:
    """Plate -> clean: de-RS, then undistort + CA removal (per-channel undistort maps), then divide by the gain.
    ders and the undistort maps are composed (one filter per channel). Image must be linear if gain is given."""
    h, w = img.shape[:2]
    out = np.asarray(img, np.float32)
    lens = {c: resize_map(m, w, h) for c, m in (lens or {}).items()}
    if ders is not None or lens:
        ders = resize_map(ders, w, h) if ders is not None else None
        chans = _chans(out.shape[2])
        res = np.empty_like(out)
        for ci, c in enumerate(chans):
            L = lens.get(c)
            st = ders if L is None else (L if ders is None else compose(L, ders))
            res[..., ci] = warp(out[..., ci:ci + 1], st, edge)[..., 0]
        out = res
    else:
        out = out.copy()
    if gain is not None:
        out[..., :3] /= np.maximum(_resize_gain(gain, w, h), 1e-6)[..., None]
    return out


def _chans(n):
    return (["R", "G", "B"] + ["G"] * (n - 3)) if n >= 3 else ["G"] * n


def apply(img: np.ndarray, *, gain: np.ndarray | None = None, lens: dict | None = None,
          rs: list | None = None, edge: str = "edge") -> np.ndarray:
    """img (H, W, C) float. gain (h, w) vignette gain (image must be linear). lens: {"R","G","B"} redistort STMaps.
    rs: list of rs STMaps (one, or one per motion-blur sub-sample; averaged). Alpha / extra channels use G's map."""
    h, w = img.shape[:2]
    out = np.asarray(img, np.float32).copy()
    if gain is not None:
        out[..., :3] *= _resize_gain(gain, w, h)[..., None]
    rs = [resize_map(m, w, h) for m in (rs or [])]
    lens = {c: resize_map(m, w, h) for c, m in (lens or {}).items()}
    if not rs and not lens:
        return out
    chans = _chans(out.shape[2])
    result = np.zeros_like(out)
    for m in (rs or [None]):
        for ci, c in enumerate(chans):
            L = lens.get(c)
            if m is None:
                st = L
            elif L is None:
                st = m
            else:
                st = compose(m, L)
            result[..., ci] += warp(out[..., ci:ci + 1], st, edge)[..., 0]
    return result / max(1, len(rs))
