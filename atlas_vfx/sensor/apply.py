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
Maps are resolution independent (Nuke s/t); they are resampled to the output size.

Overscan: the clean frame can be bigger than the plate (barrel corners kept). A map lives on its OUTPUT grid and its
s/t point into its SOURCE frame: undistort is clean-grid -> plate s/t, redistort is plate-grid -> clean s/t. With
out_size given, maps are resized to out_size (not to the input image) and lookups use the source's own size.
"""
from __future__ import annotations

import numpy as np

from .lens import from_st


_RESIZED: dict = {}         # (id(st), w, h) -> (st, resized): the same map object resized once, not every frame


def resize_map(st: np.ndarray, w: int, h: int) -> np.ndarray:
    if st.shape[1] == w and st.shape[0] == h:
        return st
    key = (id(st), w, h)
    hit = _RESIZED.get(key)
    if hit is not None and hit[0] is st:
        return hit[1]
    import cv2
    out = cv2.resize(st.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out.flags.writeable = False                # shared between calls: nobody may change it in place
    if len(_RESIZED) >= 6:                     # bounded: per-frame RS maps pass through too
        _RESIZED.pop(next(iter(_RESIZED)))
    _RESIZED[key] = (st, out)
    return out


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """STMap of `inner` applied after `outer` was used to look up: result(p) = inner(outer(p)).
    i.e. outer is the map of the LAST operation (rs), inner the one before it (lens). outer's s/t point into the
    frame inner lives on, so they are read at inner's size (the same as outer's without overscan)."""
    import cv2
    h, w = inner.shape[:2]
    x, y = from_st(outer, w, h)
    out = cv2.remap(inner.astype(np.float32), (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


_COORDS: dict = {}          # (id(st), w, h, ow, oh) -> (st, mx, my): the lens maps are the same every frame


def _coords(st: np.ndarray, w: int, h: int, ow: int, oh: int):
    """cv2.remap coordinates for an STMap, cached while the same map object comes back (Maps keeps the lens maps,
    so a shot converts them once instead of every frame and channel). Same maths as before, same result."""
    key = (id(st), w, h, ow, oh)
    hit = _COORDS.get(key)
    if hit is not None and hit[0] is st:
        return hit[1], hit[2]
    x, y = from_st(resize_map(st, ow, oh), w, h)
    mx, my = (x - 0.5).astype(np.float32), (y - 0.5).astype(np.float32)
    if len(_COORDS) >= 6:
        _COORDS.pop(next(iter(_COORDS)))
    _COORDS[key] = (st, mx, my)
    return mx, my


def warp(img: np.ndarray, st: np.ndarray, edge: str = "edge", out_size=None) -> np.ndarray:
    """Sample img (H, W, C) at an STMap (bilinear). edge: 'edge' repeats border pixels, 'black' is Nuke's default.
    out_size (w, h): the output grid (the map is resized to it); default the image's own size."""
    import cv2
    h, w = img.shape[:2]
    ow, oh = out_size or (w, h)
    mx, my = _coords(st, w, h, ow, oh)
    border = cv2.BORDER_REPLICATE if edge == "edge" else cv2.BORDER_CONSTANT
    src = np.ascontiguousarray(img, np.float32)
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=border, borderValue=0)
    return out.reshape((oh, ow) + img.shape[2:])


def _by_map(chans, map_for):
    """[(st, [channel indices])]: channels that use the same map object are warped together (cv2.remap takes up to
    4 channels; alpha uses G's map). map_for(c) is called once per distinct channel name."""
    groups, seen = [], {}
    for ci, c in enumerate(chans):
        if c not in seen:
            seen[c] = map_for(c)
        st = seen[c]
        for g in groups:
            if g[0] is st and len(g[1]) < 4:
                g[1].append(ci)
                break
        else:
            groups.append((st, [ci]))
    # cv2.remap's 2-channel path interpolates differently (up to 0.024 off on random data); 1, 3 and 4 channels
    # match a channel at a time exactly, so a pair is split to keep the result what it was
    return [g for st, idx in groups for g in ([(st, [i]) for i in idx] if len(idx) == 2 else [(st, idx)])]


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
           ders: np.ndarray | None = None, edge: str = "edge", out_size=None) -> np.ndarray:
    """Plate -> clean: de-RS, then undistort + CA removal (per-channel undistort maps), then divide by the gain.
    ders and the undistort maps are composed (one filter per channel). Image must be linear if gain is given.
    out_size: the clean frame (w, h) when it is bigger than the plate (overscan); gain is then on the clean grid."""
    h, w = img.shape[:2]
    ow, oh = out_size or (w, h)
    out = np.asarray(img, np.float32)
    lens = {c: resize_map(m, ow, oh) for c, m in (lens or {}).items()}
    if ders is not None or lens:
        ders = resize_map(ders, w, h) if ders is not None else None      # ders lives on the plate grid
        res = np.empty((oh, ow, out.shape[2]), np.float32)

        def map_for(c):
            L = lens.get(c)
            return ders if L is None else (L if ders is None else compose(L, ders))

        for st, idx in _by_map(_chans(out.shape[2]), map_for):
            res[..., idx] = warp(out[..., idx], st, edge, (ow, oh))
        out = res
    else:
        out = out.copy()
    if gain is not None:
        out[..., :3] /= np.maximum(_resize_gain(gain, ow, oh), 1e-6)[..., None]
    return out


def _chans(n):
    return (["R", "G", "B"] + ["G"] * (n - 3)) if n >= 3 else ["G"] * n


def apply(img: np.ndarray, *, gain: np.ndarray | None = None, lens: dict | None = None,
          rs: list | None = None, edge: str = "edge", out_size=None) -> np.ndarray:
    """img (H, W, C) float. gain (h, w) vignette gain on the image's grid (image must be linear). lens: {"R","G",
    "B"} redistort STMaps. rs: list of rs STMaps (one, or one per motion-blur sub-sample; averaged). Alpha / extra
    channels use G's map. out_size: the plate (w, h) when the clean image is bigger (overscan)."""
    h, w = img.shape[:2]
    ow, oh = out_size or (w, h)
    out = np.asarray(img, np.float32).copy()
    if gain is not None:
        out[..., :3] *= _resize_gain(gain, w, h)[..., None]
    rs = [resize_map(m, ow, oh) for m in (rs or [])]
    lens = {c: resize_map(m, ow, oh) for c, m in (lens or {}).items()}
    if not rs and not lens:
        return out
    chans = _chans(out.shape[2])
    result = np.zeros((oh, ow, out.shape[2]), np.float32)
    for m in (rs or [None]):
        def map_for(c, m=m):
            L = lens.get(c)
            return L if m is None else (m if L is None else compose(m, L))

        for st, idx in _by_map(chans, map_for):
            result[..., idx] += warp(out[..., idx], st, edge, (ow, oh))
    return result / max(1, len(rs))
