"""Re-photograph the AI background through the plate camera.

The plate is blue behind the performers, so there is nothing to warp the AI
background onto. Instead each AI background pixel is lifted to 3D with its
depth, moved into the plate's world with the aligned solve, and projected
through the plate camera row by row (rolling shutter), optionally integrated
over the shutter for motion blur.

Output per plate frame: the re-shot image, the backward map into the AI frame
(so Nuke can redo it from the STMap), and a hole mask for disocclusions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .align import Sim3
from .camera import CameraTrack
from . import warp


@dataclass
class ReshootResult:
    image: np.ndarray        # HxWxC
    map_x: np.ndarray        # HxW, into the AI source frame (centre of shutter)
    map_y: np.ndarray
    valid: np.ndarray        # HxW float, 1 where AI pixels landed
    source_frame: int


def _splat_map(src_track: CameraTrack, src_frame: int, depth: np.ndarray, src_valid: np.ndarray,
               dst_track: CameraTrack, dst_time: float, g: Sim3, fill_radius: float):
    """Forward-project source pixels, z-buffer, and return a sub-pixel backward map."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
    Xs = src_track.backproject(us, vs, depth, src_frame)
    Xd = g.apply(Xs)
    u, v, z = dst_track.project(Xd, dst_time)
    u = u.reshape(H, W)
    v = v.reshape(H, W)
    z = z.reshape(H, W)

    # Jacobian of target position wrt source pixel, for sub-pixel inversion
    du_dx, du_dy = np.gradient(u, axis=1), np.gradient(u, axis=0)
    dv_dx, dv_dy = np.gradient(v, axis=1), np.gradient(v, axis=0)
    det = du_dx * dv_dy - du_dy * dv_dx
    bad = np.abs(det) < 1e-6
    det = np.where(bad, 1.0, det)

    th, tw = dst_track.intr.height, dst_track.intr.width
    ix = np.rint(u).astype(np.int64)
    iy = np.rint(v).astype(np.int64)
    ok = (src_valid > 0.5) & (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & ~bad
    ok &= (ix >= 0) & (ix < tw) & (iy >= 0) & (iy < th)

    lin = (iy * tw + ix)[ok]
    zz = z[ok]
    zbuf = np.full(th * tw, np.inf)
    np.minimum.at(zbuf, lin, zz)
    win = zz <= zbuf[lin] * (1 + 1e-6)

    ex = (ix[ok] - u[ok])[win]
    ey = (iy[ok] - v[ok])[win]
    a, b, c, d = du_dx[ok][win], du_dy[ok][win], dv_dx[ok][win], dv_dy[ok][win]
    dd = det[ok][win]
    sx = us[ok][win] + (d * ex - b * ey) / dd
    sy = vs[ok][win] + (-c * ex + a * ey) / dd
    lw = lin[win]

    mx = np.full(th * tw, np.nan)
    my = np.full(th * tw, np.nan)
    mx[lw] = sx
    my[lw] = sy
    mx = mx.reshape(th, tw)
    my = my.reshape(th, tw)
    hit = np.isfinite(mx)

    valid = hit.astype(np.float32)
    if fill_radius > 0 and hit.any():
        dist, (jy, jx) = distance_transform_edt(~hit, return_indices=True)
        fill = (~hit) & (dist <= fill_radius)
        mx[fill] = mx[jy[fill], jx[fill]]
        my[fill] = my[jy[fill], jx[fill]]
        valid[fill] = 1.0
    mx = np.where(np.isfinite(mx), mx, -1e4).astype(np.float32)
    my = np.where(np.isfinite(my), my, -1e4).astype(np.float32)
    return mx, my, valid


def reshoot(src_img: np.ndarray, src_depth: np.ndarray, src_track: CameraTrack, src_frame: int,
            dst_track: CameraTrack, dst_frame: int, g: Sim3, *, src_valid: np.ndarray | None = None,
            shutter: float = 0.5, samples: int = 1, fill_radius: float = 1.5,
            depth_scale: float | None = None) -> ReshootResult:
    """Re-shoot one AI frame into one plate frame.

    g: Sim3 from the AI solve's world into the plate solve's world
       (from align.anchor_align). Depth is in AI camera units; it is used in
       the AI world, so no scaling is needed here.
    shutter: exposure as a fraction of a frame (180 degrees = 0.5).
    samples: sub-frame samples across the shutter for motion blur (1 = none).
    """
    depth = np.asarray(src_depth, float)
    if depth_scale is not None:
        depth = depth * depth_scale
    if src_valid is None:
        src_valid = np.ones(depth.shape, np.float32)
    img = np.asarray(src_img, np.float32)

    n = max(int(samples), 1)
    offsets = [0.0] if n == 1 else list(shutter * ((np.arange(n) + 0.5) / n - 0.5))
    acc = None
    wacc = None
    centre = None
    for o in offsets:
        mx, my, val = _splat_map(src_track, src_frame, depth, src_valid, dst_track, dst_frame + o, g, fill_radius)
        im = warp.remap(img, mx, my, border="replicate")
        w3 = val[..., None] if im.ndim == 3 else val
        acc = im * w3 if acc is None else acc + im * w3
        wacc = w3.copy() if wacc is None else wacc + w3
        if centre is None or abs(o) < abs(centre[3]):
            centre = (mx, my, val, o)
    out = acc / np.maximum(wacc, 1e-6)
    return ReshootResult(out.astype(np.float32), centre[0], centre[1], centre[2], src_frame)


def reshoot_merge(src_imgs, src_depths, src_valids, src_track: CameraTrack, src_frames,
                  dst_track: CameraTrack, dst_frame: int, g: Sim3, min_frames: int = 1, **kw) -> ReshootResult:
    """Re-shoot several AI frames into one plate frame and fill each other's holes.

    Pass the nearest frame first: it wins wherever it has data, and later frames
    only fill what is still missing (behind the performers, at frame edges).
    Stops early once the frame is full and at least `min_frames` were used.
    """
    best = None
    for k, (img, dep, val, f) in enumerate(zip(src_imgs, src_depths, src_valids, src_frames)):
        if best is not None and k >= min_frames and (best.valid > 0.5).all():
            break
        r = reshoot(img, dep, src_track, f, dst_track, dst_frame, g, src_valid=val, **kw)
        if best is None:
            best = r
            continue
        need = (best.valid < 0.5) & (r.valid > 0.5)
        if not need.any():
            continue
        best.image[need] = r.image[need]
        best.valid[need] = 1.0
        # maps point into the first source frame; record filled pixels as holes in
        # the map sense by marking them invalid there (they came from another frame)
        best.map_x[need] = -1e4
        best.map_y[need] = -1e4
    return best
