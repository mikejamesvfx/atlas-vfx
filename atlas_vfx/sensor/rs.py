"""Rolling shutter as per-frame STMaps that depend on depth and motion.

Physics. Sensor row y (contract pixels, 0 at the top edge, H at the bottom) is captured at
    t(y) = t_frame + (y / H - 0.5) * T_readout
so the middle of the frame is the solved pose's time (what solvers land on; platelock calls it rs_anchor "centre").
Row time is indexed on SENSOR rows, i.e. distorted y, not undistorted y.

Two maps per frame, both Nuke STMaps (s = x / W, t = 1 - y / H, like platelock warp.to_stmap):
  rs_map    output = the sensor (rolling shutter) image; samples the clean global-shutter image.  Apply RS.
  ders_map  output = the clean image; samples the rolling-shutter plate.                         Remove RS.

Camera term. The plate depth (MoGe z depth, metres, in the plate's own pixels, so in RS sensor space) is what ties
a sensor pixel to a scene point:
  rs_map:   sensor pixel p, row time t(p_y). Undistort, unproject with depth D(p) through the pose at t(p_y), then
            project through the pose at t_frame and distort. Closed form, because the depth is indexed on p.
  ders_map: clean pixel q. The scene point is on q's ray at t_frame; its sensor position p decides its row time,
            which moves the camera, which moves p. Solved by fixed-point iteration (`iters`): at the current p take
            the row time and the depth D(p), put the point on q's ray at that z depth in the row's camera, project.
The two are exact inverses of each other at convergence; `roundtrip` measures how close they are.

Poses between frames: rotation slerped, translation lerped, between frame f and its neighbour on the side of t
(extrapolated from the inner pair at the first / last frame). Uses the shot-camera@0.1 contract: metres, Y up,
camera looking down -Z, column-vector cam_to_world.

Object term (optional): `object_flow` is the RESIDUAL flow in plate pixels per frame (measured flow minus the
camera-induced flow), (H, W, 2) at plate or any size, indexed on the plate. A point on row y was seen tau(y) =
(y / H - 0.5) * T frames off the frame time, so it is displaced by tau(y) * flow. No flow model in phase 1.

Motion blur (optional, off by default): `time_offset` shifts every row's time; Sensor RS Maps writes one rs_map per
sub-sample across each row's exposure window (shutter angle) and Sensor Apply averages them.
"""
from __future__ import annotations

import numpy as np

from ..camera_side import rs_time
from .lens import BrownConrady, from_st, pixel_grid, to_st

CHUNK_ROWS = 256


def readout_frames(readout_s: float, fps: float) -> float:
    return rs_time.readout_frames(float(readout_s) * 1000.0, fps)


def row_tau(y, h: int, readout_fr: float):
    """Frames from the frame time to the capture time of sensor row y (contract pixels)."""
    return rs_time.row_tau(y, h, readout_fr)


# ------------------------------------------------------------------------------------- poses

def pose_at(cam, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(R c2w 3x3, C 3) at fractional Nuke frame t: slerp rotation, lerp translation between whole frames."""
    frames = cam.frames
    if len(frames) < 2:
        M = cam.cam_to_world(frames[0])
        return M[:3, :3], M[:3, 3]
    f0 = int(np.floor(t))
    f0 = min(max(f0, frames[0]), frames[-2])
    a = t - f0
    if f0 not in cam._by_frame or f0 + 1 not in cam._by_frame:
        raise KeyError(f"pose_at({t}): frames {f0}/{f0 + 1} not both in {cam.name}")
    A, B = cam.cam_to_world(f0), cam.cam_to_world(f0 + 1)
    return rs_time.interp_pose(A[:3, :3], B[:3, :3], A[:3, 3], B[:3, 3], a)


class RowPoses:
    """Pose of every sensor row relative to the frame-time camera, as a fine table in y (linear in between).

    M(y) = R0^T R(t(y)), b(y) = R0^T (C(t(y)) - C0): a point X_t in the row camera is X_0 = M X_t + b in the frame
    camera. Sampled every `step` px of y, which is far finer than the motion inside one row."""

    def __init__(self, cam, frame: int, h: int, readout_fr: float, time_offset: float = 0.0, step: float = 0.25):
        self.h = h
        self.readout_fr = readout_fr
        M0 = cam.cam_to_world(frame)
        R0, C0 = M0[:3, :3], M0[:3, 3]
        self.ys = np.arange(0.0, h + step, step)
        n = len(self.ys)
        self.M = np.empty((n, 3, 3))
        self.b = np.empty((n, 3))
        moving = readout_fr != 0 or time_offset != 0
        for i, y in enumerate(self.ys):
            if moving:
                R, C = pose_at(cam, frame + time_offset + float(row_tau(y, h, readout_fr)))
            else:
                R, C = R0, C0
            self.M[i] = R0.T @ R
            self.b[i] = R0.T @ (C - C0)
        self.flatM = self.M.reshape(n, 9)

    def at(self, y):
        """(M (…,3,3), b (…,3)) at continuous rows y."""
        y = np.clip(np.asarray(y, np.float64), self.ys[0], self.ys[-1])
        f = (y - self.ys[0]) / (self.ys[1] - self.ys[0])
        i0 = np.minimum(np.floor(f).astype(np.int64), len(self.ys) - 2)
        a = (f - i0)[..., None]
        M = self.flatM[i0] * (1 - a) + self.flatM[i0 + 1] * a
        b = self.b[i0] * (1 - a) + self.b[i0 + 1] * a
        return M.reshape(y.shape + (3, 3)), b


# ------------------------------------------------------------------------------- sampling

def sample(img: np.ndarray, x, y, w: int, h: int) -> np.ndarray:
    """Bilinear sample of a map stored at any resolution, at contract pixel positions of a w x h image.
    img (hi, wi) or (hi, wi, c). Clamped at the edges."""
    hi, wi = img.shape[:2]
    xs = np.clip(np.asarray(x) * (wi / w) - 0.5, 0, wi - 1)
    ys = np.clip(np.asarray(y) * (hi / h) - 0.5, 0, hi - 1)
    x0 = np.minimum(np.floor(xs).astype(np.int64), max(wi - 2, 0))
    y0 = np.minimum(np.floor(ys).astype(np.int64), max(hi - 2, 0))
    x1, y1 = np.minimum(x0 + 1, wi - 1), np.minimum(y0 + 1, hi - 1)
    fx, fy = xs - x0, ys - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


def prepare_depth(depth: np.ndarray, valid: np.ndarray | None = None, far_m: float | None = None) -> np.ndarray:
    """Depth with holes filled: nearest valid pixel (default), or `far_m` everywhere invalid (sky)."""
    z = np.asarray(depth, np.float64).copy()
    bad = ~np.isfinite(z) | (z <= 0)
    if valid is not None:
        bad |= np.asarray(valid) <= 0.5
    if not bad.any():
        return z
    if bad.all():
        raise ValueError("depth has no valid pixels")
    if far_m is not None:
        z[bad] = far_m
        return z
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
    return z[iy, ix]


# ---------------------------------------------------------------------------------- maps

class _Setup:
    def __init__(self, cam, frame, w, h, readout_s, fps, lens, time_offset):
        self.w, self.h = w, h
        pw = cam.data["plate"]["width"]
        k = cam.intrinsics(frame)
        s = w / pw
        self.fx, self.fy, self.cx, self.cy = k.fx * s, k.fy * s, k.cx * s, k.cy * s
        if lens is None:
            lens = BrownConrady.from_camera(cam, frame)          # lenses are always given at plate resolution
        self.lens = lens.scaled(s) if s != 1 else lens
        self.T = readout_frames(readout_s, fps)
        self.poses = RowPoses(cam, frame, h, self.T, time_offset)
        self.time_offset = time_offset


def _project(S: _Setup, X):
    """Camera-space points (…,3) -> distorted pixels."""
    z = -X[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xu = S.cx + S.fx * X[..., 0] / z
        yu = S.cy - S.fy * X[..., 1] / z
    return S.lens.distort(xu, yu)


def _rs_chunk(S, X, Y, depth, flow):
    xu, yu = S.lens.undistort(X, Y)
    D = sample(depth, X, Y, S.w, S.h)
    Xt = np.stack([(xu - S.cx) / S.fx * D, -(yu - S.cy) / S.fy * D, -D], -1)
    M, b = S.poses.at(Y)
    X0 = np.einsum("...ij,...j->...i", M, Xt) + b
    qx, qy = _project(S, X0)
    if flow is not None:
        tau = row_tau(Y, S.h, S.T) + S.time_offset
        f = sample(flow, X, Y, S.w, S.h)
        qx, qy = qx - tau * f[..., 0], qy - tau * f[..., 1]
    return qx, qy


def _ders_chunk(S, X, Y, depth, flow, iters):
    xu, yu = S.lens.undistort(X, Y)
    d = np.stack([(xu - S.cx) / S.fx, -(yu - S.cy) / S.fy, -np.ones_like(xu)], -1)   # q's ray, z depth 1, frame cam
    px, py = X.copy(), Y.copy()
    for _ in range(max(1, iters)):
        M, b = S.poses.at(py)
        D = sample(depth, px, py, S.w, S.h)
        Mtd = np.einsum("...ji,...j->...i", M, d)                      # M^T d
        Mtb = np.einsum("...ji,...j->...i", M, b)                      # M^T b
        lam = (Mtb[..., 2] - D) / Mtd[..., 2]                          # z depth of the point in the row camera = D
        Xt = lam[..., None] * Mtd - Mtb
        nx, ny = _project(S, Xt)
        if flow is not None:
            tau = row_tau(py, S.h, S.T) + S.time_offset
            f = sample(flow, px, py, S.w, S.h)
            nx, ny = nx + tau * f[..., 0], ny + tau * f[..., 1]
        px, py = nx, ny
    return px, py


def rs_maps(cam, frame: int, depth: np.ndarray, readout_s: float, fps: float | None = None, object_flow=None,
            iters: int = 3, *, width: int | None = None, height: int | None = None, lens: BrownConrady | None = None,
            valid: np.ndarray | None = None, far_m: float | None = None, time_offset: float = 0.0,
            want_ders: bool = True):
    """(rs_map, ders_map) float32 STMaps (H, W, 2) for one frame, at width x height (default the plate size).

    depth: z depth in metres on the plate (any resolution; sampled bilinearly, i.e. upsampled to the map size).
    readout_s: top-to-bottom sensor readout in seconds; fps defaults to the camera's. lens: defaults to the camera's
    (pinhole if it has none). ders_map is None if want_ders is False."""
    fps = float(fps or cam.data["shot"]["fps"])
    w = int(width or cam.data["plate"]["width"])
    h = int(height or cam.data["plate"]["height"])
    S = _Setup(cam, frame, w, h, readout_s, fps, lens, time_offset)
    depth = prepare_depth(depth, valid, far_m)
    rs = np.empty((h, w, 2), np.float32)
    de = np.empty((h, w, 2), np.float32) if want_ders else None
    xs = np.arange(w, dtype=np.float64) + 0.5
    for r0 in range(0, h, CHUNK_ROWS):
        r1 = min(h, r0 + CHUNK_ROWS)
        X, Y = np.meshgrid(xs, np.arange(r0, r1, dtype=np.float64) + 0.5)
        qx, qy = _rs_chunk(S, X, Y, depth, object_flow)
        rs[r0:r1] = to_st(qx, qy, w, h)
        if want_ders:
            px, py = _ders_chunk(S, X, Y, depth, object_flow, iters)
            de[r0:r1] = to_st(px, py, w, h)
    return rs, de


def identity_map(w: int, h: int) -> np.ndarray:
    X, Y = pixel_grid(w, h)
    return to_st(X, Y, w, h)


def roundtrip(rs_map: np.ndarray, ders_map: np.ndarray, margin_px: float = 2.0) -> dict:
    """How far rs_map(ders_map(q)) lands from q, in pixels, over the pixels whose RS position is inside the frame.
    rs_map is sampled bilinearly at the de-RS positions (what two STMap nodes in a row do)."""
    h, w = rs_map.shape[:2]
    X, Y = pixel_grid(w, h)
    px, py = from_st(ders_map, w, h)
    inside = (px > margin_px) & (px < w - margin_px) & (py > margin_px) & (py < h - margin_px)
    st = sample(rs_map.astype(np.float64), px, py, w, h)
    qx, qy = from_st(st, w, h)
    e = np.hypot(qx - X, qy - Y)[inside]
    if not e.size:
        return {"median_px": None, "p95_px": None, "max_px": None, "pixels": 0}
    return {"median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)), "max_px": float(e.max()),
            "pixels": int(e.size)}


def skew_px_per_row(st_map: np.ndarray, band: float = 0.1, centre: float = 0.1) -> dict:
    """Horizontal / vertical displacement slope (px per row) of a map from the top band to the bottom band, in the
    central `centre` fraction of the columns (off-centre columns move faster in a pan, 1 + x^2/f^2).
    For an rs_map: how far a vertical edge at the image centre leans per sensor row (sign: map displacement, so an
    edge in the RS image leans by minus this). dx/dy_mean_px are over the whole frame."""
    h, w = st_map.shape[:2]
    X, Y = pixel_grid(w, h)
    sx, sy = from_st(st_map, w, h)
    dx, dy = sx - X, sy - Y
    n = max(1, int(h * band))
    c0, c1 = int(w * (0.5 - centre / 2)), max(int(w * (0.5 + centre / 2)), int(w * (0.5 - centre / 2)) + 1)
    top, bot = (slice(0, n), slice(c0, c1)), (slice(h - n, h), slice(c0, c1))
    rows = (Y[bot].mean() - Y[top].mean())
    return {"dx_per_row": float((dx[bot].mean() - dx[top].mean()) / rows),
            "dy_per_row": float((dy[bot].mean() - dy[top].mean()) / rows),
            "dx_mean_px": float(dx.mean()), "dy_mean_px": float(dy.mean())}


def row_times(h: int, frame: float, readout_fr: float) -> np.ndarray:
    """Capture time (fractional frames) of each sensor row's centre."""
    return frame + row_tau(np.arange(h) + 0.5, h, readout_fr)


def slit_scan(source_at, h: int, frame: float, readout_fr: float, exposure_fr: float = 0.0, samples: int = 1):
    """RS frame built row by row from a time-continuous source: source_at(t) -> image (h, w, c) at time t.
    Exposure (frames, = shutter angle / 360) averages `samples` times across each row's window. Pure time model, no
    geometry: for flashes, strobes and (phase 2) VFI frames where the camera term is not enough."""
    ts = row_times(h, frame, readout_fr)
    offs = [0.0] if samples <= 1 or exposure_fr <= 0 else list(((np.arange(samples) + 0.5) / samples - 0.5) * exposure_fr)
    out = None
    cache = {}
    for y, t in enumerate(ts):
        acc = 0.0
        for o in offs:
            key = round(float(t + o), 9)
            if key not in cache:
                cache[key] = source_at(key)
            acc = acc + cache[key][y]
        row = acc / len(offs)
        if out is None:
            out = np.empty((h,) + np.shape(row), np.float32)
        out[y] = row
    return out
