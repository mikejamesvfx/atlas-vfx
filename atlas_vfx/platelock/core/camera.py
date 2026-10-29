"""Pinhole cameras with rolling shutter.

Conventions (OpenCV / Kornia, same as Atlas Camera):
  x_cam = R @ X_world + t        (R, t are world -> camera)
  pixel  = K @ x_cam / z          origin top-left, pixel centres on integers
  time   = fractional frame index

Rolling shutter (camera_side/rs_time.py, the one model all atlas-vfx RS code shares): row v (OpenCV pixel,
centre of row i at i) is exposed at
  time = frame + readout * ((v + 0.5) / H - offset)
where readout is the full top-to-bottom readout in frames
(readout_seconds * fps) and offset is 0.5 when the solved pose for a frame
represents mid-readout ("centre", the default and what most solvers land on)
or 0.0 when it represents the top row ("top"). Until 2026-10-22 this was v / (H - 1), 0.05 % off the others.
Outside the track pose_at clamps (holds the end pose); rs_time's users extrapolate from the inner pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np

from ...camera_side import rs_time


# ----------------------------------------------------------------- rotations

def quat_from_rot(R: np.ndarray) -> np.ndarray:
    """(…,3,3) rotation -> (…,4) unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for n in range(R.shape[0]):
        m = R[n]
        if tr[n] > 0:
            s = np.sqrt(tr[n] + 1.0) * 2
            q[n] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[n] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[n] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[n] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shp + (4,))


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Vectorised slerp; q0, q1 (…,4), a (…)."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float).copy()
    a = np.asarray(a, float)[..., None]
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    lin = d > 1 - 1e-12          # true slerp down to ~1e-6 rad (was 0.9995: nlerp below 3.6 deg, ~0.004 px off)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    s = np.where(s < 1e-12, 1.0, s)
    w0 = np.where(lin, 1 - a, np.sin((1 - a) * th) / s)
    w1 = np.where(lin, a, np.sin(a * th) / s)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation from small Euler angles (radians), applied X then Y then Z."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# ------------------------------------------------------------------ cameras

@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_fov(cls, width: int, height: int, focal_px: float) -> "Intrinsics":
        return cls(width, height, focal_px, focal_px, (width - 1) / 2, (height - 1) / 2)


@dataclass
class CameraTrack:
    """A per-frame pinhole camera track, optionally with rolling shutter."""

    intr: Intrinsics
    R: np.ndarray            # (N,3,3) world -> camera
    t: np.ndarray            # (N,3)
    readout: float = 0.0     # top-to-bottom readout, in frames
    rs_anchor: str = "centre"  # "centre" or "top"
    _q: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.R = np.asarray(self.R, float).reshape(-1, 3, 3)
        self.t = np.asarray(self.t, float).reshape(-1, 3)
        self._q = quat_from_rot(self.R)
        if self.rs_anchor not in ("centre", "top"):
            raise ValueError("rs_anchor must be 'centre' or 'top'")

    @property
    def n(self) -> int:
        return self.R.shape[0]

    @property
    def centres(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)

    # -- time

    def row_time(self, frame: float, v: np.ndarray) -> np.ndarray:
        if self.readout == 0:
            return np.full(np.shape(v), float(frame))
        return rs_time.row_time(frame, v, self.intr.height, self.readout, pixels="opencv", anchor=self.rs_anchor)

    def pose_at(self, time: np.ndarray):
        """Interpolated (R, t) at fractional times; clamps outside the track."""
        time = np.asarray(time, float)
        tt = np.clip(time, 0, self.n - 1)
        i0 = np.clip(np.floor(tt).astype(int), 0, max(self.n - 2, 0))
        i1 = np.minimum(i0 + 1, self.n - 1)
        a = tt - i0
        q = slerp(self._q[i0], self._q[i1], a)
        R = rot_from_quat(q)
        c = self.centres
        C = c[i0] * (1 - a)[..., None] + c[i1] * a[..., None]
        t = -np.einsum("...ij,...j->...i", R, C)
        return R, t

    # -- projection

    def project(self, Xw: np.ndarray, frame: float, iters: int = 4):
        """World points (M,3) -> u, v, z at a frame, honouring rolling shutter."""
        Xw = np.asarray(Xw, float).reshape(-1, 3)
        K = self.intr.K
        R, t = self.pose_at(np.full(Xw.shape[0], float(frame)))
        for k in range(iters if self.readout else 1):
            Xc = np.einsum("mij,mj->mi", R, Xw) + t
            z = Xc[:, 2]
            zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
            u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
            if self.readout and k < iters - 1:
                R, t = self.pose_at(self.row_time(frame, v))
        return u, v, z

    def backproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray, frame: float) -> np.ndarray:
        """Pixels + camera-space z depth -> world points (M,3)."""
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        z = np.asarray(depth, float).ravel()
        K = self.intr.K
        Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
        R, t = self.pose_at(self.row_time(frame, v))
        return np.einsum("mji,mj->mi", R, Xc - t)

    def ray(self, u, v, frame):
        """Ray origins and world directions for pixels."""
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        K = self.intr.K
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=1)
        R, t = self.pose_at(self.row_time(frame, v))
        o = -np.einsum("mji,mj->mi", R, t)
        dw = np.einsum("mji,mj->mi", R, d)
        return o, dw

    # -- io

    def to_dict(self) -> dict:
        i = self.intr
        return {
            "schema": "platelock.track/1",
            "convention": "opencv world_to_camera",
            "width": i.width, "height": i.height,
            "fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy,
            "readout_frames": self.readout, "rs_anchor": self.rs_anchor,
            "frames": [{"R": self.R[n].tolist(), "t": self.t[n].tolist()} for n in range(self.n)],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraTrack":
        intr = Intrinsics(int(d["width"]), int(d["height"]), d["fx"], d["fy"], d["cx"], d["cy"])
        R = np.array([f["R"] for f in d["frames"]])
        t = np.array([f["t"] for f in d["frames"]])
        return cls(intr, R, t, float(d.get("readout_frames", 0.0)), d.get("rs_anchor", "centre"))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "CameraTrack":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def scaled(self, width: int, height: int) -> "CameraTrack":
        """Same cameras for frames resized to width x height (pixel centres on integers)."""
        i = self.intr
        sx, sy = width / i.width, height / i.height
        intr = Intrinsics(width, height, i.fx * sx, i.fy * sy, (i.cx + 0.5) * sx - 0.5, (i.cy + 0.5) * sy - 0.5)
        return CameraTrack(intr, self.R.copy(), self.t.copy(), self.readout, self.rs_anchor)

    def with_readout(self, readout: float, rs_anchor: str | None = None) -> "CameraTrack":
        return CameraTrack(self.intr, self.R.copy(), self.t.copy(), readout, rs_anchor or self.rs_anchor)


def readout_frames(readout_ms: float, fps: float) -> float:
    """Convert a sensor readout time in milliseconds to frames."""
    return rs_time.readout_frames(readout_ms, fps)
