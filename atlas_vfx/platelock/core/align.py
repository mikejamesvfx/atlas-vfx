"""Put two independent solves (AI video, plate) in one space and measure the delta.

Both solves come out of their own gauge (origin, rotation, scale). The AI take
was generated from the plate's first frame, so frame 0 is where they agree:
anchor_align makes the AI camera at the anchor frame coincide with the plate
camera there, with a scale you supply (body height ratio, a known object,
or scale_from_depth). umeyama is there for well-conditioned moving cameras.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraTrack, rot_angle_deg


@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return self.s * X @ self.R.T + self.t

    def inverse(self) -> "Sim3":
        Ri = self.R.T
        return Sim3(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)

    def compose(self, other: "Sim3") -> "Sim3":
        """self ∘ other."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * self.R @ other.t + self.t)

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / src.shape[0]
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def anchor_align(src: CameraTrack, dst: CameraTrack, frame: int = 0, scale: float = 1.0) -> Sim3:
    """Sim3 taking src-world points to dst-world so the cameras agree at `frame`.

    `scale` multiplies src units into dst units (dst_metres / src_units).
    """
    Rs, ts = src.R[frame], src.t[frame]
    Rd, td = dst.R[frame], dst.t[frame]
    R = Rd.T @ Rs
    T = Rd.T @ (scale * ts - td)
    return Sim3(float(scale), R, T)


def transform_track(track: CameraTrack, g: Sim3) -> CameraTrack:
    """Re-express a camera track in the world that `g` maps into.

    Camera-space units are scaled by g.s too, so depths from that solve must
    be multiplied by g.s to stay consistent.
    """
    Rg_T = g.R.T
    R2 = np.einsum("nij,jk->nik", track.R, Rg_T)
    t2 = g.s * track.t - np.einsum("nij,j->ni", R2, g.t)
    return CameraTrack(track.intr, R2, t2, track.readout, track.rs_anchor)


def camera_delta(a: CameraTrack, b: CameraTrack) -> dict:
    """Per-frame difference between two tracks already in the same world."""
    n = min(a.n, b.n)
    dR = np.einsum("nij,nkj->nik", b.R[:n], a.R[:n])
    rot = rot_angle_deg(dR)
    trans = np.linalg.norm(a.centres[:n] - b.centres[:n], axis=1)
    return {"rotation_deg": rot, "translation": trans,
            "max_rotation_deg": float(rot.max()), "max_translation": float(trans.max())}


def scale_from_heights(src_heights, dst_heights) -> float:
    """Scale from matched lengths (e.g. performer heights) in each solve."""
    s = np.asarray(dst_heights, float) / np.asarray(src_heights, float)
    return float(np.median(s))


def scale_from_depth(src_depth: np.ndarray, dst_depth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Scale from two depth maps of the same frame (median ratio)."""
    a = np.asarray(src_depth, float)
    b = np.asarray(dst_depth, float)
    m = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= np.asarray(mask) > 0.5
    return float(np.median(b[m] / a[m]))
