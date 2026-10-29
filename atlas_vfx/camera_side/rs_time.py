"""The one rolling-shutter time model for atlas-vfx. Every RS code path imports it (sensor/rs.py maps,
camera_side/rs_solve.py, camera_side/grid_rs.py, platelock/core/camera.py), so a camera solved by one is read the
same way by the others. tests/test_rs_time.py checks that all four agree.

Row time. A frame's pose is the camera at the frame time; the sensor reads top to bottom over `readout` frames:

    t(y) = f + (y / H - anchor) * readout          anchor 0.5 ("centre", default) or 0.0 ("top")

y is a SENSOR row (distorted, as recorded), in CONTRACT pixels: 0 = top edge of the image, row i's centre at
i + 0.5 (shot-camera@0.1, COLMAP, the Sensor maps). OpenCV pixels put row i's centre at i; pass pixels="opencv"
and they are shifted by +0.5 here. y is clipped to the image, [0, H].

Pose between frames: rotation along the geodesic (slerp), centre linear, between whole frames i0 and i0 + 1.
The geodesic is the same for camera-to-world and world-to-camera matrices (one is the transpose of the other), so
interp_rot works on either. Pairing: an observation in frame f pairs with f + 1 if its row is at or below the anchor,
else with f - 1 (= floor(f + tau) for |tau| < 1), clipped to the run so the ends extrapolate from the inner pair.

Pure numpy / scipy. No imports from layers/, deliver/, sensor/ or platelock/.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

ANCHORS = {"centre": 0.5, "top": 0.0}
PIXELS = {"contract": 0.0, "opencv": 0.5}      # shift that takes the convention to contract pixels


def readout_frames(readout_ms: float, fps: float) -> float:
    """Sensor readout in milliseconds -> frames."""
    return float(readout_ms) / 1000.0 * float(fps)


def readout_ms(readout_fr: float, fps: float) -> float:
    """Sensor readout in frames -> milliseconds."""
    return float(readout_fr) / float(fps) * 1000.0


def row_frac(y, h: int, *, pixels: str = "contract", anchor: str = "centre") -> np.ndarray:
    """tau / readout for sensor rows y: -0.5 at the top edge, +0.5 at the bottom edge (anchor centre)."""
    yc = np.asarray(y, np.float64) + PIXELS[pixels]
    return np.clip(yc, 0.0, float(h)) / float(h) - ANCHORS[anchor]


def row_tau(y, h: int, readout_fr: float, *, pixels: str = "contract", anchor: str = "centre") -> np.ndarray:
    """Frames from the frame time to the capture time of sensor row y."""
    return row_frac(y, h, pixels=pixels, anchor=anchor) * float(readout_fr)


def row_time(frame, y, h: int, readout_fr: float, *, pixels: str = "contract", anchor: str = "centre"):
    """Capture time (fractional frame) of sensor row y in `frame`."""
    return np.asarray(frame, np.float64) + row_tau(y, h, readout_fr, pixels=pixels, anchor=anchor)


def pair_index(fi, rowfrac, lo=0, hi=None) -> np.ndarray:
    """First frame of the pair each observation interpolates between: fi if rowfrac >= 0 else fi - 1, clipped to
    [lo, hi - 1] (lo, hi: first and last frame index of the observation's run; scalars or arrays)."""
    fi = np.asarray(fi)
    i0 = np.where(np.asarray(rowfrac) >= 0, fi, fi - 1)
    if hi is None:
        hi = int(fi.max())
    return np.clip(i0, lo, np.asarray(hi) - 1)


def rel_rotvecs(R: np.ndarray) -> np.ndarray:
    """log(R_i^T R_{i+1}) for consecutive rotations (n,3,3) -> (n-1,3)."""
    return Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).as_rotvec()


def interp_rot(R: np.ndarray, rel: np.ndarray, i0, a) -> np.ndarray:
    """Geodesic rotation a of the way from R[i0] to R[i0 + 1]: R[i0] Exp(a log(R[i0]^T R[i0+1])). rel = rel_rotvecs(R).
    a outside [0, 1] extrapolates along the same geodesic."""
    a = np.asarray(a, np.float64)
    return R[i0] @ Rotation.from_rotvec(a[..., None] * rel[i0]).as_matrix()


def interp_centre(C: np.ndarray, i0, a) -> np.ndarray:
    a = np.asarray(a, np.float64)[..., None]
    return C[i0] * (1 - a) + C[np.asarray(i0) + 1] * a


def interp_pose(RA, RB, CA, CB, a):
    """Single pose between two (R, C): geodesic rotation, linear centre."""
    rel = Rotation.from_matrix(np.asarray(RA).T @ np.asarray(RB)).as_rotvec()
    R = np.asarray(RA) @ Rotation.from_rotvec(float(a) * rel).as_matrix()
    return R, np.asarray(CA) * (1 - a) + np.asarray(CB) * a
