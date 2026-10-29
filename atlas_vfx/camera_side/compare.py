"""Solve Compare: how well two cameras of the same shot agree (ported from prototypes/vfx_probe/align_nuke.py).

Similarity fit (Umeyama) of camera centres on the frames both have: B ~ s * R @ A + t. Reports the scale between
them, centre residuals (in B's units and as % of B's path length), and the angle between view directions after the
fit. Both cameras are shot-camera@0.1, so both look down their own -Z.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_vfx.contracts import shot_camera as sc


def umeyama(A: np.ndarray, B: np.ndarray):
    """(s, R, t) minimising |B - (s R A + t)|^2 over rows. Needs 3+ non-collinear points."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ma, mb = A.mean(0), B.mean(0)
    Ac, Bc = A - ma, B - mb
    U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    var_a = (Ac ** 2).sum(1).mean()
    if var_a <= 0:
        raise ValueError("camera A does not move: cannot fit a similarity")
    s = float(np.trace(np.diag(S) @ D) / var_a)
    t = mb - s * R @ ma
    return s, R, t


def _stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max()),
            "mean": float(x.mean())}


def compare(a: sc.ShotCamera | dict, b: sc.ShotCamera | dict, frames: tuple[int, int] | None = None) -> dict:
    a = a if isinstance(a, sc.ShotCamera) else sc.ShotCamera(a)
    b = b if isinstance(b, sc.ShotCamera) else sc.ShotCamera(b)
    common = sorted(set(a.frames) & set(b.frames))
    if frames:
        common = [f for f in common if frames[0] <= f <= frames[1]]
    if len(common) < 3:
        raise ValueError(f"only {len(common)} frames in common between {a.name} and {b.name}")
    A = np.array([a.position(f) for f in common])
    B = np.array([b.position(f) for f in common])
    s, R, t = umeyama(A, B)
    res = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    ang, roll = [], []
    for f in common:
        Ra, Rb = R @ a.cam_to_world(f)[:3, :3], b.cam_to_world(f)[:3, :3]
        fa, fb = -Ra[:, 2], -Rb[:, 2]
        ang.append(math.degrees(math.acos(float(np.clip(fa @ fb, -1, 1)))))
        Rrel = Ra.T @ Rb                                   # whole relative rotation, view + roll
        roll.append(math.degrees(math.acos(float(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))))
    fa = np.array([a.intrinsics(f).focal_mm for f in common])
    fb = np.array([b.intrinsics(f).focal_mm for f in common])
    return {
        "a": {"name": a.name, "source": a.data["source"]}, "b": {"name": b.name, "source": b.data["source"]},
        "frames_compared": len(common), "frame_range": [common[0], common[-1]],
        "scale_b_per_a": s, "rotation_b_from_a": R.tolist(), "translation_b": t.tolist(),
        "centre_residual_b_units": _stats(res),
        "b_path_length": path,
        "residual_pct_of_path": float(np.median(res) / path * 100) if path > 0 else None,
        "view_angle_deg": _stats(ang),
        "rotation_angle_deg": _stats(roll),
        "focal_mm": {"a_median": float(np.median(fa)), "b_median": float(np.median(fb)),
                     "ratio_b_over_a": float(np.median(fb / fa))},
    }


def summary_line(r: dict) -> str:
    return (f"{r['a']['name']} vs {r['b']['name']}: {r['frames_compared']} frames {r['frame_range'][0]}-"
            f"{r['frame_range'][1]}, scale b/a {r['scale_b_per_a']:.6g}, centre residual median "
            f"{r['residual_pct_of_path']:.2f} % of path, view angle median {r['view_angle_deg']['median']:.2f} deg "
            f"(p95 {r['view_angle_deg']['p95']:.2f}), focal {r['focal_mm']['a_median']:.2f} vs "
            f"{r['focal_mm']['b_median']:.2f} mm")
