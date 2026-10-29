"""Readout Solve: measure the sensor readout from tracked points, given the shot camera and 3D points.

There was no rolling-shutter readout solver in atlas-vfx or Atlas Camera (platelock has an RS camera MODEL, not a
solver), so this is the smallest one that does the job. It uses platelock's CameraTrack for the RS projection, a
separate implementation from sensor/rs.py, so the validation test (rs.py makes an RS clip, this measures it) is not
the code checking itself.

For a candidate readout r (frames), every frame's points are projected with the RS model, row times from the
projected row, poses slerped between frames. A global-shutter solve has already bent each frame's pose to absorb
part of the skew, so with `refine_pose` each frame gets a small rigid nuisance correction (rotation + translation,
applied to the world points, so every row shares it): what it cannot absorb is the row-dependent part, which is
the rolling shutter. The readout is the r with the lowest robust residual (grid, then a parabola through the best
three). The curve is returned so a flat or noisy minimum is visible.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from atlas_vfx.platelock.adapters import track_from_shot_camera


def _residuals(track, idx, X, xy, refine, soft):
    def proj(p):
        Xw = X
        if p is not None:
            Xw = Rotation.from_rotvec(p[:3]).apply(X - X.mean(0)) + X.mean(0) + p[3:]
        u, v, z = track.project(Xw, idx)
        return np.concatenate([u - xy[:, 0], v - xy[:, 1]])

    if not refine:
        r = proj(None)
        return r
    sol = least_squares(proj, np.zeros(6), loss="soft_l1", f_scale=soft, x_scale=[1e-3] * 3 + [1e-2] * 3)
    return sol.fun


def estimate_readout(cam, xyz: np.ndarray, observations: dict, *, frames=None, grid=None, refine_pose: bool = True,
                     soft_px: float = 2.0, min_points: int = 30, inner_only: bool = True) -> dict:
    """observations: {nuke frame: (point indices (K,), xy plate px (K,2))} (camera_io.colmap_points layout; contract
    pixels = OpenCV pixels + 0.5, both are handled). Returns {"readout_frames", "readout_ms", "curve", ...}."""
    fps = float(cam.data["shot"]["fps"])
    allf = cam.frames
    use = [f for f in (frames or allf) if f in observations and len(observations[f][0]) >= min_points]
    if inner_only:
        use = [f for f in use if allf[0] < f < allf[-1]]      # need neighbours on both sides for the motion
    if not use:
        raise ValueError("no frames with enough observed points")
    grid = np.linspace(0.0, 1.2, 25) if grid is None else np.asarray(grid, float)
    base = track_from_shot_camera(cam, 0.0)
    f0 = allf[0]
    costs = []
    for r in grid:
        tr = base.with_readout(float(r))
        tot, n = 0.0, 0
        for f in use:
            k, xy = observations[f]
            res = _residuals(tr, f - f0, xyz[k], np.asarray(xy, float) - 0.5, refine_pose, soft_px)
            e = np.hypot(res[: len(k)], res[len(k):])
            tot += float(np.sum(soft_px ** 2 * 2 * (np.sqrt(1 + (e / soft_px) ** 2) - 1)))
            n += len(e)
        costs.append(tot / max(n, 1))
    costs = np.array(costs)
    i = int(np.argmin(costs))
    best = float(grid[i])
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = costs[i - 1], costs[i], costs[i + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best = float(grid[i] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0]))
    return {"readout_frames": best, "readout_ms": best / fps * 1000.0, "fps": fps, "frames_used": len(use),
            "curve": [[float(g), float(c)] for g, c in zip(grid, costs)], "at_grid_edge": i in (0, len(grid) - 1),
            "refine_pose": refine_pose, "model": "platelock CameraTrack, rs_anchor centre"}
