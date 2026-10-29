"""One rolling-shutter time model (camera_side/rs_time.py), and every RS code path agrees with it.

  1. Conventions: contract row i + 0.5 == OpenCV row i; top edge -r/2, bottom edge +r/2 (anchor centre); anchor top;
     ms <-> frames; pairing = floor(f + tau).
  2. Geodesic interpolation is the same for camera-to-world and world-to-camera rotations, and matches platelock's
     quaternion slerp.
  3. One camera (a shot-camera@0.1 with a fast pan), random points, readout 0.74 frames. A reference RS projection
     is made here from rs_time alone (fixed point on the row). Then the same pixels must come out of:
       - sensor/rs.py         pose_at + row_tau                         (the maps)
       - camera_side/rs_solve project_rs on those observations        (the RS bundle adjustment)
       - sensor/rs_proof row_poses                                     (the S5 proof's RS triangulation)
       - platelock CameraTrack.project via adapters.track_from_shot_camera (Plate Lock, Readout Solve)
       - camera_side/grid_rs  fit_readout with the readout held, and free (the chart tool)
     A mismatch in row time, pixel convention, anchor or interpolation in any of them fails here.
Run: <V141 python> tests/test_rs_time.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import grid_rs, rs_solve, rs_time  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.platelock.adapters import track_from_shot_camera  # noqa: E402
from atlas_vfx.platelock.core.camera import quat_from_rot, rot_from_quat, slerp  # noqa: E402
from atlas_vfx.sensor import rs, rs_proof  # noqa: E402

W, H, FPS = 4096, 2160, 24.0
R_FR = rs_time.readout_frames(30.9, FPS)        # 0.7416 frames
N_FR, F0 = 12, 1001
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def make_camera():
    M = []
    for i in range(N_FR):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_euler("YXZ", [2.5 * i + 3 * np.sin(i / 3), -6 + 2 * np.sin(i / 2), 1.2 * np.sin(i / 4)],
                                        degrees=True).as_matrix()
        m[:3, 3] = [0.03 * i, 1.6, -0.05 * i]
        M.append(m)
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    sensor = {"width_mm": 23.5, "height_mm": 23.5 * H / W}
    return sc.build(name="rs_time_xtest", frame_start=F0, fps=FPS, plate=plate, sensor=sensor, cam_to_world=M,
                    focal_mm=16.47, source={"kind": "synthetic"})


def reference_rs(cam, X, f):
    """Contract-pixel RS projection of X in frame f from rs_time only: fixed point on the observed row."""
    k = cam.intrinsics(f)
    frames = cam.frames
    R = np.array([cam.cam_to_world(g)[:3, :3] for g in frames])
    C = np.array([cam.cam_to_world(g)[:3, 3] for g in frames])
    rel = rs_time.rel_rotvecs(R)
    fi = np.full(len(X), frames.index(f))
    v = np.full(len(X), H / 2.0)
    for _ in range(60):
        rf = rs_time.row_frac(v, H)
        i0 = rs_time.pair_index(fi, rf, 0, len(frames) - 1)
        a = (fi - i0) + rf * R_FR
        Rt, Ct = rs_time.interp_rot(R, rel, i0, a), rs_time.interp_centre(C, i0, a)
        Xc = np.einsum("nji,nj->ni", Rt, X - Ct)
        z = -Xc[:, 2]
        u, v = k.cx + k.fx * Xc[:, 0] / z, k.cy - k.fy * Xc[:, 1] / z
    return u, v, z


def main() -> int:
    # 1. conventions
    i = np.arange(H)
    check(np.allclose(rs_time.row_frac(i + 0.5, H), rs_time.row_frac(i, H, pixels="opencv"), atol=0, rtol=0),
          "contract row i + 0.5 == OpenCV row i")
    check(np.isclose(rs_time.row_tau(0, H, 1.0), -0.5) and np.isclose(rs_time.row_tau(H, H, 1.0), 0.5)
          and np.isclose(rs_time.row_tau(H / 2, H, 1.0), 0.0), "anchor centre: top edge -r/2, middle 0, bottom +r/2")
    check(np.isclose(rs_time.row_tau(0, H, 1.0, anchor="top"), 0.0)
          and np.isclose(rs_time.row_tau(H, H, 1.0, anchor="top"), 1.0), "anchor top: 0 .. r")
    check(np.isclose(rs_time.readout_ms(rs_time.readout_frames(30.9, 24), 24), 30.9)
          and np.isclose(R_FR, 0.7416), "ms <-> frames")
    rf = np.linspace(-0.5, 0.5, 101)
    fi = np.full_like(rf, 5, dtype=int)
    check(np.array_equal(rs_time.pair_index(fi, rf, 0, 10), np.floor(5 + rf * R_FR).astype(int)),
          "pair_index == floor(f + tau) for |tau| < 1")

    # 2. geodesic
    A = Rotation.from_euler("xyz", [10, -20, 5], degrees=True).as_matrix()
    B = Rotation.from_euler("xyz", [13, -17, 4], degrees=True).as_matrix()
    for a in (0.0, 0.3, 1.0, -0.4, 1.3):
        Rc, _ = rs_time.interp_pose(A, B, np.zeros(3), np.zeros(3), a)
        Rw, _ = rs_time.interp_pose(A.T, B.T, np.zeros(3), np.zeros(3), a)
        if not np.allclose(Rc, Rw.T, atol=1e-12):
            check(False, f"geodesic c2w vs w2c at a = {a}")
            break
    else:
        check(True, "geodesic: same for camera-to-world and world-to-camera (incl. extrapolation)")
    Bs = Rotation.from_euler("xyz", [10.4, -19.8, 5.1], degrees=True).as_matrix()     # small step (< 1 deg)
    ok = all(np.allclose(rot_from_quat(slerp(quat_from_rot(A), quat_from_rot(Bx), np.array(0.37))),
                         rs_time.interp_pose(A, Bx, 0, 0, 0.37)[0], atol=1e-12) for Bx in (B, Bs))
    check(ok, "matches platelock quaternion slerp (large and sub-degree steps)")

    # 3. one camera, four code paths
    cam = make_camera()
    rng = np.random.default_rng(0)
    k = cam.intrinsics(F0)
    obs, Xs = {}, []
    X = np.column_stack([rng.uniform(-6, 9, 600), rng.uniform(-1, 4, 600), rng.uniform(-14, -6, 600)])
    for f in cam.frames:
        u, v, z = reference_rs(cam, X, f)
        vis = np.flatnonzero((z > 0.5) & (u > 2) & (u < W - 2) & (v > 2) & (v < H - 2))
        obs[f] = (vis, np.column_stack([u[vis], v[vis]]))
    n_obs = sum(len(o[0]) for o in obs.values())
    print(f"     {n_obs} observations over {N_FR} frames, readout {R_FR:.4f} frames")

    # sensor/rs.py: pose_at + row_tau at the reference row reproduces the reference pixel
    e = []
    for f in cam.frames[1:-1]:
        idx, xy = obs[f]
        for j, (uu, vv) in zip(idx[::7], xy[::7]):
            Rt, Ct = rs.pose_at(cam, f + float(rs.row_tau(vv, H, R_FR)))
            Xc = (X[j] - Ct) @ Rt
            e.append(np.hypot(k.cx + k.fx * Xc[0] / -Xc[2] - uu, k.cy - k.fy * Xc[1] / -Xc[2] - vv))
    check(max(e) < 1e-6, f"sensor/rs.py pose_at + row_tau: max {max(e):.2e} px")

    # rs_solve.project_rs on the same observations, at the input poses
    prob = rs_solve.Problem(cam, X, obs, min_track=1, max_init_px=1e9)
    u, v = rs_solve.project_rs(prob, prob.R0, prob.C0, prob.X0[prob.pi], R_FR)
    e = np.hypot(u - prob.xy[:, 0], v - prob.xy[:, 1])
    check(e.max() < 1e-6, f"rs_solve.project_rs: max {e.max():.2e} px over {len(e)} obs")
    u0, v0 = rs_solve.project_rs(prob, prob.R0, prob.C0, prob.X0[prob.pi], 0.0)
    gs = np.hypot(u0 - prob.xy[:, 0], v0 - prob.xy[:, 1])
    check(np.median(gs) > 5, f"(the readout matters here: GS projection is {np.median(gs):.1f} px off, median)")

    # sensor/rs_proof.row_poses (the S5 proof's RS triangulation): the row's pose sends X to the observed pixel
    e = []
    for f in cam.frames:
        idx, xy = obs[f]
        Rt, Ct = rs_proof.row_poses(cam, np.full(len(idx), f), xy[:, 1], R_FR)
        Xc = np.einsum("nji,nj->ni", Rt, X[idx] - Ct)
        e.append(np.hypot(k.cx + k.fx * Xc[:, 0] / -Xc[:, 2] - xy[:, 0], k.cy - k.fy * Xc[:, 1] / -Xc[:, 2] - xy[:, 1]).max())
    check(max(e) < 1e-6, f"sensor/rs_proof.row_poses: max {max(e):.2e} px (all frames, ends extrapolated)")

    # platelock CameraTrack (OpenCV pixels = contract - 0.5), interior frames (it clamps at the track ends)
    tr = track_from_shot_camera(cam, readout=R_FR)
    e = []
    for n, f in enumerate(cam.frames):
        if n in (0, N_FR - 1):
            continue
        idx, xy = obs[f]
        u, v, _ = tr.project(X[idx], float(n), iters=60)
        e.append(np.hypot(u + 0.5 - xy[:, 0], v + 0.5 - xy[:, 1]).max())
    check(max(e) < 1e-6, f"platelock CameraTrack.project: max {max(e):.2e} px")

    # grid_rs.fit_readout: planar chart in an OpenCV world, reference made with rs_time (pixels="opencv")
    Kcv = np.array([[k.fx, 0, k.cx - 0.5], [0, k.fy, k.cy - 0.5], [0, 0, 1.0]])
    jj, ii = np.mgrid[0:15, 0:31]
    P = np.column_stack([ii.ravel() * 0.0762, jj.ravel() * 0.0762, np.zeros(ii.size)])
    Rw = np.array([Rotation.from_euler("YXZ", [2.0 * n - 10, 3 * np.sin(n / 2), 0.5 * n], degrees=True).as_matrix()
                   for n in range(N_FR)])                                  # camera-to-world, OpenCV axes
    Cw = np.array([[1.14 + 0.02 * n, 0.53, -3.0] for n in range(N_FR)])
    rel = rs_time.rel_rotvecs(Rw)
    gobs = {}
    for n in range(N_FR):
        v = np.full(len(P), H / 2.0)
        fi = np.full(len(P), n)
        for _ in range(60):
            rfr = rs_time.row_frac(v, H, pixels="opencv")
            i0 = rs_time.pair_index(fi, rfr, 0, N_FR - 1)
            a = (fi - i0) + rfr * R_FR
            Rt, Ct = rs_time.interp_rot(Rw, rel, i0, a), rs_time.interp_centre(Cw, i0, a)
            Xc = np.einsum("nji,nj->ni", Rt, P - Ct)
            u, v = Kcv[0, 2] + k.fx * Xc[:, 0] / Xc[:, 2], Kcv[1, 2] + k.fy * Xc[:, 1] / Xc[:, 2]
        xy = np.column_stack([u, v])
        gobs[n] = (xy, P, xy)
    held = grid_rs.fit_readout(gobs, Kcv, H, readout_start=R_FR, solve_readout=False, fps=FPS)
    check(held["rms_px"] < 1e-4, f"grid_rs fit_readout, readout held: rms {held['rms_px']:.2e} px")
    free = grid_rs.fit_readout(gobs, Kcv, H, fps=FPS)
    check(abs(free["readout_frames"] - R_FR) < 1e-4,
          f"grid_rs fit_readout, readout free: {free['readout_ms']:.4f} ms (truth 30.9), rms {free['rms_px']:.2e} px")

    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
