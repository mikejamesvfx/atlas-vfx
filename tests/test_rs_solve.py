"""RS-aware bundle adjustment (camera_side/rs_solve.py) against ground truth.

A handheld-ish camera (walk forward, pan and tilt wobble) films random points through a rolling shutter (readout
0.74 frames, the X-H2 4K HQ 30.9 ms at 24p). Observations are made by a separate, continuous-time projection
(fixed-point on the row time, analytic trajectory, not the solver's slerp). Then:
  1. GS solve (readout held at 0) from the truth: the bent solution a normal solver gives.
  2. RS solve from that GS solve with the readout held at the truth: poses must get much closer to the truth.
  3. RS solve with the readout free, starting at 0.3 frames: must find the readout.
  4. Putting the readout on the GS solve (what the deck test did) must be worse than the GS solve alone.
  5. RS Solve node (catalogue 47): the GS solve written as a COLMAP text model and imported, then readout free
     (start = the preset's spec) within 1 % of the truth, output camera validates and carries the readout, Sensor
     RS Maps' preset rule takes it, profile valley at the free readout, held keeps its value, bad widgets refused.
Run: <V141 python> tests/test_rs_solve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import rs_solve  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

W, H, FPS = 4096, 2160, 24.0
READOUT = 0.0309 * FPS          # 0.7416 frames
N_FR = 48
F0 = 1001


def traj(t):
    """Continuous camera pose at time t (frames from F0): c2w rotation, centre. Y up, looking down -Z."""
    yaw = np.radians(1.6 * t + 6.0 * np.sin(t / 7.0))           # pan ~1.6 deg/frame plus wobble
    pitch = np.radians(-8.0 + 4.0 * np.sin(t / 5.0 + 1.0))
    roll = np.radians(1.5 * np.sin(t / 4.0))
    R = Rotation.from_euler("YXZ", [yaw, pitch, roll]).as_matrix()
    C = np.array([0.02 * t, 1.6 + 0.01 * np.sin(t / 3.0), -0.04 * t])
    return R, C


def make_camera():
    Rs, Cs = zip(*(traj(i) for i in range(N_FR)))
    M = []
    for R, C in zip(Rs, Cs):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, C
        M.append(m)
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    sensor = {"width_mm": 23.5, "height_mm": 23.5 * H / W}
    return sc.build(name="synthetic_rs", frame_start=F0, fps=FPS, plate=plate, sensor=sensor, cam_to_world=M,
                    focal_mm=16.47, source={"kind": "synthetic"})


def project_true(R, C, X, fx, cx, cy):
    Xc = (X - C) @ R
    z = -Xc[:, 2]
    return cx + fx * Xc[:, 0] / z, cy - fx * Xc[:, 1] / z, z


def observe(cam, X, noise_px, rng):
    k = cam.intrinsics(F0)
    obs = {}
    for i in range(N_FR):
        R, C = traj(i)
        u, v, z = project_true(R, C, X, k.fx, k.cx, k.cy)
        for _ in range(8):                                    # row time from the row the point lands on
            ok = np.isfinite(v)
            t = i + (np.clip(v, 0, H) / H - 0.5) * READOUT
            uu, vv = np.empty_like(u), np.empty_like(v)
            for j in np.flatnonzero(ok):
                Rt, Ct = traj(t[j])
                a, b, zz = project_true(Rt, Ct, X[j:j + 1], k.fx, k.cx, k.cy)
                uu[j], vv[j], z[j] = a[0], b[0], zz[0]
            u, v = uu, vv
        vis = (z > 0.3) & (u > 4) & (u < W - 4) & (v > 4) & (v < H - 4)
        idx = np.flatnonzero(vis)
        xy = np.stack([u[idx], v[idx]], 1) + rng.normal(0, noise_px, (len(idx), 2))
        obs[F0 + i] = (idx, xy)
    return obs


def pose_error(result, cam_true, X_true):
    """Rotation (deg) and centre (mm) error after aligning the solved points to the true points."""
    s, Rs, ts, _ = rs_solve.robust_umeyama(result["xyz"], X_true[result["point_ids"]])
    rot, cen = [], []
    for i, f in enumerate(result["frames"]):
        M = cam_true.cam_to_world(f)
        Ra = Rs @ result["R"][i]
        Ca = s * Rs @ result["C"][i] + ts
        rot.append(np.degrees(Rotation.from_matrix(M[:3, :3].T @ Ra).magnitude()))
        cen.append(1000 * np.linalg.norm(Ca - M[:3, 3]))
    return float(np.median(rot)), float(np.median(cen))


def with_poses(cam, result):
    return rs_solve.to_shot_camera(cam, result, name="tmp")


def check_jacobian(cam, X, obs, fails):
    """Analytic Jacobian vs central differences along random directions, per parameter group, at a perturbed
    point with the readout free. Points, centres and readout are exact; rotations use Jr ~ I (small error)."""
    prob = rs_solve.Problem(cam, X, obs)
    lay = rs_solve._Layout(prob, True)
    L0 = rs_solve._path_length(prob.C0)
    fun = rs_solve._residual_fn(prob, lay, 0.0, 1e4, L0, (0.7, 0.2))
    jac = rs_solve._jacobian_fn(prob, lay, 0.0, 1e4, L0, (0.7, 0.2))
    rng = np.random.default_rng(1)
    x = np.zeros(lay.n)
    x[0] = 0.6
    x[lay.o_pose:lay.o_pts] = rng.normal(0, 1e-3, lay.o_pts - lay.o_pose)
    x[lay.o_pts:] = prob.X0.ravel() + rng.normal(0, 0.01, 3 * prob.n_points)
    J = jac(x)
    groups = {"readout": [0], "pose rot": [], "pose centre": [], "points": list(range(lay.o_pts, lay.n))}
    for s in range(len(lay.free)):
        groups["pose rot"] += [lay.o_pose + 6 * s + c for c in range(3)]
        groups["pose centre"] += [lay.o_pose + 6 * s + c for c in range(3, 6)]
    tol = {"readout": 1e-4, "pose rot": 0.05, "pose centre": 1e-4, "points": 1e-4}
    for g, idx in groups.items():
        d = np.zeros(lay.n)
        d[idx] = rng.normal(0, 1, len(idx))
        h = 1e-6 / max(np.abs(d).max(), 1e-12)
        fd = (fun(x + h * d) - fun(x - h * d)) / (2 * h)
        an = J @ d
        err = float(np.linalg.norm(fd - an) / max(np.linalg.norm(fd), 1e-12))
        print(f"jacobian {g:12s} relative error {err:.2e}")
        if err > tol[g]:
            fails.append(f"analytic jacobian {g} off by {err:.2e} (> {tol[g]})")


def write_colmap_text(folder: Path, cam, result, obs):
    """The GS solve as a COLMAP text model at plate size (PINHOLE), so the node reads it the way it reads a real one:
    images plate_NNNN.png (NNNN = frame - F0), each frame's observations of the solved points, tracks to match."""
    folder.mkdir(parents=True, exist_ok=True)
    k = cam.intrinsics(F0)
    num = lambda vals: " ".join(repr(float(v)) for v in vals)          # noqa: E731
    (folder / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {num([k.fx, k.fy, k.cx, k.cy])}\n")
    D = np.diag([1.0, -1.0, -1.0])                        # Nuke <-> COLMAP camera axes
    keep = {int(p): i for i, p in enumerate(result["point_ids"])}
    tracks = {p: [] for p in keep}
    lines = []
    for img_id, (i, f) in enumerate(enumerate(result["frames"]), 1):
        Rcw = (result["R"][i] @ D).T
        t = -Rcw @ result["C"][i]
        q = Rotation.from_matrix(Rcw).as_quat()            # x y z w
        idx, xy = obs[f]
        pts2d = []
        for j, (p, (u, v)) in enumerate((p, uv) for p, uv in zip(idx, xy) if int(p) in keep):
            pts2d.append(f"{float(u)!r} {float(v)!r} {int(p) + 1}")
            tracks[int(p)].append(f"{img_id} {j}")
        lines.append(f"{img_id} {num([q[3], q[0], q[1], q[2]])} {num(t)} 1 plate_{f - F0:04d}.png\n"
                     + " ".join(pts2d) + "\n")
    (folder / "images.txt").write_text("".join(lines))
    pl = []
    for p, i in keep.items():
        if len(tracks[p]) >= 2:
            x, y, z = result["xyz"][i]
            pl.append(f"{p + 1} {float(x)!r} {float(y)!r} {float(z)!r} 128 128 128 1.0 " + " ".join(tracks[p]) + "\n")
    (folder / "points3D.txt").write_text("".join(pl))


def check_node(gs, obs, fails):
    """RS Solve node (catalogue 47) on the synthetic scene: GS solve written as COLMAP -> Camera Import -> node,
    readout free from the preset's spec start and held at the truth; profile valley; output validates; Sensor RS
    Maps' preset rule then takes the camera's readout."""
    import json
    import os
    import tempfile

    from atlas_vfx.camera_side import camera_io as cio
    from atlas_vfx.camera_side.node_rs_solve import AtlasVFXRSSolve, parse_profile
    from atlas_vfx.camera_side.sensors import sensor_block
    from atlas_vfx.sensor import jobs

    truth_ms = READOUT / FPS * 1000.0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        os.environ["ATLAS_VFX_OUT_ROOTS"] = str(td)
        ref = make_camera()
        write_colmap_text(td / "solve" / "sparse" / "0", ref, gs, obs)
        plate = {"path": "", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
        cam = cio.import_camera(td / "solve", plate=plate, sensor=sensor_block("fujifilm_xh2_dci4k_24p_dis_off"),
                                frame_offset=F0, fps=FPS, name="synthetic_gs")
        node = AtlasVFXRSSolve()
        res = node.run(cam.data, "", "", "free", 0.0, 0, 0, 0, 100, "20, 30.9, 40", 2500, "", str(td / "out"))
        out_data, ro, path, report = res["result"]
        rep = json.loads(report)
        err = abs(ro - truth_ms) / truth_ms
        print(f"RS Solve node, free from {rep['readout_start_ms']:.2f} ms ({rep['readout_start_from'][:30]}): "
              f"{ro:.3f} ms (truth {truth_ms:.3f}), {rep['solve']['message']}")
        if not err < 0.01:
            fails.append(f"RS Solve node free readout {ro:.3f} ms not within 1 % of {truth_ms:.3f}")
        if not rep["readout_start_from"].startswith("preset"):
            fails.append(f"RS Solve node start not from the preset: {rep['readout_start_from']}")
        try:
            back = sc.load(path)
            sc.validate(json.loads(json.dumps(out_data)))
        except Exception as e:                            # noqa: BLE001 (any validation failure is the failure)
            fails.append(f"RS Solve node output camera does not validate: {e}")
            back = None
        if back is not None:
            ext = back.data["extensions"].get("atlas_vfx_rs_solve", {})
            if not (ext.get("readout_solved") is True and abs(ext.get("readout_ms", 0) - ro) < 1e-9):
                fails.append(f"RS Solve node camera extension wrong: {ext.get('readout_ms')} "
                             f"{ext.get('readout_solved')}")
            d = jobs.decide_readout(back.data, 30.9, jobs.PRESET_RULE)
            if not (abs(d["readout_ms"] - ro) < 1e-9 and d["winner"]["kind"] == "rs_solve"):
                fails.append(f"Sensor RS Maps preset rule did not take the RS Solve readout: {d['readout_from']}")
        prof = rep["profile"]
        best = min(prof, key=lambda r: r["cost"])
        print("  profile:", ", ".join(f"{r['readout_ms']:g} ms +{r['cost_above_best_pct']:.1f} %" for r in prof))
        if not (len(prof) == 4 and abs(best["readout_ms"] - ro) < 1.0 and prof[0]["cost_above_best_pct"] > 5):
            fails.append("RS Solve node profile: expected 4 rows, best at the free readout, 20 ms clearly worse")
        # world kept: no warning here; the DSCF2973 numbers (15.6 deg onto the input world) warn, DSCF3000's don't
        from atlas_vfx.camera_side import rs_solve as rsm
        if any("did not keep the input world" in w for w in rep["warnings"]):
            fails.append(f"RS Solve node warned about the world on the synthetic shot: {rep['solve']['align_to_input']}")
        bad = {"align_to_input": {"scale": 1.04, "rotation_deg": 15.6, "translation_m": 3.06,
                                  "point_shift_median_m": 1.04},
               "pose_change": {"rot_deg_median": 15.3, "centre_mm_median": 3083.0}}
        good = {"align_to_input": {"scale": 0.9997, "rotation_deg": 0.29, "translation_m": 0.053},
                "pose_change": {"rot_deg_median": 0.23, "centre_mm_median": 26.0}}
        if not (len(rsm.world_warnings(bad)) == 1 and not rsm.world_warnings(good)):
            fails.append("rs_solve.world_warnings: 15.6 deg must warn, 0.29 deg must not")
        print(f"  world kept: align {rep['solve']['align_to_input']['rotation_deg']:.3f} deg, no warning; "
              f"world_warnings flags a 15.6 deg move")
        side = json.loads((td / "out" / "synthetic_gs_rs.shot_camera.sidecar.json").read_text(encoding="utf-8"))
        if side["inputs"]["readout"] != "free" or "reproj_solved" not in side["outputs"]:
            fails.append("RS Solve node sidecar missing inputs/outputs")

        res = node.run(cam.data, "", "", "held", 30.9, F0 + 6, F0 + 40, 0, 100, "", 2500, "held", str(td / "out"))
        hro = res["result"][1]
        held = sc.load(res["result"][2])
        hext = held.data["extensions"]["atlas_vfx_rs_solve"]
        if not (abs(hro - 30.9) < 1e-9 and hext["readout_solved"] is False):
            fails.append(f"RS Solve node held readout came back {hro} ms (solved {hext['readout_solved']})")
        if held.frames != list(range(F0 + 6, F0 + 41)) or res["result"][0]["frames"][0]["frame"] != F0 + 6:
            fails.append(f"RS Solve node on a sub-range kept unsolved frames: {held.frames[0]}-{held.frames[-1]}")
        for bad in ("1,2,x", "500", ",".join(["1"] * 25)):
            try:
                parse_profile(bad)
                fails.append(f"RS Solve profile_ms {bad[:20]!r} not refused")
            except ValueError:
                pass
        try:
            node.run(cam.data, "", "", "free", 0.0, 0, 0, 0, 100, "", 2500, "../escape", str(td / "out"))
            fails.append("RS Solve node accepted a name with a path in it")
        except ValueError:
            pass


def main() -> int:
    import time

    rng = np.random.default_rng(3)
    cam = make_camera()
    # points in a big box around the path, so every frame sees a few hundred
    X = np.column_stack([rng.uniform(-12, 12, 2500), rng.uniform(-1, 6, 2500), rng.uniform(-14, 10, 2500)])
    obs = observe(cam, X, 0.5, rng)
    seen = np.median([len(v[0]) for v in obs.values()])
    print(f"synthetic: {N_FR} frames, {len(X)} points, median {seen:.0f} seen per frame, readout {READOUT:.4f} fr")
    fails = []
    check_jacobian(cam, X, obs, fails)
    t_start = time.time()

    gs = rs_solve.solve(cam, X, obs, readout_frames=0.0)
    g_rot, g_cen = pose_error(gs, cam, X)
    print("GS solve:", gs["report"]["reproj_solved"], f"pose err {g_rot:.4f} deg {g_cen:.1f} mm")
    gs_cam = with_poses(cam, gs)
    Xg = np.zeros_like(X)
    Xg[gs["point_ids"]] = gs["xyz"]

    # 4. readout on top of the GS solve, no re-solve: reprojection must get worse
    prob = gs["problem"]
    on_top = rs_solve.rms_px(prob, gs["R"], gs["C"], gs["xyz"], READOUT)
    print("GS poses + readout on top:", on_top)
    if not on_top["median_px"] > gs["report"]["reproj_solved"]["median_px"]:
        fails.append("readout on top of the GS solve did not make reprojection worse")

    fixed = rs_solve.solve(gs_cam, Xg, obs, readout_frames=READOUT)
    f_rot, f_cen = pose_error(fixed, cam, X)
    print("RS solve, readout fixed:", fixed["report"]["reproj_solved"], f"pose err {f_rot:.4f} deg {f_cen:.1f} mm")
    if not fixed["report"]["reproj_solved"]["median_px"] < 0.8:
        fails.append(f"fixed-readout RS solve median {fixed['report']['reproj_solved']['median_px']:.3f} px >= 0.8")
    if not (f_rot < 0.5 * g_rot):
        fails.append(f"fixed-readout RS solve rotation error {f_rot:.4f} not under half the GS {g_rot:.4f} deg")

    free = rs_solve.solve(gs_cam, Xg, obs, readout_frames=0.3, solve_readout=True)
    print("  free solve:", free["report"]["nfev"], "evaluations,", free["report"]["message"])
    r = free["report"]["readout_frames"]
    print(f"RS solve, readout free: {r:.4f} fr = {free['report']['readout_ms']:.2f} ms (truth "
          f"{READOUT / FPS * 1000:.2f} ms)", free["report"]["reproj_solved"])
    if abs(r - READOUT) > 0.05 * READOUT:
        fails.append(f"free readout {r:.4f} not within 5 % of {READOUT:.4f}")

    # contract output keeps the readout and validates
    out = rs_solve.to_shot_camera(gs_cam, fixed, name="synthetic_rs_solved")
    ext = out.data["extensions"]["atlas_vfx_rs_solve"]
    if abs(ext["readout_ms"] - 30.9) > 0.01:
        fails.append(f"shot camera extension readout {ext['readout_ms']}")

    print(f"three solves: {time.time() - t_start:.1f} s (analytic jacobian)")
    t_node = time.time()
    check_node(gs, obs, fails)
    print(f"node runs: {time.time() - t_node:.1f} s")
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
