"""sensor/rs_proof.py (Phase 1b S5 core) on a synthetic shot with a known rolling shutter.

  1. A textured ground plane rendered row by row through a fast, accelerating pan with a 30.9 ms readout (each
     sensor row sees the plane from the camera at its own row time; rs_time decides the pose).
  2. Surface from tracks: points on the plane plus a box standing on it, observed through the same RS camera,
     triangulated with triangulate_rs. Exact with the true readout, worse with the readout ignored. The plane fit
     recovers y = 0 and the box points are labelled off the plane.
  3. The proof: reference -> target through the plane, RS on (Sensor maps at the true readout) vs off (control).
     RS on must win by at least 20 % (median of references) with no reference worse; with readout-0 maps RS on
     and off are the same image (0 %); with the readout's sign flipped RS on must lose. against_baseline with a
     row as its own baseline gives back RS on vs off.
Run: <V141 python> tests/test_rs_proof.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import rs_time  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.sensor import rs_proof as rp  # noqa: E402

W, H, FPS, N_FR, F0 = 640, 360, 24.0, 14, 1001
R_MS = 30.9
R_FR = rs_time.readout_frames(R_MS, FPS)
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def make_camera():
    M = []
    for i in range(N_FR):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_euler("YXZ", [-(3.0 * i + 0.25 * i * i), -38 + 7 * np.sin(i / 2.2), 1.5 * np.sin(i / 3)],
                                        degrees=True).as_matrix()
        m[:3, 3] = [0.04 * i, 1.6, -0.02 * i]
        M.append(m)
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    sensor = {"width_mm": 23.5, "height_mm": 23.5 * H / W}
    return sc.build(name="rs_proof_synth", frame_start=F0, fps=FPS, plate=plate, sensor=sensor, cam_to_world=M,
                    focal_mm=16.47, source={"kind": "synthetic"})


rng = np.random.default_rng(3)
G1 = gaussian_filter(rng.random((900, 900)), 1.0)      # fine texture, 4 cm cells over x -18..18, z -30..6
G2 = gaussian_filter(rng.random((120, 120)), 1.0)      # coarse, 30 cm cells


def texture(x, z):
    a = map_coordinates(G1, [(z + 30) / 0.04, (x + 18) / 0.04], order=1, mode="reflect")
    b = map_coordinates(G2, [(z + 30) / 0.3, (x + 18) / 0.3], order=1, mode="reflect")
    return 0.03 + 0.5 * (0.6 * a + 0.4 * b) ** 2


def render_rs(cam, f, readout_fr):
    """Rolling-shutter image of the plane y = 0: row y from the camera at its row time."""
    k = cam.intrinsics(f)
    ys = np.arange(H) + 0.5
    R, C = rp.row_poses(cam, np.full(H, f), ys, readout_fr)
    xs = np.arange(W) + 0.5
    d = np.stack(np.broadcast_arrays((xs[None, :] - k.cx) / k.fx, -(ys[:, None] - k.cy) / k.fy, -np.ones((H, W))), -1)
    d = np.einsum("yij,yxj->yxi", R, d)
    t = -C[:, None, 1] / d[..., 1]
    X = C[:, None, :] + t[..., None] * d
    img = np.where(t > 0, texture(X[..., 0], X[..., 2]), 0.03).astype(np.float32)
    return np.repeat(img[..., None], 3, -1)


def project_rs(cam, X, f, readout_fr):
    k = cam.intrinsics(f)
    v = np.full(len(X), H / 2.0)
    for _ in range(50):
        R, C = rp.row_poses(cam, np.full(len(X), f), v, readout_fr)
        Xc = np.einsum("nji,nj->ni", R, X - C)
        z = -Xc[:, 2]
        u, v = k.cx + k.fx * Xc[:, 0] / z, k.cy - k.fy * Xc[:, 1] / z
    return u, v, z


def main() -> int:
    cam = make_camera()
    frames = cam.frames

    # 2. surface from tracks
    # plane points where random pixels of random frames see the plane; a box standing on it mid-shot
    P = []
    for f in frames:
        Xf, t = rp.plane_hits(cam, f, W, H, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        ok = np.flatnonzero(np.isfinite(t.ravel()))
        P.append(Xf.reshape(-1, 3)[rng.choice(ok, 250, replace=False)])
    P = np.vstack(P)
    f_mid = frames[N_FR // 2]
    Xc, _ = rp.plane_hits(cam, f_mid, W, H, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    bc = Xc[int(H * 0.6), W // 2]
    box = bc + np.column_stack([rng.uniform(-0.4, 0.4, 300), rng.uniform(0.3, 1.0, 300), rng.uniform(-0.4, 0.4, 300)])
    X = np.vstack([P, box])
    obs = {}
    for f in frames:
        u, v, z = project_rs(cam, X, f, R_FR)
        vis = np.flatnonzero((z > 0.3) & (u > 1) & (u < W - 1) & (v > 1) & (v < H - 1))
        obs[f] = (vis, np.column_stack([u[vis], v[vis]]))
    tri = rp.triangulate_rs(cam, obs, R_FR, max_px=1e9)
    err3d = np.linalg.norm(tri["xyz"] - X[tri["ids"]], axis=1)
    check(len(tri["ids"]) > 1500 and np.max(err3d) < 1e-6,
          f"triangulate_rs with the true readout: {len(tri['ids'])} points, max 3D error {np.max(err3d):.1e}")
    gs = rp.triangulate_rs(cam, obs, 0.0, max_px=1e9)
    check(np.median(gs["err_px"]) > 20 * max(np.median(tri["err_px"]), 1e-9),
          f"(readout ignored: ray error {np.median(gs['err_px']):.2f} px median vs {np.median(tri['err_px']):.1e})")
    plane = rp.fit_plane_from_tracks(cam, obs, tri, frames, (0, 0, W, H))
    ang = np.degrees(np.arccos(min(1.0, abs(plane["normal"][1]))))
    check(ang < 0.01 and abs(plane["centre"][1]) < 1e-6 and plane["normal"][1] > 0,
          f"plane from tracks: normal {ang:.1e} deg off +y, offset {plane['centre'][1]:.1e}")
    lab = rp.track_label_map(obs, tri, plane, f_mid, W, H, W, radius_px=12)
    bi = np.isin(obs[f_mid][0], np.arange(len(P), len(X)))
    bxy = obs[f_mid][1][bi]
    on_box = lab[np.clip(bxy[:, 1].astype(int), 0, H - 1), np.clip(bxy[:, 0].astype(int), 0, W - 1)]
    check(len(bxy) > 10 and on_box.mean() < 0.05 and lab.mean() > 0.3,
          f"label map: {on_box.mean() * 100:.1f} % of {len(bxy)} box pixels kept, {lab.mean() * 100:.0f} % of frame kept")

    # 3. the proof
    imgs = {f: render_rs(cam, f, R_FR) for f in frames}
    c, n = np.zeros(3), np.array([0.0, 1.0, 0.0])
    maps = {f: rp.rs_maps_for_plane(cam, f, W, H, R_MS, c, n) for f in frames}
    zero = {f: rp.rs_maps_for_plane(cam, f, W, H, 0.0, c, n) for f in frames}
    rows, rows0 = [], []
    for ref in (frames[3], frames[7], frames[10]):
        s_clean, s_ctrl = rp.reference_sources(imgs[ref], maps[ref])
        z_clean, z_ctrl = rp.reference_sources(imgs[ref], zero[ref])
        for f in frames:
            if f == ref:
                continue
            r = rp.compare(cam, ref, f, imgs[ref], s_clean, s_ctrl, imgs[f], maps[f], c, n, min_px=500)
            if r:
                rows.append(r)
            r0 = rp.compare(cam, ref, f, imgs[ref], z_clean, z_ctrl, imgs[f], zero[f], c, n, min_px=500)
            if r0:
                rows0.append(r0)
    s = rp.summarise(rows)
    acc = rp.acceptance(s)
    for p in s["per_ref"]:
        print(f"     ref {p['ref']}: {p['targets']} targets, RS on vs off {p['on_vs_off_median_pct']:+.1f} %, "
              f"vs plain {p['vs_plain_median_pct']:+.1f} %")
    check(len(rows) >= 15 and acc["median_gain_pass"] and s["worst_ref_pct"] > 0,
          f"RS on beats off: median of refs {s['median_of_refs_pct']:+.1f} %, worst ref {s['worst_ref_pct']:+.1f} %, "
          f"{s['on_better_pairs']} of {s['pairs']} pairs")
    s0 = rp.summarise(rows0)
    check(abs(s0["median_all_pairs_pct"]) < 0.5 and max(abs(r["better_vs_control_pct"]) for r in rows0) < 1.0,
          f"readout-0 maps: on vs off {s0['median_all_pairs_pct']:+.2f} % (same image)")

    # flipped sign (bottom-to-top readout): the maps skew the wrong way, so RS on must lose
    flip = {f: rp.rs_maps_for_plane(cam, f, W, H, -R_MS, c, n) for f in frames}
    rowsf = []
    for ref in (frames[3], frames[7], frames[10]):
        f_clean, f_ctrl = rp.reference_sources(imgs[ref], flip[ref])
        for f in frames:
            if f != ref:
                r = rp.compare(cam, ref, f, imgs[ref], f_clean, f_ctrl, imgs[f], flip[f], c, n, min_px=500)
                if r:
                    rowsf.append(r)
    sf = rp.summarise(rowsf)
    accf = rp.acceptance(sf)
    check(sf["median_of_refs_pct"] < 0 and not accf["median_gain_pass"],
          f"flipped readout loses: median of refs {sf['median_of_refs_pct']:+.1f} %, "
          f"{sf['on_better_pairs']} of {sf['pairs']} pairs better")

    # against_baseline: the same row as its own baseline scores RS on vs GS off = RS on vs RS off
    r1 = rp.compare(cam, frames[3], frames[8], imgs[frames[3]], *rp.reference_sources(imgs[frames[3]], maps[frames[3]]),
                    imgs[frames[8]], maps[frames[8]], c, n, min_px=500, want_images=True)
    b = rp.against_baseline(imgs[frames[8]], r1, r1, min_px=500)
    check(b is not None and abs(b["on_vs_gs_pct"] - b["on_vs_off_common_pct"]) < 1e-9
          and abs(b["on_vs_off_common_pct"] - r1["better_vs_control_pct"]) < 0.01,
          f"against_baseline on itself: {b and b['on_vs_gs_pct']:+.2f} % = on vs off {r1['better_vs_control_pct']:+.2f} %")

    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
