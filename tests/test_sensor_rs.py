"""Sensor pass, rolling shutter (synthetic).

- zero motion: the RS and de-RS maps are the identity
- pan at constant depth: a vertical edge put through the rs map leans by the predicted px per row, where the
  prediction is the edge's image velocity (ShotCamera.project at the neighbouring frames) * T / H
- forward check against an independent RS model: platelock CameraTrack projects world points row by row; the rs map at
  that sensor pixel must point at the point's global-shutter projection
- RS then de-RS round trip < 0.05 px median (moving, rotating camera over a slanted floor and a wall)
- flash at mid-readout splits the frame at the right row (slit scan, row times only)
- validation hook: an RS clip made with rs.py is measured by readout_solve (platelock's RS model); the readout must
  come back within 5 %

Run: <V141 python> tests\\test_sensor_rs.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side.sensors import sensor_block  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.platelock.adapters import track_from_shot_camera  # noqa: E402
from atlas_vfx.sensor import apply as ap  # noqa: E402
from atlas_vfx.sensor import readout_solve, rs  # noqa: E402
from atlas_vfx.sensor.lens import from_st, pixel_grid  # noqa: E402

FAILS = []
W, H = 512, 270
PLATE = cio.plate_block(None, W, H)
SENSOR = sensor_block("fujifilm_xh2_dci4k_24p_dis_off")
FPS = 24.0
XH2_HQ_S = 0.0309                                 # X-H2 8K / 6.2K / 4K HQ readout


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def rot(axis, deg):
    a = math.radians(deg)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def make_cam(n=9, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, move=(0, 0, 0), wobble=0.0, f0=1001):
    mats = []
    for i in range(n):
        M = np.eye(4)
        M[:3, :3] = (rot((0, 1, 0), yaw_deg * i + wobble * math.sin(1.3 * i))
                     @ rot((1, 0, 0), -5 + pitch_deg * i + 0.6 * wobble * math.cos(0.9 * i))
                     @ rot((0, 0, 1), roll_deg * i))
        M[:3, 3] = np.asarray(move, float) * i + [0, 1.6, 0]
        mats.append(M)
    return sc.build(name="syn_rs", frame_start=f0, fps=FPS, plate=PLATE, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=16.47, source={"kind": "synthetic"})


def scene_depth(cam, frame, readout_fr, w=W, h=H):
    """Exact z depth on the RS sensor: each row's ray, at its own capture time, hits the floor (y = 0) or a wall at
    z = -12, whichever is nearer."""
    k = cam.intrinsics(frame)
    s = w / k.width
    X, Y = pixel_grid(w, h)
    out = np.empty((h, w))
    for j in range(h):
        R, C = rs.pose_at(cam, frame + float(rs.row_tau(Y[j, 0], h, readout_fr)))
        d = np.stack([(X[j] / s - k.cx) / k.fx, -(Y[j] / s - k.cy) / k.fy, -np.ones(w)], -1)
        dw = d @ R.T
        best = np.full(w, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = (0.0 - C[1]) / dw[:, 1]
            t_wall = (-12.0 - C[2]) / dw[:, 2]
        for t in (t_floor, t_wall):
            best = np.where((t > 0) & (t < best), t, best)
        out[j] = best                                     # ray z component is -1, so t is z depth
    return np.where(np.isfinite(out), out, 1e4)


def test_zero_motion_identity():
    cam = make_cam(yaw_deg=0.0)
    depth = np.full((H, W), 5.0)
    m_rs, m_de = rs.rs_maps(cam, 1005, depth, XH2_HQ_S, FPS)
    ident = rs.identity_map(W, H)
    e_rs = np.abs(np.stack(from_st(m_rs, W, H)) - np.stack(from_st(ident, W, H))).max()
    e_de = np.abs(np.stack(from_st(m_de, W, H)) - np.stack(from_st(ident, W, H))).max()
    check(e_rs < 1e-3 and e_de < 1e-3, "zero motion: RS and de-RS maps are the identity",
          f"max {e_rs:.2e} / {e_de:.2e} px")


def edge_x(img):
    """Sub-pixel x of a white-left / black-right step, per row (area under the row)."""
    return img.sum(axis=1)


def test_pan_skew():
    yaw = 2.0                                                    # deg per frame, a brisk pan
    cam = make_cam(yaw_deg=yaw)
    f = 1004
    depth = np.full((H, W), 8.0)
    T = XH2_HQ_S * FPS
    m_rs, _ = rs.rs_maps(cam, f, depth, XH2_HQ_S, FPS, want_ders=False)
    x0 = W / 2.0
    X, _ = pixel_grid(W, H)
    clean = np.clip(x0 - (X - 0.5), 0, 1)[..., None].astype(np.float32)     # 1 left of x0, exact area coverage
    out = ap.warp(clean, m_rs)[..., 0]
    rows = np.arange(H) + 0.5
    mid = slice(20, H - 20)
    slope = np.polyfit(rows[mid], edge_x(out)[mid], 1)[0]
    # prediction: the edge's image velocity from the camera alone (central difference), * T / H
    R0, C0 = rs.pose_at(cam, f)
    k = cam.intrinsics(f)
    P = C0 + R0 @ np.array([(x0 - k.cx) / k.fx * 8.0, 0.0, -8.0])
    u_prev = cam.project(f - 1, P[None])[0][0, 0]
    u_next = cam.project(f + 1, P[None])[0][0, 0]
    pred = (u_next - u_prev) / 2.0 * T / H
    check(abs(slope - pred) < 0.01 * abs(pred), "pan: vertical edge leans by the predicted px per row",
          f"measured {slope:.5f}, predicted {pred:.5f} px/row, {abs(slope / pred - 1) * 100:.2f} % off")
    sk = rs.skew_px_per_row(m_rs)
    check(abs(-sk["dx_per_row"] - pred) < 0.02 * abs(pred), "pan: map's own skew report agrees",
          f"{-sk['dx_per_row']:.5f} px/row")


def test_forward_against_platelock():
    cam = make_cam(yaw_deg=1.2, pitch_deg=0.3, roll_deg=0.4, move=(0.05, 0.0, -0.25), wobble=0.8)
    f, T = 1004, XH2_HQ_S * FPS
    depth = scene_depth(cam, f, T)
    m_rs, _ = rs.rs_maps(cam, f, depth, XH2_HQ_S, FPS, want_ders=False)
    # world points on the floor / wall, projected by platelock's RS model (pixel centres on integers -> +0.5)
    rng = np.random.default_rng(3)
    pts = np.concatenate([np.c_[rng.uniform(-4, 4, 300), np.zeros(300), rng.uniform(-11, -3, 300)],
                          np.c_[rng.uniform(-6, 6, 300), rng.uniform(0.2, 5, 300), np.full(300, -12.0)]])
    tr = track_from_shot_camera(cam, T)
    u, v, z = tr.project(pts, f - cam.frames[0], iters=8)
    p = np.c_[u + 0.5, v + 0.5]
    ok = (z > 0) & (p[:, 0] > 3) & (p[:, 0] < W - 3) & (p[:, 1] > 3) & (p[:, 1] < H - 3)
    # only points that are the visible surface at that pixel (floor in front of the wall is fine either way)
    q_gs, _ = cam.project(f, pts)
    st = rs.sample(m_rs.astype(np.float64), p[ok, 0], p[ok, 1], W, H)
    qx, qy = from_st(st, W, H)
    e = np.hypot(qx - q_gs[ok, 0], qy - q_gs[ok, 1])
    disp = np.hypot(*(p[ok] - q_gs[ok]).T)
    check(np.median(e) < 0.05 and np.percentile(e, 95) < 0.2,
          "rs map agrees with platelock's independent RS projection",
          f"median {np.median(e):.4f} px, p95 {np.percentile(e, 95):.4f} px; RS displacement median "
          f"{np.median(disp):.2f} px over {ok.sum()} points")


def test_roundtrip():
    cam = make_cam(yaw_deg=1.5, pitch_deg=0.4, roll_deg=0.5, move=(0.08, 0.01, -0.3), wobble=0.4)
    f, T = 1004, XH2_HQ_S * FPS
    depth = scene_depth(cam, f, T)
    m_rs, m_de = rs.rs_maps(cam, f, depth, XH2_HQ_S, FPS, iters=3)
    r = rs.roundtrip(m_rs, m_de)
    check(r["median_px"] < 0.05, "RS then de-RS round trip < 0.05 px median",
          f"median {r['median_px']:.4f}, p95 {r['p95_px']:.4f}, max {r['max_px']:.3f} px")
    X, Y = pixel_grid(W, H)
    qx, qy = from_st(m_rs, W, H)
    disp = np.hypot(qx - X, qy - Y)
    check(np.median(disp) > 0.5 and disp.max() - disp.min() > 0.5, "round-trip case has real rolling shutter in it",
          f"RS displacement median {np.median(disp):.2f} px, range {disp.min():.2f}-{disp.max():.2f} px")


def test_flash():
    h, w = 216, 8
    for readout_fr, t_flash in ((0.74, 0.0), (0.5, 0.1), (0.37, -0.12)):
        src = lambda t: np.full((h, w), 1.0 if t >= 1000 + t_flash else 0.1, np.float32)
        img = rs.slit_scan(src, h, 1000, readout_fr)
        bright = np.nonzero(img[:, 0] > 0.5)[0]
        split = bright[0] if len(bright) else h
        expect = (t_flash / readout_fr + 0.5) * h                # first row whose centre time >= flash
        check(abs(split - expect) <= 1.0, f"flash at {t_flash:+.2f} frames, readout {readout_fr}: split at the "
              "right row", f"row {split}, expected {expect:.1f}")


def dots_scene(n=140, seed=5):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-7, 7, n), rng.uniform(0.3, 5.5, n), np.full(n, -12.0)]


def render_dots(cam, f, pts, sigma=1.4):
    uv, z = cam.project(f, pts)
    X, Y = pixel_grid(W, H)
    img = np.zeros((H, W), np.float32)
    for (u, v), zz in zip(uv, z):
        if zz <= 0 or not (-5 < u < W + 5 and -5 < v < H + 5):
            continue
        x0, x1 = max(0, int(u) - 6), min(W, int(u) + 7)
        y0, y1 = max(0, int(v) - 6), min(H, int(v) + 7)
        img[y0:y1, x0:x1] += np.exp(-((X[y0:y1, x0:x1] - u) ** 2 + (Y[y0:y1, x0:x1] - v) ** 2) / (2 * sigma ** 2))
    return img


def detect(img, thresh=0.25):
    from scipy import ndimage
    lab, n = ndimage.label(img > thresh)
    cms = ndimage.center_of_mass(img, lab, range(1, n + 1))
    return np.array([(x + 0.5, y + 0.5) for y, x in cms]).reshape(-1, 2)


def test_readout_validation():
    T_true = XH2_HQ_S * FPS
    cam = make_cam(n=11, yaw_deg=2.5, pitch_deg=0.5, move=(0.06, 0, -0.1), wobble=0.6)
    pts = dots_scene()
    obs = {}
    for f in cam.frames[1:-1]:
        depth = scene_depth(cam, f, T_true)
        m_rs, _ = rs.rs_maps(cam, f, depth, XH2_HQ_S, FPS, want_ders=False)
        img = ap.warp(render_dots(cam, f, pts)[..., None], m_rs)[..., 0]
        found = detect(img)
        # match each detection to the nearest global-shutter projection (RS moves dots by a few px only)
        uv, z = cam.project(f, pts)
        idx, xy = [], []
        for d in found:
            dist = np.hypot(*(uv - d).T)
            j = int(np.argmin(dist))
            crowded = np.sort(np.hypot(*(uv - uv[j]).T))[1] < 10      # two dots this close merge into one blob
            if dist[j] < 8 and z[j] > 0 and not crowded and 6 < d[0] < W - 6 and 6 < d[1] < H - 6:
                idx.append(j)
                xy.append(d)
        obs[f] = (np.array(idx), np.array(xy))
    est = readout_solve.estimate_readout(cam, pts, obs, refine_pose=False, min_points=10,
                                         grid=np.linspace(0, 1.2, 49))
    err = abs(est["readout_frames"] / T_true - 1)
    check(err < 0.05, "validation: readout solve recovers the RS clip's readout within 5 %",
          f"{est['readout_ms']:.2f} ms vs {XH2_HQ_S * 1000:.1f} ms ({err * 100:.2f} %), "
          f"{sum(len(o[0]) for o in obs.values())} tracks")
    est2 = readout_solve.estimate_readout(cam, pts, obs, refine_pose=True, min_points=10,
                                          grid=np.linspace(0, 1.2, 25))
    err2 = abs(est2["readout_frames"] / T_true - 1)
    check(err2 < 0.05, "validation: same with per-frame pose refinement (what the real plate uses)",
          f"{est2['readout_ms']:.2f} ms ({err2 * 100:.2f} %)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
