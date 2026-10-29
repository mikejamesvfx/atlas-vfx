"""T4: Measure & Scale (synthetic, ground truth known).

Scene in metres, Y up, ground y = 0: a 2.04 m door jamb and a 0.18 m step riser, seen from a camera walking in.
Everything is moved into "solve units" by an unknown similarity (scale 1/7.3, odd rotation, offset), as a raw solve
would be. Picks are projected with 0.3 px noise and written in Nuke's bottom-left pixel coordinates. The ground plane
is fitted from solve points in a picked region (with wall points mixed in as outliers).

Checks (plan): scale within 1 %, up axis within 0.5 deg, ground y within 1 cm. Also: a single-frame pick on the
ground plane lands on it, a bad length is refused (spread > 5 %) unless forced, Nuke's origin conversion is right.

Run: <V141 python> tests\\test_measure.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from atlas_vfx.camera_side import measure as ms  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from test_camera_io import PLATE, SENSOR, rot  # noqa: E402

W, H = PLATE["width"], PLATE["height"]
S0 = 1 / 7.3                                    # solve units per metre (unknown to measure)
Q = rot(np.array([0.2, -0.9, 0.4]) / np.linalg.norm([0.2, -0.9, 0.4]), 131.0)
T0 = np.array([3.0, -1.2, 0.7])
FRAMES = list(range(1000, 1040))
DOOR = (np.array([1.2, 0.0, -8.0]), np.array([1.2, 2.04, -8.0]))
STEP = (np.array([-1.0, 0.0, -4.0]), np.array([-1.0, 0.18, -4.0]))
SPOT = np.array([0.5, 0.0, -5.0])
BENCH_Y = 0.9                                     # a raised platform, for plane_distance


def to_solve(X):
    return S0 * (np.asarray(X) @ Q.T) + T0


def truth_cameras():
    mats = []
    for i, f in enumerate(FRAMES):
        a = i / (len(FRAMES) - 1)
        R = rot((0, 1, 0), 6 * math.sin(3 * a)) @ rot((1, 0, 0), -10 + 3 * a) @ rot((0, 0, 1), 1.5 * math.cos(5 * a))
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [0.4 * a, 1.6 - 0.05 * a + 0.01 * math.sin(9 * a), -2.5 * a]
        mats.append(M)
    return mats


def build(mats, name):
    return sc.build(name=name, frame_start=FRAMES[0], fps=24.0, plate=PLATE, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=16.47, source={"kind": "synthetic"})


def scene():
    truth = build(truth_cameras(), "truth")
    solve_mats = []
    for M in truth_cameras():
        N = np.eye(4)
        N[:3, :3] = Q @ M[:3, :3]
        N[:3, 3] = to_solve(M[:3, 3])
        solve_mats.append(N)
    solve = build(solve_mats, "solve")
    rng = np.random.default_rng(11)
    ground = np.column_stack([rng.uniform(-3, 3, 400), np.zeros(400), rng.uniform(-7.5, -2.5, 400)])
    ground[:, 1] += rng.normal(0, 0.003, 400)                       # 3 mm of solve noise
    wall = np.column_stack([rng.uniform(-3, 3, 120), rng.uniform(0, 2.5, 120), np.full(120, -8.0)])
    bench = np.column_stack([rng.uniform(1.8, 3.0, 250), np.full(250, BENCH_Y), rng.uniform(-6.5, -5.0, 250)])
    bench[:, 1] += rng.normal(0, 0.003, 250)
    pts = {"xyz": to_solve(np.vstack([ground, wall, bench]))}
    return truth, solve, pts, ground


def nuke_px(cam, f, X, rng, noise=0.3):
    uv = cam.project(f, X)[0][0] + rng.normal(0, noise, 2)
    return [float(uv[0]), float(H - uv[1])]                        # Nuke: origin bottom-left, y up


def make_picks(truth, door_len=2.04):
    rng = np.random.default_rng(5)
    fa, fb = [1000, 1015, 1030], [1005, 1020, 1035]
    return {
        "origin": "nuke_bottom_left",
        "measurements": [
            {"label": "door_jamb", "length_m": door_len, "vertical": True,
             "a": {str(f): nuke_px(truth, f, DOOR[0], rng) for f in fa},
             "b": {str(f): nuke_px(truth, f, DOOR[1], rng) for f in fa}},
            {"label": "step_riser", "length_m": 0.18, "vertical": True,
             "a": {str(f): nuke_px(truth, f, STEP[0], rng) for f in fb},
             "b": {str(f): nuke_px(truth, f, STEP[1], rng) for f in fb}},
        ],
        "points": [{"label": "deck_spot", "frames": {"1010": nuke_px(truth, 1010, SPOT, rng)},
                    "plane": {"frame": 1010, "region": [0, 0, W, 0.45 * H]}}],
        "ground": {"frame": 1010, "region": [0, 0, W, 0.45 * H]},   # bottom 45 % of the frame, Nuke coords
    }


def main() -> int:
    truth, solve, pts, ground_m = scene()
    picks = ms.load_picks(make_picks(truth), H)
    r = ms.measure(solve, picks, pts)
    new = ms.apply(solve, r, name="measured")

    scale_err = abs(r["m_per_unit"] * S0 - 1) * 100
    assert scale_err < 1.0, f"scale off by {scale_err:.2f} %"

    R = np.array(r["transform"]["rotation"])
    up_true_new = R @ (Q @ np.array([0, 1.0, 0]))
    up_err = math.degrees(math.acos(np.clip(up_true_new @ [0, 1, 0], -1, 1)))
    assert up_err < 0.5, f"up axis off by {up_err:.3f} deg"

    gy = ms.transform_points(r, to_solve(ground_m))[:, 1]
    assert abs(float(np.median(gy))) < 0.01 and float(np.percentile(np.abs(gy), 95)) < 0.012, (np.median(gy),)

    h_err = max(abs(new.position(f)[1] - truth.position(f)[1]) for f in FRAMES)
    assert h_err < 0.01, f"camera height off by {h_err * 100:.2f} cm"
    c0 = new.position(FRAMES[0])
    assert abs(c0[0]) < 1e-9 and abs(c0[2]) < 1e-9, c0
    fwd = -new.cam_to_world(FRAMES[0])[:3, 2]
    assert abs(fwd[0]) < 1e-9 and fwd[2] < 0, fwd                                      # looks towards -Z
    spot = np.array(r["points_m"]["deck_spot"])
    assert abs(spot[1]) < 0.01, spot
    for m in r["measurements"]:
        assert abs(m["length_after_m"] / m["length_m"] - 1) < 0.01, m
    sc.validate(new.data)
    assert new.data["scale"]["method"] == "known_lengths" and new.data["scale"]["ground_y_m"] == 0.0

    print(f"ok scale: {r['m_per_unit']:.5f} m/unit vs truth {1 / S0:.5f} ({scale_err:.3f} %), spread "
          f"{r['spread_pct']:.2f} %")
    print(f"ok up axis: {up_err:.3f} deg off (verticals agree within {r['up_spread_deg']:.3f} deg)")
    print(f"ok ground: true ground points at y median {np.median(gy) * 1000:+.1f} mm, p95 |y| "
          f"{np.percentile(np.abs(gy), 95) * 1000:.1f} mm; fitted from {r['ground_method']}, normal vs up "
          f"{r['ground_normal_vs_up_deg']:.3f} deg")
    print(f"ok camera: height error max {h_err * 1000:.1f} mm, frame {FRAMES[0]} at x=z=0 looking -Z; "
          f"single-frame pick on the ground lands at y {spot[1] * 1000:+.1f} mm")
    for m in r["measurements"]:
        print(f"    {m['label']}: {m['length_after_m']:.4f} m (given {m['length_m']}), ends "
              f"{m['a']['method']} rms {m['a']['rms_px']:.2f}/{m['b']['rms_px']:.2f} px")

    bad = ms.load_picks(make_picks(truth, door_len=2.40), H)
    try:
        ms.measure(solve, bad, pts)
    except ms.MeasureError as e:
        assert "spread" in str(e), e
        msg = str(e)
    else:
        raise AssertionError("2.40 m door with a 0.18 m step was not refused")
    forced = ms.measure(solve, bad, pts, force=True)
    assert forced["forced"] and forced["spread_pct"] > 5
    print(f"ok refuses bad length: {msg.splitlines()[0][:120]}")
    print(f"    forced: spread {forced['spread_pct']:.1f} %, recorded as forced")

    # top-left origin gives the same answer as Nuke bottom-left once converted
    tl = make_picks(truth)
    tl["origin"] = "top_left"
    for m in tl["measurements"]:
        for end in ("a", "b"):
            m[end] = {f: [xy[0], H - xy[1]] for f, xy in m[end].items()}
    for p in tl["points"]:
        p["frames"] = {f: [xy[0], H - xy[1]] for f, xy in p["frames"].items()}
        p["plane"]["region"] = [0, 0.55 * H, W, H]
    tl["ground"]["region"] = [0, 0.55 * H, W, H]
    r2 = ms.measure(solve, ms.load_picks(tl, H), pts)
    assert abs(r2["m_per_unit"] - r["m_per_unit"]) < 1e-9
    print("ok pick origins: top_left and nuke_bottom_left agree")

    test_guards(truth, solve, pts)
    test_plane_distance(truth, solve, pts)
    test_lock_check(new, r)
    return 0


def test_plane_distance(truth, solve, pts):
    """Scale and up from the perpendicular distance between two fitted planes (ground y=0 and a bench at BENCH_Y),
    no point picks at all (what the real stairs needed: deck to pad = 4 risers)."""
    corners = np.array([[1.8, BENCH_Y, -6.5], [3.0, BENCH_Y, -6.5], [1.8, BENCH_Y, -5.0], [3.0, BENCH_Y, -5.0]])
    frames = [1000, 1005, 1010]
    uv = np.vstack([truth.project(f, corners)[0] for f in frames])
    region = [float(uv[:, 0].min()) - 20, float(H - uv[:, 1].max()) - 20, float(uv[:, 0].max()) + 20,
              float(H - uv[:, 1].min()) + 20]                                   # Nuke bottom-left box around it
    p = {"origin": "nuke_bottom_left",
         "measurements": [{"label": "ground_to_bench", "kind": "plane_distance", "length_m": BENCH_Y,
                           "vertical": True, "a": {"frames": [1010, 1020], "region": [0, 0, W, 0.45 * H]},
                           "b": {"frames": frames, "region": region}}],
         "ground": {"frames": [1010, 1020], "region": [0, 0, W, 0.45 * H]}}
    r = ms.measure(solve, ms.load_picks(p, H), pts)
    err = abs(r["m_per_unit"] * S0 - 1) * 100
    R = np.array(r["transform"]["rotation"])
    up_err = math.degrees(math.acos(np.clip((R @ (Q @ np.array([0, 1.0, 0])))[1], -1, 1)))
    par = r["measurements"][0]["b"]["planes_parallel_within_deg"]
    assert err < 1.0 and up_err < 0.5, (err, up_err)
    print(f"ok plane distance: ground to bench gives scale {err:.3f} % off, up {up_err:.3f} deg off (planes parallel "
          f"within {par:.2f} deg), no point picks")


def test_guards(truth, solve, pts):
    """Ground region on the wall (the real-data failure: a plane that is not the ground) is refused; pooled frames
    work; a line of points is refused as a plane."""
    p = make_picks(truth)
    p["ground"] = {"frame": 1000, "region": [0, 0.55 * H, 0.45 * W, H]}  # top left: the wall, not the ground
    try:
        ms.measure(solve, ms.load_picks(p, H), pts)
    except ms.MeasureError as e:
        assert "off the verticals" in str(e), e
        msg = str(e)
    else:
        raise AssertionError("ground fitted on the wall was accepted")
    p = make_picks(truth)
    p["ground"] = {"frames": [1000, 1010, 1020, 1030], "region": [0, 0, W, 0.45 * H]}
    r = ms.measure(solve, ms.load_picks(p, H), pts)
    assert r["ground_normal_vs_up_deg"] < 0.5 and "4 pooled" in r["ground_method"], r["ground_method"]
    line = np.column_stack([np.linspace(0, 5, 50), np.zeros(50), np.zeros(50)]) + \
        np.random.default_rng(1).normal(0, 1e-4, (50, 3))
    try:
        ms.fit_plane(line)
    except ms.MeasureError as e:
        assert "nearly a line" in str(e)
    else:
        raise AssertionError("a line of points was fitted as a plane")
    print(f"ok guards: wall-as-ground refused ({msg[:70]}...), pooled frames fit ({r['ground_method']}), "
          f"line refused")


def test_lock_check(cam, r):
    """Card segment projection equals ShotCamera.project of the ends; near clipping keeps the visible part;
    render writes an mp4 over a synthetic plate."""
    import subprocess
    import tempfile

    from atlas_vfx.camera_side import lock_check as lc
    from atlas_vfx.camera_side import plate as pl
    from test_plate import make_plate

    spot = r["points_m"]["deck_spot"]
    lines = lc.card_lines(cam, spot, "ground", 1.0, 1.0, 4)
    f = FRAMES[10]
    segs = lc.project_segments(cam, f, lines, 1.0)
    assert len(segs) == len(lines)
    for (a, b), s in zip(lines, segs):
        uv = cam.project(f, np.array([a, b]))[0]
        assert np.allclose(uv, s, atol=1e-6)
    # a line from in front of the camera to behind it: the clipped end sits on the near plane
    C, fwd = cam.position(f), -cam.cam_to_world(f)[:3, 2]
    seg = lc.project_segments(cam, f, [(C + 3 * fwd, C - 3 * fwd + np.array([0.1, 0, 0]))], 1.0)
    assert len(seg) == 1 and np.all(np.isfinite(seg[0]))
    assert lc.project_segments(cam, f, [(C - fwd, C - 2 * fwd)], 1.0) == []

    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        # synthetic plate: one frame per camera frame, 512x270 (exactly 4096x2160 aspect), index 0 = shot frame 1000
        import numpy as _np
        planes = []
        for k in range(len(FRAMES)):
            y = _np.full((270, 512), 300 + k, _np.uint16)
            c = _np.full((270, 256), 512, _np.uint16)
            planes.append((y, c, c))
        mp = make_plate(td / "plates" / "syn.mov", "hevc", "pc", planes)
        ref = pl.plate_ref(mp, frame_offset=1000)
        out, kept = lc.render(ref, cam, td / "lock.mp4", lines=lines, marks={"deck_spot": spot}, width=256,
                              keep_every=10)
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True,
                               text=True).stdout.strip())
        assert n == len(FRAMES) and len(kept) == 4, (n, len(kept))
        green = (kept[1][..., 1] > 0.9) & (kept[1][..., 0] < 0.2)
        assert green.sum() > 50, "card not drawn"
    print(f"ok lock check: card projection exact, near clip, mp4 with {n} frames over a synthetic plate")


if __name__ == "__main__":
    sys.exit(main())
