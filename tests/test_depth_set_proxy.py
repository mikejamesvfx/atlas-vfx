"""T5: Plate Depth scaling and Set Proxy (synthetic, ground truth known).

Scene (metres, Y up): deck y = 0 (x -3..3, z -7..-1), wall z = -7 (x -3..3, y 0..3), lawn y = -0.6 in front (clutter).
A camera walks in. Depth maps are rendered exactly, then each frame is multiplied by an unknown factor (0.3-2.0, like
MoGe's own scale) with 0.5 % noise. Sparse "solve" points on the deck and wall (1 mm noise) are the reference.

Checks: per-frame scale recovered within 0.5 %; frames with too few points get interpolated scales; the planes fitted
from unprojected depth are within 1 cm / 0.5 deg (plan); EXR round trip keeps Z, A and the attributes; OBJ/USDA
written with outlines inside the true extents.

Run: <V141 python> tests\\test_depth_set_proxy.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from atlas_vfx.camera_side import plate_depth as pd  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import set_proxy as sp  # noqa: E402
from test_camera_io import PLATE, SENSOR, rot  # noqa: E402

DW, DH = 512, 270                 # depth map size (proxy)
FRAMES = list(range(1130, 1160))


def camera():
    mats = []
    for i, f in enumerate(FRAMES):
        a = i / (len(FRAMES) - 1)
        R = rot((0, 1, 0), 8 * math.sin(2 * a)) @ rot((1, 0, 0), -14 + 4 * a) @ rot((0, 0, 1), 1.0 * math.cos(4 * a))
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [0.3 * a, 1.7 - 0.05 * a, 1.5 - 4.5 * a]
        mats.append(M)
    return sc.build(name="syn_s2s3", frame_start=FRAMES[0], fps=24.0, plate=PLATE, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=16.47, source={"kind": "synthetic"},
                    scale={"method": "known_lengths", "m_per_source_unit": 1.0})


def render_depth(cam, f):
    """Exact z depth of the scene at DWxDH, inf where nothing is hit."""
    k = cam.intrinsics(f)
    s = DW / k.width
    v, u = np.mgrid[0:DH, 0:DW]
    up, vp = (u + 0.5) / s, (v + 0.5) / s
    d_cam = np.stack([(up - k.cx) / k.fx, -(vp - k.cy) / k.fy, -np.ones_like(up)], -1)
    M = cam.cam_to_world(f)
    d = d_cam @ M[:3, :3].T
    o = M[:3, 3]
    best = np.full((DH, DW), np.inf)

    def hit(axis, value, lo, hi):
        t = (value - o[axis]) / d[..., axis]
        P = o + d * t[..., None]
        ok = (t > 0)
        for ax, (a, b) in zip(lo, hi):
            ok &= (P[..., ax] >= a) & (P[..., ax] <= b)
        return np.where(ok, t, np.inf)

    best = np.minimum(best, hit(1, 0.0, [0, 2], [(-3, 3), (-7, -1)]))          # deck
    best = np.minimum(best, hit(2, -7.0, [0, 1], [(-3, 3), (0, 3)]))           # wall
    best = np.minimum(best, hit(1, -0.6, [0, 2], [(-6, 6), (-1, 3)]))          # lawn
    return best          # t along d_cam whose z component is -1, so t == z depth


def solve_points(rng):
    deck = np.column_stack([rng.uniform(-3, 3, 500), np.zeros(500), rng.uniform(-7, -1, 500)])
    wall = np.column_stack([rng.uniform(-3, 3, 400), rng.uniform(0, 3, 400), np.full(400, -7.0)])
    return {"xyz": np.vstack([deck, wall]) + rng.normal(0, 0.001, (900, 3))}


def angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b)))))


def main() -> int:
    rng = np.random.default_rng(21)
    cam = camera()
    pts = solve_points(rng)
    factors = rng.uniform(0.3, 2.0, len(FRAMES))
    depths, stats = {}, []
    for f, kf in zip(FRAMES, factors):
        true = render_depth(cam, f)
        moge = true * kf * (1 + rng.normal(0, 0.005, true.shape))
        depths[f] = (true, moge)
        st = pd.frame_scale(cam, f, np.where(np.isfinite(moge), moge, 0.0), pts)
        stats.append(st)
    worst = max(abs(s["scale"] * kf - 1) for s, kf in zip(stats, factors))
    assert worst < 0.005, f"scale off by {worst * 100:.2f} %"
    print(f"ok depth scale: {len(FRAMES)} frames, unknown factors 0.3-2.0 recovered within {worst * 100:.3f} % "
          f"(median {int(np.median([s['points'] for s in stats]))} points per frame)")

    # too few points in two frames -> interpolated
    st2 = [dict(s) for s in stats]
    for k in (5, 6):
        st2[k]["scale"] = None
    st2 = pd.fill_scales(st2)
    assert st2[5].get("filled") and abs(st2[5]["scale"] - np.interp(FRAMES[5], [FRAMES[4], FRAMES[7]],
                                                                    [stats[4]["scale"], stats[7]["scale"]])) < 1e-12
    print("ok fill: frames without enough points take the scale interpolated from their neighbours")

    # unproject the scaled depth, fit the set
    cloud = []
    for (f, (true, moge)), s in zip(depths.items(), stats):
        if (f - FRAMES[0]) % 3:
            continue
        z = np.where(np.isfinite(moge), moge * s["scale"], 0.0)
        cloud.append(pd.unproject(cam, f, z, step=6))
    cloud = np.vstack(cloud + [pts["xyz"]])
    cloud = np.vstack([cloud, rng.uniform([-4, -1, -8], [4, 3, 2], (len(cloud) // 20, 3))])   # 5 % junk
    res = sp.fit_set(cloud, [cam.position(f) for f in FRAMES], deck_y=0.0)
    d, w = res["deck"], res["wall"]
    deck_h = -d["d"]                      # n=(0,1,0): y + d = 0
    assert angle(d["normal"], np.array([0, 1.0, 0])) < 0.5 and abs(deck_h) < 0.01, (d["normal"], deck_h)
    wall_z = w["centre"] @ w["normal"] / w["normal"][2]
    assert angle(w["normal"], np.array([0, 0, 1.0])) < 0.5 and abs(wall_z + 7.0) < 0.01, (w["normal"], wall_z)
    assert w["normal"][2] > 0, "wall normal should face the camera (+Z)"
    for name, p, lo, hi in (("deck", d, [-3.05, -0.02, -7.05], [3.05, 0.02, -0.95]),
                            ("wall", w, [-3.05, -0.05, -7.02], [3.05, 3.05, -6.98])):
        o = p["outline"]
        assert np.all(o >= np.array(lo) - 0.02) and np.all(o <= np.array(hi) + 0.02), (name, o.min(0), o.max(0))
    print(f"ok set proxy: deck normal {angle(d['normal'], [0, 1, 0]):.3f} deg, height {deck_h * 1000:+.1f} mm, "
          f"rms {d['rms'] * 1000:.1f} mm; wall normal {angle(w['normal'], [0, 0, 1]):.3f} deg, z {wall_z:+.4f} "
          f"(true -7), rms {w['rms'] * 1000:.1f} mm; from {res['points_used']} points with 5 % junk")

    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        f = FRAMES[10]
        z = np.where(np.isfinite(depths[f][1]), depths[f][1] * stats[10]["scale"], 0.0)
        valid = z > 0
        pd.write_exr(td / "depth.1140.exr", z, valid, {"camera": cam.name, "frame": f, "units": "m",
                                                       "scale": stats[10]["scale"]})
        Z, A, attrs = pd.read_exr(td / "depth.1140.exr")
        assert np.array_equal(Z, z.astype(np.float32)) and np.array_equal(A > 0.5, valid)
        assert attrs["camera"] == "syn_s2s3" and attrs["frame"] == f and attrs["units"] == "m"
        assert attrs["colorspace"] == "data" and attrs["ocio_colorspace"] == "Raw"
        obj = sp.write_obj(td / "set.obj", {"deck": d, "wall": w})
        usda = sp.write_usda(td / "set.usda", {"deck": d, "wall": w})
        txt = obj.read_text()
        assert txt.count("\no ") == 2 and txt.count("\nv ") == len(d["outline"]) + len(w["outline"])
        assert 'upAxis = "Y"' in usda.read_text() and usda.read_text().count("def Mesh") == 2
        try:
            from pxr import Usd  # noqa: F401  (optional: validate if USD is installed)
            stage = Usd.Stage.Open(str(usda))
            assert stage.GetPrimAtPath("/set_proxy/deck")
            usd_note = "opened with pxr"
        except ImportError:
            usd_note = "pxr not installed, text checked"
    print(f"ok files: EXR Z/A exact with attributes (colorInteropID data, OCIO Raw); OBJ and USDA with 2 meshes ({usd_note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
