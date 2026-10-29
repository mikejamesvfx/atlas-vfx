"""Plate lock core: every stage against a known answer.

- rotations, rolling-shutter project/backproject round trip
- anchor_align recovers the AI solve's gauge exactly
- re-shoot geometry: the backward map matches the analytic answer (plate RS ray -> wall -> AI camera) to a
  small fraction of a pixel, and modelling rolling shutter beats ignoring it
- retime recovers each performer's known curve
- correspondence field, map composition, STMap round trip
- edge snap closes a known silhouette offset
- EXR write/read with header attributes; seqio reads the metadata back
- push-pull fill stays inside the data range

Run: <V141 python> tests\\test_platelock_core.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.platelock.core import (align, camera, exr, fill, key, qc, reshoot, retime, seqio,  # noqa: E402
                                      synth, warp)

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


SHOT = synth.make_shot()


def test_rotations():
    R = camera.rot_xyz(0.3, -0.7, 1.1)
    q = camera.quat_from_rot(R)
    check(np.abs(camera.rot_from_quat(q) - R).max() < 1e-12, "quaternion round trip")
    q0, q1 = camera.quat_from_rot(np.eye(3)), camera.quat_from_rot(R)
    mid = camera.rot_from_quat(camera.slerp(q0, q1, 0.5))
    check(abs(camera.rot_angle_deg(mid) - camera.rot_angle_deg(R) / 2) < 1e-9, "slerp halfway is half the angle")


def test_rs_roundtrip():
    tr = SHOT.plate_track
    rng = np.random.default_rng(1)
    u = rng.uniform(0, tr.intr.width - 1, 500)
    v = rng.uniform(0, tr.intr.height - 1, 500)
    z = rng.uniform(2, 20, 500)
    X = tr.backproject(u, v, z, 11)
    u2, v2, z2 = tr.project(X, 11, iters=8)
    err = np.hypot(u2 - u, v2 - v).max()
    check(err < 1e-6, "rolling-shutter backproject -> project", f"max {err:.2e} px")


def test_anchor_align():
    g = align.anchor_align(SHOT.ai_track, SHOT.plate_track, 0, 1 / SHOT.damage.gauge_scale)
    P = np.random.default_rng(0).normal(size=(64, 3)) * 5
    err = np.abs(g.apply(P) - SHOT.true_gauge.apply(P)).max()
    check(err < 1e-9, "anchor_align recovers the AI gauge", f"{err:.1e}")
    moved = align.transform_track(SHOT.ai_track, g)
    check(np.abs(moved.R[0] - SHOT.plate_track.R[0]).max() < 1e-12, "aligned AI camera equals plate at frame 0")


def _true_map(i, j, plate_track):
    """Analytic: for each plate pixel, where the wall point it sees lands in AI frame j."""
    H, W = SHOT.plate.shape[1:3]
    vs, us = np.mgrid[0:H, 0:W].astype(float)
    o, d = plate_track.ray(us, vs, i)
    lam = (SHOT.meta["plane_z"] - o[:, 2]) / d[:, 2]
    Xp = o + lam[:, None] * d
    Xa = SHOT.true_gauge.inverse().apply(Xp)
    u, v, _ = SHOT.ai_track.project(Xa, j)
    return u.reshape(H, W), v.reshape(H, W)


def test_reshoot_geometry():
    g = align.anchor_align(SHOT.ai_track, SHOT.plate_track, 0, 2.0)
    for i in (8, 20):
        r = reshoot.reshoot(SHOT.ai_bg[i], SHOT.ai_depth[i], SHOT.ai_track, i, SHOT.plate_track, i, g, fill_radius=0)
        tu, tv = _true_map(i, i, SHOT.plate_track)
        m = r.valid > 0.5
        m[:3], m[-3:], m[:, :3], m[:, -3:] = False, False, False, False
        e = np.hypot(r.map_x - tu, r.map_y - tv)[m]
        check(np.median(e) < 0.02 and np.percentile(e, 99) < 0.1, f"re-shoot map matches analytic, frame {i}",
              f"median {np.median(e):.4f} px, p99 {np.percentile(e, 99):.4f} px")
        r0 = reshoot.reshoot(SHOT.ai_bg[i], SHOT.ai_depth[i], SHOT.ai_track, i,
                             SHOT.plate_track.with_readout(0), i, g, fill_radius=0)
        m0 = m & (r0.valid > 0.5)
        e0 = np.hypot(r0.map_x - tu, r0.map_y - tv)[m0]
        check(np.median(e0) > 5 * max(np.median(e), 1e-3), f"ignoring rolling shutter is clearly worse, frame {i}",
              f"median {np.median(e0):.3f} px without RS")


def test_retime():
    for k in range(2):
        rc = retime.solve_retime(SHOT.plate_masks[k], SHOT.ai_masks[k])
        d = np.abs(rc.tau - SHOT.tau_true[k])
        check(d.mean() < 0.3 and d.max() < 1.0, f"retime performer {k}",
              f"mean {d.mean():.3f}, max {d.max():.3f} frames")
    check(retime.isotonic(np.array([3, 1, 2, 5, 4.0])).tolist() == [2, 2, 2, 4.5, 4.5], "isotonic regression")


def test_warp_maps():
    h, w = 60, 90
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    fx = lambda x, y: x + 2.5 + 0.03 * y  # noqa: E731
    fy = lambda x, y: y - 1.5 + 0.02 * x  # noqa: E731
    rng = np.random.default_rng(3)
    dst = np.stack([rng.uniform(0, w - 1, 300), rng.uniform(0, h - 1, 300)], 1)
    src = np.stack([fx(dst[:, 0], dst[:, 1]), fy(dst[:, 0], dst[:, 1])], 1)
    mx, my = warp.correspondence_field(dst, src, (h, w), smoothing=0.0)
    e = np.hypot(mx - fx(xs, ys), my - fy(xs, ys)).max()
    check(e < 0.05, "correspondence field recovers an affine warp", f"max {e:.4f} px")
    a = warp.from_displacement(np.full((h, w), 1.25, np.float32), np.zeros((h, w), np.float32))
    b = warp.from_displacement(np.zeros((h, w), np.float32), np.full((h, w), -0.5, np.float32))
    cx, cy = warp.compose(a, b)
    check(abs(float(np.median(cx - xs)) - 1.25) < 1e-4 and abs(float(np.median(cy - ys)) + 0.5) < 1e-4,
          "map composition adds shifts")
    st = warp.to_stmap((mx, my), w, h)
    bx, by = warp.from_stmap(st, w, h)
    check(max(np.abs(bx - mx).max(), np.abs(by - my).max()) < 1e-3, "STMap round trip")


def test_edge_snap():
    h, w = 120, 120
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    plate = ((((xs - 60) / 22) ** 2 + ((ys - 60) / 30) ** 2) < 1).astype(np.float32)
    ai = ((((xs - 63) / 24) ** 2 + ((ys - 58) / 31) ** 2) < 1).astype(np.float32)
    maps = key.edge_snap_iterative(ai, plate, iters=3, max_shift=10, sigma=2.0)
    snapped = warp.remap(ai, *maps)
    b, a = qc.matte_iou(ai, plate), qc.matte_iou(snapped, plate)
    ed = qc.edge_distance(snapped, plate)
    check(a > 0.97 and ed < 0.8, "edge snap closes a 3px offset and size change",
          f"IoU {b:.3f} -> {a:.3f}, edge {qc.edge_distance(ai, plate):.2f} -> {ed:.2f} px")
    k = key.screen_key(SHOT.plate[5])
    union = np.maximum(SHOT.plate_masks[0, 5], SHOT.plate_masks[1, 5])
    check(qc.matte_iou(k, union) > 0.95, "screen key on the synthetic plate", f"IoU {qc.matte_iou(k, union):.3f}")


def test_exr_and_metadata():
    with tempfile.TemporaryDirectory() as d:
        rgb = np.random.default_rng(0).random((9, 13, 3)).astype(np.float32) * 4 - 1
        for f in range(3):
            exr.write_rgba(f"{d}/plate.{1001 + f:04d}.exr", rgb + f, rgb[..., 0], attrs={
                "framesPerSecond": ("rational", (24000, 1001)), "focalLength": ("float", 18.0),
                "cameraModel": ("string", "test body"), "expTime": ("float", 1 / 48.0)})
        back = exr.read_exr(f"{d}/plate.1002.exr")
        check(np.abs(back["G"] - (rgb[..., 1] + 1)).max() == 0, "EXR float pixels exact, negatives and >1 kept")
        frames, headers, files = seqio.read_sequence(f"{d}/plate.####.exr")
        meta = seqio.sequence_report(headers)["first"]
        check(frames.shape == (3, 9, 13, 3), "sequence reads in frame order")
        check(abs(meta["fps"] - 23.976) < 1e-3 and meta["focal_mm"] == 18.0 and meta["camera"] == "test body",
              "header metadata (fps rational, focal, camera)")
        check(abs(meta["shutter_angle"] - 179.82) < 0.1, "shutter angle derived from exposure", f"{meta['shutter_angle']:.2f}")


def test_fill():
    img = np.stack([np.linspace(0, 1, 50, dtype=np.float32)[None].repeat(40, 0)] * 3, -1)
    holes = np.zeros((40, 50), bool)
    holes[10:30, 15:35] = True
    out = fill.pushpull_fill(img, holes)
    check(out[holes].min() >= -1e-6 and out[holes].max() <= 1 + 1e-6 and np.abs(out - img)[holes].max() < 0.2,
          "push-pull fill is smooth and bounded", f"max err {np.abs(out - img)[holes].max():.3f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
