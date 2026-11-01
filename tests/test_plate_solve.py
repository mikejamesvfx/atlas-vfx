"""camera_side/plate_solve.py (Solve Plate engine) on rendered shots with a known camera and lens.

A courtyard of textured walls, ground and boxes is ray traced through a moving camera with a known focal and
Brown k1/k2 (COLMAP RADIAL convention, which the contract stores as brown). Checks:
  1. dolly + pan: every frame registers, the solved camera (through camera_io.from_colmap) matches the truth after a
     similarity fit (Solve Compare), the focal comes back within 1 % and k1 within 0.01, distortion maps to brown.
  2. a textured block sliding through the frame: the motion test drops its points; with a mask it never gets
     features, and the masked solve is at least as good as the unmasked one.
  3. a nodal pan (rotation only): the degeneracy check warns.
Run: <V141 python> tests/test_plate_solve.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = Path(tempfile.mkdtemp(prefix="atlas_plate_solve_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import compare  # noqa: E402
from atlas_vfx.camera_side import plate_solve as ps  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

W, H, FPS, F0 = 960, 540, 24.0, 1001
SENSOR = {"width_mm": 23.5, "height_mm": 23.5 * H / W}
F_PX = 16.47 / 23.5 * W                     # 672.8 px
K1, K2 = -0.08, 0.012
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


# ------------------------------------------------------------------------------------------ scene

rng = np.random.default_rng(7)
TEX = [gaussian_filter(rng.random((700, 700)), 1.2) for _ in range(6)]
TEXC = [gaussian_filter(rng.random((90, 90)), 1.0) for _ in range(6)]

# axis-aligned rectangles: (axis, value, (lo_a, hi_a), (lo_b, hi_b), texture id); a, b = the other two axes in order
RECTS = [
    (1, 0.0, (-8, 8), (-14, 6), 0),          # ground y = 0 (x, z)
    (2, -12.0, (-8, 8), (0, 6), 1),          # back wall z = -12 (x, y)
    (0, -6.0, (0, 6), (-14, 6), 2),          # left wall x = -6 (y, z)
    (0, 6.0, (0, 6), (-14, 6), 3),           # right wall x = 6
    (2, -5.0, (-2.5, -0.5), (0, 1.6), 4),    # box front faces
    (0, -0.5, (0, 1.6), (-7, -5), 4),
    (2, -8.0, (1.0, 3.0), (0, 2.4), 5),
    (0, 1.0, (0, 2.4), (-10, -8), 5),
]


def texture(tid, a, b, cell=0.03):
    s = map_coordinates(TEX[tid], [b / cell % 700, a / cell % 700], order=1, mode="wrap")
    c = map_coordinates(TEXC[tid], [b / 0.25 % 90, a / 0.25 % 90], order=1, mode="wrap")
    return 0.05 + 0.9 * np.clip((0.55 * s + 0.45 * c - 0.3) * 2.5, 0, 1)


def trace(C, D, movers=()):
    """Nearest hit of rays C + t D against RECTS (+ moving rects); returns the shade."""
    best = np.full(D.shape[:-1], np.inf)
    shade = np.full(D.shape[:-1], 0.05)
    for rect in list(RECTS) + list(movers):
        axis, val, ra, rb, tid = rect[:5]
        da, db = rect[5] if len(rect) > 5 else (0.0, 0.0)      # a mover carries its texture with it
        oth = [i for i in range(3) if i != axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - C[axis]) / D[..., axis]
        P = C + t[..., None] * D
        a, b = P[..., oth[0]], P[..., oth[1]]
        hit = (t > 0.05) & (t < best) & (a >= ra[0]) & (a <= ra[1]) & (b >= rb[0]) & (b <= rb[1])
        if hit.any():
            best = np.where(hit, t, best)
            shade = np.where(hit, texture(tid, np.where(hit, a - da, 0), np.where(hit, b - db, 0)), shade)
    return shade


def undistort_norm(xd, yd, k1, k2):
    x, y = xd.copy(), yd.copy()
    for _ in range(25):
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2
        x, y = xd / f, yd / f
    return x, y


def render(M, f_px=F_PX, k1=K1, k2=K2, movers=(), ss=2):
    """Distorted image through camera-to-world M (Nuke axes: -Z forward, Y up), supersampled ss x ss."""
    xs = (np.arange(W * ss) + 0.5) / ss
    ys = (np.arange(H * ss) + 0.5) / ss
    u, v = np.meshgrid(xs, ys)
    x, y = undistort_norm((u - W / 2) / f_px, (v - H / 2) / f_px, k1, k2)
    d = np.stack([x, -y, -np.ones_like(x)], -1) @ M[:3, :3].T
    img = trace(M[:3, 3], d, movers)
    return img.reshape(H, ss, W, ss).mean((1, 3))


def cam_matrix(pos, yaw, pitch, roll=0.0):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("YXZ", [yaw, pitch, roll], degrees=True).as_matrix()
    m[:3, 3] = pos
    return m


def write_shot(folder: Path, mats, movers_at=None):
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, M in enumerate(mats):
        img = render(M, movers=movers_at(i) if movers_at else ())
        n = f"plate_{i:04d}.png"
        cv2.imwrite(str(folder / n), np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))
        names.append(n)
    return names


def truth_camera(mats, name):
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    return sc.build(name=name, frame_start=F0, fps=FPS, plate=plate, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=16.47, source={"kind": "synthetic"})


def rot_error(cam, truth):
    """Rotation error per frame (deg) after the best global rotation: the chordal mean of R_truth R_solved^T.
    Works for a nodal camera too (its centres don't move, so a centre fit can't align it)."""
    fr = [f for f in cam.frames if f in set(truth.frames)]
    Rs = [cam.cam_to_world(f)[:3, :3] for f in fr]
    Rt = [truth.cam_to_world(f)[:3, :3] for f in fr]
    U, _s, Vt = np.linalg.svd(sum(t @ s.T for s, t in zip(Rs, Rt)))
    A = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    e = [np.degrees(Rotation.from_matrix(t.T @ A @ s).magnitude()) for s, t in zip(Rs, Rt)]
    return float(np.median(e)), float(np.max(e))


def solved_camera(res, name):
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    return cio.from_colmap(res.model_dir, plate=plate, sensor=SENSOR, frame_offset=F0, name=name,
                           frames=(res.run[0] + F0, res.run[1] + F0))


# ------------------------------------------------------------------------------------------ cases

def dolly_pan(n=36):
    return [cam_matrix([-2.0 + 4.0 * i / (n - 1), 1.6 + 0.2 * np.sin(i / 6), 3.0 - 2.0 * i / (n - 1)],
                       12 - 24 * i / (n - 1), -8 + 3 * np.sin(i / 5), 1.0 * np.sin(i / 7)) for i in range(n)]


def case_dolly():
    mats = dolly_pan()
    names = write_shot(TMP / "dolly", mats)
    opts = ps.SolveOptions(focal_px=F_PX * 1.08)        # an 8 % wrong guess, as from a zoom mark
    # manual tracks: box and wall corners projected through the true camera and lens (as Track 2D would give)
    lensed = sc.ShotCamera({**truth_camera(mats, "truth").data,
                            "lens": {"distortion_model": "brown", "distortion_params": [K1, K2, 0, 0, 0]}})
    corners = {"box_a": [-2.5, 1.6, -5.0], "box_b": [-0.5, 1.6, -5.0], "box_c": [1.0, 2.4, -8.0],
               "wall": [-6.0, 3.0, -12.0]}
    tracks = {"kind": "atlas.tracks_2d", "version": 1, "width": W, "height": H, "tracks": {}}
    for label, X in corners.items():
        t = {}
        for i, f in enumerate(lensed.frames):
            uv, z = lensed.project(f, [X])
            if z[0] > 0 and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
                t[str(i)] = uv[0].tolist()
        tracks["tracks"][label] = t
    res = ps.solve_folder(TMP / "dolly", names, TMP / "dolly_solve", opts, tracks=tracks)
    r = res.report
    check(r["manual_track_matches"] > 50, f"dolly: manual tracks added {r['manual_track_matches']} matches")
    check(r["registered"] == len(names), f"dolly: {r['registered']}/{len(names)} frames registered")
    check(r["mean_reproj_px"] < 0.8, f"dolly: mean reprojection {r['mean_reproj_px']:.3f} px < 0.8")
    f = r["camera"]["focal_px"]
    check(abs(f / F_PX - 1) < 0.01, f"dolly: focal {f:.2f} px vs truth {F_PX:.2f} ({100 * (f / F_PX - 1):+.2f} %)")
    k1, k2 = r["camera"]["params"][3:5]
    check(abs(k1 - K1) < 0.01, f"dolly: k1 {k1:.4f} vs truth {K1}  (k2 {k2:.4f} vs {K2})")
    check(not r["degeneracy"]["warn"], f"dolly: no degeneracy warning ({r['degeneracy']})")
    cam = solved_camera(res, "dolly_solve")
    check(cam.data["lens"]["distortion_model"] == "brown"
          and abs(cam.data["lens"]["distortion_params"][0] - k1) < 1e-12,
          f"dolly: from_colmap maps RADIAL to brown {cam.data['lens']['distortion_params']}")
    check(abs(cam.intrinsics(cam.frames[0]).focal_mm / 16.47 - 1) < 0.01,
          f"dolly: focal_mm {cam.intrinsics(cam.frames[0]).focal_mm:.3f} vs 16.47")
    cmp = compare.compare(cam, truth_camera(mats, "truth"))
    print("     " + compare.summary_line(cmp))
    check(cmp["residual_pct_of_path"] < 0.5, f"dolly: centre residual {cmp['residual_pct_of_path']:.3f} % of path < 0.5")
    med, mx = rot_error(cam, truth_camera(mats, "truth"))
    check(mx < 0.05, f"dolly: rotation error median {med:.4f}, max {mx:.4f} deg < 0.05")
    return res


def case_mover():
    mats = dolly_pan(30)
    # a textured board sliding left to right 3 m in front of the back wall, 25 cm per frame, texture riding on it
    def movers(i):
        x0 = -4.0 + 0.25 * i
        return [(2, -9.0, (x0, x0 + 2.5), (0.2, 3.2), 3, (x0, 0.0))]
    names = write_shot(TMP / "mover", mats, movers)
    truth = truth_camera(mats, "truth")
    out = {}
    for label, masks in (("unmasked", None), ("masked", "auto")):
        m = None
        if masks:
            m = []
            for i, M in enumerate(mats):          # truth mask: render the board alone, white where it is
                d = render_mask(M, movers(i))
                m.append(d)
        res = ps.solve_folder(TMP / "mover", names, TMP / f"mover_{label}", ps.SolveOptions(focal_px=F_PX),
                              masks=m, mask_dilate_px=4)
        cmp = compare.compare(solved_camera(res, label), truth)
        out[label] = (res, cmp)
        print(f"     {label}: {compare.summary_line(cmp)}; reject {res.report['reject']['rounds']}")
    um, ma = out["unmasked"], out["masked"]
    check(ma[0].report["masked"] and ma[0].report["registered"] == len(names),
          f"mover: masked solve registers all {len(names)} frames")
    for label, (res, _cmp) in out.items():
        med, mx = rot_error(solved_camera(res, label), truth)
        check(mx < 0.05, f"mover {label}: rotation error median {med:.4f}, max {mx:.4f} deg < 0.05")
    on_board = {label: points_on_board(res, mats, movers) for label, (res, _c) in out.items()}
    check(on_board["masked"] == 0, f"mover: masked solve has no points on the board ({on_board['masked']})")
    check(um[0].report["reject"]["rounds"][0]["dropped"] > 0 and on_board["unmasked"] < 20,
          f"mover: unmasked solve drops the board's points (dropped {um[0].report['reject']['rounds'][0]['dropped']},"
          f" {on_board['unmasked']} left on it)")


def points_on_board(res, mats, movers):
    """Solved points whose observations fall on the moving board in more than half their frames."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(res.model_dir))
    masks = {}
    hits = {}
    for img in rec.images.values():
        f = ps.frame_of(img.name)
        if f not in masks:
            masks[f] = render_mask(mats[f], movers(f)) > 0
        for p in img.points2D:
            if p.has_point3D():
                x, y = int(p.xy[0]), int(p.xy[1])
                on = 0 <= x < W and 0 <= y < H and masks[f][y, x]
                a = hits.setdefault(int(p.point3D_id), [0, 0])
                a[0] += int(on)
                a[1] += 1
    return sum(1 for a, b in hits.values() if a > b / 2)


def render_mask(M, movers):
    xs, ys = np.arange(W) + 0.5, np.arange(H) + 0.5
    u, v = np.meshgrid(xs, ys)
    x, y = undistort_norm((u - W / 2) / F_PX, (v - H / 2) / F_PX, K1, K2)
    d = np.stack([x, -y, -np.ones_like(x)], -1) @ M[:3, :3].T
    with_m = trace(M[:3, 3], d, movers)
    without = trace(M[:3, 3], d)
    return (np.abs(with_m - without) > 1e-6).astype(np.float32)


def case_nodal():
    n = 24
    mats = [cam_matrix([0.0, 1.6, 2.0], 20 - 40 * i / (n - 1), -6.0 + 2 * np.sin(i / 4)) for i in range(n)]
    names = write_shot(TMP / "nodal", mats)
    res = ps.solve_folder(TMP / "nodal", names, TMP / "nodal_solve", ps.SolveOptions())   # no focal guess
    r = res.report
    check(r["mode"] == "nodal" and r["registered"] == n, f"nodal: falls back to a nodal solve, {r['registered']}/{n}")
    f = r["camera"]["focal_px"]
    check(abs(f / F_PX - 1) < 0.01, f"nodal: focal {f:.2f} px vs truth {F_PX:.2f} ({100 * (f / F_PX - 1):+.2f} %), "
          f"no guess")
    k1 = r["camera"]["params"][3]
    check(abs(k1 - K1) < 0.015, f"nodal: k1 {k1:.4f} vs truth {K1}")
    cam = solved_camera(res, "nodal")
    med, mx = rot_error(cam, truth_camera(mats, "truth"))
    check(mx < 0.05, f"nodal: rotation error median {med:.4f}, max {mx:.4f} deg < 0.05")
    check(any("nodal solve" in w for w in r["warnings"]), "nodal: report says it is a nodal solve")
    pts = cio.colmap_points(res.model_dir, plate_width=W, frame_offset=F0)
    f0 = cam.frames[n // 2]
    k, xy = pts["observations"][f0]
    uv, _ = cam.project(f0, pts["xyz"][k])
    e = float(np.median(np.linalg.norm(uv - xy, axis=1)))
    check(e < 0.5, f"nodal: points reproject through the shot camera (lens applied) at {e:.3f} px")


def main() -> int:
    try:
        check(ps.longest_run([3, 4, 5, 9, 10]) == (3, 5) and ps.gaps([1, 2, 4], 1, 5) == [3, 5],
              "longest_run / gaps")
        check(ps.frame_of("plate_0042.png") == 42, "frame_of")
        import pycolmap

        pc = pycolmap.Camera.create_from_model_name(1, "RADIAL", 500.0, 960, 540)
        check(ps._project(pc, np.array([0.1, 0.2, 2.0])) == (505.0, 320.0)
              and np.isnan(ps._project(pc, np.array([0.1, 0.2, -2.0]))[0]),
              "a point behind the camera projects to NaN (pycolmap gives None), not a crash")
        case_dolly()
        case_mover()
        case_nodal()
    finally:
        if not fails:
            shutil.rmtree(TMP, ignore_errors=True)
        else:
            print("kept", TMP)
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
