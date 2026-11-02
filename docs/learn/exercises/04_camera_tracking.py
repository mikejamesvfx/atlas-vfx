"""Exercise 04: the 3D camera tracker (Solve Plate's engine) on a shot where we know the answer.

What it does, in numbered steps you can watch in the console:
  1. builds a small textured courtyard and renders two synthetic plates through a camera with a KNOWN path, focal
     (16.47 mm on a 23.5 mm sensor) and lens distortion (k1 -0.08, k2 0.012):
       a) a handheld-style walk with a pan (the camera translates, so there is parallax)
       b) a tripod pan (the camera only turns about its lens: no parallax at all)
  2. solves the walk with camera_side/plate_solve.py (SIFT + sequential matching + pycolmap incremental SfM, focal and
     k1/k2 solved in the bundle adjust, the motion test that drops points that moved), starting from a focal guess
     8 % wrong, like a zoom mark
  3. turns the COLMAP model into a shot camera (camera_io.from_colmap) and grades it against the truth:
     reprojection error, focal and k1 error, camera centres and view angles after a similarity fit (Solve Compare)
  4. does the COLMAP -> Nuke conversion by hand for one frame (R_c2w_nuke = R_c2w_colmap @ diag(1,-1,-1)) and turns
     the matrix into Nuke ZXY Euler angles, then checks it matches what from_colmap wrote
  5. solves the tripod pan with no focal guess at all, twice. camera_move auto first: SfM trades focal against
     distortion into a wrong lens. If that lens is past the plausibility limits (|k| > 0.25 and so on) auto goes
     nodal by itself; if it is wrong but still plausible, auto keeps SfM and only warns (the auto rule's blind spot;
     this scene sits right on that line). Then camera_move nodal: the rotation-only solve, which finds the focal
     from the rotation alone
  6. runs the camera checks the downstream nodes use: the walk is unscaled (Measure & Scale first), the pan is nodal
     (no depth, so no metres ever)

Synthetic data only: no plate, no model weights, no GPU, no network. Writes only into a temp folder (set as
ATLAS_VFX_OUT_ROOTS) and deletes it at the end. Needs pycolmap, opencv, scipy (the ComfyUI venv has them).
Run from the repo root:  <ComfyUI venv python> docs/learn/exercises/04_camera_tracking.py
Takes about two minutes on a desktop CPU (most of it pycolmap and the nodal solve).

Try changing (one at a time, then rerun):
  - FOCAL_GUESS_ERR = 1.30: a 30 % wrong zoom mark. Does the focal still come back? (SfM refines it; the lens check
    only calls a lens impossible past 2x the guess.)
  - REJECT_PX = 0.3: a very aggressive motion test. Watch "points" fall and ask whether the error you gained is worth
    it (the NukeX lesson in the guide: culling to a pretty number can leave frames with nothing to hold them).
  - WALK_STEP_M = 0.0: the walk stops walking and becomes a pan. Watch the triangulation angle collapse, the
    low-parallax warning appear and maybe the mode flip to nodal (step 2's checks will fail: that's the point).
  - PAN_FRAMES = 24 and PAN_DEG = 40 (the repo test's case is 24 frames, 40 deg, 960 wide): SfM's lens can go far
    enough off (focal about 2x) that auto catches it and goes nodal by itself. Where that line sits is the lesson:
    auto is a heuristic, and the warning is there for you to read.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="atlas_learn_04_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)          # every output the repo's code writes must land in here

from atlas_vfx.camera_side import camera_checks as cc  # noqa: E402
from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import compare  # noqa: E402
from atlas_vfx.camera_side import plate_solve as ps  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

# ---------------------------------------------------------------------------------------- knobs to play with
W, H = 640, 360                       # the solver's image size (Solve Plate's proxy_width is 1024 on real plates)
SENSOR = {"width_mm": 23.5, "height_mm": 23.5 * H / W}
FOCAL_MM = 16.47
F_PX = FOCAL_MM / SENSOR["width_mm"] * W            # focal in pixels at this width: 448.5 px
K1, K2 = -0.08, 0.012                               # a mild barrel lens, COLMAP RADIAL / Brown convention
FOCAL_GUESS_ERR = 1.08                              # the guess we hand the solver (8 % long)
REJECT_PX = 2.0                                     # Solve Plate's reject_px default
WALK_FRAMES, PAN_FRAMES = 24, 16
WALK_STEP_M = 3.0                                   # how far the walk travels in total (metres)
PAN_DEG = 36.0                                      # how far the tripod pan turns
F0 = 1001                                           # plate frame 0 is shot frame 1001, as in Nuke

fails: list[str] = []


def step(n, text):
    print(f"\n[{n}] {text}", flush=True)


def check(ok, msg):
    print(("   ok    " if ok else "   FAIL  ") + msg, flush=True)
    if not ok:
        fails.append(msg)


# ---------------------------------------------------------------------------------------- a tiny ray tracer
rng = np.random.default_rng(7)
TEX = [gaussian_filter(rng.random((600, 600)), 1.2) for _ in range(6)]       # fine grain (SIFT likes it)
TEXC = [gaussian_filter(rng.random((90, 90)), 1.0) for _ in range(6)]       # coarse blotches

# axis-aligned rectangles: (axis, value, (lo, hi) on the first other axis, (lo, hi) on the second, texture)
RECTS = [
    (1, 0.0, (-8, 8), (-14, 6), 0),          # ground, y = 0
    (2, -12.0, (-8, 8), (0, 6), 1),          # back wall, z = -12
    (0, -6.0, (0, 6), (-14, 6), 2),          # left wall
    (0, 6.0, (0, 6), (-14, 6), 3),           # right wall
    (2, -5.0, (-2.5, -0.5), (0, 1.6), 4),    # a box near camera (strong parallax against the wall)
    (0, -0.5, (0, 1.6), (-7, -5), 4),
    (2, -8.0, (1.0, 3.0), (0, 2.4), 5),      # a taller box further back
    (0, 1.0, (0, 2.4), (-10, -8), 5),
]


def shade(tid, a, b):
    s = map_coordinates(TEX[tid], [b / 0.03 % 600, a / 0.03 % 600], order=1, mode="wrap")
    c = map_coordinates(TEXC[tid], [b / 0.25 % 90, a / 0.25 % 90], order=1, mode="wrap")
    return 0.05 + 0.9 * np.clip((0.55 * s + 0.45 * c - 0.3) * 2.5, 0, 1)


def trace(C, D):
    best = np.full(D.shape[:-1], np.inf)
    out = np.full(D.shape[:-1], 0.05)
    for axis, val, ra, rb, tid in RECTS:
        oth = [i for i in range(3) if i != axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - C[axis]) / D[..., axis]
        P = C + t[..., None] * D
        a, b = P[..., oth[0]], P[..., oth[1]]
        hit = (t > 0.05) & (t < best) & (a >= ra[0]) & (a <= ra[1]) & (b >= rb[0]) & (b <= rb[1])
        best = np.where(hit, t, best)
        out = np.where(hit, shade(tid, np.where(hit, a, 0), np.where(hit, b, 0)), out)
    return out


def undistort_norm(xd, yd):
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x * x + y * y
        f = 1 + K1 * r2 + K2 * r2 * r2
        x, y = xd / f, yd / f
    return x, y


def render(M, ss=1):
    """The image the lensed camera sees. M = camera-to-world in Nuke axes (Y up, looking down -Z)."""
    xs, ys = (np.arange(W * ss) + 0.5) / ss, (np.arange(H * ss) + 0.5) / ss
    u, v = np.meshgrid(xs, ys)
    x, y = undistort_norm((u - W / 2) / F_PX, (v - H / 2) / F_PX)      # pixel -> the ray the lens bent there
    d = np.stack([x, -y, -np.ones_like(x)], -1) @ M[:3, :3].T            # image y down, Nuke y up, looking down -Z
    return trace(M[:3, 3], d).reshape(H, ss, W, ss).mean((1, 3))


def cam(pos, yaw, pitch, roll=0.0):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("YXZ", [yaw, pitch, roll], degrees=True).as_matrix()
    m[:3, 3] = pos
    return m


def write_plate(folder: Path, mats) -> list[str]:
    import cv2

    folder.mkdir(parents=True)
    names = []
    for i, M in enumerate(mats):
        n = f"plate_{i:04d}.png"                    # the frame number is the last number in the name
        cv2.imwrite(str(folder / n), np.clip(render(M) * 255 + 0.5, 0, 255).astype(np.uint8))
        names.append(n)
    return names


def truth(mats, name):
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    return sc.build(name=name, frame_start=F0, fps=24.0, plate=plate, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=FOCAL_MM, source={"kind": "synthetic"})


def solved(res, name):
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    return cio.from_colmap(res.model_dir, plate=plate, sensor=SENSOR, frame_offset=F0, name=name,
                           frames=(res.run[0] + F0, res.run[1] + F0))


def rotation_error(a, b):
    """Per-frame view rotation error (deg) after the best single global rotation (works for nodal cameras too)."""
    fr = [f for f in a.frames if f in set(b.frames)]
    Ra = [a.cam_to_world(f)[:3, :3] for f in fr]
    Rb = [b.cam_to_world(f)[:3, :3] for f in fr]
    U, _s, Vt = np.linalg.svd(sum(t @ s.T for s, t in zip(Ra, Rb)))
    A = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    e = [np.degrees(Rotation.from_matrix(t.T @ A @ s).magnitude()) for s, t in zip(Ra, Rb)]
    return float(np.median(e)), float(np.max(e))


def quiet(_msg):                     # the solver talks a lot; keep the console for the lesson
    pass


# ---------------------------------------------------------------------------------------- the lesson
def main() -> int:
    t_all = time.time()
    step(1, f"render two plates at {W}x{H}: lens {FOCAL_MM} mm on a {SENSOR['width_mm']} mm sensor "
            f"= {F_PX:.1f} px, k1 {K1}, k2 {K2}")
    n = WALK_FRAMES
    walk = [cam([-WALK_STEP_M / 2 + WALK_STEP_M * i / (n - 1), 1.6 + 0.15 * np.sin(i / 4), 3.0 - 1.5 * i / (n - 1)],
                12 - 24 * i / (n - 1), -8 + 3 * np.sin(i / 5), 1.0 * np.sin(i / 6)) for i in range(n)]
    pan = [cam([0.0, 1.6, 2.0], PAN_DEG / 2 - PAN_DEG * i / (PAN_FRAMES - 1), -6.0 + 2 * np.sin(i / 4))
           for i in range(PAN_FRAMES)]
    t = time.time()
    walk_names = write_plate(TMP / "walk", walk)
    pan_names = write_plate(TMP / "pan", pan)
    print(f"   walk: {len(walk_names)} frames, camera travels {WALK_STEP_M} m while turning 24 deg")
    print(f"   pan:  {len(pan_names)} frames, camera stays put and turns {PAN_DEG:g} deg (a tripod pan)")
    print(f"   rendered in {time.time() - t:.1f} s")

    # ---------------------------------------------------------------- 2. solve the walk
    step(2, f"Solve Plate's engine on the walk, focal guess {FOCAL_GUESS_ERR:.2f} x truth, reject_px {REJECT_PX}")
    opts = ps.SolveOptions(focal_px=F_PX * FOCAL_GUESS_ERR, reject_px=REJECT_PX)
    t = time.time()
    res = ps.solve_folder(TMP / "walk", walk_names, TMP / "walk_solve", opts, log=quiet)
    r = res.report
    f_px = r["camera"]["focal_px"]
    k1, k2 = r["camera"]["params"][3:5]
    print(f"   mode {r['mode']}, {r['registered']}/{r['of']} frames, {r['points3D']} points, "
          f"mean reprojection {r['mean_reproj_px']:.3f} px (at {W} wide), solved in {time.time() - t:.1f} s")
    print(f"   motion test: " + ", ".join(f"round {i + 1} dropped {x['dropped']}" for i, x in
                                         enumerate(r["reject"]["rounds"])))
    print(f"   focal {f_px:.2f} px vs truth {F_PX:.2f} ({100 * (f_px / F_PX - 1):+.2f} %), started "
          f"{100 * (FOCAL_GUESS_ERR - 1):+.0f} % off")
    print(f"   k1 {k1:+.4f} (truth {K1}), k2 {k2:+.4f} (truth {K2})")
    print(f"   parallax: panoramic pairs {r['degeneracy']['panoramic_pair_frac']:.0%}, median triangulation angle "
          f"{r['degeneracy']['median_tri_angle_deg']:.1f} deg, warnings: {r['warnings'] or 'none'}")
    check(r["mode"] == "sfm", "a translating shot is solved by SfM (free camera)")
    check(r["registered"] == len(walk_names), "every frame registered")
    check(r["mean_reproj_px"] < 1.0, f"mean reprojection {r['mean_reproj_px']:.3f} px < 1.0")
    check(abs(f_px / F_PX - 1) < 0.02, "focal back within 2 % of the truth")
    check(abs(k1 - K1) < 0.02, "k1 back within 0.02 of the truth")

    # ---------------------------------------------------------------- 3. grade the camera
    step(3, "into a shot camera (camera_io.from_colmap) and graded against the true path (Solve Compare)")
    cam_w = solved(res, "walk_solve")
    k = cam_w.intrinsics(cam_w.frames[0])
    print(f"   shot camera frames {cam_w.frames[0]}-{cam_w.frames[-1]}, focal {k.focal_mm:.3f} mm, lens "
          f"{cam_w.data['lens']['distortion_model']} {[round(v, 4) for v in cam_w.data['lens']['distortion_params']]}"
          f", scale.method '{(cam_w.data.get('scale') or {}).get('method', 'unscaled')}'")
    cmp = compare.compare(cam_w, truth(walk, "truth"))
    med, mx = rotation_error(cam_w, truth(walk, "truth"))
    print(f"   {compare.summary_line(cmp)}")
    print(f"   rotation error after a global fit: median {med:.4f} deg, max {mx:.4f} deg")
    print(f"   scale truth/solve {cmp['scale_b_per_a']:.3f}: a solve has no idea how big the world is")
    check(cmp["residual_pct_of_path"] < 1.0, f"centres within {cmp['residual_pct_of_path']:.3f} % of the path")
    check(mx < 0.1, f"view rotation within 0.1 deg on every frame ({mx:.4f})")

    # ---------------------------------------------------------------- 4. COLMAP -> Nuke by hand
    step(4, "COLMAP -> Nuke axes by hand, one frame")
    import pycolmap

    rec = pycolmap.Reconstruction(str(res.model_dir))
    img = sorted((i for i in rec.images.values() if i.has_pose), key=lambda i: ps.frame_of(i.name))[len(walk) // 2]
    pose = img.cam_from_world()
    R_w2c = np.asarray(pose.rotation.matrix())          # COLMAP stores world -> camera, x right, y DOWN, z FORWARD
    t_w2c = np.asarray(pose.translation)
    R_c2w_colmap = R_w2c.T
    centre = -R_w2c.T @ t_w2c
    R_c2w_nuke = R_c2w_colmap @ np.diag([1.0, -1.0, -1.0])    # flip y and z: y UP, looking down -Z
    shot_f = ps.frame_of(img.name) + F0
    M = cam_w.cam_to_world(shot_f)
    rx, ry, rz = sc.matrix_to_euler_zxy(R_c2w_nuke)
    print(f"   {img.name} -> shot frame {shot_f}")
    print(f"   camera centre {np.round(centre, 4).tolist()} (solve units, unchanged by the flip)")
    print(f"   Nuke rotate (rot_order ZXY) = ({rx:.3f}, {ry:.3f}, {rz:.3f}) deg")
    check(np.allclose(M[:3, :3], R_c2w_nuke, atol=1e-9) and np.allclose(M[:3, 3], centre, atol=1e-9),
          "by-hand conversion matches what from_colmap wrote into the shot camera")
    check(np.allclose(sc.euler_zxy_to_matrix(rx, ry, rz), R_c2w_nuke, atol=1e-9),
          "ZXY Euler round trip: R = Ry @ Rx @ Rz gives the same matrix back")
    look = R_c2w_nuke @ np.array([0, 0, -1.0])
    print(f"   the camera looks along {np.round(look, 3).tolist()} in the solve's world (the world is not "
          f"gravity aligned until Measure & Scale)")

    # ---------------------------------------------------------------- 5. the tripod pan
    step(5, "the tripod pan, NO focal guess: first camera_move auto, then camera_move nodal")
    t = time.time()
    ra = ps.solve_folder(TMP / "pan", pan_names, TMP / "pan_auto", ps.SolveOptions(camera_move="auto"),
                         log=quiet).report
    fa = ra["camera"]["focal_px"]
    print(f"   auto:  panoramic pairs {ra['degeneracy'].get('panoramic_pair_frac', 0):.0%}, SfM lens check: "
          f"{ra['sfm_lens_check'] or 'plausible'} -> mode {ra['mode']}")
    print(f"          focal {fa:.2f} px vs truth {F_PX:.2f} ({100 * (fa / F_PX - 1):+.1f} %), k1 "
          f"{ra['camera']['params'][3]:+.4f}, {ra['mean_reproj_px']:.3f} px, in {time.time() - t:.1f} s")
    for w in ra["warnings"]:
        print(f"          WARNING {w[:105]}...")
    if ra["mode"] == "sfm":
        print("          a tiny error and a wrong lens: a low reprojection error does NOT mean a right camera")
        check(any("rotation-only" in w for w in ra["warnings"]),
              "auto kept SfM on a pure pan, so it must warn that the pairs look rotation-only")
    t = time.time()
    rp = ps.solve_folder(TMP / "pan", pan_names, TMP / "pan_solve", ps.SolveOptions(camera_move="nodal"),
                         log=quiet).report
    fp = rp["camera"]["focal_px"]
    print(f"   nodal: {rp['registered']}/{rp['of']} frames, rms {rp['mean_reproj_px']:.3f} px, focal {fp:.2f} px vs "
          f"truth {F_PX:.2f} ({100 * (fp / F_PX - 1):+.2f} %), k1 {rp['camera']['params'][3]:+.4f} (truth {K1}), "
          f"in {time.time() - t:.1f} s")
    check(rp["mode"] == "nodal", "camera_move nodal runs the rotation-only solve")
    check(abs(fp / F_PX - 1) < 0.02, "the nodal solve finds the focal with no guess (a pure rotation pins it)")
    check(ra["mode"] == "nodal" or abs(fp / F_PX - 1) < abs(fa / F_PX - 1),
          "on a tripod pan the nodal focal beats the free SfM focal")

    # ---------------------------------------------------------------- 6. what the next nodes will say
    step(6, "the checks every camera-taking node runs (camera_side/camera_checks.py)")
    pan_cam = solved(ps.SolveResult(TMP / "pan_solve" / "sparse" / "0", [], (0, PAN_FRAMES - 1)), "pan_solve")
    for label, c in (("walk", cam_w), ("pan", pan_cam)):
        msg = cc.require_metric(c, "exercise", "Measure widgets")
        print(f"   {label}: nodal={cc.is_nodal(c)}, lens={cc.has_lens(c)}")
        print(f"     -> {msg}")
    check(not cc.is_nodal(cam_w) and cc.is_nodal(pan_cam),
          "is_nodal tells the walk from the pan")

    print(f"\ntotal {time.time() - t_all:.1f} s")
    print("PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
