"""Lens + readout from a filmed checker grid (camera_side/grid_rs.py) against ground truth.

Renders the 96 x 48 in chart (32 x 16 squares of 76.2 mm) through a known lens (focal, k1, k2) and a rolling
shutter, row by row: each output (sensor, distorted) pixel is undistorted to a ray at its own row time, the pose
at that time comes from a continuous pan (not the fit's slerp), and the ray hits the chart. Then:
  1. calibrate from 10 static views: focal within 0.3 %, k1 within 0.01, k2 within 0.02, rms < 0.3 px
  2. a pan at 30.9 ms readout: fitted readout within 3 %, and the RS fit beats the GS fit (r = 0)
  3. the same pan at 15.3 ms: within 3 % (the sub-sampled 4K value, so the fit tells the modes apart)
  4. a locked-off (no motion) shot: the readout is not measurable and the sigma must say so (> 10 ms)
  5. Chart Calibrate node (catalogue 49, Phase 1b S4): the same renders written as lossless HEVC takes (10-bit full
     range, like the X-H2) and run through the node: lens as in 1, readout within 1 % on both pans, the locked-off
     take reported not measurable (0 on the wire), bad names and paths refused, tools/grid_rs_shoot.py agreeing
Half-res frames (2048 x 1080) to keep it quick; readout is a fraction of the frame, so resolution doesn't matter.
Run: <V141 python> tests/test_grid_rs.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import grid_rs, rs_time  # noqa: E402

W, H, FPS = 2048, 1080, 24.0
F_PX = 16.47 / 23.5 * W                     # 1435 px: XF16-55 at 16 mm on the DCI 4K sensor area
K = np.array([[F_PX, 0, W / 2 + 6.0], [0, F_PX, H / 2 - 4.0], [0, 0, 1]])
DIST = np.array([-0.03, 0.012, 0, 0, 0])
SQ = grid_rs.CHART["square_m"]
CW, CH = grid_rs.CHART["cols"] * SQ, grid_rs.CHART["rows"] * SQ
CENTRE = np.array([15 * SQ, 7 * SQ, 0.0])              # middle inner corner (chart Y down, inner corners from 0)


def undistort_norm(u, v):
    """Distorted pixels -> normalised pinhole coords (fixed-point inverse of k1, k2)."""
    xd, yd = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(12):
        r2 = x * x + y * y
        s = 1 + DIST[0] * r2 + DIST[1] * r2 * r2
        x, y = xd / s, yd / s
    return x, y


def render(pose_at, frame, readout_frames, ss=2):
    """pose_at(t) -> (R world->cam, C). Chart white 0.9 / black 0.05 on a 0.45 wall; ss x ss supersampling."""
    acc = np.zeros((H, W), np.float32)
    off = (np.arange(ss) + 0.5) / ss - 0.5
    rows = np.arange(H, dtype=float)
    for oy in off:
        for ox in off:
            v = rows[:, None] + oy + np.zeros((1, W))                  # OpenCV: pixel centres on integers
            u = np.arange(W, dtype=float)[None, :] + ox + np.zeros((H, 1))
            xn, yn = undistort_norm(u, v)
            img = np.full((H, W), 0.45, np.float32)
            t_rows = rs_time.row_time(frame, rows, H, readout_frames, pixels="opencv")
            for band in np.array_split(np.arange(H), 54):          # pose per 20-row band (sub-pixel at 2 deg/fr)
                R, C = pose_at(t_rows[band].mean())
                d = np.stack([xn[band], yn[band], np.ones_like(xn[band])], -1) @ R      # world dirs (R^T d)
                s = -C[2] / d[..., 2]
                X = C[0] + s * d[..., 0] + SQ            # outer-edge coords: inner corner (i, j) at world (i, j) * SQ
                Y = C[1] + s * d[..., 1] + SQ
                ins = (s > 0) & (X >= 0) & (X < CW) & (Y >= 0) & (Y < CH)
                cell = (np.floor(X / SQ) + np.floor(Y / SQ)) % 2
                b = img[band]
                b[ins] = np.where(cell[ins] == 0, 0.9, 0.05)
                img[band] = b
            acc += img
    acc /= ss * ss
    noise = np.random.default_rng(frame).normal(0, 0.01, acc.shape)
    return np.clip((acc + noise) * 255, 0, 255).astype(np.uint8)


def look(yaw_deg, pitch_deg, roll_deg=0.0, dist=3.0, offset=(0.0, 0.0)):
    """Camera dist metres in front of the chart centre (chart at Z = 0, camera at -Z, looking +Z)."""
    Rc2w = Rotation.from_euler("YXZ", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    C = CENTRE + np.array([offset[0], offset[1], -dist])
    return Rc2w.T, C


def pan(amp_deg=9.0, period=24.0):
    def pose_at(t):
        yaw = amp_deg * np.sin(2 * np.pi * t / period)
        pitch = 1.5 * np.sin(2 * np.pi * t / 17.0)
        return look(yaw, pitch, 0.4 * np.sin(t / 5.0))
    return pose_at


def write_mov(path: Path, imgs, hold: int = 1) -> Path:
    """8-bit renders -> 10-bit FULL-range luma codes (x 1023/255), neutral chroma, lossless HEVC Rext 4:2:2 (the
    X-H2's own format), each frame repeated `hold` times. As test_plate.py: no -color_range anywhere (ffmpeg would
    insert a range conversion, NOTES 2026-10-19); the range goes in as HEVC metadata only."""
    c = np.full((H, W // 2), 512, np.uint16).tobytes()
    raw = b"".join((np.round(img.astype(np.float32) * (1023 / 255)).astype(np.uint16).tobytes() + c + c) * hold
                   for img in imgs)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-s", f"{W}x{H}",
                    "-r", "24", "-i", "-", "-c:v", "libx265", "-x265-params", "lossless=1:log-level=error",
                    "-tag:v", "hvc1", "-bsf:v", "hevc_metadata=video_full_range_flag=1:matrix_coefficients=1",
                    str(path)], input=raw, check=True)
    return path


def check_node(static_imgs, takes, cal, fails):
    """Chart Calibrate (catalogue 49) on the renders written as MOVs: lens take (each static view held 2 frames),
    then readout on each pan and the locked-off take with the lens JSON the lens run wrote. Truth tolerances as above,
    readout within 1 %, locked off reported as not measurable with 0 on the wire, bad paths and names refused, the
    readout output type-matched to Sensor RS Maps' readout_chart_ms and ranked first by the preset rule, and
    tools/grid_rs_shoot.py (a thin caller of the same code) giving the same lens."""
    import os
    import tempfile

    from atlas_vfx.camera_side import plate as pl
    from atlas_vfx.camera_side.node_chart_calibrate import AtlasVFXChartCalibrate
    from atlas_vfx.sensor import jobs
    from atlas_vfx.sensor.node_rs_maps import AtlasVFXSensorRSMaps

    refused = []

    def bad(what, fn):
        try:
            fn()
        except (ValueError, pl.PlateError) as e:
            refused.append(f"{what}: {str(e)[:90]}")
            return
        fails.append(f"node: {what} was accepted")

    node = AtlasVFXChartCalibrate()
    names = node.RETURN_NAMES
    preset = "fujifilm_xh2_dci4k_24p_dis_off"
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        os.environ["ATLAS_VFX_OUT_ROOTS"] = str(td)
        t1 = time.time()
        lens_mov = write_mov(td / "plates" / "GRID_A.mov", static_imgs, hold=2)
        movs = {name: write_mov(td / "plates" / f"GRID_{k}.mov", imgs) for k, (name, _, imgs) in
                zip("BCE", takes)}
        print(f"node: wrote {1 + len(movs)} HEVC takes in {time.time() - t1:.0f}s")
        out = td / "grid"

        # lens (take A)
        r = dict(zip(names, node.run(str(lens_mov), "lens", preset, "", 0, -1, 2, 1.0, 13.0, "", str(out))["result"]))
        f_mm = F_PX * 23.5 / W
        fe = r["focal_mm"] / f_mm - 1
        pp_err = np.hypot(r["pp_offset_x_px"] - (K[0, 2] - W / 2), r["pp_offset_y_px"] - (K[1, 2] - H / 2))
        lens_doc = json.loads(Path(r["lens_path"]).read_text(encoding="utf-8"))
        print(f"node lens: focal {r['focal_mm']:.4f} mm (truth {f_mm:.4f}, {100 * fe:+.3f} %), k1 {r['k1']:+.4f} "
              f"k2 {r['k2']:+.4f}, pp offset {r['pp_offset_x_px']:+.2f}, {r['pp_offset_y_px']:+.2f} px (truth +6, -4), "
              f"rms {lens_doc['rms_px']:.3f} px, {lens_doc['n_views']} still views; in-memory focal "
              f"{cal['K'][0, 0]:.2f} px vs node {lens_doc['focal_px']:.2f} px")
        if abs(fe) > 0.003:
            fails.append(f"node lens: focal off by {100 * fe:.2f} %")
        if abs(r["k1"] - DIST[0]) > 0.01 or abs(r["k2"] - DIST[1]) > 0.02:
            fails.append(f"node lens: distortion {r['k1']:.4f} {r['k2']:.4f} vs {DIST[:2]}")
        if lens_doc["rms_px"] > 0.3 or lens_doc["n_views"] != len(static_imgs) or pp_err > 2.0:
            fails.append(f"node lens: rms {lens_doc['rms_px']:.3f}, {lens_doc['n_views']} views, pp off {pp_err:.2f} px")
        if not Path(r["lens_path"]).with_suffix(".sidecar.json").is_file() or r["readout_path"] or r["measurable"]:
            fails.append("node lens: no sidecar, or readout outputs set in lens mode")

        # readout (takes B, C, E)
        chart_type = AtlasVFXSensorRSMaps.INPUT_TYPES()["optional"]["readout_chart_ms"][0]
        if node.RETURN_TYPES[names.index("readout_ms")] != chart_type:
            fails.append(f"node readout_ms is {node.RETURN_TYPES[names.index('readout_ms')]}, RS Maps wants {chart_type}")
        cam = {"sensor": {"preset": preset}, "extensions": {}}
        lens_json = r["lens_path"]                  # lens_path wired into lens_json, as in the workflow
        for (name, ms, _), mov in zip(takes, movs.values()):
            r = dict(zip(names, node.run(str(mov), "readout", preset, lens_json, 0, -1, 12, 1.0, 16.0, "",
                                         str(out))["result"]))
            doc = json.loads(Path(r["readout_path"]).read_text(encoding="utf-8"))
            print(f"node {name}: measurable {r['measurable']}, readout {r['readout_ms']:.3f} ms (fit "
                  f"{doc['readout_ms_fit']:.3f} +- {doc['sigma_ms'] if doc['sigma_ms'] is None else round(doc['sigma_ms'], 3)}"
                  f" ms, truth {ms}), rms RS {r['rms_rs_px']:.3f} / GS {r['rms_gs_px']:.3f} px, turn "
                  f"{doc['rot_deg_per_frame_median']:.2f} deg/frame")
            won = jobs.decide_readout(cam, 30.9, jobs.PRESET_RULE, chart_ms=r["readout_ms"])
            if name.startswith("pan"):
                if not r["measurable"] or abs(r["readout_ms"] / ms - 1) > 0.01:
                    fails.append(f"node {name}: readout {r['readout_ms']:.3f} ms not within 1 % of {ms}")
                if not r["rms_rs_px"] < 0.7 * r["rms_gs_px"]:
                    fails.append(f"node {name}: RS rms {r['rms_rs_px']:.3f} not clearly below GS {r['rms_gs_px']:.3f}")
                if won["winner"]["kind"] != "chart" or abs(won["readout_ms"] - r["readout_ms"]) > 1e-9:
                    fails.append(f"node {name}: RS Maps preset rule did not take the chart value: {won['readout_from']}")
            else:
                if r["measurable"] or r["readout_ms"] != 0.0 or doc["readout_ms"] is not None or \
                        "not measurable" not in (doc["not_measurable_reason"] or ""):
                    fails.append(f"node {name}: a locked-off take gave a readout ({r['readout_ms']}, {doc['readout_ms']})")
                if won["winner"]["kind"] != "spec":
                    fails.append(f"node {name}: 0 on the wire should leave RS Maps on the preset spec, got "
                                 f"{won['readout_from']}")

        # refused: names with a path, outputs outside the roots / in the repo / next to the take, bad inputs
        run = node.run
        bad("a name with a path in it", lambda: run(str(lens_mov), "lens", preset, "", 0, -1, 2, 1.0, 16.0,
                                                     "../escape", str(out)))
        bad("out_dir outside the work roots", lambda: run(str(lens_mov), "lens", preset, "", 0, -1, 2, 1.0, 16.0,
                                                           "", str(Path(tempfile.gettempdir()).parent / "elsewhere")))
        bad("out_dir in the repo", lambda: run(str(lens_mov), "lens", preset, "", 0, -1, 2, 1.0, 16.0, "",
                                               str(pl.REPO_ROOT / "workflows")))
        bad("out_dir next to the take", lambda: run(str(lens_mov), "lens", preset, "", 0, -1, 2, 1.0, 16.0, "",
                                                    str(lens_mov.parent)))
        bad("a missing take", lambda: run(str(td / "nope.mov"), "lens", preset, "", 0, -1, 2, 1.0, 16.0, "",
                                          str(out)))
        bad("readout without a lens JSON", lambda: run(str(lens_mov), "readout", preset, "", 0, -1, 2, 1.0, 16.0,
                                                       "", str(out)))
        junk = td / "junk.json"
        junk.write_text('{"K": "import os"}', encoding="utf-8")
        bad("a lens JSON that isn't one", lambda: run(str(lens_mov), "readout", preset, str(junk), 0, -1, 2, 1.0,
                                                      16.0, "", str(out)))
        bad("a lens take with no motion-free pairs (every frame its own view)",
            lambda: run(str(movs[takes[0][0]]), "lens", preset, "", 0, -1, 2, 1.0, 16.0, "pan_as_lens", str(out)))
        print(f"node refused {len(refused)} of 8 bad inputs:\n  " + "\n  ".join(refused))
        if list(td.glob("plates/*.json")) or (td.parent / "escape_lens.json").exists():
            fails.append("node wrote next to the take or outside out_dir")

        # the tool runs the same code: same lens from the same take
        tool = Path(__file__).resolve().parents[1] / "tools" / "grid_rs_shoot.py"
        p = subprocess.run([sys.executable, str(tool), "lens", str(lens_mov), "--every", "2", "--out",
                            str(td / "tool")], capture_output=True, text=True)
        tl = td / "tool" / "GRID_A_lens.json"
        if p.returncode or not tl.is_file():
            fails.append(f"tools/grid_rs_shoot.py lens failed: {(p.stdout + p.stderr)[-400:]}")
        else:
            tdoc = json.loads(tl.read_text(encoding="utf-8"))
            d = abs(tdoc["focal_mm"] - lens_doc["focal_mm"])
            print(f"tool lens: focal {tdoc['focal_mm']:.4f} mm (node {lens_doc['focal_mm']:.4f}, diff {d:.2e})")
            if d > 1e-3 or abs(tdoc["k1"] - lens_doc["k1"]) > 1e-4:
                fails.append(f"tool and node disagree on the lens: {tdoc['focal_mm']} vs {lens_doc['focal_mm']}")


def main() -> int:
    t0 = time.time()
    fails = []

    # 1. calibration from static views
    views, static_imgs, takes = [], [], []
    for yaw, pitch, roll, dist, off in [(0, 0, 0, 1.7, (0, 0)), (25, 0, 0, 2.0, (-0.4, 0)), (-25, 0, 0, 2.0, (0.4, 0)),
                                        (0, 20, 0, 2.0, (0, 0.2)), (0, -20, 0, 2.0, (0, -0.2)), (15, 15, 10, 2.2, (0, 0)),
                                        (-15, -10, -12, 2.2, (0, 0)), (8, 0, 0, 1.4, (0.5, 0.2)),
                                        (-8, 5, 0, 1.4, (-0.5, -0.2)), (30, -15, 5, 2.6, (-0.6, 0.1))]:
        R, C = look(yaw, pitch, roll, dist, off)
        img = render(lambda t, R=R, C=C: (R, C), 0, 0.0)
        static_imgs.append(img)
        d = grid_rs.detect(img)
        if d is None:
            fails.append(f"no chart detected in static view yaw {yaw} pitch {pitch}")
            continue
        views.append(d)
    cal = grid_rs.calibrate(views, (W, H), focal_guess_px=0.8 * F_PX)
    fe = cal["K"][0, 0] / F_PX - 1
    print(f"calibration: {cal['n_views']} views, rms {cal['rms_px']:.3f} px, focal {cal['K'][0, 0]:.1f} "
          f"(truth {F_PX:.1f}, {100 * fe:+.3f} %), pp {cal['K'][0, 2]:.1f}, {cal['K'][1, 2]:.1f}, "
          f"k1 {cal['dist'][0]:+.4f} (truth {DIST[0]}), k2 {cal['dist'][1]:+.4f} (truth {DIST[1]})  [{time.time() - t0:.0f}s]")
    if abs(fe) > 0.003:
        fails.append(f"focal off by {100 * fe:.2f} %")
    if abs(cal["dist"][0] - DIST[0]) > 0.01 or abs(cal["dist"][1] - DIST[1]) > 0.02:
        fails.append(f"distortion off: {cal['dist'][:2]} vs {DIST[:2]}")
    if cal["rms_px"] > 0.3:
        fails.append(f"calibration rms {cal['rms_px']:.2f} px")

    # 2-4. readout from a pan (and from a locked-off shot)
    for name, ms, pose_at, n in [("pan 30.9 ms", 30.9, pan(), 20), ("pan 15.3 ms", 15.3, pan(), 20),
                                 ("locked off 30.9 ms", 30.9, lambda t: look(0, 0), 8)]:
        r = ms / 1000 * FPS
        imgs = [render(pose_at, f, r) for f in range(n)]
        takes.append((name, ms, imgs))
        dets = {f: grid_rs.detect(img) for f, img in enumerate(imgs)}
        miss = [f for f, d in dets.items() if d is None]
        obs = grid_rs.index_sequence(dets, cal["K"], cal["dist"])
        rs = grid_rs.fit_readout(obs, cal["K"], H, fps=FPS)
        gs = grid_rs.fit_readout(obs, cal["K"], H, readout_start=0.0, solve_readout=False, fps=FPS)
        print(f"{name}: readout {rs['readout_ms']:.2f} +- {rs['sigma_ms']:.2f} ms (truth {ms}), rms RS "
              f"{rs['rms_px']:.3f} / GS {gs['rms_px']:.3f} px, {rs['n_frames']} frames ({len(miss)} undetected), "
              f"{rs['n_obs']} corners, turn {rs['rot_deg_per_frame_median']:.2f} deg/frame  [{time.time() - t0:.0f}s]")
        if name.startswith("pan"):
            if abs(rs["readout_ms"] / ms - 1) > 0.03:
                fails.append(f"{name}: readout {rs['readout_ms']:.2f} ms vs {ms}")
            if not rs["rms_px"] < 0.7 * gs["rms_px"]:
                fails.append(f"{name}: RS fit {rs['rms_px']:.3f} px not clearly better than GS {gs['rms_px']:.3f}")
        elif not rs["sigma_ms"] > 10:
            fails.append(f"{name}: sigma {rs['sigma_ms']:.2f} ms claims a readout from a shot with no motion")

    # 5. the same renders as takes on disk, through the Chart Calibrate node (and the tool, which runs its code)
    check_node(static_imgs, takes, cal, fails)
    print(f"node checks done  [{time.time() - t0:.0f}s]")

    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else f"{len(fails)} failure(s)", f"({time.time() - t0:.0f}s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
