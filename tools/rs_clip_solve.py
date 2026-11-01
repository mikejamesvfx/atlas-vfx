"""Rolling-shutter test on any X-H2 clip: tracking proxies -> pycolmap GS solve -> shot camera -> RS solve with the
readout free -> readout profile (how deep the valley is). Same recipe as DSCF3000 (track_proxy.py, the f16 solve,
tools/rs_solve_dscf3000.py, tools/rs_readout_profile.py), wrapped so a new clip is one command.

  proxies   1024 px log luma + CLAHE, decoded with the stream's range (these clips are FULL range)
  GS solve  camera_side/plate_solve.py (Solve Plate's engine): SIMPLE_RADIAL, single camera, focal starts at 717.7 px @1024 (16.47 mm on 23.5 mm), refined unless
            --fix-focal; distortion pinned at 0 and principal point fixed (XF video is corrected in camera)
  camera    the longest run of registered frames without gaps (rs_solve needs contiguous frames), frame 1001 = 0
  RS        free readout from 0.3 frames (12.5 ms); then the profile re-solves at fixed readouts

Writes atlas-vfx-work/rs_clips/<clip>/: track/, solve/ (database + sparse), <clip>_gs.shot_camera.json,
<clip>_rs_free.report.json, <clip>_rs_free.shot_camera.json, <clip>_gs.readout_profile.json, summary.json.
The plate is only read.

Run: <V141 python> tools\\rs_clip_solve.py DSCF2973 [--fix-focal] [--first N --last N] [--max-points 8000]
     [--ms 0,10,15.3,20,26.6,30.9,40] [--skip-profile]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import measure, rs_solve, sensors  # noqa: E402
from atlas_vfx.camera_side import plate_solve as ps  # noqa: E402
from atlas_vfx.camera_side.plate import fuji_recording_mode  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

FOOTAGE = Path(r"C:\Users\miike\Videos\atlas-vfx_footage")
WORK = Path(r"C:\Users\miike\atlas-vfx-work\rs_clips")
F16_PX_1024 = 717.7
PW, PH, PROXY_W = 4096, 2160, 1024
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def probe(mov: Path) -> dict:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate,nb_frames,color_range:stream_tags=timecode",
                                   "-of", "json", str(mov)])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": float(num) / float(den),
            "frames": int(s["nb_frames"]), "range": s.get("color_range", "unknown"),
            "timecode": s.get("tags", {}).get("timecode")}


def proxies(mov: Path, out: Path, info: dict, first: int, last: int) -> list[str]:
    """plate_NNNN.png for frames first..last-1 (log luma, per-frame stretch, CLAHE), as prototypes/track_proxy.py
    but with the stream's own range."""
    import cv2

    names = [f"plate_{i:04d}.png" for i in range(first, last)]
    if all((out / n).is_file() for n in names):
        return names
    out.mkdir(parents=True, exist_ok=True)
    W = PROXY_W
    H = int(round(W * info["height"] / info["width"] / 2)) * 2
    rng = {"pc": "pc", "jpeg": "pc", "tv": "tv", "mpeg": "tv"}.get(info["range"], "pc")
    cmd = ["ffmpeg", "-v", "error", "-i", str(mov), "-vf",
           f"select='between(n\\,{first}\\,{last - 1})',scale={W}:{H}:in_range={rng}:out_range=pc:flags=area,"
           "format=gray16le", "-vsync", "0", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    n, i = W * H * 2, first
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        y = np.frombuffer(buf, np.uint16).reshape(H, W)
        lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
        y = np.clip((y.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        cv2.imwrite(str(out / f"plate_{i:04d}.png"), (clahe.apply((y * 65535).astype(np.uint16)) >> 8).astype(np.uint8))
        i += 1
    p.wait()
    log(f"{i - first} proxies ({W}x{H}, range {rng})")
    return names


def gs_solve(track: Path, out: Path, names: list[str], fix_focal: bool):
    """The GS solve, through camera_side/plate_solve.py (Solve Plate's engine) with this tool's recipe kept as it
    was for the RS evidence: SIMPLE_RADIAL with k1 held at 0, focal seeded at 16.47 mm, no gap bridging, no
    motion rejection, no nodal fallback."""
    opts = ps.SolveOptions(camera_model="SIMPLE_RADIAL", focal_px=F16_PX_1024, refine_focal=not fix_focal,
                           refine_distortion=False, max_image_size=PROXY_W, bridge_gaps=False, reject_rounds=0,
                           nodal_fallback=False, min_points=500)
    try:
        res = ps.solve_folder(track, names, out, opts, log=log)
    except ps.SolveError as e:
        raise SystemExit(f"GS solve too weak for an RS test ({e}); a pan with no baseline needs --fix-focal")
    r = res.report
    rep = {"registered": r["registered"], "of": r["of"], "points3D": r["points3D"],
           "mean_reproj_px_1024": r["mean_reproj_px"], "focal_px_1024": r["camera"]["focal_px"],
           "focal_mm": r["camera"]["focal_px"] * 23.5 / r["image_size"][0], "passes": r["passes"]}
    log("GS", json.dumps(rep, default=str))
    return res.model_dir, res.frames, rep


longest_run = ps.longest_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None, help="exclusive")
    ap.add_argument("--fix-focal", action="store_true")
    ap.add_argument("--max-points", type=int, default=None, help="free RS solve (default all points)")
    ap.add_argument("--profile-points", type=int, default=8000)
    ap.add_argument("--ms", default="0,10,15.3,20,26.6,30.9,40")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the work folder (e.g. a frame range)")
    a = ap.parse_args()

    mov = FOOTAGE / f"{a.clip}.MOV"
    info = probe(mov)
    info["recording_mode"] = fuji_recording_mode(mov)
    last = min(a.last or info["frames"], info["frames"])
    wd = WORK / (a.clip + a.tag)
    wd.mkdir(parents=True, exist_ok=True)
    log(a.clip, json.dumps(info), "frames", a.first, last - 1)

    names = proxies(mov, wd / "track", info, a.first, last)
    model, frames, gs_rep = gs_solve(wd / "track", wd / "solve", names, a.fix_focal)
    run = longest_run(frames)
    log("longest contiguous run", run, f"({run[1] - run[0] + 1} of {len(frames)} registered)")

    plate = cio.plate_block(None, PW, PH)
    plate.update({"path": str(mov), "colorspace": "F-Log F-Gamut" if info["recording_mode"] == "F-Log" else
                  "F-Log2 F-Gamut", "range": "full" if info["range"] == "pc" else "legal"})
    if info["timecode"]:
        plate["timecode"] = info["timecode"]
    cam = cio.from_colmap(model, plate=plate, sensor=sensors.sensor_block("fujifilm_xh2_dci4k_24p_dis_off"),
                          frame_offset=1001, name=f"{a.clip}_gs", frames=(run[0] + 1001, run[1] + 1001),
                          fps=info["fps"])
    gs_path = wd / f"{a.clip}_gs.shot_camera.json"
    sc.save(cam, gs_path)
    cam = sc.load(gs_path)
    pts, note = measure.points_for_camera(cam, max_reproj_px=6.0)    # 1.5 px at proxy size
    log("points:", note)

    # the same code as the RS Solve node: readout free from 0.3 frames (12.5 ms at 24p), then the profile
    out = rs_solve.solve_shot(cam, pts, readout_ms=0.3 / info["fps"] * 1000.0, solve_readout=True,
                              max_points=a.max_points, name=f"{a.clip}_rs_free", verbose=1,
                              profile_ms=() if a.skip_profile else [float(v) for v in a.ms.split(",")],
                              profile_points=a.profile_points, log=lambda row: log("profile", json.dumps(row)))
    rep, rows = out["report"], out["profile"]
    (wd / f"{a.clip}_rs_free.report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    sc.save(out["camera"], wd / f"{a.clip}_rs_free.shot_camera.json")
    log(f"free readout {rep['readout_ms']:.2f} ms | reproj GS {rep['reproj_input_gs']} -> RS {rep['reproj_solved']}")
    if rows:
        (wd / f"{a.clip}_gs.readout_profile.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        log("profile:", ", ".join(f"{r['readout_ms']:g} ms +{r['cost_above_best_pct']:.2f} %" for r in rows))

    summary = {"clip": a.clip, "plate": info, "gs": gs_rep, "run": [run[0], run[1]],
               "free_readout_ms": rep["readout_ms"], "reproj_gs": rep["reproj_input_gs"],
               "reproj_gs_plus_free": rep["reproj_input_with_rs"], "reproj_rs": rep["reproj_solved"],
               "pose_change": rep["pose_change"], "align_to_input": rep["align_to_input"], "rs_message": rep["message"],
               "profile": rows, "seconds": round(time.time() - T0)}
    (wd / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log("wrote", wd / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
