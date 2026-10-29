"""Lens + readout from the checker-grid shoot. Thin caller of camera_side/chart_calibrate.py, the same code as the
Chart Calibrate node (catalogue 49). Two steps, one clip each:

  lens     static holds of the chart (take A). Every Nth frame is detected together with the next frame; a view is
           kept only if the chart barely moved between them (< --still-px), so rolling shutter can't bend it.
           Writes <out>/<clip>_lens.json: K at plate size, k1 k2, focal in mm on the sensor preset, rms.
  readout  fast pans / tilts with the whole chart in view (takes B-E). Every frame is detected, labelled, and
           fitted with per-frame poses plus one readout. Writes <out>/<clip>_readout.json. A take with no motion
           comes back "not measurable" instead of a number.

Frames come through the plate reader (range from the stream), luma only, stretched per frame (the chart is black
and white; the log curve doesn't matter for corners). The plate is only read. Each JSON gets a .sidecar.json.

Run: <V141 python> tools\\grid_rs_shoot.py lens    GRID_A.MOV [--every 12] [--first N --last N]
     <V141 python> tools\\grid_rs_shoot.py readout GRID_B.MOV --lens <out>\\GRID_A_lens.json [--first N --last N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import chart_calibrate as cc  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lens", "readout"])
    ap.add_argument("clip", help="MOV path")
    ap.add_argument("--lens", help="lens json from the lens step (readout step)")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None, help="exclusive")
    ap.add_argument("--every", type=int, default=12, help="lens step: sample every N frames")
    ap.add_argument("--still-px", type=float, default=1.0, help="lens step: max chart motion to the next frame")
    ap.add_argument("--preset", default=cc.DEFAULT_PRESET, help="sensor preset (focal in mm)")
    ap.add_argument("--focal-guess-mm", type=float, default=16.47, help="lens step: starting focal")
    ap.add_argument("--out", default="", help="output folder (default atlas-vfx-work/grid)")
    a = ap.parse_args()
    if a.step == "readout" and not a.lens:
        raise SystemExit("readout step needs --lens <clip>_lens.json")
    ref = cc.chart_ref(a.clip, a.first, None if a.last is None else a.last - 1)
    out = pl.resolve_out_dir(a.out, cc.default_out_dir(), plate_path=ref["path"])
    name = pl.safe_name(Path(ref["path"]).stem)
    log(Path(ref["path"]).name, cc.plate_info(ref))
    seen = {"n": 0}

    def progress(f, d):
        seen["n"] += 1
        if seen["n"] % 24 == 0:
            log(f"frame {f}: {'none' if d is None else d['shape']}")

    args = vars(a)
    if a.step == "lens":
        res = cc.calibrate_lens(ref, preset=a.preset, every=a.every, still_px=a.still_px,
                                focal_guess_mm=a.focal_guess_mm, progress=progress)
        log(f"{res['n_views']} still views of {res['n_sampled']} sampled", res["views"])
        for w in res["warnings"]:
            log("warning:", w)
        log(f"focal {res['focal_mm']:.3f} mm ({res['focal_px']:.1f} px), k1 {res['k1']:+.5f} k2 {res['k2']:+.5f}, "
            f"pp offset {res['principal_point_offset_px'][0]:+.1f}, {res['principal_point_offset_px'][1]:+.1f} px, "
            f"rms {res['rms_px']:.3f} px")
    else:
        res = cc.measure_readout(ref, cc.read_lens(a.lens), progress=progress)
        log(f"chart in {res['frames_detected']} frames ({res['frames_full_chart']} with the whole chart)")
        ro = f"{res['readout_ms']:.2f} +- {res['sigma_ms']:.2f} ms" if res["measurable"] else \
            res["not_measurable_reason"]
        log(f"readout {ro} | rms RS {res['rms_px_rs']:.3f} vs GS {res['rms_px_gs']:.3f} px | turn "
            f"{res['rot_deg_per_frame_median']:.2f} deg/frame | {res['n_frames']} frames, {res['n_corners']} corners")
    log("wrote", cc.write_result(out, name, a.step, ref, args, res, node="tools/grid_rs_shoot.py"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
