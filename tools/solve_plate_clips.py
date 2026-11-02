"""Run the 3D camera tracker on real clips headless (the node classes, no ComfyUI server), and compare with the
cameras we already have.

  Plate Read -> Solve Mask (optional) -> Solve Plate -> report
  DSCF3000: also Solve Compare against Mike's NukeX camera (Camera1.json) and the pycolmap f16 solve.

Writes atlas-vfx-work/<clip>/solve/<name>/ (the node's own outputs) and atlas-vfx-work/tracker_runs/<tag>.json.
Plates are only read.

Run: <V141 python> tools\\solve_plate_clips.py DSCF3000:0-296 DSCF2977 DSCF2978 DSCF2968 [--no-mask] [--focal 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import compare  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.camera_side.node_camera_import import AtlasVFXCameraImport  # noqa: E402
from atlas_vfx.camera_side.node_plate_read import AtlasVFXPlateRead  # noqa: E402
from atlas_vfx.camera_side.node_solve_mask import AtlasVFXSolveMask  # noqa: E402
from atlas_vfx.camera_side.node_solve_plate import AtlasVFXSolvePlate  # noqa: E402

FOOTAGE = Path.home() / "Videos" / "atlas-vfx_footage"
REFS = {   # clip -> cameras to compare with (kind, path)
    "DSCF3000": [("nuke_json", Path.home() / "howling-yowies" / "nuke" / "export" / "Camera1.json"),
                 ("colmap", Path.home() / "ComfyUI_V135" / "ComfyUI" / "output" / "_pick" / "vfx_probe" /
                  "solve_s1s3_f16")],
}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def run_clip(spec: str, mask: bool, focal: float) -> dict:
    clip, _, rng = spec.partition(":")
    first, last = (int(v) for v in rng.split("-")) if rng else (0, -1)
    ref = AtlasVFXPlateRead().run(str(FOOTAGE / f"{clip}.MOV"), first, last, "", "auto", "auto", 1001, 256, 1000)[0]
    log(clip, f"frames {ref['first']}-{ref['last']}", ref["colorspace"])
    out = {"clip": clip, "frames": [ref["first"], ref["last"]], "masked": mask}
    masks = None
    if mask:
        t = time.time()
        m = AtlasVFXSolveMask().run(ref, "person. car. bicycle. dog. tree. palm tree. bush. water. wave. flag.", 16,
                                    0.35, 0.25, 6, 1024)
        masks = m["result"][0]
        rep = json.loads(m["result"][2])
        out["mask"] = {"coverage": rep["coverage"], "seconds": round(time.time() - t, 1),
                       "boxes": sum(w["boxes"] for w in rep["windows"])}
        log("mask", out["mask"])
    t = time.time()
    name = f"{clip}_{ref['first']:04d}-{ref['last']:04d}_tracker" + ("" if mask else "_nomask")
    res = AtlasVFXSolvePlate().run(ref, "auto", 23.5, 12.39, focal, True, True, "k1 k2 (RADIAL)", 1024, 8192, 15,
                                   2.0, name, "", masks=masks, mask_dilate_px=4)
    cam = res["result"][0]
    rep = json.loads(res["result"][2])
    out["solve"] = {k: rep[k] for k in ("mode", "registered", "of", "missing", "longest_run", "points3D",
                                         "mean_reproj_px", "camera", "degeneracy", "warnings", "seconds")}
    out["solve"]["wall_seconds"] = round(time.time() - t, 1)
    out["camera_path"] = res["result"][1]
    log("solve", res["ui"]["text"])
    out["compare"] = []
    for kind, p in REFS.get(clip, []):
        if not p.exists():
            out["compare"].append({"ref": str(p), "missing": True})
            continue
        b = AtlasVFXCameraImport().run(str(p), kind, ref, "auto", 23.5, 12.39, 1.0, "ZXY", 0.0, True,
                                       f"{clip}_ref_{kind}", "")[0]
        try:
            c = compare.compare(cam, b)
            out["compare"].append({"ref": str(p), "kind": kind, "summary": compare.summary_line(c),
                                   "residual_pct_of_path": c["residual_pct_of_path"],
                                   "view_angle_deg": c["view_angle_deg"], "focal_mm": c["focal_mm"]})
            log("compare", compare.summary_line(c))
        except ValueError as e:
            out["compare"].append({"ref": str(p), "kind": kind, "error": str(e)})
            log("compare failed", e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="CLIP or CLIP:first-last (plate frame indices)")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--focal", type=float, default=16.0, help="focal guess mm (0 = none); the lens is solved")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M"))
    a = ap.parse_args()
    runs = []
    for spec in a.clips:
        try:
            runs.append(run_clip(spec, not a.no_mask, a.focal))
        except Exception as e:  # noqa: BLE001 - one bad clip must not stop the batch
            log(spec, "FAILED", repr(e))
            runs.append({"clip": spec, "error": repr(e)})
    out = pl.resolve_out_dir("", pl.WORK_ROOT / "tracker_runs") / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=1, default=str), encoding="utf-8")
    log("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
