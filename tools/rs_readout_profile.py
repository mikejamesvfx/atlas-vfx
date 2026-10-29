"""How well does a shot pin down the readout? Re-solve (RS bundle adjustment) with the readout held at each value
of a sweep and print the converged robust cost. A deep, narrow valley = the shot measures it; flat = it does not.

Run: <V141 python> tools\\rs_readout_profile.py [shot_camera.json] [--max-points N] [--ms 0,10,20,30.9,40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import measure, rs_solve  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

DEFAULT = r"C:\Users\miike\atlas-vfx-work\DSCF3000\camera\DSCF3000_f16_GUESS_stairs.shot_camera.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", nargs="?", default=DEFAULT)
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--ms", default="0,10,20,26.6,30.9,40")
    a = ap.parse_args()
    cam = sc.load(a.camera)
    pts, note = measure.points_for_camera(cam)
    fps = float(cam.data["shot"]["fps"])
    rows = []
    for ms in [float(v) for v in a.ms.split(",")]:
        res = rs_solve.solve(cam, pts["xyz"], pts["observations"], readout_frames=ms / 1000 * fps,
                             max_points=a.max_points)
        rep = res["report"]
        rows.append({"readout_ms": ms, "cost": rep["cost"], "median_px": rep["reproj_solved"]["median_px"],
                     "rms_px": rep["reproj_solved"]["rms_px"], "iters": rep["nfev"], "seconds": rep["seconds"]})
        print(json.dumps(rows[-1]), flush=True)
    c0 = min(r["cost"] for r in rows)
    print("relative cost above the best:", ", ".join(f"{r['readout_ms']:g} ms +{100 * (r['cost'] / c0 - 1):.2f} %"
                                                      for r in rows))
    out = Path(a.camera).with_suffix("").with_suffix(".readout_profile.json")
    out.write_text(json.dumps({"camera": cam.name, "points": note, "rows": rows}, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
