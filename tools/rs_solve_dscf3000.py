"""RS-aware re-solve of DSCF3000 from the pycolmap f16 solve (camera_side/rs_solve.py).

Runs (each writes a report; the fixed one writes cameras):
  free   readout free, start 0.3 frames: does the plate itself say ~30.9 ms? (cross-check, --free)
  fixed  readout held at the spec 30.9 ms (X-H2 4K HQ): the camera to use with Sensor RS Maps at 30.9 ms
Writes to atlas-vfx-work/DSCF3000/camera/:
  DSCF3000_f16_rs.shot_camera.json        1001-1297, poses at the centre-row time, readout in extensions
  DSCF3000_S2S3_rs.shot_camera.json       1130-1297 (the .abc range; same world as the .abc and f16_GUESS)
  DSCF3000_f16_rs.points.npz              solved points + their solve ids, for Readout/RS checks
Checks printed: reprojection before/after, the solved readout, how far points and poses moved, and that the deck
points stay planar at y = 0 (an RS solve can bend structure: Albl et al., degeneracies of RS bundle adjustment).

Run: <V141 python> tools\\rs_solve_dscf3000.py [--free] [--max-points N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import measure, rs_solve  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

WORK = Path(r"C:\Users\miike\atlas-vfx-work\DSCF3000")
SRC = WORK / "camera" / "DSCF3000_f16_GUESS_stairs.shot_camera.json"
READOUT_MS = 30.9
DECK_X, DECK_Z = (-3.5, 2.5), (-9.3, -6.0)    # deck boards in the .abc / f16 world (doormat at (-1.76, 0, -9.02))


def deck_stats(xyz):
    m = (xyz[:, 0] > DECK_X[0]) & (xyz[:, 0] < DECK_X[1]) & (xyz[:, 2] > DECK_Z[0]) & (xyz[:, 2] < DECK_Z[1]) \
        & (np.abs(xyz[:, 1]) < 0.15)
    P = xyz[m]
    if len(P) < 20:
        return {"points": int(len(P))}
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    n = Vt[2] * np.sign(Vt[2][1])
    d = (P - c) @ n
    return {"points": int(len(P)), "y_median_m": float(np.median(P[:, 1])),
            "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[1]))))),
            "flatness_mm_mad": float(1000 * np.median(np.abs(d - np.median(d))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", action="store_true", help="solve the readout too (cross-check)")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=100)
    a = ap.parse_args()

    cam = sc.load(SRC)
    pts, note = measure.points_for_camera(cam)
    print("input:", cam.name, note)
    fps = float(cam.data["shot"]["fps"])
    r_spec = READOUT_MS / 1000 * fps
    res = rs_solve.solve(cam, pts["xyz"], pts["observations"], readout_frames=0.3 if a.free else r_spec,
                         solve_readout=a.free, max_points=a.max_points, max_iter=a.max_iter, verbose=1)
    rep = res["report"]
    rep["deck_input"] = deck_stats(pts["xyz"][res["point_ids"]])
    rep["deck_solved"] = deck_stats(res["xyz"])
    tag = "free" if a.free else "fixed"
    out = WORK / "camera"
    (out / f"DSCF3000_f16_rs_{tag}.report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    if a.free:
        return 0

    new = rs_solve.to_shot_camera(cam, res, name="DSCF3000_f16_rs")
    new.data["plate"]["colorspace"] = "F-Log F-Gamut"        # the f16 import predates the F-Log fix
    sc.save(new, out / "DSCF3000_f16_rs.shot_camera.json")
    s2s3 = cio.crop_frames(new, (1130, 1297))
    s2s3.data["shot"]["name"] = "DSCF3000_S2S3_rs"
    sc.save(s2s3, out / "DSCF3000_S2S3_rs.shot_camera.json")
    np.savez_compressed(out / "DSCF3000_f16_rs.points.npz", xyz=res["xyz"], point_ids=pts["ids"][res["point_ids"]])
    print("wrote", out / "DSCF3000_f16_rs.shot_camera.json", "and DSCF3000_S2S3_rs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
