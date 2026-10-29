"""Phase 1b S5, regression half: the 2026-10-20 deck test (tools/rs_deck_proof.py, unchanged) re-run with the
S3 node's camera, DSCF3000_S2S3_rs_free (readout solved free, 26.46 ms).

Makes Sensor RS maps for that camera at its own readout from the MoGe depth the earlier runs used
(depth/DSCF3000_f16_GUESS_stairs, same world), 2048 wide, frames 1150-1215 (sensor/DSCF3000_S2S3_rs_free_rs), then
runs rs_deck_proof.py at refs 1160-1200 and prints RS on vs off (sensor vs control) per ref, the median of the refs
and the median over all pairs, next to the earlier 30.9 ms held camera's CSVs.

Run: <V141 python> tools\\rs_deck_regression.py [--skip-maps]
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.sensor import jobs  # noqa: E402

WORK = pl.WORK_ROOT / "DSCF3000"
CAM = WORK / "camera" / "DSCF3000_S2S3_rs_free.shot_camera.json"
DEPTH = WORK / "depth" / "DSCF3000_f16_GUESS_stairs"
MAPS = WORK / "sensor" / "DSCF3000_S2S3_rs_free_rs"
REFS = (1160, 1170, 1180, 1190, 1200)
EARLIER = "DSCF3000_S2S3_rs_rs_DSCF3000_S2S3_rs"          # 30.9 ms held camera, 2026-10-21


def table(tag):
    per, allv = {}, []
    for r in REFS:
        p = WORK / "review" / f"rs_deck_proof_ref{r}_{tag}.csv"
        if not p.is_file():
            continue
        v = [float(x["better_vs_control_pct"]) for x in csv.DictReader(open(p))]
        per[r] = (float(np.median(v)), len(v), int(sum(x > 0 for x in v)))
        allv += v
    return per, (float(np.median([m for m, _, _ in per.values()])) if per else None,
                 float(np.median(allv)) if allv else None, len(allv), int(sum(x > 0 for x in allv)))


def main() -> int:
    cam = sc.load(CAM)
    readout = cam.data["extensions"]["atlas_vfx_rs_solve"]["readout_ms"]
    if "--skip-maps" not in sys.argv:
        rep = jobs.make_rs_maps(cam, DEPTH, MAPS, readout, readout_from="RS Solve free (S3 node)",
                                frames=list(range(1150, 1216)), scale=0.5, direction="both")
        print("maps", rep["count"], "frames, readout", readout, "ms, skew median", rep["skew_px_per_row_median"],
              "roundtrip", rep["roundtrip_median_px"], flush=True)
    for r in REFS:
        subprocess.run([sys.executable, str(ROOT / "tools" / "rs_deck_proof.py"), str(r), str(MAPS), str(CAM)],
                       check=True)
    out = {}
    for name, tag in (("rs_free 26.46 ms (S3 node)", f"{MAPS.name}_{cam.name}"), ("rs held 30.9 ms (10-21)", EARLIER)):
        per, (mref, mall, n, nb) = table(tag)
        out[name] = {"per_ref": per, "median_of_refs_pct": mref, "median_all_pairs_pct": mall, "pairs": n,
                     "on_better_pairs": nb}
        print(name, json.dumps(out[name]))
    (WORK / "review" / "s5_deck_regression.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
