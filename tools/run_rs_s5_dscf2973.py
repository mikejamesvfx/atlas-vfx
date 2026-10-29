"""Phase 1b S5, fast-pan half: the DSCF2973 proof runs and the headline table, as run on 2026-10-29.

Camera: the RS Solve node's free solve (tools/run_rs_solve_dscf2973.py, 27.73 ms, 151 LM iterations). Tracks and the
baseline: the GS camera it started from (DSCF2973_gs). Two rigid surfaces, because the shot looks at two places:
  early: the sand the camera looks down on before the whip pan, plane fitted in 1001-1013, refs 1004 and 1010
  main:  the plane through the tracks in the lower 38 % of frame over the whole shot (the tool's defaults), refs
         1016-1046 every 6 (the single plane leaves no surface in 1004/1010, so those refs come from 'early')
Each at the camera's readout with --baseline, and with the sign flipped (must lose). Then one table over both.

Writes atlas-vfx-work/rs_clips/DSCF2973_s5/proof_v2/: per-run csv, summary json, review jpgs, s5_headline.json.
Run: <V141 python> tools\\run_rs_s5_dscf2973.py [--combine-only]
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
from atlas_vfx.sensor import rs_proof as rp  # noqa: E402

D = pl.WORK_ROOT / "rs_clips" / "DSCF2973_s5"
OUT = D / "proof_v2"
CAM = D / "DSCF2973_rs_free_it500.shot_camera.json"
GS = pl.WORK_ROOT / "rs_clips" / "DSCF2973_f16" / "DSCF2973_gs.shot_camera.json"
MAIN_REFS = "1016,1022,1028,1034,1040,1046"
RUNS = {
    "DSCF2973_s5_early": ["--fit-frames", "1001-1013", "--refs", "1004,1010", "--baseline", str(GS)],
    "DSCF2973_s5": ["--refs", MAIN_REFS, "--baseline", str(GS)],
    "DSCF2973_s5_early_flip": ["--fit-frames", "1001-1013", "--refs", "1004,1010", "--readout-ms", "-27.73"],
    "DSCF2973_s5_flip": ["--refs", MAIN_REFS, "--readout-ms", "-27.73"],
}
VARIANTS = {"27.73 ms": ["DSCF2973_s5_early", "DSCF2973_s5"],
            "-27.73 ms (flipped)": ["DSCF2973_s5_early_flip", "DSCF2973_s5_flip"]}


def rows(tag):
    out = []
    for r in csv.DictReader(open(OUT / f"{tag}.csv", encoding="utf-8")):
        d = {k: (float(v) if v not in ("", None) else None) for k, v in r.items()}
        d["ref"], d["frame"] = int(d["ref"]), int(d["frame"])
        out.append(d)
    return out


def combine() -> dict:
    res = {}
    for name, tags in VARIANTS.items():
        rr = [x for t in tags for x in rows(t)]
        s = rp.summarise(rr)
        res[name] = {"on_vs_off": s, "acceptance": rp.acceptance(s)}
        if any(x.get("on_vs_gs_pct") is not None for x in rr):
            g = rp.summarise(rr, key="on_vs_gs_pct")
            res[name].update({
                "on_vs_gs_camera": g, "acceptance_vs_gs": rp.acceptance(g),
                "gs_plus_readout_vs_gs_off_median_pct": float(np.median(
                    [x["gs_on_vs_gs_off_pct"] for x in rr if x.get("gs_on_vs_gs_off_pct") is not None])),
                "on_vs_off_common_px_median_pct": float(np.median(
                    [x["on_vs_off_common_pct"] for x in rr if x.get("on_vs_off_common_pct") is not None]))})
        gp = {p["ref"]: p for p in res[name].get("on_vs_gs_camera", {}).get("per_ref", [])}
        print(f"== {name}")
        for p in s["per_ref"]:
            q = gp.get(p["ref"])
            print(f"  ref {p['ref']}: on vs off {p['on_vs_off_median_pct']:+.1f} % ({p['on_better_in']}/{p['targets']})"
                  + (f" | vs GS camera {q['on_vs_off_median_pct']:+.1f} % ({q['on_better_in']}/{q['targets']})"
                     if q else ""))
        print(f"  median of refs {s['median_of_refs_pct']:+.1f} %, all pairs {s['median_all_pairs_pct']:+.1f} % "
              f"({s['on_better_pairs']}/{s['pairs']}), worst {s['worst_ref_pct']:+.1f} %",
              json.dumps(res[name]["acceptance"]))
        if "on_vs_gs_camera" in res[name]:
            g = res[name]["on_vs_gs_camera"]
            print(f"  vs GS camera: median of refs {g['median_of_refs_pct']:+.1f} %, all pairs "
                  f"{g['median_all_pairs_pct']:+.1f} % ({g['on_better_pairs']}/{g['pairs']}), worst "
                  f"{g['worst_ref_pct']:+.1f} %", json.dumps(res[name]["acceptance_vs_gs"]))
    (OUT / "s5_headline.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main() -> int:
    if "--combine-only" not in sys.argv:
        for tag, extra in RUNS.items():
            subprocess.run([sys.executable, str(ROOT / "tools" / "rs_s5_proof.py"), "--camera", str(CAM),
                            "--tracks", str(GS), "--out", str(OUT), "--tag", tag, *extra], check=True)
    combine()
    return 0


if __name__ == "__main__":
    sys.exit(main())
