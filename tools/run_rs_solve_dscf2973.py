"""Phase 1b S5, fast-pan half: RS Solve (catalogue 47) on DSCF2973, run through the node class itself (no ComfyUI
server, same code path). The GS camera is the 2026-10-22 pycolmap solve from tools/rs_clip_solve.py --fix-focal
(48 of 48 frames, 55.7k points, 1.38 px @1024, focal held at 16.47 mm); its points come from its own solve.

Readout free, started at the sensor preset's spec (readout_ms 0, as the node does by default), node default
max_iter 100, profile at fixed readouts. If the free solve hits the iteration cap, it is run again with the node's
max_iter ceiling (500) and no profile, so the report says whether the cap mattered.

Writes atlas-vfx-work/rs_clips/DSCF2973_s5/: DSCF2973_rs_free[_it500].shot_camera.json, .rs_solve.json, sidecars.
Run: <V141 python> tools\\run_rs_solve_dscf2973.py [--skip-default]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.camera_side.node_rs_solve import AtlasVFXRSSolve  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

GS = pl.WORK_ROOT / "rs_clips" / "DSCF2973_f16" / "DSCF2973_gs.shot_camera.json"
OUT = pl.WORK_ROOT / "rs_clips" / "DSCF2973_s5"
PROFILE = "0, 10, 20, 26.6, 30.9, 40"


def run(name: str, max_iter: int, profile: str) -> dict:
    t0 = time.time()
    cam = sc.load(GS)
    res = AtlasVFXRSSolve().run(cam.data, "", "", "free", 0.0, 0, 0, 0, max_iter, profile, 8000, name, str(OUT))
    full = json.loads(res["result"][3])
    print(res["ui"]["text"][0], f"({time.time() - t0:.0f} s)", flush=True)
    return full


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--skip-default" not in sys.argv:
        full = run("DSCF2973_rs_free", 100, PROFILE)
        capped = "iteration cap" in full["solve"]["message"]
        for r in full["profile"]:
            print("profile", json.dumps(r), flush=True)
    else:
        capped = True
    if capped:
        run("DSCF2973_rs_free_it500", 500, "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
