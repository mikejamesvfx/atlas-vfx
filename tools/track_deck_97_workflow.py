"""Save the GUI workflow track_dscf3000_deck_97: the 3D camera tracker on the 97 frames of the deck light shot
(DSCF3000 Nuke frames 1130-1226, plate frames 129-225: the steps up to the deck).

Plate Read -> Solve Mask -> Solve Plate (focal locked at the measured 16.42 mm, k1 k2 solved) -> Solve View (points, camera
path, error per frame), Lock Check (a ground grid locked to the plate), Camera Export (.chan / .usda / Nuke JSON) and
Track 2D (seeded from Solve View). Built from tools/solve_plate_workflow.py's graph, without its f16 comparison.
Run: <V141 python> tools\\track_deck_97_workflow.py [--queue]
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import solve_plate_workflow as spw  # noqa: E402
import ui_workflow  # noqa: E402

NAME = "DSCF3000_deck_97_solve"
GRAPH = {k: json.loads(json.dumps(v)) for k, v in spw.GRAPH.items() if k not in ("5", "6", "7")}
GRAPH["1"]["inputs"].update(first=129, last=225, preview_every=24)
GRAPH["3"]["inputs"].update(name=NAME, focal_guess_mm=16.42, solve_focal=False)   # 97 frames of this move leave
# focal loose (a free solve gave 12.0 mm); 16.42 mm is the lens as NukeX and the 297-frame solves measure it
GRAPH["10"]["inputs"]["name"] = NAME
GRAPH["4"]["inputs"]["plate_frames"] = 97                         # every frame: the Plate tab plays
spw.GRAPH = GRAPH

if __name__ == "__main__":
    spw.seed_atlas_definitions()
    print("saved", ui_workflow.save(GRAPH, "track_dscf3000_deck_97",
                                    "3D camera tracker, 97 frames (1130-1226): Solve Mask -> Solve Plate -> View / Lock / Export"))
    if "--queue" in sys.argv:
        api = {k: {"class_type": v["class_type"], "inputs": {
            n: (str(Path(x).expanduser()) if isinstance(x, str) and x.startswith("~") else x)
            for n, x in v["inputs"].items()}} for k, v in GRAPH.items()}
        req = urllib.request.Request(ui_workflow.API + "/prompt", json.dumps({"prompt": api}).encode(),
                                     {"Content-Type": "application/json"})
        print("queued", json.loads(urllib.request.urlopen(req, timeout=30).read()))
