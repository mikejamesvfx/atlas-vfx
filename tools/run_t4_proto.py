"""T4 smoke on real data BEFORE Mike's picks: the prototype's door picks (door_scale3.py, frames 0/40/60, door
ASSUMED 2.08 m, lawn as ground) on the pycolmap f16 camera, then Lock Check through S2-S3 with a ground card at the
door threshold (deck level). Provisional: the real run uses Mike's measured picks (docs/PICKING.md).

Run: <V141 python> tools\\run_t4_proto.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_t2_s2s3 import PLATE_READ  # noqa: E402
from run_t3_s2s3 import cam_import  # noqa: E402

F16 = r"C:\Users\miike\ComfyUI_V135\ComfyUI\output\_pick\vfx_probe\solve_s1s3_f16"
PICKS = r"C:\Users\miike\atlas-vfx-work\DSCF3000\picks\proto_door_f16_picks.json"

imp = cam_import(F16, "DSCF3000_pycolmap_f16_full")
imp["inputs"]["crop_to_plate"] = False          # the door picks are at 1001-1061, outside S2-S3

GRAPH = {
    "1": {**PLATE_READ, "inputs": {**PLATE_READ["inputs"], "preview_every": 56}},
    "2": imp,
    "3": {"class_type": "AtlasVFXMeasureScale", "inputs": {
        "shot_camera": ["2", 0], "picks_path": PICKS, "solve_points": "", "max_spread_pct": 5.0, "force": False,
        "name": "DSCF3000_f16_PROVISIONAL_door208", "out_dir": ""}},
    "4": {"class_type": "AtlasVFXLockCheck", "inputs": {
        "plate_ref": ["1", 0], "shot_camera": ["3", 0], "centre": "door_bottom", "orientation": "ground",
        "width_m": 2.0, "height_m": 2.0, "divisions": 8, "proxy_width": 1024, "preview_every": 24, "out_path": "",
        "measure_report": ["3", 2]}},
    "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 1]}},
}

if __name__ == "__main__":
    rec = run(GRAPH, "t4_measure_lockcheck_PROVISIONAL", "T4 Measure & Scale + Lock Check (provisional door picks)")
