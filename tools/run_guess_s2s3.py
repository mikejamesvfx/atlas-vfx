"""T4 + T5 on real data with Claude's GUESSED picks (Mike: "just guess"): scale from the stairs (pad to deck = 4 risers
guessed at 0.18 m, plane distance), deck as ground. Then Lock Check (card on the deck at actor_spot), Plate Depth (MoGe-3)
and Set Proxy (deck_y 0). Picks: atlas-vfx-work/DSCF3000/picks/DSCF3000_S2S3_picks_GUESS.json.

Run: <V141 python> tools\\run_guess_s2s3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_t4_proto import GRAPH as T4  # noqa: E402

PICKS = r"C:\Users\miike\atlas-vfx-work\DSCF3000\picks\DSCF3000_S2S3_picks_GUESS.json"
NAME = "DSCF3000_f16_GUESS_stairs"

GRAPH = {
    "1": T4["1"], "2": T4["2"],
    "3": {"class_type": "AtlasVFXMeasureScale", "inputs": {
        "shot_camera": ["2", 0], "picks_path": PICKS, "solve_points": "", "max_spread_pct": 5.0, "force": False,
        "name": NAME, "out_dir": ""}},
    "4": {"class_type": "AtlasVFXLockCheck", "inputs": {
        "plate_ref": ["1", 0], "shot_camera": ["3", 0], "centre": "actor_spot", "orientation": "ground",
        "width_m": 1.5, "height_m": 1.5, "divisions": 6, "proxy_width": 1024, "preview_every": 24, "out_path": "",
        "measure_report": ["3", 2]}},
    "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 1]}},
    "6": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_3_vitl_fp16.safetensors"}},
    "7": {"class_type": "AtlasVFXPlateDepth", "inputs": {
        "moge_model": ["6", 0], "plate_ref": ["1", 0], "shot_camera": ["3", 0], "solve_points": "",
        "proxy_width": 1024, "resolution_level": 9, "refine_steps": 3, "frame_step": 1, "preview_every": 24,
        "out_dir": ""}},
    "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 1]}},
    "9": {"class_type": "AtlasVFXSetProxy", "inputs": {
        "shot_camera": ["3", 0], "depth_folder": ["7", 0], "solve_points": "", "deck_y": 0.0, "deck_band": 0.3,
        "max_tilt_deg": 3.0, "inlier_m": 0.03, "frame_every": 8, "pixel_step": 8, "name": "", "out_dir": ""}},
}

if __name__ == "__main__":
    run(GRAPH, "s2s3_guess_measure_lock_depth_set", "S2-S3 guessed picks: Measure, Lock Check, Depth, Set Proxy",
        timeout=3600)
