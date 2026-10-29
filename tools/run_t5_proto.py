"""T5 on real data with the PROVISIONAL measured camera (T4 prototype door picks, lawn ground): Plate Depth (MoGe-3)
over S2-S3, then Set Proxy with deck_y 1.41 (the threshold's height above the lawn in that camera). With Mike's picks
the deck becomes ground (y = 0) and deck_y goes back to 0.

Run: <V141 python> tools\\run_t5_proto.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_t4_proto import GRAPH as T4  # noqa: E402

GRAPH = {
    "1": T4["1"], "2": T4["2"], "3": T4["3"],
    "6": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_3_vitl_fp16.safetensors"}},
    "7": {"class_type": "AtlasVFXPlateDepth", "inputs": {
        "moge_model": ["6", 0], "plate_ref": ["1", 0], "shot_camera": ["3", 0], "solve_points": "",
        "proxy_width": 1024, "resolution_level": 9, "refine_steps": 3, "frame_step": 1, "preview_every": 24,
        "out_dir": ""}},
    "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 1]}},
    "9": {"class_type": "AtlasVFXSetProxy", "inputs": {
        "shot_camera": ["3", 0], "depth_folder": ["7", 0], "solve_points": "", "deck_y": 1.41, "deck_band": 0.4,
        "max_tilt_deg": 3.0, "inlier_m": 0.03, "frame_every": 8, "pixel_step": 8, "name": "", "out_dir": ""}},
}

if __name__ == "__main__":
    run(GRAPH, "t5_plate_depth_set_proxy_PROVISIONAL", "T5 Plate Depth + Set Proxy (provisional camera)",
        timeout=3600)
