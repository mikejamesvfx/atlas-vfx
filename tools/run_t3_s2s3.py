"""T3 on the real data: Plate Read (S2-S3) -> Camera Import x2 (Nuke Camera1.json, pycolmap solve_s1s3_f16, both
cropped to the plate's frames 1130-1297) -> Solve Compare (a = solve, b = Nuke). Run through V141.

Run: <V141 python> tools\\run_t3_s2s3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_t2_s2s3 import PLATE_READ  # noqa: E402


def cam_import(src, name):
    return {"class_type": "AtlasVFXCameraImport", "inputs": {
        "source_path": src, "kind": "auto", "plate_ref": ["1", 0], "sensor_preset": "auto",
        "sensor_width_mm": 23.5, "sensor_height_mm": 12.39, "m_per_unit": 1.0, "rot_order": "ZXY",
        "chan_focal_mm": 0.0, "crop_to_plate": True, "name": name, "out_dir": ""}}


GRAPH = {
    "1": {**PLATE_READ, "inputs": {**PLATE_READ["inputs"], "preview_every": 56}},
    "2": cam_import(r"C:\Users\miike\howling-yowies\nuke\export\Camera1.json", "DSCF3000_S2S3_nuke_Camera1"),
    "3": cam_import(r"C:\Users\miike\ComfyUI_V135\ComfyUI\output\_pick\vfx_probe\solve_s1s3_f16",
                    "DSCF3000_S2S3_pycolmap_f16"),
    "4": {"class_type": "AtlasVFXSolveCompare", "inputs": {"camera_a": ["3", 0], "camera_b": ["2", 0]}},
}

if __name__ == "__main__":
    run(GRAPH, "t3_camera_import_compare_s2s3", "T3 Camera Import + Solve Compare, S2-S3")
