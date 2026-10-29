"""RS Solve (catalogue 47) on DSCF3000 S2-S3 in V141, readout free: the measured GS camera (DSCF3000_f16_GUESS_stairs,
the pycolmap f16 solve through Measure & Scale) cropped to the plate, its own solve's points, readout started at the
sensor preset's spec, plus a readout profile. Gives a saved RS camera with a solved readout (the two RS cameras saved
before held 30.9 ms). Saves the GUI workflow rs_solve_s2s3 first.

Run: <V141 python> tools\\run_rs_solve_s2s3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_run  # noqa: E402
import ui_workflow  # noqa: E402

PORT = "http://127.0.0.1:8188"                # the owner's test instance (some helpers still say 8288)
comfy_run.API = ui_workflow.API = PORT

GS_CAMERA = "~/atlas-vfx-work/DSCF3000/camera/DSCF3000_f16_GUESS_stairs.shot_camera.json"

GRAPH = {
    "1": {"class_type": "AtlasVFXPlateRead", "inputs": {
        "path": "~/Videos/atlas-vfx_footage/DSCF3000.MOV", "first": 129, "last": 296, "ocio_config": "",
        "input_space": "auto", "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 56}},
    "2": {"class_type": "AtlasVFXCameraImport", "inputs": {
        "source_path": GS_CAMERA, "kind": "shot_camera", "plate_ref": ["1", 0], "sensor_preset": "auto",
        "sensor_width_mm": 23.5, "sensor_height_mm": 12.39, "m_per_unit": 1.0, "rot_order": "ZXY",
        "chan_focal_mm": 0.0, "crop_to_plate": True, "name": "DSCF3000_S2S3_GUESS_gs", "out_dir": ""}},
    "3": {"class_type": "AtlasVFXRSSolve", "inputs": {
        "shot_camera": ["2", 0], "solve_folder": "", "points_camera": GS_CAMERA, "readout": "free",
        "readout_ms": 0.0, "first_frame": 0, "last_frame": 0, "max_points": 0, "max_iter": 100,
        "profile_ms": "0, 10, 20, 26.6, 30.9, 40", "profile_points": 8000, "name": "DSCF3000_S2S3_rs_free",
        "out_dir": ""}},
    "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 3]}},
}

if __name__ == "__main__":
    comfy_run.run(GRAPH, "rs_solve_s2s3", "RS Solve, readout free: DSCF3000 S2-S3 (measured GS camera)", timeout=3600)
