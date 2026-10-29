"""Save the GUI workflow chart_calibrate_s2s3: Chart Calibrate (catalogue 49) on the chart takes, lens (take A) then
readout (take B) with A's lens JSON wired in, and B's readout_ms into Sensor RS Maps' readout_chart_ms on the
DSCF3000 S2-S3 RS camera (the S3 run), so the preset rule's chart kind wins. Nothing is queued: the takes don't
exist yet (Mike shoots them in S1); drop them in as GRID_A / GRID_B in the footage folder, or retype the paths.

Atlas VFX node definitions come from the classes here (V141 loads the main checkout, which may not have this node
yet); core nodes from V141's /object_info.

Run: <V141 python> tools\\chart_calibrate_workflow.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ui_workflow  # noqa: E402
from atlas_vfx import registry  # noqa: E402

ui_workflow.API = "http://127.0.0.1:8188"                 # the owner's test instance
FOOTAGE = "~/Videos/atlas-vfx_footage"

GRAPH = {
    "1": {"class_type": "AtlasVFXChartCalibrate", "inputs": {
        "video_path": f"{FOOTAGE}/GRID_A.MOV", "mode": "lens", "sensor_preset": "fujifilm_xh2_dci4k_24p_dis_off",
        "lens_json": "", "first_frame": 0, "last_frame": -1, "every": 12, "still_px": 1.0, "focal_guess_mm": 16.0,
        "name": "GRID_A", "out_dir": ""}},
    "2": {"class_type": "AtlasVFXChartCalibrate", "inputs": {
        "video_path": f"{FOOTAGE}/GRID_B.MOV", "mode": "readout", "sensor_preset": "fujifilm_xh2_dci4k_24p_dis_off",
        "lens_json": ["1", 0], "first_frame": 0, "last_frame": -1, "every": 12, "still_px": 1.0,
        "focal_guess_mm": 16.0, "name": "GRID_B", "out_dir": ""}},
    "3": {"class_type": "PreviewAny", "inputs": {"source": ["1", 12]}},
    "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 12]}},
    "5": {"class_type": "AtlasVFXPlateRead", "inputs": {
        "path": f"{FOOTAGE}/DSCF3000.MOV", "first": 129, "last": 296, "ocio_config": "", "input_space": "auto",
        "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 56}},
    "6": {"class_type": "AtlasVFXCameraImport", "inputs": {
        "source_path": "~/atlas-vfx-work/DSCF3000/camera/DSCF3000_S2S3_rs_free.shot_camera.json",
        "kind": "shot_camera", "plate_ref": ["5", 0], "sensor_preset": "auto", "sensor_width_mm": 23.5,
        "sensor_height_mm": 12.39, "m_per_unit": 1.0, "rot_order": "ZXY", "chan_focal_mm": 0.0,
        "crop_to_plate": True, "name": "DSCF3000_S2S3_rs_free_in", "out_dir": ""}},
    "7": {"class_type": "AtlasVFXSensorRSMaps", "inputs": {
        "shot_camera": ["6", 0], "depth_folder": "~/atlas-vfx-work/DSCF3000/depth/DSCF3000_f16_GUESS_stairs",
        "readout_ms": 30.9, "readout_source": "preset rule", "direction": "both", "shutter_angle": 0.0,
        "blur_samples": 5, "map_scale": 0.5, "iters": 3, "first_frame": 0, "last_frame": 0, "out_dir": "",
        "readout_chart_ms": ["2", 7]}},
    "8": {"class_type": "PreviewAny", "inputs": {"source": ["7", 2]}},
}


def seed_atlas_definitions():
    """ui_workflow reads /object_info; give it ours for the Atlas VFX nodes in this graph."""
    classes, _, _ = registry.load_nodes()
    for node in GRAPH.values():
        cls = classes.get(node["class_type"])
        if cls is not None:
            ui_workflow._INFO[node["class_type"]] = {"input": cls.INPUT_TYPES(), "output": list(cls.RETURN_TYPES),
                                                     "output_name": list(cls.RETURN_NAMES)}


if __name__ == "__main__":
    seed_atlas_definitions()
    print("saved", ui_workflow.save(GRAPH, "chart_calibrate_s2s3",
                                    "Chart Calibrate: lens (A) -> readout (B) -> Sensor RS Maps (chart wins)"))
