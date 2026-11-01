"""Save the GUI workflow solve_plate_s1s3: the 3D camera tracker on DSCF3000 S1-S3 (plate frames 0-296).

Plate Read -> Solve Mask (people, cars, trees) -> Solve Plate (focal + k1 k2 solved from the 16 mm mark) ->
Solve View (points, camera path, error per frame; click a plate frame to seed Track 2D) -> Track 2D, whose
tracks go into Solve Plate's tracks input for a second run (wire it by hand: a graph can't loop). Solve Compare against the pycolmap f16 solve (vfx_probe solve_s1s3_f16,
imported with Camera Import) shows how close the free-lens solve lands; Lock Check draws a ground grid; Camera
Export writes the .chan / .usda / Nuke JSON for Nuke.

Atlas VFX node definitions come from the classes here; core nodes from V141's /object_info.
Run: <V141 python> tools\\solve_plate_workflow.py
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
F16 = "~/ComfyUI_V135/ComfyUI/output/_pick/vfx_probe/solve_s1s3_f16"

GRAPH = {
    "1": {"class_type": "AtlasVFXPlateRead", "inputs": {
        "path": f"{FOOTAGE}/DSCF3000.MOV", "first": 0, "last": 296, "ocio_config": "", "input_space": "auto",
        "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 48}},
    "2": {"class_type": "AtlasVFXSolveMask", "inputs": {
        "plate_ref": ["1", 0], "classes": "person. car. bicycle. dog. tree. bush. water. flag.", "window": 16,
        "box_threshold": 0.35, "text_threshold": 0.25, "dilate_px": 6, "proxy_width": 1024}},
    "3": {"class_type": "AtlasVFXSolvePlate", "inputs": {
        "plate_ref": ["1", 0], "sensor_preset": "auto", "sensor_width_mm": 23.5, "sensor_height_mm": 12.39,
        "focal_guess_mm": 16.0, "solve_focal": True, "solve_distortion": True, "lens_model": "k1 k2 (RADIAL)",
        "proxy_width": 1024, "max_features": 8192, "overlap": 15, "reject_px": 2.0, "name": "DSCF3000_S1S3_solve",
        "out_dir": "", "masks": ["2", 0], "mask_dilate_px": 4, "seed": 0}},
    "4": {"class_type": "AtlasVFXSolveView", "inputs": {
        "shot_camera": ["3", 0], "max_points": 20000, "plate_frames": 8, "seeds": "[]"}},
    "5": {"class_type": "AtlasVFXCameraImport", "inputs": {
        "source_path": F16, "kind": "colmap", "plate_ref": ["1", 0], "sensor_preset": "auto",
        "sensor_width_mm": 23.5, "sensor_height_mm": 12.39, "m_per_unit": 1.0, "rot_order": "ZXY",
        "chan_focal_mm": 0.0, "crop_to_plate": True, "name": "DSCF3000_S1S3_f16", "out_dir": ""}},
    "6": {"class_type": "AtlasVFXSolveCompare", "inputs": {"camera_a": ["3", 0], "camera_b": ["5", 0]}},
    "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
    "8": {"class_type": "PreviewAny", "inputs": {"source": ["3", 2]}},
    "9": {"class_type": "AtlasVFXLockCheck", "inputs": {
        "plate_ref": ["1", 0], "shot_camera": ["3", 0], "centre": "auto", "orientation": "ground", "width_m": 4.0,
        "height_m": 4.0, "divisions": 8, "proxy_width": 1024, "preview_every": 24, "out_path": "",
        "measure_report": ""}},
    "10": {"class_type": "AtlasVFXCameraExport", "inputs": {
        "shot_camera": ["3", 0], "units_per_m": 1.0, "name": "DSCF3000_S1S3_solve", "out_dir": ""}},
    "11": {"class_type": "AtlasVFXTrack2D", "inputs": {
        "plate_ref": ["1", 0], "seeds": ["4", 0], "scale": 0.5, "span": 0, "fb_px": 1.0, "window": 41, "levels": 4}},
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
    print("saved", ui_workflow.save(GRAPH, "solve_plate_s1s3",
                                    "3D camera tracker: Solve Mask -> Solve Plate -> Solve View / Compare / Lock / Export"))
