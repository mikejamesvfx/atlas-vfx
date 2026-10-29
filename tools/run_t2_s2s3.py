"""T2 on the real plate: Plate Read (S2-S3, frames 129-296) -> Tracking Proxy, run through V141.
Saves the GUI workflow (t2_plate_read_s2s3) and prints what came back.

Run: <V141 python> tools\\run_t2_s2s3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402

PLATE_READ = {"class_type": "AtlasVFXPlateRead", "inputs": {
    "path": r"C:\Users\miike\Videos\atlas-vfx_footage\DSCF3000.MOV", "first": 129, "last": 296,
    "ocio_config": r"C:\Users\miike\atlas-vfx\config\atlas_aces2.0_studio_flog2.ocio",
    "input_space": "auto", "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 8}}

GRAPH = {
    "1": PLATE_READ,
    "2": {"class_type": "AtlasVFXTrackingProxy", "inputs": {
        "plate_ref": ["1", 0], "width": 1024, "out_dir": "", "clip_limit": 3.0, "preview_every": 8}},
    "3": {"class_type": "PreviewImage", "inputs": {"images": ["1", 1]}},
    "4": {"class_type": "PreviewImage", "inputs": {"images": ["2", 1]}},
}

if __name__ == "__main__":
    run(GRAPH, "t2_plate_read_s2s3", "T2 Plate Read + Tracking Proxy, S2-S3")
