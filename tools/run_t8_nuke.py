"""T8 on real data: export the GUESS camera (chan in metres, Nuke settings, usda) and write the Nuke build script for
S2-S3 with the T6 grass layers (back to front), plate from the DPX sequence, Set Proxy OBJ for
lining up.

Run: <V141 python> tools\\run_t8_nuke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_guess_s2s3 import GRAPH as G  # noqa: E402

W = r"C:\Users\miike\atlas-vfx-work\DSCF3000"
LAYERS = "\n".join(rf"{W}\layers\DSCF3000_f16_GUESS_stairs_{n}" for n in ("grass_far", "grass_near"))

GRAPH = {
    "1": G["1"], "2": G["2"], "3": G["3"],
    "90": {"class_type": "AtlasVFXCameraExport", "inputs": {"shot_camera": ["3", 0], "units_per_m": 1.0, "name": "",
                                                            "out_dir": ""}},
    "91": {"class_type": "AtlasVFXNukeBuilder", "inputs": {
        "plate_ref": ["1", 0], "nuke_camera_json": ["90", 2], "layer_folders": LAYERS, "shot": "DSCF3000_S2S3",
        "plate_kind": "dpx", "set_proxy_obj": rf"{W}\set_proxy\DSCF3000_f16_GUESS_stairs_set.obj",
        "prefix": "AVFX_", "set_root": True, "out_dir": ""}},
}

if __name__ == "__main__":
    run(GRAPH, "t8_camera_export_nuke_builder_s2s3", "T8 Camera Export + Nuke Builder (S2-S3, grass)",
        timeout=1800)
