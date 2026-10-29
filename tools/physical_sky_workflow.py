"""Build workflows/physical_sky_dscf2977.json: Atlas VFX Physical Sky for the DSCF2977 check shot (Burning Palms,
25 Dec 2025 6:39 pm AEDT, sun 252.4 az / 16.1 el), its preview into a Preview Image.

Uses tools/ui_workflow.py's convert + portable, with the node definitions of THIS checkout for the atlas-vfx nodes
(the live V141 loads the main checkout, which has no Physical Sky until this branch is merged) and V141's
/object_info for core nodes. Writes only the repo copy; ui_workflow.save would also drop it in V141's sidebar, which
waits for the merge.

Run: <V141 python> tools\\physical_sky_workflow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ui_workflow  # noqa: E402
from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky  # noqa: E402

ui_workflow.API = "http://127.0.0.1:8188"
NAME = "physical_sky_dscf2977"


def local_info(cls) -> dict:
    """The /object_info entry ComfyUI would serve for a node class of this checkout."""
    return json.loads(json.dumps({"input": cls.INPUT_TYPES(), "output": list(cls.RETURN_TYPES),
                                  "output_name": list(cls.RETURN_NAMES), "name": cls.__name__,
                                  "category": cls.CATEGORY, "output_node": getattr(cls, "OUTPUT_NODE", False)}))


def graph() -> dict:
    return {
        "1": {"class_type": "AtlasVFXPhysicalSky", "inputs": {
            "latitude": -34.1906, "longitude": 151.0425, "altitude_m": 5.0, "capture_utc": "2025-12-25T07:39:25Z",
            "time_offset_h": 0.0, "heading_deg": 180.0, "fov_x": 69.5, "tilt_deg": 0.0, "visibility_km": 40.0,
            "ground_albedo": 0.25, "exposure_ev": 0.0, "sun_mode": "metadata", "sun_azimuth_offset": 0.0,
            "sun_elevation_offset": 0.0, "sun_azimuth": 250.0, "sun_elevation": 15.0, "width": 2048,
            "latlong_centre": "north", "name": "DSCF2977_physical_sky", "out_dir": "", "dataset": "",
            "ocio_config": "", "display": "sRGB - Display", "view": "ACES 2.0 - SDR 100 nits (Rec.709)"}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 2]}},
    }


def main() -> int:
    ui_workflow._INFO["AtlasVFXPhysicalSky"] = local_info(AtlasVFXPhysicalSky)
    wf = ui_workflow.convert(graph(), "Physical Sky: DSCF2977 (Burning Palms, 6:39 pm AEDT)")
    out = ui_workflow.REPO / f"{NAME}.json"
    out.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
