"""Build the Relight Transfer starter (catalogue 53): workflows/relight_transfer_element.json

  element sequence (linear EXR)  --+------------------------------------------+
  element sequence (as sRGB) -> MoGe-3 -> Atlas VFX Normals (in the matte) --+--> Relight Transfer (sh) -> EXR Write
  relit take (your LTX relight IC-LoRA render, any size) ---------------------+                           -> gain EXR
  matte sequence -> ImageToMask ----------------------------------------------+

The relight itself (LTX-2.3 Relight at 1280x704, LTX-2.5 Day-to-Night / Golden-Hour) runs in its own graph and
writes the relit sequence; this graph only reads it, so it works with any relight model. MoGe-3 runs on the element
at its own size, which is what gives the sh mode its full-res normals.

graph() is the API graph. Uses tools/ui_workflow.py's convert + portable with this checkout's node definitions for
the atlas-vfx nodes and a running ComfyUI's /object_info for the core ones (MoGe, ImageToMask). Writes only the repo
copy.

Run: <V141 python> tools\\relight_transfer_workflow.py [--api http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ui_workflow  # noqa: E402
from physical_sky_workflow import local_info  # noqa: E402
from atlas_vfx.camera_side.node_exr_write import AtlasVFXEXRWrite  # noqa: E402
from atlas_vfx.layers.node_normals import AtlasVFXNormals  # noqa: E402
from atlas_vfx.layers.node_relight_transfer import AtlasVFXRelightTransfer  # noqa: E402
from atlas_vfx.layers.node_sequence_read_chunk import AtlasVFXSequenceReadChunk  # noqa: E402

ELEMENT = "~/atlas-vfx-work/element/element.####.exr"
MATTE = "~/atlas-vfx-work/element/matte.####.exr"
RELIT = "~/atlas-vfx-work/relit/relit.####.exr"
SRGB = "sRGB Encoded Rec.709 (sRGB)"


def read(path, input_colorspace="", start=1001, length=121, numbering="frame"):
    return {"class_type": "AtlasVFXSequenceReadChunk", "inputs": {
        "path": path, "start": start, "length": length, "numbering": numbering, "width": 0,
        "input_colorspace": input_colorspace, "colorspace": SRGB, "ocio_config": ""}}


def exr(images, name):
    return {"class_type": "AtlasVFXEXRWrite", "inputs": {
        "images": images, "first_frame": ["1", 1], "input_colorspace": "ACEScg", "colorspace": "ACEScg",
        "input_premultiplied": False, "ocio_config": "", "name": name, "out_dir": ""}}


def graph() -> dict:
    return {
        "1": read(ELEMENT),                                   # linear ACEScg, as stored
        "2": read(ELEMENT, input_colorspace="ACEScg"),        # the same frames as sRGB, for MoGe
        "3": read(RELIT, numbering="sequential"),             # LTX output numbered from 0 or 1 is fine
        "4": read(MATTE),
        "5": {"class_type": "ImageToMask", "inputs": {"image": ["4", 0], "channel": "red"}},
        "6": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_3_vitl_fp16.safetensors"}},
        "7": {"class_type": "MoGeInference", "inputs": {
            "moge_model": ["6", 0], "image": ["2", 0], "resolution_level": 9, "fov_x_degrees": 0.0,
            "batch_size": 2, "force_projection": True, "apply_mask": False, "refine_steps": 3}},
        "8": {"class_type": "AtlasVFXNormals", "inputs": {
            "moge_geometry": ["7", 0], "temporal": 1, "sigma": 1.5, "matte": ["5", 0]}},
        "9": {"class_type": "AtlasVFXRelightTransfer", "inputs": {
            "element": ["1", 0], "matte": ["5", 0], "relit": ["3", 0], "encoding": "linear", "mode": "sh",
            "analysis_width": 1280, "blur_px": 2.0, "max_stops": 3.0, "temporal": 1, "eps": 0.01, "subsample": 2,
            "geo_weight": 0.5, "min_r2": 0.3, "normals": ["8", 0], "moge_geometry": ["7", 0],
            "first_frame": ["1", 1]}},
        "10": exr(["9", 0], "element_relit"),
        "11": exr(["9", 1], "element_relit_gain"),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    for cls in (AtlasVFXSequenceReadChunk, AtlasVFXNormals, AtlasVFXRelightTransfer, AtlasVFXEXRWrite):
        ui_workflow._INFO[cls.__name__] = local_info(cls)
    wf = ui_workflow.convert(graph(), "Relight Transfer: LTX relit take (low res) onto the full-res element, "
                                      "MoGe-3 normals")
    path = ui_workflow.REPO / "relight_transfer_element.json"
    path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
