"""Build the frame chunk workflows (catalogue 50-52):

  workflows/chunked_sequence_starter.json   Chunk Plan (chunk -1, all) -> Sequence Read Chunk -> EXR Write, one folder
                                            per chunk (shot_c00, ...). Put your generator between the read and the
                                            write; for long runs set chunk 0 and queue with tools/queue_chunks.py.
  workflows/chunked_sequence_stitch.json    Chunk Plan -> Chunk Stitch: the chunk folders joined into one sequence.

graph_*() are the API graphs. Uses tools/ui_workflow.py's convert + portable with this checkout's node definitions
for the atlas-vfx nodes and a running ComfyUI's /object_info for the core ones (StringConcatenate). Writes only the
repo copies.

Run: <V141 python> tools\\chunks_workflow.py [--api http://127.0.0.1:8188]
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
from atlas_vfx.layers.node_chunk_plan import AtlasVFXChunkPlan  # noqa: E402
from atlas_vfx.layers.node_chunk_stitch import AtlasVFXChunkStitch  # noqa: E402
from atlas_vfx.layers.node_sequence_read_chunk import AtlasVFXSequenceReadChunk  # noqa: E402

SEQ = "~/atlas-vfx-work/shot/shot.####.exr"
PLAN = {"first": 1001, "last": 1400, "max_len": 121, "overlap": 24}


def graph_chunks(seq: str = SEQ, prefix: str = "shot_") -> dict:
    return {
        "1": {"class_type": "AtlasVFXChunkPlan", "inputs": {**PLAN, "chunk": -1}},
        "2": {"class_type": "AtlasVFXSequenceReadChunk", "inputs": {
            "path": seq, "start": ["1", 1], "length": ["1", 2], "numbering": "frame", "width": 0,
            "input_colorspace": "", "colorspace": "sRGB Encoded Rec.709 (sRGB)", "ocio_config": ""}},
        "3": {"class_type": "StringConcatenate", "inputs": {"string_a": prefix, "string_b": ["1", 4],
                                                            "delimiter": ""}},
        "4": {"class_type": "AtlasVFXEXRWrite", "inputs": {
            "images": ["2", 0], "first_frame": ["1", 1], "input_colorspace": "ACEScg", "colorspace": "ACEScg",
            "input_premultiplied": False, "ocio_config": "", "name": ["3", 0], "out_dir": ""}},
    }


def graph_stitch(prefix: str = "shot_") -> dict:
    return {
        "1": {"class_type": "AtlasVFXChunkPlan", "inputs": {**PLAN, "chunk": 0}},
        "2": {"class_type": "AtlasVFXChunkStitch", "inputs": {
            "chunk_plan": ["1", 0],
            "pattern": f"~/atlas-vfx-work/exr_write/{prefix}{{tag}}/{prefix}{{tag}}.####.exr",
            "numbering": "frame", "colorspace": "auto", "name": f"{prefix}stitched", "out_dir": ""}},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    for cls in (AtlasVFXChunkPlan, AtlasVFXSequenceReadChunk, AtlasVFXChunkStitch, AtlasVFXEXRWrite):
        ui_workflow._INFO[cls.__name__] = local_info(cls)
    for file, title, g in (
            ("chunked_sequence_starter", "Frame chunks: 1001-1400 read as 121-frame chunks", graph_chunks()),
            ("chunked_sequence_stitch", "Frame chunks: stitch the chunk folders back to 1001-1400", graph_stitch())):
        wf = ui_workflow.convert(g, title)
        path = ui_workflow.REPO / f"{file}.json"
        path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
