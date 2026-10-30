"""Atlas VFX Chunk Plan: a long frame range (e.g. 1001-1400) split into overlapping LTX-sized chunks (8k+1 frames).
Chunk k's start / length / end wire into LTX Prep, the LTX latent and audio lengths, or Sequence Read Chunk.
chunk -1 gives every chunk as a list, so ComfyUI runs the rest of the graph once per chunk in one queue."""
from __future__ import annotations

from . import chunks as ch


class AtlasVFXChunkPlan:
    CATEGORY = "Atlas VFX/Layers/Chunks"
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_VFX_CHUNK_PLAN", "INT", "INT", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("chunk_plan", "start", "length", "end", "tag", "chunk_count", "info")
    OUTPUT_IS_LIST = (False, True, True, True, True, False, False)
    DESCRIPTION = ("Splits first-last into chunks of the same 8k+1 length (at most max_len) that overlap by at least "
                   "`overlap` frames; the last chunk is moved back to end on `last`, so no frame is padded. chunk = "
                   "which one to output (0 = first); -1 = all of them as lists, which runs the downstream nodes once "
                   "per chunk. tag is c00, c01, ... for output names. Chunk Stitch joins the results.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("INT", {"default": 1001, "min": 0, "max": 1_000_000, "tooltip": "first shot frame"}),
                "last": ("INT", {"default": 1400, "min": 0, "max": 1_000_000, "tooltip": "last shot frame"}),
                "max_len": ("INT", {"default": 121, "min": 9, "max": 1001, "step": 8,
                                    "tooltip": "LTX frames per run, 8k+1"}),
                "overlap": ("INT", {"default": 24, "min": 0, "max": 1000,
                                    "tooltip": "frames each chunk shares with the next (crossfaded by Chunk Stitch)"}),
                "chunk": ("INT", {"default": 0, "min": -1, "max": 999,
                                  "tooltip": "which chunk to output; -1 = all, as lists"}),
            },
        }

    def run(self, first, last, max_len, overlap, chunk):
        p = ch.plan(first, last, max_len, overlap)
        if chunk >= len(p.chunks):
            raise ValueError(f"chunk {chunk}: the plan has {len(p.chunks)} (0-{len(p.chunks) - 1})\n{p.table()}")
        pick = list(p.chunks) if chunk < 0 else [p.chunks[chunk]]
        info = p.table() + "\n" + ("output: all chunks" if chunk < 0 else f"output: {pick[0].tag}")
        return (p.to_dict(), [c.start for c in pick], [c.length for c in pick], [c.end for c in pick],
                [c.tag for c in pick], len(p.chunks), info)


NODE_CLASS_MAPPINGS = {"AtlasVFXChunkPlan": AtlasVFXChunkPlan}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXChunkPlan": "Atlas VFX Chunk Plan"}
