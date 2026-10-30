"""Atlas VFX Chunk Stitch: the chunks of a Chunk Plan, rendered to image sequences, joined back into one sequence
numbered from the plan's first frame (1001...), with the overlaps crossfaded. Writes <name>.####.exr (RGBA half),
<name>.stitch.json (which chunks and weights made each frame) and a sidecar. Reads the chunks, never writes them."""
from __future__ import annotations

import json

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl

from . import chunks as ch
from .node_sequence_read_chunk import read_image


def chunk_path(pattern: str, c: ch.Chunk) -> str:
    if "{chunk}" not in pattern and "{tag}" not in pattern:
        raise ValueError("pattern needs {chunk} (00, 01, ...) or {tag} (c00, c01, ...) to tell the chunks apart")
    return pattern.replace("{chunk}", f"{c.index:02d}").replace("{tag}", c.tag)


def as_rgba(img: np.ndarray) -> np.ndarray:
    C = img.shape[-1]
    if C == 4:
        return img
    rgb = np.repeat(img, 3, axis=-1) if C == 1 else img[..., :3]
    return np.dstack([rgb, np.ones(img.shape[:2], np.float32)])


def gather(p: ch.ChunkPlan, pattern: str, numbering: str) -> dict[int, list]:
    """Every chunk's files, checked before anything is written: one error lists every short or missing chunk."""
    files, problems = {}, []
    for c in p.chunks:
        src = chunk_path(pattern, c)
        try:
            files[c.index] = ch.pick(ch.scan(src), c.start, c.length, numbering)
        except (ValueError, OSError) as e:
            problems.append(f"{c.tag} ({c.start}-{c.end}) {src}: {e}")
    if problems:
        raise ValueError("chunks not ready:\n  " + "\n  ".join(problems))
    return files


class AtlasVFXChunkStitch:
    CATEGORY = "Atlas VFX/Layers/Chunks"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("pattern", "first_frame", "info")
    DESCRIPTION = ("Joins the plan's chunks into one sequence from `first`: pattern names each chunk's sequence with "
                   "{chunk} (00, 01, ...) or {tag} (c00, ...) plus a frame token or folder, e.g. "
                   "~/atlas-vfx-work/elem_{tag}/elem.####.exr. numbering frame = chunk files carry shot frames; "
                   "sequential = each chunk's files in order (plain SaveImage output). Overlaps crossfade "
                   "(smoothstep) in the files' own values: scene-linear for EXR, display-encoded for PNG/JPEG. Every "
                   "chunk is checked before a frame is written.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_plan": ("ATLAS_VFX_CHUNK_PLAN",),
                "pattern": ("STRING", {"default": "~/atlas-vfx-work/elem_{tag}/elem.####.exr"}),
                "numbering": (["frame", "sequential"],),
                "colorspace": ("STRING", {"default": "auto",
                                          "tooltip": "label written in the EXR headers; auto = the chunks' own"}),
                "name": ("STRING", {"default": "stitched"}),
                "out_dir": ("STRING", {"default": ""}),
            },
        }

    def run(self, chunk_plan, pattern, numbering, colorspace, name, out_dir):
        p = ch.from_dict(chunk_plan)
        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / "chunks" / name)
        files = gather(p, pl.clean_path(pattern), numbering)
        cs, shape, record = None, None, []
        for frame in range(p.first, p.last + 1):
            acc, used = None, []
            for k, w in ch.weights(p, frame):
                src = files[k][frame - p.chunks[k].start]
                img, info = read_image(src)
                img = as_rgba(img)
                if shape is None:
                    shape = img.shape
                    cs = info["colorspace"] if colorspace.strip() in ("", "auto") else colorspace.strip()
                elif img.shape != shape:
                    raise ValueError(f"{src}: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}, first chunk frame was "
                                     f"{shape[1]}x{shape[0]}x{shape[2]}")
                acc = img * w if acc is None else acc + img * w
                used.append({"chunk": p.chunks[k].tag, "file": src.name, "weight": round(w, 6)})
            exr_io.write_rgba(out / f"{name}.{frame:04d}.exr", acc, colorspace=cs or "",
                              attrs={"layer": name, "frame": frame, "stitched_from": [u["chunk"] for u in used]})
            record.append({"frame": frame, "sources": used})
        report = {"plan": p.to_dict(), "pattern": pattern, "numbering": numbering, "colorspace": cs,
                  "frames": [p.first, p.last], "size": [shape[1], shape[0]],
                  "blended_frames": sum(1 for r in record if len(r["sources"]) > 1)}
        (out / f"{name}.stitch.json").write_text(json.dumps({**report, "per_frame": record}, indent=1),
                                                 encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXChunkStitch",
                         inputs={"plan": p.to_dict(), "pattern": pattern, "numbering": numbering,
                                 "colorspace": colorspace},
                         outputs=report)
        out_pattern = str(out / f"{name}.%04d.exr")
        text = (f"{p.frame_count} frames {p.first}-{p.last} from {len(p.chunks)} chunks, "
                f"{report['blended_frames']} crossfaded -> {out_pattern}")
        return {"ui": {"text": [text]}, "result": (out_pattern, p.first, text)}


NODE_CLASS_MAPPINGS = {"AtlasVFXChunkStitch": AtlasVFXChunkStitch}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXChunkStitch": "Atlas VFX Chunk Stitch"}
