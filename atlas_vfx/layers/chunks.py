"""Frame chunks: split a long frame range into overlapping LTX-sized pieces, and the crossfade weights that join them.

LTX 2.x takes 8k+1 frames per run (121 is the usual cap). A shot of 1001-1400 is planned as equal chunks of
`length` frames, each starting `stride = length - overlap` after the last; the last chunk is moved back so it ends
exactly on the shot's last frame (no padded frames), which gives its seam more overlap than asked. Every chunk has
the same length, so one graph (latent size, guide frames) serves them all.

Seams crossfade with smoothstep weights. Chunk b takes over from chunk a across their shared frames; with three
chunks on one frame (a short last stride) the handovers chain, and the weights still sum to 1 and are 0 outside a
chunk.

Also the image-sequence helpers the Chunk nodes share: `shot.####.exr`, `shot.%04d.exr`, `shot.{frame}.exr` or a
folder. Pure python (pathlib, re), no ComfyUI or numpy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

RULE = 8                     # LTX: frame counts are 8k+1
MIN_LENGTH = RULE + 1
SEQ_EXTS = (".exr", ".dpx", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def tag(self) -> str:
        return f"c{self.index:02d}"


@dataclass(frozen=True)
class ChunkPlan:
    first: int
    last: int
    max_len: int
    overlap: int
    chunks: tuple[Chunk, ...]

    @property
    def length(self) -> int:
        return self.chunks[0].length

    @property
    def frame_count(self) -> int:
        return self.last - self.first + 1

    def seams(self) -> list[int]:
        """Shared frames between chunk k-1 and k, for k = 1..n-1."""
        c = self.chunks
        return [c[k - 1].end - c[k].start + 1 for k in range(1, len(c))]

    def to_dict(self) -> dict:
        return {"kind": "atlas_vfx_chunk_plan", "version": 1, "first": self.first, "last": self.last,
                "max_len": self.max_len, "overlap": self.overlap, "length": self.length,
                "chunks": [{"index": c.index, "start": c.start, "end": c.end, "tag": c.tag} for c in self.chunks],
                "seams": self.seams()}

    def table(self) -> str:
        rows = [f"{self.frame_count} frames {self.first}-{self.last} -> {len(self.chunks)} chunks of {self.length} "
                f"(max {self.max_len}, overlap {self.overlap})"]
        seams = [0] + self.seams()
        for c, s in zip(self.chunks, seams):
            rows.append(f"  {c.tag}  {c.start}-{c.end}" + (f"  seam {s}" if s else ""))
        return "\n".join(rows)


def ltx_length(n: int) -> int:
    """The largest 8k+1 that is <= n."""
    return (n - 1) // RULE * RULE + 1


def plan(first: int, last: int, max_len: int = 121, overlap: int = 24) -> ChunkPlan:
    first, last, max_len, overlap = int(first), int(last), int(max_len), int(overlap)
    if last < first:
        raise ValueError(f"last {last} is before first {first}")
    if max_len < MIN_LENGTH or (max_len - 1) % RULE:
        raise ValueError(f"max_len {max_len}: LTX takes 8k+1 frames (9, 17, ... 97, 121)")
    n = last - first + 1
    if n < MIN_LENGTH:
        raise ValueError(f"{n} frames ({first}-{last}): LTX needs at least {MIN_LENGTH}")
    length = min(max_len, ltx_length(n))
    if not 0 <= overlap < length:
        raise ValueError(f"overlap {overlap}: must be 0 to {length - 1} for {length}-frame chunks")
    stride = length - overlap
    count = 1 if n == length else 1 + math.ceil((n - length) / stride)
    starts = [first + k * stride for k in range(count - 1)] + [last - length + 1]
    chunks = tuple(Chunk(k, s, s + length - 1) for k, s in enumerate(starts))
    return ChunkPlan(first, last, max_len, overlap, chunks)


def from_dict(d: dict) -> ChunkPlan:
    """Rebuilds the plan from its widgets (and checks the stored chunks agree, so a hand-edited dict can't drift)."""
    if not isinstance(d, dict) or d.get("kind") != "atlas_vfx_chunk_plan":
        raise ValueError("not an Atlas VFX chunk plan")
    p = plan(d["first"], d["last"], d["max_len"], d["overlap"])
    if [(c["start"], c["end"]) for c in d.get("chunks", [])] != [(c.start, c.end) for c in p.chunks]:
        raise ValueError("chunk plan: stored chunks don't match its first/last/max_len/overlap")
    return p


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def weights(p: ChunkPlan, frame: int) -> list[tuple[int, float]]:
    """[(chunk index, weight)] for one shot frame, weights > 0 only, summing to 1."""
    if not p.first <= frame <= p.last:
        raise ValueError(f"frame {frame} outside {p.first}-{p.last}")
    c = p.chunks
    # t[k]: how far chunk k has taken over from chunk k-1 at this frame (0 before its start, 1 after k-1's end)
    t = [1.0]
    for k in range(1, len(c)):
        a, b = c[k - 1], c[k]
        t.append(_smooth((frame - b.start + 1) / (a.end - b.start + 2)))
    t.append(0.0)
    out, carried = [], 1.0
    for k in range(len(c)):
        carried *= t[k]
        w = carried * (1.0 - t[k + 1])
        if w > 0.0:
            out.append((k, w))
    s = sum(w for _, w in out)
    return [(k, w / s) for k, w in out]


# --------------------------------------------------------------------------------------------- image sequences

_HASHES = re.compile(r"#+")
_PRINTF = re.compile(r"%0?(\d*)d")


def frame_template(path: str) -> str | None:
    """`shot.####.exr` / `shot.%04d.exr` / `shot.{frame}.exr` -> a str.format template with `{frame:0Nd}`; None for a
    path with no frame token (a folder)."""
    if "{frame" in path:
        return path
    m = _HASHES.search(path)
    if m:
        return path[:m.start()] + "{frame:0%dd}" % len(m.group()) + path[m.end():]
    m = _PRINTF.search(path)
    if m:
        return path[:m.start()] + "{frame:0%dd}" % int(m.group(1) or 1) + path[m.end():]
    return None


_TRAIL = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def scan(path: str) -> dict[int, Path]:
    """{frame number: file} for a sequence pattern or a folder (files ending in a number before the extension). A
    folder must hold one sequence: two names or extensions is an error, so a stray file can't slip in."""
    tmpl = frame_template(path)
    if tmpl is not None:
        p = Path(tmpl)
        folder = p.parent
        pre, _, post = p.name.partition("{frame")
        post = post.split("}", 1)[1]
        rx = re.compile(re.escape(pre) + r"(\d+)" + re.escape(post) + "$")
        files = {int(m.group(1)): f for f in (folder.iterdir() if folder.is_dir() else [])
                 if (m := rx.match(f.name))}
        return dict(sorted(files.items()))
    folder = Path(path)
    if not folder.is_dir():
        raise ValueError(f"{path}: not a folder and no frame token (####, %04d or {{frame}})")
    found, stems = {}, set()
    for f in folder.iterdir():
        m = _TRAIL.search(f.name)
        if f.is_file() and f.suffix.lower() in SEQ_EXTS and m:
            stems.add((f.name[:m.start()], f.suffix.lower()))
            found[int(m.group(1))] = f
    if len(stems) > 1:
        raise ValueError(f"{path}: more than one sequence in the folder ({sorted(stems)[:3]}); give a pattern")
    return dict(sorted(found.items()))


def pick(files: dict[int, Path], start: int, length: int, numbering: str = "frame") -> list[Path]:
    """The files for frames start..start+length-1. numbering "frame": file numbers are shot frames. "sequential": the
    files in number order are start, start+1, ... (plain ComfyUI SaveImage output, 00001_...). Missing frames are
    one error listing them."""
    if numbering == "sequential":
        ordered = list(files.values())
        if len(ordered) < length:
            raise ValueError(f"{len(ordered)} files, need {length} (sequential numbering)")
        return ordered[:length]
    if numbering != "frame":
        raise ValueError(f"numbering {numbering!r}: frame or sequential")
    want = range(start, start + length)
    missing = [n for n in want if n not in files]
    if missing:
        raise ValueError(f"{len(missing)} of {length} frames missing ({start}-{start + length - 1}), first "
                         f"{missing[:5]}")
    return [files[n] for n in want]
