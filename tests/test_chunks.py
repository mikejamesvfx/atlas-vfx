"""Frame chunks (layers/chunks.py, Chunk Plan, Sequence Read Chunk, Chunk Stitch, tools/queue_chunks.py) on synthetic
sequences with known values.

Run: <V141 python> tests\\test_chunks.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

TMP = Path(tempfile.mkdtemp(prefix="atlas_vfx_chunks_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.layers import chunks as ch  # noqa: E402
from atlas_vfx.layers.node_chunk_plan import AtlasVFXChunkPlan  # noqa: E402
from atlas_vfx.layers.node_chunk_stitch import AtlasVFXChunkStitch  # noqa: E402
from atlas_vfx.layers.node_sequence_read_chunk import AtlasVFXSequenceReadChunk  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def raises(fn, label, needle=""):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        check(needle in str(e), label, str(e).splitlines()[0][:100])
        return
    check(False, label, "no error")


def covered(p):
    return all(any(c.start <= f <= c.end for c in p.chunks) for f in range(p.first, p.last + 1))


def test_plan():
    p = ch.plan(1001, 1400, 121, 24)
    got = [(c.start, c.end) for c in p.chunks]
    check(got == [(1001, 1121), (1098, 1218), (1195, 1315), (1280, 1400)], "1001-1400 / 121 / 24: four chunks", got)
    check(p.seams() == [24, 24, 36], "seams 24, 24 and 36 on the moved-back last chunk", p.seams())
    check([c.tag for c in p.chunks] == ["c00", "c01", "c02", "c03"], "tags c00-c03")

    cases = [(1001, 1121, 121, 24), (1001, 1122, 121, 24), (1001, 1400, 97, 16), (1001, 1528, 121, 0),
             (1001, 1100, 121, 24), (1001, 1218, 121, 8), (1001, 1009, 121, 0), (0, 999, 121, 60)]
    ok = True
    for args in cases:
        q = ch.plan(*args)
        lens = {c.length for c in q.chunks}
        good = (covered(q) and len(lens) == 1 and (q.length - 1) % 8 == 0 and q.length <= args[2]
                and q.chunks[0].start == args[0] and q.chunks[-1].end == args[1]
                and all(s >= args[3] for s in q.seams()))
        ok &= good
        if not good:
            print("   bad plan", args, [(c.start, c.end) for c in q.chunks])
    check(ok, f"{len(cases)} plans: every frame covered, one 8k+1 length <= max, ends on first/last, seams >= overlap")
    check(len(ch.plan(1001, 1121, 121, 24).chunks) == 1, "exactly 121 frames: one chunk")
    q = ch.plan(1001, 1100, 121, 24)
    check(q.length == 97 and len(q.chunks) == 2, "100 frames: two 97-frame chunks (no 100-frame LTX run)",
          [(c.start, c.end) for c in q.chunks])
    raises(lambda: ch.plan(1001, 1005), "5 frames refused", "at least 9")
    raises(lambda: ch.plan(1001, 1400, 120), "max_len 120 refused (not 8k+1)", "8k+1")
    raises(lambda: ch.plan(1001, 1400, 121, 121), "overlap = length refused", "overlap")
    raises(lambda: ch.plan(1400, 1001), "last before first refused", "before")
    d = p.to_dict()
    check(ch.from_dict(d) == p, "to_dict / from_dict round trip")
    d["chunks"][1]["start"] += 1
    raises(lambda: ch.from_dict(d), "hand-edited chunk list refused", "don't match")


def test_weights():
    for args in [(1001, 1400, 121, 24), (1001, 1218, 121, 8), (1001, 1100, 121, 24), (1001, 1240, 121, 60)]:
        p = ch.plan(*args)
        ok, inside, blended = True, True, 0
        for f in range(p.first, p.last + 1):
            w = ch.weights(p, f)
            ok &= abs(sum(x for _, x in w) - 1.0) < 1e-12
            inside &= all(p.chunks[k].start <= f <= p.chunks[k].end for k, _ in w)
            blended += len(w) > 1
        check(ok and inside, f"weights {args}: sum to 1, only chunks that hold the frame", f"{blended} blended")
    p = ch.plan(1001, 1400, 121, 24)
    check(ch.weights(p, 1050) == [(0, 1.0)] and ch.weights(p, 1150) == [(1, 1.0)], "1.0 away from seams")
    w = ch.weights(p, 1098)
    check([k for k, _ in w] == [0, 1] and w[1][1] < 0.05, "seam start leans on the outgoing chunk", w)
    w = ch.weights(p, 1121)
    check(w[1][1] > 0.95, "seam end leans on the incoming chunk", w)
    mid = [ch.weights(p, f)[-1][1] for f in range(1098, 1122)]
    check(all(b >= a for a, b in zip(mid, mid[1:])), "incoming weight rises through the seam")
    tri = ch.plan(1001, 1219, 121, 24)              # last chunk one frame after the second: three on some frames
    check(max(len(ch.weights(tri, f)) for f in range(1001, 1220)) == 3, "three-chunk frames handled",
          [(c.start, c.end) for c in tri.chunks])


def write_seq(folder: Path, name: str, frames, value_of, size=(8, 4), digits=4):
    folder.mkdir(parents=True, exist_ok=True)
    for f in frames:
        img = np.dstack([np.full(size[::-1], value_of(f), np.float32)] * 3 + [np.ones(size[::-1], np.float32)])
        exr_io.write_rgba(folder / f"{name}.{f:0{digits}d}.exr", img)


def test_sequences():
    seq = TMP / "plate"
    write_seq(seq, "shot", range(1001, 1031), lambda f: f / 2000.0)
    files = ch.scan(str(seq / "shot.####.exr"))
    check(list(files) == list(range(1001, 1031)), "scan shot.####.exr: 1001-1030")
    check(ch.scan(str(seq / "shot.%04d.exr")) == files == ch.scan(str(seq)) == ch.scan(str(seq / "shot.{frame}.exr")),
          "####, %04d, {frame} and the folder agree")
    check(ch.pick(files, 1005, 9) == [files[n] for n in range(1005, 1014)], "pick 1005 + 9")
    raises(lambda: ch.pick(files, 1025, 9), "pick past the end lists the missing frames", "missing")
    (seq / "notes.0001.png").write_bytes(b"")
    raises(lambda: ch.scan(str(seq)), "folder with two sequences refused", "more than one")
    (seq / "notes.0001.png").unlink()
    node = AtlasVFXSequenceReadChunk()
    imgs, first, info = node.run(str(seq / "shot.####.exr"), 1010, 9, "frame", 0, "", "", "")
    a = imgs.numpy()
    check(a.shape == (9, 4, 8, 3) and first == 1010, "Read Chunk: 9 RGB frames from 1010", a.shape)
    check(np.allclose(a[:, 0, 0, 0], np.arange(1010, 1019) / 2000.0, atol=1e-3), "Read Chunk: frame values in order")
    imgs, _, _ = node.run(str(seq), 1001, 9, "sequential", 4, "", "", "")
    check(tuple(imgs.shape) == (9, 2, 4, 3), "Read Chunk: folder, sequential, width 4", tuple(imgs.shape))


def test_stitch():
    p = ch.plan(1001, 1060, 25, 8)            # small: chunks of 25, seams >= 8
    val = {c.index: 0.1 * (c.index + 1) for c in p.chunks}
    for c in p.chunks:                          # chunk k is a flat value, so every output frame is its weight mix
        write_seq(TMP / f"elem_{c.tag}", "elem", range(c.start, c.end + 1), lambda f, v=val[c.index]: v)
    plan_dict = AtlasVFXChunkPlan().run(1001, 1060, 25, 8, -1)[0]
    node = AtlasVFXChunkStitch()
    res = node.run(plan_dict, str(TMP / "elem_{tag}" / "elem.####.exr"), "frame", "auto", "joined", str(TMP / "out"))
    pattern, first, _ = res["result"]
    out = Path(pattern).parent
    got = sorted(int(f.name.split(".")[1]) for f in out.glob("joined.*.exr"))
    check(got == list(range(1001, 1061)) and first == 1001, "stitched 1001-1060", f"{got[0]}-{got[-1]} ({len(got)})")
    worst, exact = 0.0, True
    for f in range(1001, 1061):
        img, _ = exr_io.read_rgba(out / f"joined.{f:04d}.exr")
        want = sum(val[k] * w for k, w in ch.weights(p, f))
        worst = max(worst, float(np.abs(img[..., 0] - want).max()))
        if len(ch.weights(p, f)) == 1:
            exact &= bool(np.all(img[..., 0] == np.float16(want)))
    check(worst < 1e-3, "every frame is the weighted mix of its chunks", f"max err {worst:.2e}")
    check(exact, "frames outside seams are the chunk's own values")
    check((out / "joined.stitch.json").is_file() and (out / "joined.sidecar.json").is_file(), "stitch.json + sidecar")

    # sequential numbering: chunks saved as plain 00001-style files
    for c in p.chunks:
        write_seq(TMP / f"seq_{c.tag}", "ComfyUI", range(1, c.length + 1), lambda f, v=val[c.index]: v, digits=5)
    res = node.run(plan_dict, str(TMP / "seq_{tag}"), "sequential", "auto", "joined_seq", str(TMP / "out_seq"))
    check(len(list(Path(res["result"][0]).parent.glob("joined_seq.*.exr"))) == 60, "sequential numbering: 60 frames")

    # a missing chunk fails before anything is written
    for f in (TMP / "elem_c01").glob("*.exr"):
        f.unlink()
    raises(lambda: node.run(plan_dict, str(TMP / "elem_{tag}" / "elem.####.exr"), "frame", "auto", "broken",
                            str(TMP / "out_broken")), "missing chunk refused", "c01")
    check(not list((TMP / "out_broken").glob("*.exr")) if (TMP / "out_broken").exists() else True,
          "nothing written for the broken stitch")
    raises(lambda: node.run(plan_dict, str(TMP / "elem_c00" / "elem.####.exr"), "frame", "auto", "x", str(TMP / "o")),
           "pattern without {chunk}/{tag} refused", "{chunk}")
    raises(lambda: node.run(plan_dict, str(TMP / "elem_{tag}" / "elem.####.exr"), "frame", "auto", "x",
                            str(ROOT / "out")), "out_dir in the repo refused", "repo")
    raises(lambda: node.run(plan_dict, str(TMP / "elem_{tag}" / "elem.####.exr"), "frame", "auto", "..\\x", ""),
           "unsafe name refused", "name")


def test_plan_node_and_queue():
    plan_node = AtlasVFXChunkPlan()
    d, start, length, end, tag, n, info = plan_node.run(1001, 1400, 121, 24, 2)
    check((start, length, end, tag, n) == ([1195], [121], [1315], ["c02"], 4), "Chunk Plan chunk 2",
          (start, length, end, tag, n))
    _, start, *_ = plan_node.run(1001, 1400, 121, 24, -1)
    check(start == [1001, 1098, 1195, 1280], "Chunk Plan chunk -1: every start as a list")
    raises(lambda: plan_node.run(1001, 1400, 121, 24, 4), "chunk past the end refused", "has 4")

    import queue_chunks as qc
    graph = {"1": {"class_type": "AtlasVFXChunkPlan",
                   "inputs": {"first": 1001, "last": 1400, "max_len": 121, "overlap": 24, "chunk": 0}},
             "2": {"class_type": "AtlasVFXSequenceReadChunk", "inputs": {"path": "~/shot/shot.####.exr",
                                                                          "start": ["1", 1], "length": ["1", 2]}},
             "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "elem_{tag}/elem"}},
             "4": {"class_type": "AtlasVFXChunkStitch", "inputs": {"chunk_plan": ["1", 0],
                                                                    "pattern": "~/out/elem_{tag}"}}}
    nid = qc.find_plan(graph)
    p = qc.plan_of(graph, nid)
    g, changed = qc.per_chunk(graph, nid, p.chunks[3])
    check(g["1"]["inputs"]["chunk"] == 3 and g["3"]["inputs"]["filename_prefix"] == "elem_c03/elem",
          "queue_chunks: chunk 3 sets chunk and the {tag} name", changed)
    check("4" not in g and graph["4"]["inputs"]["pattern"] == "~/out/elem_{tag}", "queue_chunks: stitch left out")


if __name__ == "__main__":
    test_plan()
    test_weights()
    test_sequences()
    test_stitch()
    test_plan_node_and_queue()
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all chunk checks passed'}")
    sys.exit(1 if FAILS else 0)
