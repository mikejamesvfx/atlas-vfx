"""Comp Preview (deliver/comp_preview.py): folder parsing, layer discovery, back-to-front premultiplied over, burn-in.

Run: <V141 python> tests\\test_comp_preview.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.deliver import comp_preview as cpv  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def test_parse_and_discover():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a, b = d / "DSCF_abc_element", d / "DSCF_abc_grass"
        a.mkdir()
        b.mkdir()
        rgba = np.zeros((4, 6, 4), np.float32)
        for f in (1130, 1131):
            exr_io.write_rgba(a / f"element.{f:04d}.exr", rgba)
        exr_io.write_rgba(a / "element.tex.0000.exr", rgba)          # element texture, must be ignored
        (a / "element.render.json").write_text(json.dumps({"card": "element"}))
        exr_io.write_rgba(b / "grass_far.1131.exr", rgba)
        text = f'\n# back to front\n"{a}"\n{b}\n{a}\n'
        folders = cpv.parse_folders(text)
        check(folders == [a, b], "folders: quotes, comments, blanks and duplicates handled", [f.name for f in folders])
        n1, fr1 = cpv.layer_frames(a)
        n2, fr2 = cpv.layer_frames(b)
        check(n1 == "element" and sorted(fr1) == [1130, 1131], "layer frames from render.json name, texture files ignored")
        check(n2 == "grass_far" and sorted(fr2) == [1131], "layer frames without a render.json")
        try:
            cpv.layer_frames(d / "missing")
            check(False, "missing folder is a clear error")
        except FileNotFoundError:
            check(True, "missing folder is a clear error")


def test_over_order():
    bg = np.full((2, 2, 3), 0.2, np.float32)
    back = np.zeros((2, 2, 4), np.float32)
    back[...] = [0.5, 0.0, 0.0, 1.0]                          # opaque red behind
    front = np.zeros((2, 2, 4), np.float32)
    front[...] = [0.0, 0.25, 0.0, 0.5]                        # half green in front (premultiplied)
    out = cpv.comp_frame(bg, [back, None, front])
    check(np.allclose(out, [0.25, 0.25, 0.0]), "back to front premultiplied over, None skipped", out[0, 0].tolist())
    img = np.full((200, 300, 3), 0.5, np.float32)
    cpv._burn(img, "1130")
    check(img.dtype == np.float32 and img[150:].std() == 0 and img[:40].std() > 0,
          "burn-in touches only the top strip and stays float")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
