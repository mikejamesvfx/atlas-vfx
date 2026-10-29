"""Character Insert time map: hold, ping-pong and loop past the clip's end, with a clip start.

Run: <V141 python> tests\\test_character_time.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import character_insert as ci  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def seq(n_tex, n_plate, start, mode):
    m = ci.time_map(n_tex, list(range(1001, 1001 + n_plate)), start, mode)
    return [m[str(1001 + i)] for i in range(n_plate)]


def test_modes():
    check(seq(4, 7, 0, "hold") == [0, 1, 2, 3, 3, 3, 3], "hold keeps the last frame")
    check(seq(4, 9, 0, "pingpong") == [0, 1, 2, 3, 2, 1, 0, 1, 2], "pingpong plays back and forth")
    check(seq(4, 9, 0, "loop") == [0, 1, 2, 3, 0, 1, 2, 3, 0], "loop wraps to the first frame")
    check(seq(6, 9, 2, "loop") == [2, 3, 4, 5, 2, 3, 4, 5, 2], "loop wraps to clip_start, not 0")
    check(seq(3, 4, 9, "loop") == [2, 2, 2, 2], "clip_start past the end is clamped (no crash)")
    check(seq(1, 3, 0, "loop") == [0, 0, 0], "single-frame clip holds")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
