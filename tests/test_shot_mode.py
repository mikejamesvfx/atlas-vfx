"""Shot mode (owner call 2026-11-01): the choice before anything else, background plate / foreground on a screen /
create both. layers/shot_mode.py, Shot Setup (catalogue 70), the virtual plate in camera_side/plate.py, and
Atlas Comp's shot.json mode.

  1. virtual_plate_ref: size, fps (23.976 typed as a decimal -> 24000/1001), frame range; refuses odd sizes.
  2. A virtual plate refuses decode with a clear message; decode_rgb_or_black gives black frames of the right size;
     plate_stem names folders after it.
  3. Shot Setup, each mode: generate (virtual ref, black preview, record), plate and foreground (a synthetic
     10-bit plate; foreground carries the screen colour); wrong inputs are refused.
  4. Atlas Comp: create_shot per mode writes mode (and screen / size / fps); a generate shot refuses a plate, a
     foreground shot needs one; an old shot.json without a mode reads as plate.
  5. Recipes list the modes they are offered in; list_recipes(mode) filters by it.
Run: <V141 python> tests/test_shot_mode.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="shotmode_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

from atlas_vfx.comp import shot as sh  # noqa: E402
from atlas_vfx.layers import shot_mode as sm  # noqa: E402
from atlas_vfx.layers.node_shot_setup import AtlasVFXShotSetup  # noqa: E402

FAILS: list[str] = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def raises(fn, label, needle="", exc=(ValueError, pl.PlateError)):
    try:
        fn()
    except exc as e:
        check(needle in str(e), label, str(e)[:110])
        return
    check(False, label, "no error")


def test_virtual():
    ref = pl.virtual_plate_ref(4096, 2160, "23.976", 121)
    check(pl.is_virtual(ref) and ref["fps"] == [24000, 1001] and (ref["nuke_first"], ref["nuke_last"]) == (1001, 1121),
          "virtual plate: 4096x2160, 23.976 -> 24000/1001, Nuke 1001-1121", f"{ref['fps']}")
    check(ref["colorspace"] == ref["working_space"] == "ACEScg" and ref["path"] == "", "virtual plate is in ACEScg, no file")
    check(pl.virtual_plate_ref(1920, 1080, [25, 1], 10, 2, 5)["nuke_first"] == 1003, "frame range and offset")
    raises(lambda: pl.virtual_plate_ref(1921, 1080), "odd size refused", "even")
    raises(lambda: pl.virtual_plate_ref(1920, 1080, 24, 5, 2, 9), "range outside the frames refused", "outside")
    raises(lambda: next(pl.decode_rgb(ref, 1001, 1001)), "decode refuses a virtual plate", "virtual")
    frames = list(pl.decode_rgb_or_black(ref, 0, 16, width=512, step=8))
    check([i for i, _ in frames] == [0, 8, 16] and frames[0][1].shape == (270, 512, 3) and not frames[0][1].any(),
          "decode_rgb_or_black: black frames at the proxy size")
    check(np.array_equal(pl.to_working(frames[0][1], ref), frames[0][1]), "to_working leaves a virtual plate black")
    check(pl.plate_stem(ref) == "virtual" and pl.plate_stem(pl.virtual_plate_ref(64, 64, name="dune")) == "dune",
          "plate_stem names folders after the virtual plate")


def test_node():
    import test_plate as tp

    node = AtlasVFXShotSetup()
    common = dict(first=0, last=-1, screen="green", width=1920, height=1080, fps="25", frames=48, ocio_config="",
                  input_space="auto", frame_offset=1001, preview_width=256, preview_every=8)
    ref, shot, prev, info = node.run(mode="generate", path="", **common)
    check(pl.is_virtual(ref) and shot["mode"] == "generate" and shot["virtual_plate"] and "set_build" in shot["routes"],
          "Shot Setup generate: virtual plate, record with its routes", shot["routes"])
    check(tuple(prev.shape) == (1, 144, 256, 3) and float(prev.abs().max()) == 0.0, "generate preview is black")
    check(json.loads(info)["frames"] == [1001, 1048], "info: Nuke frames 1001-1048")
    raises(lambda: node.run(mode="generate", path="x.mov", **common), "generate with a path refused", "no plate")
    raises(lambda: node.run(mode="plate", path="", **common), "plate mode without a path refused", "needs a plate")
    mov = tp.make_plate(TMP / "plates" / "screen.mov", "hevc", "pc")
    for mode in ("plate", "foreground"):
        kw = dict(common, screen="blue", input_space="F-Log2 F-Gamut")
        ref, shot, prev, _ = node.run(mode=mode, path=str(mov), **kw)
        ok = (not pl.is_virtual(ref) and ref["width"] == tp.W and shot["mode"] == mode and prev.shape[0] >= 1
              and (shot.get("screen") == "blue") == (mode == "foreground"))
        check(ok, f"Shot Setup {mode}: real plate_ref, preview, screen colour only on foreground", shot.get("screen"))
    raises(lambda: sm.record("studio", ref), "unknown mode refused", "one of")
    raises(lambda: sm.record("foreground", ref, "red"), "unknown screen colour refused", "one of")


def test_comp_shot():
    import test_plate as tp

    mov = tp.make_plate(TMP / "plates" / "fg.mov", "hevc", "pc")
    s = sh.create_shot("GEN", "", 1001, 1096, mode="generate", width=3840, height=2160, fps="24")
    check(s["mode"] == "generate" and s["size"] == [3840, 2160] and s["fps"] == [24, 1], "comp: generate shot",
          f"{s.get('size')} {s.get('fps')}")
    s = sh.create_shot("FG", str(mov), 1001, 1003, mode="foreground", screen="blue")
    check(s["mode"] == "foreground" and s["screen"] == "blue", "comp: foreground shot keeps its screen colour")
    s = sh.create_shot("BG", "", 1001, 1010)
    check(s["mode"] == "plate", "comp: default mode is plate (blank plate allowed, as before)")
    raises(lambda: sh.create_shot("G2", str(mov), 1001, 1010, mode="generate", width=64, height=64),
           "comp: generate refuses a plate", "no plate")
    raises(lambda: sh.create_shot("F2", "", 1001, 1010, mode="foreground"), "comp: foreground needs a plate",
           "screen plate")
    raises(lambda: sh.create_shot("G3", "", 1001, 1010, mode="generate", width=63, height=64),
           "comp: generate checks the virtual plate size", "even")
    p = sh._file("BG")
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("mode")
    p.write_text(json.dumps(d), encoding="utf-8")
    check(sh.load_shot("BG")["mode"] == "plate", "comp: an old shot.json without a mode reads as plate")


def test_recipes():
    from atlas_vfx.comp import recipes as rc

    every = {r["name"] for r in rc.list_recipes()}
    gen = {r["name"] for r in rc.list_recipes("generate")}
    check(every and gen <= every and all("modes" in r for r in rc.list_recipes()),
          "recipes list their modes; a mode's list is a subset", sorted(gen))
    check("sky" in gen, "the sky recipe is offered when creating both")
    raises(lambda: rc.list_recipes("studio"), "recipes: unknown mode refused", "one of")


def main() -> int:
    test_virtual()
    test_node()
    test_comp_shot()
    test_recipes()
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
