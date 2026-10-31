"""Atlas Comp server core (atlas_vfx/comp): the shot store on disk, recipes, frame compositing.

Everything runs in a temp work root, so nothing touches ~/atlas-vfx-work.

Run: <V141 python> tests\\test_comp.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="comp_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def raises(fn, label, needle=""):
    try:
        fn()
    except ValueError as e:
        check(needle in str(e), label, str(e)[:100])
        return
    check(False, label, "no error")


def write_seq(folder: Path, stem: str, frames, value=(0.5, 0.5, 0.5), alpha=1.0, size=(16, 32)):
    from atlas_vfx.camera_side import exr_io

    h, w = size
    for n in frames:
        img = np.zeros((h, w, 4), np.float32)
        img[..., :3] = np.asarray(value, np.float32) * alpha
        img[..., 3] = alpha
        exr_io.write_rgba(folder / f"{stem}.{n}.exr", img)
    return str(folder / f"{stem}.####.exr")


def test_shot_roundtrip():
    from atlas_vfx.comp import shot as sh

    s = sh.create_shot("S2S3", "", 1130, 1226)
    check(s["first"] == 1130 and s["layers"] == [], "shot: created")
    check(sh.list_shots() == ["S2S3"], "shot: listed", str(sh.list_shots()))
    check(sh.load_shot("S2S3")["last"] == 1226, "shot: loads from disk")
    raises(lambda: sh.create_shot("S2S3", "", 1130, 1226), "shot: no second create over the first", "exists")
    raises(lambda: sh.load_shot("nope"), "shot: unknown shot", "nope")


def test_names_are_safe():
    from atlas_vfx.comp import shot as sh

    try:
        s = sh.create_shot("../evil", "", 1001, 1010)
        inside = (sh.root() / s["name"]).resolve()
        check(sh.root().resolve() in inside.parents, "names: a hostile shot name lands inside the comp root", s["name"])
    except ValueError as e:
        check(True, "names: a hostile shot name is refused", str(e)[:80])
    outside = TMP / "outside"
    outside.mkdir()
    pat = write_seq(outside, "x", [1130])
    raises(lambda: sh.add_layer("S2S3", "bad", "sequence", source=pat), "names: a source outside the work root",
           "outside")


def test_layers():
    from atlas_vfx.comp import shot as sh

    pat = write_seq(pl.WORK_ROOT / "renders" / "grey", "grey", [1130, 1131])
    lay = sh.add_layer("S2S3", "grey", "sequence", source=pat)
    check(lay["blend"] == "over" and lay["opacity"] == 1.0 and lay["visible"], "layers: defaults")
    sh.update_layer("S2S3", "grey", opacity=0.5, blend="screen")
    got = [x for x in sh.load_shot("S2S3")["layers"] if x["name"] == "grey"][0]
    check(got["opacity"] == 0.5 and got["blend"] == "screen", "layers: changes are saved")
    raises(lambda: sh.update_layer("S2S3", "grey", blend="dodge"), "layers: unknown blend", "blend")
    raises(lambda: sh.update_layer("S2S3", "grey", colour="x"), "layers: unknown key", "colour")
    raises(lambda: sh.update_layer("S2S3", "grey", opacity=2), "layers: opacity outside 0-1", "opacity")
    raises(lambda: sh.add_layer("S2S3", "grey", "sequence", source=pat), "layers: no duplicate name", "exists")
    check(sh.layer_pattern(sh.load_shot("S2S3"), got) == pat, "layers: a sequence layer's pattern is its source")


def test_versions():
    from atlas_vfx.comp import shot as sh

    sh.add_layer("S2S3", "matte", "recipe", recipe="matte", params={})
    check([sh.next_version("S2S3", "matte"), sh.next_version("S2S3", "matte")] == ["v001", "v002"],
          "versions: two renders, two versions")
    s = sh.load_shot("S2S3")
    lay = [x for x in s["layers"] if x["name"] == "matte"][0]
    check(lay["versions"] == ["v001", "v002"], "versions: scanned from disk", str(lay["versions"]))
    check(sh.layer_pattern(s, lay) is None, "versions: nothing rendered yet")
    write_seq(sh.layer_dir("S2S3", "matte") / "v002", "matte", [1130])
    lay = sh.update_layer("S2S3", "matte", version="v002")
    pat = sh.layer_pattern(sh.load_shot("S2S3"), lay)
    check(pat is not None and pat.endswith("matte.####.exr") and "v002" in pat, "versions: the chosen version's EXRs",
          str(pat))
    raises(lambda: sh.update_layer("S2S3", "matte", version="v999"), "versions: unknown version refused", "v999")


def main():
    for t in (test_shot_roundtrip, test_names_are_safe, test_layers, test_versions):
        t()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
