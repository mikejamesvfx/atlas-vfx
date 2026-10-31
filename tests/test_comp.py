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


def test_beauty_and_preview():
    from atlas_vfx.comp import shot as sh

    sh.add_layer("S2S3", "rl", "recipe", recipe="relight")
    v = sh.next_version("S2S3", "rl")
    d = sh.layer_dir("S2S3", "rl") / v
    write_seq(d / "beauty_gain", "rl_gain", [1130])
    write_seq(d / "ltx", "rl", [1130])
    write_seq(d / "beauty", "rl", [1130])
    pat = sh.layer_pattern(sh.load_shot("S2S3"), [x for x in sh.load_shot("S2S3")["layers"] if x["name"] == "rl"][0])
    check(pat is not None and Path(pat).parent.name == "beauty", "versions: a relight shows its beauty folder", str(pat))
    raises(lambda: sh.update_layer("S2S3", "rl", version="preview"), "preview: none yet", "preview")
    write_seq(sh.preview_dir("S2S3", "rl"), "rl", [1130])
    s = sh.load_shot("S2S3")
    lay = [x for x in s["layers"] if x["name"] == "rl"][0]
    check(lay["has_preview"] and lay["versions"] == ["v001"], "preview: flagged, not counted as a version")
    lay = sh.update_layer("S2S3", "rl", version="preview")
    check("preview" in sh.layer_pattern(sh.load_shot("S2S3"), lay), "preview: the layer can show its preview")


def _links_ok(graph: dict) -> bool:
    for node in graph.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) and str(v[0]) not in graph:
                return False
    return True


def test_recipes_listed():
    from atlas_vfx.comp import recipes as rc

    names = {r["name"] for r in rc.list_recipes()}
    check(names == {"matte", "relight", "sky"}, "recipes: matte, relight, sky", str(names))
    check(all("graph" not in r for r in rc.list_recipes()), "recipes: the list leaves the graphs out")
    bad = []
    for n in names:
        r = rc.load_recipe(n)
        targets = [t for f in r["fields"] for t in f["targets"]] + [t for ts in r["context"].values() for t in ts]
        targets += r["outputs"] + r.get("names", [])
        bad += [(n, t) for t in targets if str(t[0]) not in r["graph"] or t[1] not in r["graph"][str(t[0])]["inputs"]]
    check(not bad, "recipes: every target is a real node input", str(bad[:3]))
    raises(lambda: rc.load_recipe("../shot"), "recipes: an unknown name", "recipe")


def test_validate():
    from atlas_vfx.comp import recipes as rc

    r = rc.load_recipe("relight")
    raises(lambda: rc.validate(r, {"direction": "../x"}), "validate: a choice outside the list", "direction")
    raises(lambda: rc.validate(r, {"seed": "abc"}), "validate: a seed that is not a number", "seed")
    raises(lambda: rc.validate(r, {"zzz": 1}), "validate: an unknown field", "zzz")
    d = rc.validate(r, {})
    check(d["direction"] == "the left" and d["height"] == "low", "validate: defaults filled in", str(d)[:80])


def _relight_context():
    from atlas_vfx.comp import recipes as rc
    from atlas_vfx.comp import shot as sh

    pat = write_seq(pl.WORK_ROOT / "renders" / "el", "el", [1130])
    if "el" not in [x["name"] for x in sh.load_shot("S2S3")["layers"]]:
        sh.add_layer("S2S3", "el", "sequence", source=pat)
        sh.add_layer("S2S3", "relit", "recipe", recipe="relight", params={"element": "el"})
    s = sh.load_shot("S2S3")
    lay = [x for x in s["layers"] if x["name"] == "relit"][0]
    return rc, sh, s, lay


def test_fill_render_vs_preview():
    rc, sh, s, lay = _relight_context()
    r = rc.load_recipe("relight")
    params = rc.validate(r, {"element": "el"})
    ctx = rc.context_for(s, lay)
    check(ctx["first_frame"] == 1130 and ctx["length"] == 97 and ctx["plate_first"] == 129, "context: shot range",
          str({k: ctx[k] for k in ("first_frame", "length", "plate_first")}))
    out = str(sh.layer_dir("S2S3", "relit") / "v009")
    g = rc.fill(r, params, ctx, out, "render")
    outs = [g[str(n)]["inputs"][i] for n, i, _ in r["outputs"]]
    check(all(o.startswith(out) for o in outs), "fill: render writes into the version folder", str(outs))
    check(any(n["class_type"] == "AtlasVFXRelightTransferFiles" for n in g.values()) and _links_ok(g),
          "fill: render keeps the 4K transfer, links intact")
    el_targets = [g[str(t[0])]["inputs"][t[1]] for f in r["fields"] if f["key"] == "element" for t in f["targets"]]
    check(el_targets and all("el" in str(v) for v in el_targets), "fill: the element layer resolves to its files",
          str(el_targets)[:120])
    p = rc.fill(r, params, ctx, out, "preview")
    check(not any(n["class_type"] == "AtlasVFXRelightTransferFiles" for n in p.values()) and _links_ok(p),
          "fill: preview drops the 4K transfer, links intact")
    check(p["54"]["inputs"]["out_dir"] == out, "fill: a relight preview writes LTX Return's layer at the folder root",
          p["54"]["inputs"]["out_dir"])
    check(r["graph"] != g and "57" in r["graph"], "fill: the recipe itself is not changed")


def test_two_renders_two_versions():
    rc, sh, s, lay = _relight_context()
    r = rc.load_recipe("relight")
    params = rc.validate(r, {"element": "el"})
    outs = []
    for _ in range(2):
        v = sh.next_version("S2S3", "relit")
        g = rc.fill(r, params, rc.context_for(s, lay), str(sh.layer_dir("S2S3", "relit") / v), "render")
        outs.append(g[str(r["outputs"][0][0])]["inputs"][r["outputs"][0][1]])
    check(outs[0] != outs[1] and "v001" in outs[0] and "v002" in outs[1], "versions: two renders, two folders",
          str(outs))


def main():
    for t in (test_shot_roundtrip, test_names_are_safe, test_layers, test_versions, test_beauty_and_preview,
              test_recipes_listed,
              test_validate, test_fill_render_vs_preview, test_two_renders_two_versions):
        t()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
