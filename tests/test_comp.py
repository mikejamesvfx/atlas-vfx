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
    check(lay["versions"] == [] and lay["empty_versions"] == ["v001", "v002"], "versions: scanned from disk",
          f"{lay['versions']} {lay['empty_versions']}")
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


def _frames_shot():
    from atlas_vfx.comp import shot as sh

    if "F" not in sh.list_shots():
        sh.create_shot("F", "", 1130, 1140)
        r = pl.WORK_ROOT / "renders"
        sh.add_layer("F", "grey", "sequence", source=write_seq(r / "fgrey", "g", [1130, 1131], (0.2, 0.2, 0.2)))
        sh.add_layer("F", "red", "sequence", source=write_seq(r / "fred", "r", [1130], (1.0, 0.0, 0.0), alpha=0.5))
    return sh


def test_blends():
    from atlas_vfx.comp import frames as fr

    sh = _frames_shot()
    s = sh.load_shot("F")
    img, used = fr.composite(s, 1130, width=32)
    check(used == ["grey", "red"] and np.allclose(img[5, 5, :3], [0.6, 0.1, 0.1], atol=2e-3),
          "blend: over (half red over grey)", f"{used} {img[5, 5, :3]}")
    want = {"plus": [0.7, 0.2, 0.2], "screen": [0.6, 0.2, 0.2], "multiply": [0.2, 0.1, 0.1]}
    for mode, v in want.items():
        sh.update_layer("F", "red", blend=mode)
        img, _ = fr.composite(sh.load_shot("F"), 1130, width=32)
        check(np.allclose(img[5, 5, :3], v, atol=3e-3), f"blend: {mode}", str(img[5, 5, :3]))
    sh.update_layer("F", "red", blend="over", opacity=0.5)
    img, _ = fr.composite(sh.load_shot("F"), 1130, width=32)
    check(np.allclose(img[5, 5, :3], [0.4, 0.15, 0.15], atol=3e-3), "blend: opacity 0.5 halves the layer",
          str(img[5, 5, :3]))
    sh.update_layer("F", "red", opacity=1.0, visible=False)
    _, used = fr.composite(sh.load_shot("F"), 1130, width=32)
    check(used == ["grey"], "blend: a hidden layer is left out")
    sh.update_layer("F", "red", visible=True)


def test_skips_missing():
    from atlas_vfx.comp import frames as fr

    sh = _frames_shot()
    img, used = fr.composite(sh.load_shot("F"), 1131, width=32)
    check(used == ["grey"] and np.allclose(img[5, 5, :3], 0.2, atol=2e-3), "frames: a layer without the frame is skipped",
          str(used))
    img, used = fr.composite(sh.load_shot("F"), 1139, width=32)
    check(used == [] and img.shape == (16, 32, 4), "frames: nothing at that frame gives an empty frame, no error",
          f"{used} {img.shape}")


def test_jpeg():
    import cv2
    from atlas_vfx.comp import frames as fr

    sh = _frames_shot()
    s = sh.load_shot("F")
    j = fr.display_jpeg(s, 1130, width=1280)
    im = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
    check(j[:2] == bytes([0xFF, 0xD8]) and im.shape[1] == 1280, "jpeg: a JPEG at the asked width", str(im.shape))
    w = cv2.imdecode(np.frombuffer(fr.display_jpeg(s, 1130, width=1280, a=["grey"], b=["red", "grey"], wipe=0.25),
                                   np.uint8), cv2.IMREAD_COLOR)
    check(w[300, 200, 2] < w[300, 500, 2] - 20 and abs(int(w[300, 200, 2]) - int(w[300, 200, 1])) < 4,
          "jpeg: the wipe shows A left of 25% and B right of it", f"{w[300, 200]} {w[300, 500]}")


def test_plate_proxy():
    from atlas_vfx.comp import frames as fr
    from atlas_vfx.comp import shot as sh

    fake = pl.WORK_ROOT / "fake_plate.mov"
    fake.write_bytes(b"not a movie")
    sh.create_shot("P", str(fake), 1130, 1131)
    s = sh.load_shot("P")
    proxy = fr.proxy_dir(s, 32)
    write_seq(proxy, "plate", [1130, 1131], (0.3, 0.3, 0.3), size=(16, 32))
    img, used = fr.composite(s, 1130, width=32)
    check(np.allclose(img[2, 2, :3], 0.3, atol=2e-3) and img[2, 2, 3] == 1.0,
          "plate: read from the proxy, never the movie, once it exists", str(img[2, 2]))


def test_no_second_proxy_build():
    from atlas_vfx.comp import frames as fr
    from atlas_vfx.comp import shot as sh

    fake = pl.WORK_ROOT / "fake2.mov"
    fake.write_bytes(b"not a movie")
    sh.create_shot("Q", str(fake), 1130, 1131)
    s = sh.load_shot("Q")
    fr.BUILDING.add("Q")
    try:
        img, used = fr.composite(s, 1130, width=32)          # would try to decode the fake movie and raise
        check(used == [] and not fr.proxy_dir(s).exists(), "plate: no second build while one is running")
    except Exception as e:  # noqa: BLE001
        check(False, "plate: no second build while one is running", repr(e)[:100])
    finally:
        fr.BUILDING.discard("Q")


def test_proxy_failure_and_partial():
    from atlas_vfx.comp import frames as fr
    from atlas_vfx.comp import shot as sh

    bad = pl.WORK_ROOT / "bad_plate.mov"
    bad.write_bytes(b"not a movie")
    sh.create_shot("BAD", str(bad), 1130, 1131)
    r = pl.WORK_ROOT / "renders"
    sh.add_layer("BAD", "grey", "sequence", source=write_seq(r / "bgrey", "g", [1130], (0.2, 0.2, 0.2)))
    s = sh.load_shot("BAD")
    try:
        img, used = fr.composite(s, 1130, width=32)
        check(used == ["grey"], "proxy: a plate that will not decode never breaks the layers", str(used))
    except Exception as e:  # noqa: BLE001
        check(False, "proxy: a plate that will not decode never breaks the layers", repr(e)[:120])
    try:
        fr.build_plate_proxy(s)
        check(False, "proxy: a failed build raises", "no error")
    except Exception as e:  # noqa: BLE001
        check(fr.proxy_state(s)[0] == "error" and str(e)[:20] in fr.proxy_state(s)[1],
              "proxy: a failed build is remembered as an error with its message", str(fr.proxy_state(s)))
    check(not fr.proxy_dir(s).exists(), "proxy: a failed build leaves no folder that looks ready")
    sh.create_shot("PART", str(bad), 1130, 1131)
    s2 = sh.load_shot("PART")
    tmp = fr.proxy_dir(s2).with_name(fr.proxy_dir(s2).name + ".tmp")
    write_seq(tmp, "plate", [1130], (0.3, 0.3, 0.3))
    check(fr.proxy_state(s2)[0] == "missing", "proxy: a half-built proxy (interrupted) is not ready",
          str(fr.proxy_state(s2)))


def test_latest_skips_empty_versions():
    from atlas_vfx.comp import shot as sh

    sh.add_layer("S2S3", "lat", "recipe", recipe="matte")
    v1 = sh.next_version("S2S3", "lat")
    write_seq(sh.layer_dir("S2S3", "lat") / v1, "lat", [1130])
    sh.next_version("S2S3", "lat")                                    # a render that failed or was cancelled
    s = sh.load_shot("S2S3")
    lay = [x for x in s["layers"] if x["name"] == "lat"][0]
    pat = sh.layer_pattern(s, lay)
    check(pat is not None and "v001" in pat, "versions: latest is the newest version with frames, not an empty one",
          str(pat))
    check(lay["versions"] == ["v001"] and lay["empty_versions"] == ["v002"],
          "versions: empty versions are listed apart", f"{lay['versions']} {lay.get('empty_versions')}")


def test_light_mix():
    from atlas_vfx.comp import frames as fr
    from atlas_vfx.comp import shot as sh

    sh.create_shot("MIX", "", 1130, 1131)
    sh.add_layer("MIX", "rl", "recipe", recipe="relight")
    d = sh.layer_dir("MIX", "rl")
    for v, val in (("v001", (0.8, 0.4, 0.1)), ("v002", (0.1, 0.2, 0.6))):
        assert sh.next_version("MIX", "rl") == v
        write_seq(d / v / "beauty", "rl", [1130], val)
    lay = sh.update_layer("MIX", "rl", mix={"v001": 0.7, "v002": 0.3})
    img, used = fr.composite(sh.load_shot("MIX"), 1130, width=32)
    want = 0.7 * np.array([0.8, 0.4, 0.1]) + 0.3 * np.array([0.1, 0.2, 0.6])
    check(used == ["rl"] and np.allclose(img[3, 3, :3], want, atol=3e-3) and abs(img[3, 3, 3] - 1.0) < 1e-3,
          "mix: the layer is the weighted sum of its versions in linear", f"{img[3, 3, :3]} want {want}")
    sh.update_layer("MIX", "rl", mix={"v001": 1.0, "v002": 1.0})
    img, _ = fr.composite(sh.load_shot("MIX"), 1130, width=32)
    check(np.allclose(img[3, 3, :3], [0.9, 0.6, 0.7], atol=3e-3), "mix: weights add like lights (no normalising)",
          str(img[3, 3, :3]))
    img, used = fr.composite(sh.load_shot("MIX"), 1131, width=32)
    check(used == [], "mix: a frame no version has is skipped, no error")
    raises(lambda: sh.update_layer("MIX", "rl", mix={"v009": 1.0}), "mix: an unknown version refused", "v009")
    raises(lambda: sh.update_layer("MIX", "rl", mix={"v001": 5.0}), "mix: a weight outside 0-2 refused", "0-2")
    raises(lambda: sh.update_layer("MIX", "rl", mix=[1, 2]), "mix: must be version -> weight", "mix")
    lay = sh.update_layer("MIX", "rl", mix={})
    img, _ = fr.composite(sh.load_shot("MIX"), 1130, width=32)
    check(np.allclose(img[3, 3, :3], [0.1, 0.2, 0.6], atol=3e-3), "mix: empty mix goes back to the chosen version")


def main():
    for t in (test_shot_roundtrip, test_names_are_safe, test_layers, test_versions, test_beauty_and_preview,
              test_recipes_listed,
              test_validate, test_fill_render_vs_preview, test_two_renders_two_versions, test_blends,
              test_skips_missing, test_jpeg, test_plate_proxy,
              test_no_second_proxy_build, test_proxy_failure_and_partial,
              test_latest_skips_empty_versions, test_light_mix):
        t()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
