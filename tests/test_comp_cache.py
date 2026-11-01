"""Light room frame cache (comp/cache.py; Light Mix plan phase 2).

  1. A take frame round trips to float16 rounding at both levels (proxy 1024 wide, full size).
  2. A changed source (new mtime / size) is remade; an unchanged one is served from the file.
  3. Names that would leave the cache folder are refused.
  4. warm builds every frame of the mixed takes in the background.
  5. under / over: the plate and lower layers / higher layers at the takes' size, equal to the comp's own blend.
  6. view_shader: OCIO's GLSL ES 3.0 for the default ACES 2.0 view, with its lookup tables.
  7. The size budget: past it, least recently used frames go, full-size first; only cache files; clear() empties a
     shot's cache and leaves everything else.
Run: <V141 python> tests/test_comp_cache.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="compcache_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

from atlas_vfx.comp import cache as cc  # noqa: E402
from atlas_vfx.comp import frames as fr  # noqa: E402
from atlas_vfx.comp import shot as sh  # noqa: E402

FAILS: list[str] = []


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


def ramp(h, w, k):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    img = np.zeros((h, w, 4), np.float32)
    img[..., 0] = xx / w * (1 + k)
    img[..., 1] = yy / h * 4.0                    # up to 4: above display white, as relit takes are
    img[..., 2] = 0.1 * k
    img[..., 3] = 1.0
    return img


def setup():
    sh.create_shot("C", "", 1001, 1003)
    sh.add_layer("C", "bg", "recipe", recipe="sky")
    sh.add_layer("C", "rl", "recipe", recipe="relight")
    sh.add_layer("C", "fg", "recipe", recipe="matte")
    H, W = 270, 2048
    for layer, n in (("bg", 1), ("rl", 3), ("fg", 1)):
        for k in range(n):
            v = sh.next_version("C", layer)
            for f in (1001, 1002, 1003):
                img = ramp(H, W, k)
                if layer == "fg":
                    img[..., :] = 0
                    img[100:150, 300:900] = [0.2, 0.1, 0.05, 0.5]
                exr_io.write_rgba(sh.layer_dir("C", layer) / v / f"{layer}.{f:04d}.exr", img, half=False)
    sh.update_layer("C", "rl", mix={"mode": "blend", "takes": {"v001": {"weight": 0.5}, "v002": {"weight": 0.5},
                                                              "v003": {"weight": 0.0}}})
    return H, W


def test_takes(H, W):
    cc.PROXY_W = 1024                                              # the checks below are written at 1024
    src = ramp(H, W, 1)
    cap = cc.FULL_MAX_W                                            # the paused frame is capped at HD by default
    hd, hd_head = cc.decode(cc.take_bytes("C", "rl", "v002", "full", 1002))
    check(hd.shape[1] == min(W, cap) and np.allclose(hd, fr._resize(src, cc.size_for(W, H, "full")), atol=2e-3),
          "full level is capped at FULL_MAX_W wide", f"{hd.shape}")
    cc.FULL_MAX_W = 0                                              # 0: the source size
    full = cc.decode(cc.take_bytes("C", "rl", "v002", "full", 1002))
    check(full[0].shape == (H, W, 4), "a smaller cap's cache file is remade at the new size", f"{full[0].shape}")
    cc.FULL_MAX_W = cap
    check(full[0].shape == (H, W, 4) and np.allclose(full[0], src, rtol=1e-3, atol=1e-3),
          "full level round trips to float16 rounding", f"max {np.abs(full[0] - src).max():.1e}")
    px, head = cc.decode(cc.take_bytes("C", "rl", "v002", "proxy", 1002))
    check(px.shape == (136, 1024, 4) and head["level"] == "proxy" and np.allclose(px, fr._resize(src, (1024, 136)),
                                                                              atol=2e-3),
          "proxy level is 1024 wide, the area-resized take", str(px.shape))
    reads = []
    real = cc.vw.read_frame
    cc.vw.read_frame = lambda *a: reads.append(a) or real(*a)
    try:
        cc.take_bytes("C", "rl", "v002", "proxy", 1002)
    finally:
        cc.vw.read_frame = real
    check(not reads, "an unchanged source is served from the cache file (no EXR read)")
    exr = sh.layer_dir("C", "rl") / "v002" / "rl.1002.exr"
    exr_io.write_rgba(exr, ramp(H, W, 5), half=False)
    os.utime(exr, None)
    again = cc.decode(cc.take_bytes("C", "rl", "v002", "proxy", 1002))[0]
    check(not np.allclose(again, px, atol=1e-3), "a changed source is remade")
    exr_io.write_rgba(exr, src, half=False)
    raises(lambda: cc.take_file("C", "../x", "v001", "proxy", 1001), "a layer name with .. refused", "")
    raises(lambda: cc.take_bytes("C", "rl", "v009", "proxy", 1001), "an unrendered version refused", "not rendered")
    raises(lambda: cc.take_bytes("C", "rl", "v001", "huge", 1001), "an unknown level refused", "level")
    check(cc.take_bytes("C", "rl", "v001", "proxy", 999) is None, "a frame the take does not have: None")


def test_warm():
    st = cc.warm("C", "rl", "proxy")
    t0 = time.time()
    while cc.status("C", "rl", "proxy")["state"] == "running" and time.time() - t0 < 60:
        time.sleep(0.05)
    st = cc.status("C", "rl", "proxy")
    files = list((cc.cache_root("C") / "rl").rglob("proxy/*.f16"))
    check(st["state"] == "done" and st["of"] == 9 and len(files) == 9, "warm builds every frame of the three takes",
          f"{st} {len(files)} files")


def test_stack(H, W):
    size = (1024, 136)
    under = cc.decode(cc.stack_bytes("C", "rl", "under", "proxy", 1001, size))[0]
    over = cc.decode(cc.stack_bytes("C", "rl", "over", "proxy", 1001, size))[0]
    bg = fr._resize(ramp(H, W, 0), size)
    check(np.allclose(under, bg, atol=2e-3), "under = the lower layer at the takes' size")
    fg = cc.decode(cc.take_bytes("C", "fg", "v001", "proxy", 1001))[0]
    check(np.allclose(over, fr.blend(np.zeros_like(fg), fg, "over", 1.0), atol=2e-3), "over = the higher layer")
    k1 = cc.stack_key(sh.load_shot("C"), "rl", "under", "proxy")
    sh.update_layer("C", "bg", opacity=0.5)
    k2 = cc.stack_key(sh.load_shot("C"), "rl", "under", "proxy")
    under2 = cc.decode(cc.stack_bytes("C", "rl", "under", "proxy", 1001, size))[0]
    check(k1 != k2 and np.allclose(under2, bg * 0.5, atol=2e-3), "a change below the layer changes the key and the image")
    raises(lambda: cc.stack_bytes("C", "rl", "middle", "proxy", 1001, size), "unknown side refused", "side")
    sh.next_version("C", "bg")                                      # v002: an empty render (no frames)
    shot = sh.load_shot("C")
    bg = [x for x in shot["layers"] if x["name"] == "bg"][0]
    bg["version"] = "v002"
    sh.save_shot(shot)
    u3 = cc.decode(cc.stack_bytes("C", "rl", "under", "proxy", 1001, size))[0]
    check(not u3.any(), "a layer whose chosen version has no frames is left out of under, not an error")


def test_shader():
    s = cc.view_shader("", fr.DEFAULT_VIEW)
    check("OCIOView" in s["shader"] and "ACES 2.0" in s["view"] and s["textures"] and
          all(len(t["values"]) == t["width"] * t["height"] * t["channels"] for t in s["textures"]),
          "view shader: GLSL ES 3.0 function + its tables", f"{len(s['shader'])} chars, {len(s['textures'])} tables")


def test_budget():
    for f in (1001, 1002, 1003):                                   # full-size frames of two takes
        for v in ("v001", "v002"):
            cc.take_bytes("C", "rl", v, "full", f)
    u = cc.usage()
    full = sorted(cc.cache_root("C").rglob("full/*.f16"), key=lambda p: p.stat().st_mtime)
    os.utime(full[-1], (time.time() + 10, time.time() + 10))      # the newest-used full frame
    keep = full[-1]
    proxies = list(cc.cache_root("C").rglob("proxy/*.f16"))
    exr = list(sh.layer_dir("C", "rl").rglob("*.exr"))
    one_full = full[0].stat().st_size
    budget = u["bytes"] - u["full_bytes"] + 2 * one_full         # room for the proxies and ~2 full frames
    r = cc.evict(budget)
    left_full = list(cc.cache_root("C").rglob("full/*.f16"))
    check(r["bytes"] <= budget and r["removed_files"] > 0, "over budget: frames removed until under it",
          f"{r['removed_files']} files, {r['freed_bytes'] / 2**20:.1f} MB freed")
    check(all(p.exists() for p in proxies) and keep.exists() and len(left_full) < len(full),
          "full-size frames go first, least recently used first; proxies kept")
    check(all(p.exists() for p in exr), "the layers' own EXRs are never touched")
    stray = cc.cache_root("C") / "rl" / "v001" / "proxy" / "001001.tmp99"
    stray.write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stray, (old, old))
    cc.evict(10 ** 12)
    check(not stray.exists(), "a temp file a crashed write left behind is removed")
    note = cc.cache_root("C") / "README.txt"
    note.write_text("not a cache frame")
    c = cc.clear("C")
    check(c["removed_files"] > 0 and not list(cc.cache_root("C").rglob("*.f16")) and note.exists()
          and all(p.exists() for p in exr), "clear empties the shot's cache frames, nothing else")


def main() -> int:
    H, W = setup()
    test_takes(H, W)
    test_warm()
    test_stack(H, W)
    test_shader()
    test_budget()
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
