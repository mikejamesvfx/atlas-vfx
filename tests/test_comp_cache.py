"""Light room frame cache (comp/cache.py; Light Mix plan phase 2).

  1. A take frame round trips to float16 rounding at both levels (proxy 1024 wide, full size).
  2. A changed source (new mtime / size) is remade; an unchanged one is served from the file.
  3. Names that would leave the cache folder are refused.
  4. warm builds every frame of the mixed takes in the background.
  5. under / over: the plate and lower layers / higher layers at the takes' size, equal to the comp's own blend.
  6. view_shader: OCIO's GLSL ES 3.0 for the default ACES 2.0 view, with its lookup tables.
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
    full = cc.decode(cc.take_bytes("C", "rl", "v002", "full", 1002))
    src = ramp(H, W, 1)
    check(full[0].shape == (H, W, 4) and np.allclose(full[0], src, rtol=1e-3, atol=1e-3),
          "full level round trips to float16 rounding", f"max {np.abs(full[0] - src).max():.1e}")
    px, head = cc.decode(cc.take_bytes("C", "rl", "v002", "proxy", 1002))
    check(px.shape == (136, 1024, 4) and head["level"] == "proxy" and np.allclose(px, fr._resize(src, (1024, 136)),
                                                                              atol=2e-3),
          "proxy level is 1024 wide, the area-resized take", str(px.shape))
    f = cc.take_file("C", "rl", "v002", "proxy", 1002)
    t1 = f.stat().st_mtime_ns
    cc.take_bytes("C", "rl", "v002", "proxy", 1002)
    check(f.stat().st_mtime_ns == t1, "an unchanged source is served from the cache file")
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


def test_shader():
    s = cc.view_shader("", fr.DEFAULT_VIEW)
    check("OCIOView" in s["shader"] and "ACES 2.0" in s["view"] and s["textures"] and
          all(len(t["values"]) == t["width"] * t["height"] * t["channels"] for t in s["textures"]),
          "view shader: GLSL ES 3.0 function + its tables", f"{len(s['shader'])} chars, {len(s['textures'])} tables")


def main() -> int:
    H, W = setup()
    test_takes(H, W)
    test_warm()
    test_stack(H, W)
    test_shader()
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
