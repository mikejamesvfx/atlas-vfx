"""Light mix v2 (comp/mix.py, comp/bake.py; owner call 2026-11-01: full relit takes, blended).

  1. Blend: weights are proportions (two identical takes at any weights = the take); stops and a luminance-1 tint
     scale RGB only; solo beats mute; nothing active -> None.
  2. Sum (light passes, and every v1 record): weights add, alpha clipped.
  3. Records: v1 {version: weight} reads as sum; validate refuses bad numbers, unknown versions, bad tints.
  4. Through the comp: a layer with a v2 mix composites as mix_frame; bake writes a new version whose EXRs equal
     mix_frame at full size.
Run: <V141 python> tests/test_light_mix.py
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
TMP = Path(tempfile.mkdtemp(prefix="lightmix_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

from atlas_vfx.comp import bake as bk  # noqa: E402
from atlas_vfx.comp import frames as fr  # noqa: E402
from atlas_vfx.comp import mix as mx  # noqa: E402
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


def flat(rgb, a=1.0, size=(8, 12)):
    img = np.zeros(size + (4,), np.float32)
    img[..., :3] = np.asarray(rgb, np.float32) * a
    img[..., 3] = a
    return img


def take(**kw):
    return {"weight": 1.0, "stops": 0.0, "tint": [1.0, 1.0, 1.0], "mute": False, "solo": False, **kw}


def test_maths():
    A, B, C = flat([0.8, 0.4, 0.1]), flat([0.1, 0.2, 0.6]), flat([0.3, 0.3, 0.3])
    same = mx.mix_frame({"v1": A, "v2": A.copy()}, {"mode": "blend", "takes": {"v1": take(weight=0.9),
                                                                               "v2": take(weight=0.2)}})
    check(np.allclose(same, A), "blend: identical takes at any weights give the take (exposure holds)")
    m = mx.mix_frame({"v1": A, "v2": B}, {"mode": "blend", "takes": {"v1": take(weight=0.75), "v2": take(weight=0.25)}})
    check(np.allclose(m[0, 0, :3], 0.75 * A[0, 0, :3] + 0.25 * B[0, 0, :3]), "blend: weights are proportions")
    up = mx.mix_frame({"v1": A}, {"mode": "blend", "takes": {"v1": take(stops=1.0)}})
    check(np.allclose(up[..., :3], 2 * A[..., :3]) and np.allclose(up[..., 3], 1.0), "stops scale RGB, not alpha")
    t = mx.tint_unit([1.0, 0.8, 0.5])
    check(abs(float(np.asarray(t) @ mx.LUMA_AP1) - 1.0) < 1e-5, "tint_unit: luminance 1", str(t))
    grey = flat([0.5, 0.5, 0.5])
    tinted = mx.mix_frame({"v1": grey}, {"mode": "blend", "takes": {"v1": take(tint=t)}})
    check(abs(float(tinted[0, 0, :3] @ mx.LUMA_AP1) - 0.5) < 1e-5, "a luminance-1 tint keeps a grey's luminance")
    solo = mx.mix_frame({"v1": A, "v2": B, "v3": C}, {"mode": "blend", "takes": {
        "v1": take(), "v2": take(solo=True), "v3": take(solo=True, mute=True)}})
    check(np.allclose(solo[0, 0, :3], (B[0, 0, :3] + C[0, 0, :3]) / 2), "solo beats mute and the rest")
    check(mx.mix_frame({"v1": A}, {"mode": "blend", "takes": {"v1": take(mute=True)}}) is None and
          mx.mix_frame({"v1": A}, {"mode": "blend", "takes": {"v1": take(weight=0.0)}}) is None,
          "nothing active: None (the layer drops out)")
    half = flat([0.5, 0.5, 0.5], a=0.5)
    s = mx.mix_frame({"v1": half, "v2": half}, {"v1": 1.5, "v2": 1.0})
    check(np.allclose(s[..., :3], 2.5 * half[..., :3]) and np.allclose(s[..., 3], 1.0),
          "sum (v1 record): weights add, alpha clipped to 1")
    miss = mx.mix_frame({"v1": A}, {"mode": "blend", "takes": {"v1": take(weight=0.5), "v2": take(weight=0.5)}})
    check(np.allclose(miss, A), "a take missing the frame is left out of both sums")


def test_records():
    check(mx.normalize({"v001": 0.5})["mode"] == "sum", "a v1 record reads as sum")
    vs = ["v001", "v002"]
    ok = mx.validate({"mode": "blend", "takes": {"v001": {"weight": 0.5, "stops": -1, "tint": [1, 1, 1]}}}, vs)
    check(ok["takes"]["v001"]["solo"] is False and ok["mode"] == "blend", "validate fills defaults")
    raises(lambda: mx.validate({"mode": "blend", "takes": {"v003": {}}}, vs), "unknown version refused", "v003")
    raises(lambda: mx.validate({"mode": "blend", "takes": {"v001": {"weight": 1.5}}}, vs), "blend weight > 1 refused", "0-1")
    raises(lambda: mx.validate({"mode": "blend", "takes": {"v001": {"stops": 9}}}, vs), "stops out of range refused", "stops")
    raises(lambda: mx.validate({"mode": "blend", "takes": {"v001": {"tint": [1, 1]}}}, vs), "short tint refused", "tint")
    raises(lambda: mx.validate({"mode": "blend", "takes": {"v001": {"mute": "yes"}}}, vs), "non-boolean mute refused", "mute")
    raises(lambda: mx.validate({"mode": "lerp", "takes": {}}, vs), "unknown mode refused", "mode")
    raises(lambda: mx.tint_unit([0, 0, 0]), "a black tint refused", "dark")


def write_seq(folder: Path, stem: str, frames, rgb, size=(16, 32)):
    for n in frames:
        exr_io.write_rgba(folder / f"{stem}.{n:04d}.exr", flat(rgb, size=size), half=False)


def test_comp_and_bake():
    sh.create_shot("LM", "", 1001, 1002)
    sh.add_layer("LM", "rl", "recipe", recipe="relight")
    d = sh.layer_dir("LM", "rl")
    cols = {"v001": (0.8, 0.4, 0.1), "v002": (0.1, 0.2, 0.6), "v003": (0.4, 0.4, 0.4)}
    for v, rgb in cols.items():
        assert sh.next_version("LM", "rl") == v
        write_seq(d / v / "beauty", "rl", [1001, 1002], rgb)
    t = mx.tint_unit([1.0, 0.9, 0.7])
    mix = {"mode": "blend", "takes": {"v001": take(weight=0.6, stops=0.5), "v002": take(weight=0.4, tint=t),
                                      "v003": take(mute=True)}}
    sh.update_layer("LM", "rl", mix=mix)
    img, used = fr.composite(sh.load_shot("LM"), 1001, width=32)
    want = mx.mix_frame({v: flat(c, size=(16, 32)) for v, c in cols.items()}, mix)
    check(used == ["rl"] and np.allclose(img[4, 4], want[4, 4], atol=2e-3), "comp: a v2 mix composites as mix_frame",
          f"{img[4, 4, :3]} vs {want[4, 4, :3]}")
    st = bk.start("LM", "rl")
    t0 = time.time()
    while bk.status("LM", "rl")["state"] == "running" and time.time() - t0 < 60:
        time.sleep(0.1)
    st = bk.status("LM", "rl")
    check(st["state"] == "done" and st["version"] == "v004", "bake: done as the next version", str(st))
    out = sorted((d / "v004").glob("*.exr"))
    rd = exr_io.read_rgba(out[0])[0] if out else None
    check(len(out) == 2 and rd is not None and np.allclose(rd, want, atol=2e-3) and (d / "v004" / "mix.json").is_file(),
          "bake: full-size EXRs equal mix_frame (half float), mix.json kept", f"{len(out)} files")
    sh.update_layer("LM", "rl", mix={"mode": "blend", "takes": {"v001": take(mute=True)}})
    raises(lambda: bk.bake_mix("LM", "rl"), "bake refuses a mix with nothing active", "no light mix")


def main() -> int:
    test_maths()
    test_records()
    test_comp_and_bake()
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
