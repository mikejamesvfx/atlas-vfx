"""Exercise 03: OCIO compliance, end to end, on synthetic numbers (no plate, no model, no GPU).

What it shows, step by step:
  1. the shipped config: its roles, the default display/view, and the two Fuji input spaces Nuke can see
  2. 18 % grey through F-Log vs F-Log2: why reading the wrong curve put the plate about a stop bright
  3. the same 10-bit code values decoded as full range and as legal range (the 'pc' mistake)
  4. ACEScg -> the ACES 2.0 SDR view -> its inverse: what survives the trip a video model takes, and what clips
  5. a generated (display) picture into ACEScg two ways, then back out through LTX Prep's view
  6. an EXR written by the repo's writer carries its colourspace; a mislabelled sequence is refused
  7. ocio_fast: the same OCIO processor over row bands on every core gives the same bits

Run from the repo root with the ComfyUI venv python that has PyOpenColorIO and OpenImageIO:
    python docs/learn/exercises/03_ocio_compliance.py
It writes only into a temp folder (set as ATLAS_VFX_OUT_ROOTS) and deletes it at the end. Exits non-zero on failure.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="atlas_learn_ocio_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)            # outputs confined to a temp folder, as the tests do

import PyOpenColorIO as ocio  # noqa: E402

from atlas_vfx.camera_side import exr_io, ocio_fast  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.layers import element_colour as ec  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("   ok   " if ok else "   FAIL ") + msg)
    if not ok:
        fails.append(msg)


def stops(a: float, b: float) -> float:
    return float(np.log2(a / b))


def grey_through(space: str, code: float) -> float:
    """One neutral code value, from `space` to ACEScg with the shipped config (what plate.to_working does)."""
    ref = {"ocio_config": pl.ocio_path(""), "colorspace": space, "working_space": "ACEScg"}
    return float(pl.to_working(np.full((1, 1, 3), code, np.float32), ref)[0, 0, 0])


cfg_path = pl.ocio_path("")                              # blank widget = the shipped config
cfg = pl.ocio_config(cfg_path)
t_start = time.perf_counter()

# ------------------------------------------------------------------------------------------------ 1. the config
print("\n1. The shipped config")
print(f"   file          {Path(cfg_path).relative_to(ROOT)}")
print(f"   OCIO          PyOpenColorIO {ocio.__version__}, config profile {cfg.getMajorVersion()}.{cfg.getMinorVersion()}")
for role in ("scene_linear", "compositing_log", "color_timing", "data"):
    print(f"   role {role:<16} -> {cfg.getRoleColorSpace(role)}")
disp = cfg.getDefaultDisplay()
print(f"   default view  {disp} / {cfg.getDefaultView(disp)}")
for name in ("F-Log F-Gamut", "F-Log2 F-Gamut"):
    cs = cfg.getColorSpace(name)
    print(f"   {name:<15} family {cs.getFamily()}, encoding {cs.getEncoding()}, categories {list(cs.getCategories())}")
    check("file-io" in list(cs.getCategories()), f"{name} is tagged file-io (or Nuke's Read menu hides it)")
check(cfg.getRoleColorSpace("scene_linear") == pl.WORKING_SPACE == "ACEScg", "working space is ACEScg")
check((disp, cfg.getDefaultView(disp)) == (pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW),
      "the code's default display/view is the config's default")

# ------------------------------------------------------------------------------------------------ 2. F-Log vs F-Log2
print("\n2. 18 % grey: F-Log vs F-Log2 (data sheet grey codes 0.4593 and 0.391)")
flog_right = grey_through("F-Log F-Gamut", 0.4593)
flog2_right = grey_through("F-Log2 F-Gamut", 0.391)
flog_as_flog2 = grey_through("F-Log2 F-Gamut", 0.4593)   # the pre-2026-10-20 mistake: an F-Log plate read as F-Log2
print(f"   F-Log  code 0.4593 read as F-Log   -> {flog_right:.4f} ACEScg")
print(f"   F-Log2 code 0.391  read as F-Log2  -> {flog2_right:.4f} ACEScg")
print(f"   F-Log  code 0.4593 read as F-Log2  -> {flog_as_flog2:.4f} ACEScg  ({stops(flog_as_flog2, flog_right):+.2f} stops)")
for code in (0.25, 0.35, 0.55, 0.70):                   # the error is not one exposure offset: the contrast is wrong too
    a, b = grey_through("F-Log F-Gamut", code), grey_through("F-Log2 F-Gamut", code)
    print(f"      code {code:.2f}: F-Log {a:8.4f}  as F-Log2 {b:8.4f}  ({stops(b, a):+.2f} stops)")
check(abs(flog_right - 0.18) < 1e-3 and abs(flog2_right - 0.18) < 1e-3, "both data-sheet greys land on 0.18")
check(0.9 < stops(flog_as_flog2, flog_right) < 1.1, "the wrong curve puts mid-grey about a stop bright")
# try changing: 0.4593 to an F-Log2 code like 0.2 (a dark dusk value) and see the error shrink. NOTES 2026-10-20 night:
# the dusk plate sat ~3 stops under grey, where the two curves are close, so the fix moved the comp only 0.15 stops.

# ------------------------------------------------------------------------------------------------ 3. ranges
print("\n3. The same 10-bit codes decoded as full range and as legal range (plate._norm_luma, H.273)")
codes = np.array([64, 470, 940, 1019], np.uint16)
full = pl._norm_luma(codes, 10, True)
legal = pl._norm_luma(codes, 10, False)
print("   code   full     legal    F-Log lin(full)  F-Log lin(legal)  error")
for c, f, g in zip(codes, full, legal):
    lf, lg = grey_through("F-Log F-Gamut", float(f)), grey_through("F-Log F-Gamut", float(g))
    err = f"{stops(lg, lf):+.2f} st" if lf > 1e-4 and lg > 1e-4 else "(toe)"
    print(f"   {int(c):4d}  {f:.4f}  {g:+.4f}   {lf:12.4f}    {lg:12.4f}      {err}")
check(abs(full[1] - 470 / 1023) < 1e-6 and abs(legal[2] - 1.0) < 1e-6, "full = code/1023, legal = (code-64)/876")
check(legal[3] > 1.0, "legal decode is not clipped: superwhites above 940 stay above 1.0")
# The real plate is FULL range (ffprobe color_range=pc). The first prototypes forced in_range=tv (legal), which
# stretched every code. plate_ref() now reads the range from the stream and records range_source; range_override
# exists only for files whose flag is known to be wrong, and the override is logged and written into the record.

# ------------------------------------------------------------------------------------------------ 4. view round trip
print("\n4. ACEScg -> ACES 2.0 SDR view -> inverse (what a video model sees, and what comes back)")
fwd, inv = li.view_processors(cfg_path)
lin = np.array([0.002, 0.05, 0.18, 1.0, 4.0, 16.0, 64.0, 100.0, 500.0], np.float32)
g = np.repeat(lin[:, None], 3, 1)[None]
d = li.to_view(g, fwd)
back = li.from_view(d, inv)                              # float, as the IMAGE tensors are
q8 = li.from_view(np.round(d * 255) / 255, inv)          # the same display values stored as 8-bit (an mp4 / png)
print("   neutral greys")
print("   ACEScg    display   back (float)  rel err   back (8-bit)  stops off")
for x, y, z, w in zip(lin, d[0, :, 1], back[0, :, 1], q8[0, :, 1]):
    off = f"{np.log2(w / x):+.2f}" if w > 0 else "lost (rounds to 0)"
    print(f"   {x:7.3f}   {y:.4f}    {z:9.3f}   {abs(z - x) / x:7.2%}   {w:9.3f}    {off}")
# try changing: 255 to 1023 (a 10-bit file) and watch the 8-bit errors near the top and in the toe shrink.
rel = np.abs(back[0, :, 1] - lin) / lin
check(rel[lin <= 64].max() < 0.01, "in float, greys up to 64 come back within 1 % (the view compresses, it barely clips)")
check(back[0, -1, 1] < 200, "500.0 comes back near 128: past display 1.0 the 0-1 clip loses it")
hot = np.array([[[8.0, 0.2, 0.05], [0.5, 0.02, 0.01], [6.0, 1.2, 0.15]]], np.float32)      # saturated hot colours
hb = li.from_view(li.to_view(hot, fwd), inv)
print("   saturated colours (ACEScg in -> back)")
for a, b in zip(hot[0], hb[0]):
    print(f"   [{a[0]:5.2f} {a[1]:5.2f} {a[2]:5.2f}] -> [{b[0]:5.2f} {b[1]:5.2f} {b[2]:5.2f}]")
check(abs(hb[0, 0, 0] - 8.0) > 1.0, "a bright saturated red does not come back (the view's gamut compression)")
yy, xx = np.mgrid[0:270, 0:512].astype(np.float32)
dusk = (0.02 + 0.3 * (xx / 512) * (0.5 + 0.5 * yy / 270))[..., None].repeat(3, 2)       # like test_ltx_integration
sunny = dusk.copy()
sunny[:60, 380:] = [6.0, 1.2, 0.15]                      # a hot orange sunset corner, about 6 % of the frame
for name, img in (("dusk plate", dusk), ("same + hot sky", sunny)):
    rt = li.roundtrip_report(img, fwd, inv, step=1)
    warn = li.roundtrip_warnings(rt, "layer")
    print(f"   {name:<15} median {rt['median_rel_err']:.2e}, share off >5 % {rt['share_over_5pct']:.3%}, "
          f"warning: {'yes' if warn else 'no'}")
check(not li.roundtrip_warnings(li.roundtrip_report(dusk, fwd, inv, step=1), "layer"), "dusk plate: no warning")
check(bool(li.roundtrip_warnings(li.roundtrip_report(sunny, fwd, inv, step=1), "layer")), "hot sky: LTX Prep warns")
# try changing: the view in li.view_processors(cfg_path, view="Un-tone-mapped") and watch where it clips instead.

# ------------------------------------------------------------------------------------------------ 5. elements
print("\n5. A generated (display) picture into ACEScg, then out through LTX Prep's view")
rng = np.random.default_rng(3)
pic = rng.uniform(0.05, 0.9, (64, 96, 3)).astype(np.float32)          # stand-in for a generator's output frame
mid = (pic > 0.3) & (pic < 0.7)
for src in ec.SOURCES:
    out = li.to_view(ec.to_acescg(pic, "", src), fwd)
    print(f"   {src:<45} max err {np.abs(out - pic).max():.4f}, mean drift in the mids {np.mean(out[mid] - pic[mid]):+.3f}")
via_inv = li.to_view(ec.to_acescg(pic, "", ec.VIEW_INVERSE), fwd)
via_tex = li.to_view(ec.to_acescg(pic, "", ec.SRGB_TEXTURE), fwd)
check(np.abs(via_inv - pic).max() < 2e-3, "inverse view: the element comes back as it was made")
check(np.mean(via_tex[mid] - pic[mid]) < -0.02, "sRGB texture: it comes back darker (tone-mapped twice)")

# ------------------------------------------------------------------------------------------------ 6. EXR headers
print("\n6. EXRs carry their colourspace")
print(f"   the config's own file rule for '*.exr': {cfg.getColorSpaceFromFilepath('layer.1001.exr')!r} "
      "(so an untagged ACEScg EXR would be read as AP0 by anything trusting the rules)")
lay = TMP / "layers" / "demo"
rgba = np.zeros((32, 48, 4), np.float32)
rgba[8:24, 12:36] = [0.18, 0.18, 0.18, 1.0]               # premultiplied grey card on black
for n in (1001, 1002, 1003):
    exr_io.write_rgba(lay / f"demo.{n:04d}.exr", rgba, colorspace="ACEScg", attrs={"frame": n})
h = exr_io.header(lay / "demo.1001.exr")
print(f"   header: format {h['format']}, channels {h['channels']}, colorInteropID {h['colorInteropID']!r}, "
      f"atlas_vfx:ocio_colorspace {h['ocio_colorspace']!r}, premultiplied {h['premultiplied']}")
check(h["colorInteropID"] == "lin_ap1_scene" and h["ocio_colorspace"] == "ACEScg", "ACEScg is named twice in the header")
ok = exr_io.check_sequence(str(lay / "demo.%04d.exr"), 1001, 1003, colorspace="ACEScg")
check(ok["frames"] == [1001, 1003], "check_sequence passes the ACEScg layer (what Nuke Builder runs before writing)")
exr_io.write_rgba(lay / "demo.1002.exr", rgba, colorspace="ACES2065-1")     # one frame quietly in AP0
try:
    exr_io.check_sequence(str(lay / "demo.%04d.exr"), 1001, 1003, colorspace="ACEScg")
    check(False, "a mislabelled frame is refused")
except ValueError as e:
    print(f"   refused: {str(e).split(': ', 1)[1]}")
    check("colorspace" in str(e), "a mislabelled frame is refused")

# ------------------------------------------------------------------------------------------------ 7. ocio_fast
print("\n7. ocio_fast: same processor, row bands on a thread pool")
frame = np.exp(np.random.default_rng(1).normal(-2.0, 2.5, (1080, 2048, 3))).astype(np.float32)
proc = cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW))
a, b = frame.copy(), frame.copy()
t0 = time.perf_counter()
proc.getDefaultCPUProcessor().applyRGB(a)
t1 = time.perf_counter()
ocio_fast.fast(proc).applyRGB(b)
t2 = time.perf_counter()
print(f"   2K frame through the ACES 2.0 SDR view: {1000 * (t1 - t0):.0f} ms single thread, "
      f"{1000 * (t2 - t1):.0f} ms banded on {ocio_fast.threads()} threads")
check(np.array_equal(a, b), "banded output is bit-identical (no LUT, no approximation)")
# try changing: set ATLAS_VFX_OCIO_THREADS=1 in the environment before running and the banded path turns itself off.

# ------------------------------------------------------------------------------------------------ done
import shutil  # noqa: E402

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'PASS' if not fails else f'FAILED {len(fails)}'}  ({time.perf_counter() - t_start:.1f} s)")
sys.exit(1 if fails else 0)
