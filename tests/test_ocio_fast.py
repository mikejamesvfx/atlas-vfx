"""camera_side/ocio_fast.py: OCIO CPU processors run over row bands on a thread pool (speed pass 2026-11-01).

  1. Banded output is bit-identical to OCIO's single-thread processor: ACES 2.0 view forward and inverse, and
     F-Log F-Gamut -> ACEScg, on a 4K HDR frame with negatives and highlights.
  2. Small arrays take the plain path (OCIO itself refuses float64 and non-contiguous arrays, as before).
  3. Every processor helper in the pack hands out the banded wrapper (plate._cpu, ltx_integration, element_colour).
  4. It is faster (printed; asserted only loosely, machines differ).
Run: <V141 python> tests/test_ocio_fast.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import PyOpenColorIO as ocio  # noqa: E402

from atlas_vfx.camera_side import ocio_fast  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.layers import element_colour as ec  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


cfg_path = pl.ocio_path("")
cfg = pl.ocio_config(cfg_path)
rng = np.random.default_rng(1)
hdr = np.exp(rng.normal(-2.0, 2.5, (2160, 4096, 3))).astype(np.float32)
hdr[::97, ::89] *= -0.01                                     # a few negatives
hdr[5, :64] = 6e4                                            # highlights near half max

view = ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW)
inv_view = ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW,
                                     direction=ocio.TRANSFORM_DIR_INVERSE)
cases = [("view forward", cfg.getProcessor(view), hdr),
         ("view inverse", cfg.getProcessor(inv_view), np.clip(hdr, 0, 1)),
         ("F-Log -> ACEScg", cfg.getProcessor(pl.FUJI_MODE_SPACE["F-Log"], "ACEScg"), np.clip(hdr, 0, 1))]
for name, proc, src in cases:
    ref = np.ascontiguousarray(src).copy()
    t0 = time.perf_counter()
    proc.getDefaultCPUProcessor().applyRGB(ref)
    t1 = time.perf_counter()
    got = np.ascontiguousarray(src).copy()
    ocio_fast.fast(proc).applyRGB(got)
    t2 = time.perf_counter()
    same = np.array_equal(ref, got, equal_nan=True)
    check(same, f"{name}: banded == single thread (max diff {np.nanmax(np.abs(ref - got)):.3g})")
    print(f"     {name}: {1000 * (t1 - t0):.0f} ms single, {1000 * (t2 - t1):.0f} ms banded "
          f"({ocio_fast.threads()} threads)")
    if name == "view forward":
        check(t2 - t1 < (t1 - t0) or ocio_fast.threads() < 3, f"{name}: banded is faster")

fwd = cfg.getProcessor(view)
for name, arr in [("small", hdr[:100, :100].copy())]:
    ref = np.ascontiguousarray(arr, np.float32).copy()
    fwd.getDefaultCPUProcessor().applyRGB(ref)
    got = arr.copy(order="K")
    ocio_fast.fast(fwd).applyRGB(got)
    check(np.allclose(np.ascontiguousarray(got, np.float32), ref, atol=1e-6), f"{name} array takes the plain path")

f, i = li.view_processors(cfg_path)
check(isinstance(f, ocio_fast.Banded) and isinstance(i, ocio_fast.Banded), "ltx_integration.view_processors is banded")
check(isinstance(pl._cpu(cfg_path, "ACEScg", None, pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW), ocio_fast.Banded),
      "plate._cpu is banded")
check(isinstance(ec.to_acescg_processor(cfg_path), ocio_fast.Banded), "element_colour processors are banded")
check(ocio_fast.fast(f) is f, "fast() of a Banded is the same object")

print("PASS" if not fails else f"FAILED {len(fails)}")
sys.exit(1 if fails else 0)
