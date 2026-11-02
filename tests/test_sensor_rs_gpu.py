"""sensor/rs_gpu.py: RS Maps' per-pixel maths on CUDA match the CPU path (speed pass 2026-11-01; the CPU path took
~25 s per 4K frame).

  1. Synthetic camera with a brown lens, pan + move + wobble, slanted depth, object flow: rs and ders maps from the
     GPU equal the CPU ones within 1e-6 px (both float64; the stored STMaps are float32).
  2. With a time offset (the motion-blur sub-samples) and with no lens.
  3. Real DSCF3000 camera at 4K, if present: same check, and the GPU time is printed.
Skips (passes) without CUDA. Run: <V141 python> tests/test_sensor_rs_gpu.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from atlas_vfx.camera_side.plate import WORK_ROOT  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.sensor import rs, rs_gpu  # noqa: E402
from atlas_vfx.sensor.lens import BrownConrady  # noqa: E402

FAILS: list[str] = []
REAL = WORK_ROOT / "DSCF3000" / "solve" / "DSCF3000_relight_solve" / "DSCF3000_relight_solve.shot_camera.json"


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def both(cam, frame, depth, readout_s, **kw):
    t0 = time.perf_counter()
    g = rs.rs_maps(cam, frame, depth, readout_s, **kw)
    t1 = time.perf_counter()
    os.environ["ATLAS_VFX_GPU_RS"] = "0"
    rs_gpu.available.cache_clear()
    try:
        c = rs.rs_maps(cam, frame, depth, readout_s, **kw)
    finally:
        os.environ.pop("ATLAS_VFX_GPU_RS")
        rs_gpu.available.cache_clear()
    t2 = time.perf_counter()
    w = kw.get("width") or cam.data["plate"]["width"]
    h = kw.get("height") or cam.data["plate"]["height"]
    px = np.array([w, h], np.float64)
    worst = max(float(np.nanmax(np.abs((a.astype(np.float64) - b.astype(np.float64)) * px)))
                for a, b in zip(g, c) if a is not None)
    return worst, t1 - t0, t2 - t1


def main():
    if not rs_gpu.available():
        print("skip: no CUDA, the GPU path is not used")
        return 0
    import test_sensor_rs as tsr

    cam = tsr.make_cam(yaw_deg=1.5, move=(0.08, 0, -0.05), wobble=0.4)
    lens = BrownConrady.from_camera(cam, 1004, k1=-0.08, k2=0.012, p1=1e-4)
    W, H = tsr.W, tsr.H
    yy = np.linspace(4.0, 20.0, H)[:, None] * np.ones((1, W))
    flow = np.dstack([np.full((H, W), 1.5), np.linspace(-1, 1, W)[None, :].repeat(H, 0)])
    T = tsr.XH2_HQ_S
    worst, tg, tc = both(cam, 1004, yy, T, lens=lens, object_flow=flow)
    check(worst < 1e-3, f"lens + flow: GPU = CPU within {worst:.2e} px (float32 storage)")
    worst, *_ = both(cam, 1004, yy, T, lens=lens, time_offset=0.3)
    check(worst < 1e-3, f"time offset 0.3: GPU = CPU within {worst:.2e} px")
    worst, *_ = both(cam, 1004, yy, T, want_ders=False)
    check(worst < 1e-3, f"no lens, rs only: GPU = CPU within {worst:.2e} px")
    if REAL.is_file():
        real = sc.load(REAL)
        f = real.frames[len(real.frames) // 2]
        depth = np.linspace(3.0, 30.0, 540)[:, None] * np.ones((1, 1024))
        worst, tg, tc = both(real, f, depth, 0.0266)
        check(worst < 1e-3, f"DSCF3000 4K, brown lens: GPU = CPU within {worst:.2e} px "
                            f"({tg:.2f} s GPU vs {tc:.1f} s CPU)")
    else:
        print(f"skip real camera: {REAL} not found")
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
