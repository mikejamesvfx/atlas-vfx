"""Clean-space LTX loop (layers/ltx_integration.py sensor helpers): LTX Prep takes the plate camera off, LTX Return
puts it back. Synthetic lens maps (barrel + vignette), no RS.

Run: <V141 python> tests\\test_ltx_sensor_loop.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import ltx_integration as li  # noqa: E402
from atlas_vfx.sensor import jobs  # noqa: E402
from atlas_vfx.sensor import lens as ln  # noqa: E402

FAILS = []
W, H = 320, 180


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def grid_image():
    """Smooth linear test image (ACEScg-ish values), so warps interpolate cleanly."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 0.18 + 0.12 * np.sin(x / 9.0) * np.cos(y / 7.0)
    g = 0.18 + 0.10 * np.cos(x / 11.0)
    b = 0.18 + 0.08 * np.sin(y / 5.0)
    return np.dstack([r, g, b]).astype(np.float32)


def main() -> int:
    check(li.sensor_maps("", "") is None and li.sensor_record(None) is None, "blank folders: no sensor, prep unchanged")

    tmp = Path(tempfile.mkdtemp(prefix="avfx_ltx_sensor_"))
    lens = ln.BrownConrady(300.0, 300.0, W / 2.0, H / 2.0, k1=-0.12, k2=0.02)
    jobs.make_lens_maps(lens, (W, H), tmp, v1=-0.15)
    maps = li.sensor_maps("", str(tmp))
    rec = li.sensor_record(maps)
    check(rec == {"rs_folder": None, "lens_folder": str(tmp)}, "sensor record names the lens folder", rec)
    again = li.sensor_maps(rec["rs_folder"] or "", rec["lens_folder"] or "")
    check(again is not None, "Return can rebuild the maps from the prep record")

    plate = grid_image()                                   # stands in for the real (distorted) plate
    clean = li.remove_camera(plate, maps, 1130)
    check(clean.shape == plate.shape and np.isfinite(clean).all(), "remove camera keeps shape, finite")
    d = np.abs(clean - plate)[10:-10, 10:-10].mean()
    check(d > 1e-3, "remove camera changes the plate (barrel + vignette taken off)", f"mean diff {d:.4f}")

    # LTX paints in clean space; Return delivers a premultiplied RGBA layer. A full-alpha layer of the clean plate
    # must land back on the original plate once the camera is put back.
    layer = np.dstack([clean, np.ones((H, W), np.float32)])
    back = li.apply_camera(layer, maps, 1130)
    inner = (slice(20, -20), slice(30, -30))              # away from the edges the undistort pushes out of frame
    err = np.abs(back[..., :3] - plate)[inner]
    check(float(np.median(err)) < 2e-3, "plate -> clean -> camera back: matches the plate", f"median {np.median(err):.5f}")
    a = back[..., 3][inner]
    check(abs(float(a.mean()) - 1.0) < 1e-3, "alpha survives the warp and is not vignetted", f"mean {a.mean():.5f}")

    # premultiplied layer with a hole: alpha 0 stays 0 after apply camera
    hole = layer.copy()
    hole[60:120, 120:200] = 0.0
    b2 = li.apply_camera(hole, maps, 1130)
    check(float(b2[85:95, 155:165, 3].max()) < 1e-3, "transparent area stays transparent")
    print("FAILED: " + ", ".join(FAILS) if FAILS else "all ok")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
