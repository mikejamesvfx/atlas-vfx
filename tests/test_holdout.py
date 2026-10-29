"""Depth holdout (layers/holdout.py): card plane depth per pixel, and the keep mask against plate depth.

Run: <V141 python> tests\\test_holdout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import holdout as ho  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def camera():
    plate = {"path": "", "width": 400, "height": 200, "colorspace": "unknown", "range": "full"}
    sensor = {"width_mm": 36.0, "height_mm": 18.0}
    return sc.build(name="t", frame_start=1001, fps=24.0, plate=plate, sensor=sensor,
                    cam_to_world=[np.eye(4)], focal_mm=36.0, source={"kind": "synthetic"})


def test_card_depth():
    cam = camera()
    # upright card 5 m straight ahead, facing the camera
    corners = [[-1, 1, -5], [1, 1, -5], [1, -1, -5], [-1, -1, -5]]
    z = ho.card_depth(cam, 1001, corners, 400, 200)
    check(np.allclose(z, 5.0, atol=1e-5), "card facing camera: depth 5 m at every pixel", f"{z.min():.4f}-{z.max():.4f}")
    # card tilted 45 deg about Y: depth varies across the frame, exact at the centre
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    tilt = [[-c, 1, -5 - s], [c, 1, -5 + s], [c, -1, -5 + s], [-c, -1, -5 - s]]
    z2 = ho.card_depth(cam, 1001, tilt, 400, 200)
    check(abs(z2[100, 200] - 5.0) < 0.05 and z2[100, 10] > z2[100, 390], "tilted card: deeper on its far side",
          f"centre {z2[100, 200]:.3f}, left {z2[100, 10]:.2f}, right {z2[100, 390]:.2f}")
    box = ho.card_depth(cam, 1001, tilt, 400, 200, (50, 60, 100, 120))
    check(np.allclose(box, z2[50:60, 100:120]), "box computes the same depths as the full frame")


def test_keep_mask():
    card = np.full((4, 4), 5.0, np.float32)
    plate = np.array([[3.0] * 4, [4.7] * 4, [4.5] * 4, [8.0] * 4], np.float32)
    valid = np.ones((4, 4), np.float32)
    k = ho.keep_mask(plate, valid, card, margin_m=0.35, soft_m=0.25)
    check(k[0].max() == 0 and k[3].min() == 1, "plate nearer -> hidden, plate behind -> kept")
    check(k[1].min() == 1 and 0 < k[2].min() < 1, "inside the margin kept, then a soft ramp", k[:, 0].tolist())
    valid[0] = 0
    k2 = ho.keep_mask(plate, valid, card)
    check(k2[0].min() == 1, "no valid plate depth -> card left alone")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
