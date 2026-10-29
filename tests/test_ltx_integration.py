"""LTX integration pass, the parts around LTX (synthetic):

- the named view (ACES 2.0 SDR) and its inverse round trip mid-tones within 1 %
- the inpaint mask grows downward (contact shadow room) and not upward, then dilates evenly
- identity: if LTX hands back exactly what it was given (the plate in the view, at work size), the returned layer over
  the plate reproduces the plate: outside the mask exactly (alpha 0), inside within the round trip + resize error
- the returned layer is premultiplied ACEScg with alpha = the feathered mask

Run: <V141 python> tests\test_ltx_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import card_project as cp  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

OCIO = str(ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio")


def plate(W=512, H=270):
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.02 + 0.3 * (xx / W) * (0.5 + 0.5 * yy / H)                  # dusk-like range, 0.02-0.32 ACEScg
    tex = 0.03 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    return np.clip(np.dstack([base + tex, base * 0.95 + tex, base * 0.9 + tex]), 0.001, None).astype(np.float32)


def test_roundtrip(fwd, inv):
    lin = plate()
    rep = li.roundtrip_report(lin, fwd, inv, step=1)
    assert rep["median_rel_err"] < 0.01 and rep["share_over_5pct"] < 0.01, rep
    print(f"ok view round trip: median {rep['median_rel_err'] * 100:.3f} %, p99 {rep['p99_rel_err'] * 100:.3f} % "
          f"(ACEScg -> ACES 2.0 SDR -> inverse)")


def test_mask():
    a = np.zeros((270, 512), np.float32)
    a[100:150, 200:240] = 1.0
    m = li.make_mask(a, (512, 270), shadow_px=20, dilate_px=0)
    ys, xs = np.nonzero(m)
    assert ys.min() == 100 and ys.max() == 169 and xs.min() == 200 and xs.max() == 239, (ys.min(), ys.max())
    m2 = li.make_mask(a, (512, 270), shadow_px=0, dilate_px=5)
    ys, xs = np.nonzero(m2)
    assert ys.min() == 95 and ys.max() == 154 and xs.min() == 195 and xs.max() == 244
    print("ok mask: shadow smear goes 20 px down only; dilation 5 px each way")


def test_identity(fwd, inv):
    W, H, ww, wh = 512, 270, 256, 128
    lin = plate(W, H)
    alpha = np.zeros((H, W), np.float32)
    alpha[90:200, 220:300] = 1.0
    mask = li.make_mask(alpha, (ww, wh), shadow_px=6, dilate_px=3)
    ltx_out = li.resize(li.to_view(lin, fwd), (ww, wh))                      # "LTX" returns its reference as is
    rgba = li.return_frame(ltx_out, mask, (W, H), inv, feather_px=1.5)
    comp = cp.over(rgba, lin)
    out = rgba[..., 3] == 0
    assert out.mean() > 0.5 and np.array_equal(comp[out], lin[out]), "plate changed outside the mask"
    inside = rgba[..., 3] > 0.99
    rel = np.abs(comp[inside] - lin[inside]).mean() / lin[inside].mean()
    assert rel < 0.05, rel
    a = rgba[..., 3:4]
    assert np.all(rgba[..., :3] <= np.maximum(a, 1e-6) * 10)               # premultiplied (bounded by alpha)
    print(f"ok identity: outside the mask the plate is bit-exact ({out.mean() * 100:.0f} % of pixels); inside, an "
          f"unchanged LTX pass comes back within {rel * 100:.2f} % (half-res resize + view round trip)")


def test_windows(fwd, inv):
    """Crop windows for a box that grows and slides right (like a walk-on over 1130-1250), on a 4096x2160 plate."""
    W, H, ww, wh = 4096, 2160, 1216, 1920
    boxes = []
    for i in range(121):
        t = i / 120
        bw, bh = 500 + 750 * min(1, t * 1.4), 1080 + 1080 * min(1, t * 1.3)
        x0 = 1950 + 1500 * t ** 1.5
        boxes.append((int(x0), 0, int(min(x0 + bw, W)), int(min(bh, H))))
    boxes[60] = None                                                      # an empty frame borrows a neighbour
    wins = li.crop_windows(boxes, (ww, wh), (W, H), margin=64)
    scales = []
    for b, (x0, y0, w, h) in zip(boxes, wins):
        assert 0 <= x0 and 0 <= y0 and x0 + w <= W and y0 + h <= H, (x0, y0, w, h)
        assert w >= ww and h >= wh and abs(w / h - ww / wh) < 0.01
        if b is not None:
            assert x0 <= max(b[0] - 64, 0) and x0 + w >= min(b[2] + 64, W) and y0 <= b[1], (b, (x0, y0, w, h))
        scales.append(h / wh)
    native = sum(1 for s in scales if s == 1.0)
    jump = max(abs(a - b) for a, b in zip(scales, scales[1:]))
    assert native >= 30 and jump < 0.02, (native, jump)
    # identity through crop and paste back
    lin = plate(W // 4, H // 4)
    lin = np.ascontiguousarray(np.repeat(np.repeat(lin, 4, 0), 4, 1))[:H, :W]
    win = wins[10]
    alpha = np.zeros((H, W), np.float32)
    b = boxes[10]
    alpha[b[1]:b[3], b[0]:b[2]] = 1
    x0, y0, w, h = win
    m = li.make_mask(alpha[y0:y0 + h, x0:x0 + w], (ww, wh), shadow_px=10, dilate_px=4)
    ltx_out = li.crop(li.to_view(lin, fwd), win, (ww, wh))
    rgba = li.return_window(ltx_out, m, win, (W, H), inv, feather_px=2)
    comp = cp.over(rgba, lin)
    out = rgba[..., 3] == 0
    assert np.array_equal(comp[out], lin[out])
    inside = rgba[..., 3] > 0.99
    rel = np.abs(comp[inside] - lin[inside]).mean() / lin[inside].mean()
    assert rel < 0.01, rel
    print(f"ok crop windows: element always inside, window >= work size and inside the plate, {native}/121 frames at "
          f"native 1:1, max scale step {jump:.4f}/frame; crop -> paste back identity within {rel * 100:.3f} %")


def main() -> int:
    fwd, inv = li.view_processors(OCIO)
    test_roundtrip(fwd, inv)
    test_mask()
    test_identity(fwd, inv)
    test_windows(fwd, inv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
