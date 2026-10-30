"""Relight Transfer (layers/light_transfer.py, catalogue 53) on a synthetic scene with known ground truth.

A textured albedo on spheres and a bumpy wall is lit by light L1 (the element, full res) and by L2 (the "relit
take", rendered full res and then shrunk 4x like an LTX pass). Truth is the albedo under L2 at full res. Route A
(guided) must beat a plain upscale of the relit take; route B (sh, true normals) must beat A where the normals carry
fine detail; pixels outside the matte stay bit-exact; a bright background next to the matte must not change the gain
inside it.

Run: <V141 python> tests\\test_relight_transfer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from atlas_vfx.layers import light_transfer as lt  # noqa: E402
from atlas_vfx.layers import relight as rl  # noqa: E402

FAILS = []
W, H, DOWN = 768, 384, 4


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


def scene(seed=0):
    """Normals (H, W, 3) OpenGL camera space, matte (H, W), albedo (H, W, 3), bump region mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n = np.zeros((H, W, 3), np.float32)
    n[..., 2] = 1.0
    matte = np.zeros((H, W), np.float32)
    bump = np.zeros((H, W), bool)
    # bumpy wall on the right: fine normal detail (period ~6 px, gone at 4x down)
    wall = (xx > W * 0.55) & (xx < W * 0.95) & (yy > H * 0.1) & (yy < H * 0.9)
    nx = 0.45 * np.sin(xx * 2 * np.pi / 6.0) * np.cos(yy * 2 * np.pi / 9.0)
    ny = 0.45 * np.cos(xx * 2 * np.pi / 7.0) * np.sin(yy * 2 * np.pi / 6.0)
    n[wall] = np.stack([nx, ny, np.ones_like(nx)], -1)[wall]
    matte[wall] = 1.0
    bump |= wall
    # two spheres on the left
    for cx, cy, rad in ((W * 0.2, H * 0.5, H * 0.32), (W * 0.42, H * 0.35, H * 0.16)):
        dx, dy = (xx - cx) / rad, -(yy - cy) / rad
        r2 = dx * dx + dy * dy
        inside = r2 < 1
        n[inside] = np.stack([dx, dy, np.sqrt(np.clip(1 - r2, 0, 1))], -1)[inside]
        matte[inside] = 1.0
    n = lt._unit(n)
    tex = rng.uniform(0.25, 0.9, (H // 2, W // 2, 3)).astype(np.float32)
    tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_NEAREST)        # sharp 2 px texture
    albedo = tex * np.array([0.9, 0.75, 0.6], np.float32)
    return n, matte, albedo, bump


def shade(n, L, col, amb):
    L = np.asarray(L, np.float32) / np.linalg.norm(L)
    lam = np.clip(n @ L, 0, None)[..., None]
    return lam * np.asarray(col, np.float32) + np.asarray(amb, np.float32)


def render(albedo, n, matte, L, col, amb, bg):
    img = albedo * shade(n, L, col, amb)
    return np.where(matte[..., None] > 0, img, bg).astype(np.float32)


def down(img):
    return cv2.resize(img, (W // DOWN, H // DOWN), interpolation=cv2.INTER_AREA)


def err_stops(a, b, sel):
    """RMS error in stops over selected pixels."""
    d = np.log2(np.maximum(a, 1e-4)) - np.log2(np.maximum(b, 1e-4))
    return float(np.sqrt((d[sel] ** 2).mean()))


def setup(bg_value=0.18):
    n, matte, albedo, bump = scene()
    bg = np.full((H, W, 3), bg_value, np.float32)
    L1, c1, a1 = (0.2, 0.3, 1.0), (1.0, 1.0, 1.0), (0.15, 0.15, 0.15)
    L2, c2, a2 = (-0.8, 0.5, 0.5), (1.4, 0.9, 0.55), (0.1, 0.12, 0.18)        # warm low sun from the left, cool fill
    element = render(albedo, n, matte, L1, c1, a1, bg)
    truth = render(albedo, n, matte, L2, c2, a2, bg)
    relit_low = down(truth)
    return n, matte, bump, element, truth, relit_low


def test_quality():
    n, matte, bump, element, truth, relit_low = setup()
    inner = cv2.erode(matte, np.ones((9, 9), np.uint8)) > 0.5
    plain = cv2.resize(relit_low, (W, H), interpolation=cv2.INTER_CUBIC)
    e_plain = err_stops(plain, truth, inner)
    outA, gainA, repA = lt.transfer(element[None], relit_low[None], matte[None], mode="guided",
                                    analysis_width=0)
    smooth = inner & ~bump                     # the spheres: texture, smooth normals (A's case)
    e_A, e_plain_s = err_stops(outA[0], truth, smooth), err_stops(plain, truth, smooth)
    check(e_A < 0.5 * e_plain_s, "guided beats a plain upscale of the relit take on the spheres",
          f"{e_A:.3f} vs {e_plain_s:.3f} stops")
    check(err_stops(outA[0], truth, inner) < e_plain, "guided beats the plain upscale overall")
    check(repA["analysis_size"] == [W // DOWN, H // DOWN], "analysis size = the relit take's when width 0",
          repA["analysis_size"])

    outB, gainB, repB = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", analysis_width=0,
                                    normals=n[None])
    sel = bump & inner
    eA_b, eB_b = err_stops(outA[0], truth, sel), err_stops(outB[0], truth, sel)
    check(eB_b < 0.8 * eA_b, "sh beats guided on the fine-normal wall", f"{eB_b:.3f} vs {eA_b:.3f} stops")
    check(err_stops(outB[0], truth, inner) < e_plain, "sh beats the plain upscale overall")
    check(repB["sh_frames"] == 1 and repB["sh_r2"][0] > 0.5, "sh fit used, r2 reported", repB["sh_r2"])

    outside = matte == 0
    check(np.array_equal(outA[0][outside], element[outside]), "guided: pixels outside the matte are bit-exact")
    check(np.array_equal(outB[0][outside], element[outside]), "sh: pixels outside the matte are bit-exact")
    check(np.all(gainA[0][outside] == 0), "gain is 0 outside the matte")


def test_halo():
    """A very bright background must not change the gain just inside the matte edge."""
    res = []
    for bg in (0.18, 8.0):
        n, matte, bump, element, truth, relit_low = setup(bg)
        _, gain, _ = lt.transfer(element[None], relit_low[None], matte[None], mode="guided", analysis_width=0)
        res.append(gain[0])
    ring = (matte > 0.5) & (cv2.dilate(1 - matte, np.ones((7, 7), np.uint8)) > 0.5)
    d = float(np.abs(res[0][ring] - res[1][ring]).max())
    check(d < 0.05, "bright background: edge gain moves < 0.05 stops", f"{d:.4f}")


def test_clamp_temporal_fallback():
    n, matte, bump, element, truth, relit_low = setup()
    hot = relit_low * 1000.0
    _, gain, rep = lt.transfer(element[None], hot[None], matte[None], max_stops=2.0, analysis_width=0)
    check(float(gain.max()) <= 2.0 + 1e-5, "gain clamped to max_stops", f"{gain.max():.3f}")
    check(rep["clamped_fraction_max"] > 0.9, "clamp fraction reported", rep["clamped_fraction_max"])

    # three frames, middle one noisy: temporal 1 pulls it toward its neighbours
    rng = np.random.default_rng(3)
    noisy = relit_low * np.exp2(rng.normal(0, 0.6, relit_low.shape)).astype(np.float32)
    seq_e = np.stack([element] * 3)
    seq_r = np.stack([relit_low, noisy, relit_low])
    seq_m = np.stack([matte] * 3)
    _, g0, _ = lt.transfer(seq_e, seq_r, seq_m, temporal=0, analysis_width=0)
    _, g1, _ = lt.transfer(seq_e, seq_r, seq_m, temporal=1, analysis_width=0)
    inner = cv2.erode(matte, np.ones((9, 9), np.uint8)) > 0.5
    flick0 = float(np.abs(g0[1] - g0[0])[inner].mean())
    flick1 = float(np.abs(g1[1] - g1[0])[inner].mean())
    check(flick1 < 0.6 * flick0, "temporal smoothing cuts frame-to-frame gain change", f"{flick1:.3f} vs {flick0:.3f}")

    # normals that have nothing to do with the light: the fit is poor, the frame falls back to guided
    rand_n = lt._unit(rng.normal(0, 1, n.shape).astype(np.float32))
    _, _, rep = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", normals=rand_n[None],
                            analysis_width=0, min_r2=0.3)
    check(rep["fallback_frames"] == [0], "sh with unrelated normals falls back to guided", rep["sh_r2"])

    # shot camera rotation: world-space normals with the matching rotation give the same result as camera space
    R = cv2.Rodrigues(np.array([0.1, 0.5, 0.0]))[0].astype(np.float32)
    o_cam, _, _ = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", normals=n[None],
                              analysis_width=0)
    o_w, _, rep = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", normals=n[None],
                              analysis_width=0, cam_rotations=R[None])
    check(rep["camera"].startswith("world"), "camera reported as world")
    check(float(np.abs(o_cam - o_w).max()) < 2e-3 * float(o_cam.max()), "one-frame sh fit is rotation invariant")


def test_guides_and_errors():
    n, matte, bump, element, truth, relit_low = setup()
    depth = np.where(matte > 0, 3.0, 20.0).astype(np.float32)
    out, _, rep = lt.transfer(element[None], relit_low[None], matte[None], analysis_width=0, normals=n[None],
                              depth=depth[None])
    check(rep["guides"] == ["luma", "depth", "normals"], "depth and normals used as guides", rep["guides"])
    inner = cv2.erode(matte, np.ones((9, 9), np.uint8)) > 0.5
    plain = cv2.resize(relit_low, (W, H), interpolation=cv2.INTER_CUBIC)
    check(err_stops(out[0], truth, inner) < err_stops(plain, truth, inner), "multi-channel guide still beats plain")
    lowres_matte = cv2.resize(matte, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    out2, _, _ = lt.transfer(element[None], relit_low[None], lowres_matte[None], analysis_width=0)
    check(out2.shape == element[None].shape, "a matte at another size is resized")
    empty = np.zeros_like(matte)
    out3, g3, _ = lt.transfer(element[None], relit_low[None], empty[None], analysis_width=0)
    check(np.array_equal(out3[0], element) and not g3.any(), "empty matte: element passes through")

    raises(lambda: lt.transfer(element[None], relit_low[None], matte[None], mode="sh"), "sh without normals",
           "needs normals")
    raises(lambda: lt.transfer(element[None], relit_low[:, : W // 8][None], matte[None]), "framing mismatch",
           "framing")
    raises(lambda: lt.transfer(np.stack([element] * 2), np.stack([relit_low] * 3), matte[None]), "frame count",
           "frames")
    raises(lambda: lt.transfer(element[None], relit_low[None], matte[None], mode="lab"), "bad mode", "one of")


def test_guided_filter_identity():
    rng = np.random.default_rng(1)
    I = rng.normal(0, 1, (40, 50, 2)).astype(np.float32)
    p = (2.0 * I[..., :1] - 0.5 * I[..., 1:2] + 0.3).astype(np.float32)
    q = lt.guided_filter(I, p, 3, 1e-6)
    check(float(np.abs(q - p).max()) < 1e-3, "guided filter reproduces a linear function of its guide",
          f"{np.abs(q - p).max():.2e}")


def test_node():
    import math

    import torch

    from atlas_vfx.contracts import shot_camera as sc
    from atlas_vfx.layers.node_relight_transfer import AtlasVFXRelightTransfer

    n, matte, bump, element, truth, relit_low = setup()
    node = AtlasVFXRelightTransfer()
    kw = dict(encoding="linear", mode="sh", analysis_width=0, blur_px=2.0, max_stops=3.0, temporal=0, eps=0.01,
              geo_weight=0.5, min_r2=0.3)
    t = (lambda a: torch.from_numpy(np.ascontiguousarray(a[None])))
    img, gain, info = node.run(t(element), t(matte), t(relit_low), normals=t(n * 0.5 + 0.5), **kw)
    want, _, _ = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", analysis_width=0,
                             normals=lt.decode_normals(n * 0.5 + 0.5)[None])
    check(np.allclose(img.numpy(), want, atol=1e-6), "node = module (sh, encoded normals)")
    check(gain.shape == img.shape and json.loads(info)["mode"] == "sh", "node gives gain and info")

    srgb = (lambda a: rl.lin_to_srgb(a))
    e2, r2 = element * 0.4, relit_low * 0.4                                         # below 1: sRGB clips there
    img_s, _, _ = node.run(t(srgb(e2)), t(matte), t(srgb(r2)), **{**kw, "mode": "guided", "encoding": "srgb"})
    lin, _, _ = node.run(t(e2), t(matte), t(r2), **{**kw, "mode": "guided"})
    d = float(np.abs(img_s.numpy() - srgb(lin.numpy())).max())
    check(d < 1e-3, "srgb encoding round-trips to the linear result", f"{d:.2e}")

    def ry(deg):
        a = math.radians(deg)
        M = np.eye(4)
        M[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        return M
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    cam = sc.build(name="rt", frame_start=1001, fps=24.0, plate=plate, sensor={"width_mm": 23.5, "height_mm": 11.75},
                   cam_to_world=[ry(20.0)], focal_mm=16.0, source={"kind": "synthetic", "solver": "test"})
    _, _, info = node.run(t(element), t(matte), t(relit_low), normals=t(n * 0.5 + 0.5), shot_camera=cam.data,
                          first_frame=1001, **kw)
    rep = json.loads(info)
    check(rep["camera"].startswith("world") and rep["shot_frames"] == [1001, 1001], "shot camera frames used")
    raises(lambda: node.run(t(element), t(matte), t(relit_low), normals=t(n * 0.5 + 0.5), shot_camera=cam.data,
                            first_frame=1100, **kw), "shot camera without the frame", "first_frame")


def main():
    test_guided_filter_identity()
    test_quality()
    test_halo()
    test_clamp_temporal_fallback()
    test_guides_and_errors()
    test_node()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
