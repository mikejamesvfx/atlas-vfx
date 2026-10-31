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
              subsample=4, geo_weight=0.5, min_r2=0.3)
    t = (lambda a: torch.from_numpy(np.ascontiguousarray(a[None])))
    img, gain, info = node.run(t(element), t(matte), t(relit_low), normals=t(n * 0.5 + 0.5), **kw)
    want, _, _ = lt.transfer(element[None], relit_low[None], matte[None], mode="sh", analysis_width=0, subsample=4,
                             normals=lt.decode_normals(n * 0.5 + 0.5)[None])
    check(np.allclose(img.numpy(), want, atol=1e-6), "node = module (sh, encoded normals)")
    check(gain.shape == img.shape and json.loads(info)["mode"] == "sh", "node gives gain and info")
    img_m, _, info_m = node.run(t(element), t(matte), t(relit_low[:, ::-1]), normals=t(n * 0.5 + 0.5),
                                relit_mirrored=True, **kw)
    check(np.array_equal(img_m.numpy(), img.numpy()) and json.loads(info_m)["relit_mirrored"],
          "relit_mirrored flips a mirrored take back (same result as the unflipped take)")

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


def test_fast_guided():
    """subsample 4 (fast guided filter) against 1 (exact): within a few hundredths of a stop, as close to the
    truth, faster."""
    import time

    n, matte, bump, element, truth, relit_low = setup()
    inner = cv2.erode(matte, np.ones((9, 9), np.uint8)) > 0.5
    depth = np.where(matte > 0, 3.0, 20.0).astype(np.float32)
    args = (element[None], relit_low[None], matte[None])
    kw = dict(analysis_width=0, normals=n[None], depth=depth[None])
    t0 = time.perf_counter()
    exact, g1, _ = lt.transfer(*args, subsample=1, **kw)
    t1 = time.perf_counter()
    fast, g4, rep = lt.transfer(*args, subsample=4, **kw)
    t2 = time.perf_counter()
    d = float(np.sqrt(((g4[0] - g1[0])[inner] ** 2).mean()))
    check(d < 0.05, "fast guided filter (subsample 4) matches exact", f"{d:.4f} stops rms")
    check(rep["subsample"] == 4, "subsample reported")
    e_exact, e_fast = err_stops(exact[0], truth, inner), err_stops(fast[0], truth, inner)
    check(e_fast < e_exact + 0.02, "fast is as close to the truth as exact", f"{e_fast:.3f} vs {e_exact:.3f} stops")
    check(t2 - t1 < t1 - t0, "fast is faster (5-channel guide)", f"{t2 - t1:.2f} s vs {t1 - t0:.2f} s")
    rng = np.random.default_rng(1)
    I = rng.normal(0, 1, (80, 100, 2)).astype(np.float32)
    I = cv2.GaussianBlur(I, (0, 0), 3.0)
    p = (2.0 * I[..., :1] - 0.5 * I[..., 1:2] + 0.3).astype(np.float32)
    q = lt.guided_filter(I, p, 4, 1e-6, subsample=2)
    check(float(np.abs(q - p).max()) < 2e-2, "fast guided filter keeps a linear function of a smooth guide",
          f"{np.abs(q - p).max():.2e}")


def test_holdout():
    """LTX paints an object the plate does not have: a bright square in the relit take. Without a holdout the transfer
    lifts the real pixels there (a soft smudge); with the square held out, the light there comes from around it."""
    n, matte, bump, element, truth, relit_low = setup()
    smooth = (cv2.erode(matte, np.ones((41, 41), np.uint8)) > 0.5) & ~bump
    ys, xs = np.nonzero(smooth[::DOWN, ::DOWN])
    k = len(ys) // 2
    cy, cx = int(ys[k]), int(xs[k])
    bad = relit_low.copy()
    bad[cy - 3:cy + 4, cx - 3:cx + 4] *= 6.0                                     # ~2.6 stops of invented content
    hold = np.zeros(matte.shape, np.float32)
    hold[(cy - 3) * DOWN:(cy + 4) * DOWN, (cx - 3) * DOWN:(cx + 4) * DOWN] = 1.0  # roto at full res
    patch = hold > 0.5
    ring = (cv2.dilate(hold, np.ones((81, 81), np.uint8)) > 0.5) & ~patch & smooth
    kw = dict(mode="guided", analysis_width=0)
    clean, _, _ = lt.transfer(element[None], relit_low[None], matte[None], **kw)
    raw, _, _ = lt.transfer(element[None], bad[None], matte[None], **kw)
    fix, _, rep = lt.transfer(element[None], bad[None], matte[None], holdout=hold[None], **kw)
    e_raw, e_fix, e_clean = (err_stops(o[0], truth, patch) for o in (raw, fix, clean))
    # guided fills the hole smoothly from its edge; on a small sphere the true light bends inside it, so about half
    check(e_fix < 0.6 * e_raw, "guided: holdout cuts the invented patch", f"{e_fix:.3f} vs {e_raw:.3f} stops "
          f"(clean {e_clean:.3f})")
    d = float(np.abs(fix[0][ring] - raw[0][ring]).max() / truth[ring].mean())
    check(err_stops(fix[0], truth, ring) <= err_stops(raw[0], truth, ring) + 1e-4,
          "around the holdout no worse", f"max change {d:.3f}")
    outside = matte <= 0
    check(np.array_equal(fix[0][outside], element[outside]), "holdout: outside the matte still bit-exact")
    check(rep["holdout"]["matte_fraction_max"] > 0, "report gives the held-out fraction", rep["holdout"])
    kws = dict(mode="sh", analysis_width=0, normals=n[None])
    sh, _, _ = lt.transfer(element[None], bad[None], matte[None], holdout=hold, **kws)
    sh_clean, _, _ = lt.transfer(element[None], relit_low[None], matte[None], **kws)
    e_sh, e_shc = err_stops(sh[0], truth, patch), err_stops(sh_clean[0], truth, patch)
    check(e_sh < e_shc + 0.05, "sh: under the holdout the normals carry the light (as good as no invention)",
          f"{e_sh:.3f} vs clean {e_shc:.3f} stops")


def test_lift_limit():
    """A near-black patch of the element that the relit take lifts by ~5 stops: the plate's noise times 30. Above
    lift_limit the relit take's own pixels take over; below it nothing changes."""
    n, matte, bump, element, truth, relit_low = setup()
    rng = np.random.default_rng(3)
    el = element.copy()
    smooth = (cv2.erode(matte, np.ones((41, 41), np.uint8)) > 0.5) & ~bump
    ys, xs = np.nonzero(smooth)
    cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    dark = np.zeros(matte.shape, bool)
    dark[cy - 24:cy + 24, cx - 24:cx + 24] = True
    el[dark] = 0.004 + 0.003 * rng.standard_normal((int(dark.sum()), 3)).astype(np.float32).clip(-1, 1)
    rl = relit_low.copy()
    lo = cv2.resize(dark.astype(np.float32), (rl.shape[1], rl.shape[0]), interpolation=cv2.INTER_AREA) > 0.99
    rl[lo] = 0.12                                                                  # lit, flat: ~5 stops up
    kw = dict(mode="guided", analysis_width=0, max_stops=6.0)
    off, _, _ = lt.transfer(el[None], rl[None], matte[None], lift_limit=0, **kw)
    on, _, rep = lt.transfer(el[None], rl[None], matte[None], lift_limit=3.0, **kw)
    core = np.zeros_like(dark)
    core[cy - 12:cy + 12, cx - 12:cx + 12] = True
    noise = lambda o: float((o[0][core] @ lt.LUMA_AP1).std() / (o[0][core] @ lt.LUMA_AP1).mean())
    check(noise(on) < 0.3 * noise(off), "lift_limit: the lifted plate noise is replaced by the relit take",
          f"relative noise {noise(on):.3f} vs {noise(off):.3f}")
    far = smooth & ~(cv2.dilate(dark.astype(np.uint8), np.ones((121, 121), np.uint8)) > 0)
    check(np.array_equal(on[0][far], off[0][far]), "lift_limit: nothing changes where the lift is small")
    check(0 < rep["relit_pixels_fraction_max"] < 0.2, "report gives the swapped fraction", rep["relit_pixels_fraction_max"])


class FakeMoGe:
    """Stands in for ComfyUI's MOGE_MODEL: returns the scene's true normals (OpenCV, as MoGe does) and a depth."""

    def __init__(self, n, depth):
        self.n, self.depth, self.calls = n, depth, 0

    def infer(self, x, **kw):
        import torch

        self.calls += 1
        assert x.shape == (1, 3, H, W), x.shape
        return {"normal": torch.from_numpy(rl.opencv_to_opengl(self.n)[None].copy()),     # its own inverse
                "depth": torch.from_numpy(self.depth[None].copy())}


def test_files():
    """Relight Transfer Files streams EXRs from disk and gives what the in-memory transfer gives on the same
    (half-float) frames, runs MoGe once per frame, and its RAM does not grow with the number of frames."""
    import os
    import tempfile
    import tracemalloc

    from atlas_vfx.camera_side import exr_io
    from atlas_vfx.layers.node_relight_transfer_files import AtlasVFXRelightTransferFiles

    n, matte, bump, element, truth, relit_low = setup()
    tmp = Path(tempfile.mkdtemp(prefix="rtf_"))
    os.environ["ATLAS_VFX_OUT_ROOTS"] = str(tmp)
    T, F0 = 5, 1001
    for t in range(T):
        k = 1.0 + 0.05 * t                                                      # the light changes a little
        exr_io.write_rgba(tmp / "el" / f"el.{F0 + t}.exr", np.dstack([element, matte]))
        rgba = np.dstack([relit_low * k, np.ones(relit_low.shape[:2], np.float32)])
        exr_io.write_rgba(tmp / "rl" / f"rl.{F0 + t}.exr", rgba)
        exr_io.write_rgba(tmp / "rlm" / f"rlm.{F0 + t}.exr", np.ascontiguousarray(rgba[:, ::-1]))
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    depth = (2.0 + yy / H).astype(np.float32)
    moge = FakeMoGe(n, depth)
    node = AtlasVFXRelightTransferFiles()
    kw = dict(element_path=str(tmp / "el" / "el.####.exr"), relit_path=str(tmp / "rl" / "rl.####.exr"),
              matte_path="", first_frame=F0, length=T, relit_numbering="frame", relit_encoding="linear", mode="sh",
              analysis_width=0, blur_px=2.0, max_stops=3.0, temporal=1, eps=0.01, subsample=2, geo_weight=0.5,
              min_r2=0.3, lift_limit=3.0, name="rt", out_dir=str(tmp / "out"))
    res = node.run(**kw, moge_model=moge, normals_temporal=1, normals_sigma=1.5)
    pattern, gpattern, info = res["result"]
    rep = json.loads(info)
    check(moge.calls == T and rep["moge"]["runs"] == T, "files: MoGe runs once per frame (normals cached on disk)",
          f"{moge.calls} runs for {T} frames")
    check(not (tmp / "out" / ".normals_cache").exists(), "files: the normals cache is deleted at the end")

    read = (lambda p: exr_io.read_rgba(p)[0])
    el = np.stack([read(tmp / "el" / f"el.{F0 + t}.exr") for t in range(T)])
    rel = np.stack([read(tmp / "rl" / f"rl.{F0 + t}.exr")[..., :3] for t in range(T)])
    m = el[..., 3]
    ns = rl.smooth_normals(np.repeat(n[None], T, 0), m, 0, 1.5).astype(np.float16).astype(np.float32)
    want, wgain, _ = lt.transfer(el[..., :3], rel, m, mode="sh", analysis_width=0, temporal=1, subsample=2,
                                 normals=ns, depth=np.repeat(depth[None], T, 0))
    got = np.stack([read(pattern % (F0 + t)) for t in range(T)])
    inside = m > 0.5
    d = np.abs(np.log2(np.maximum(got[..., :3], 1e-4)) - np.log2(np.maximum(want, 1e-4)))[inside]
    check(float(d.max()) < 0.01 and float(d.mean()) < 1e-3, "files = in-memory transfer (sh, MoGe normals, depth)",
          f"max {d.max():.4f} mean {d.mean():.5f} stops")
    check(np.array_equal(got[..., :3][m == 0], el[..., :3][m == 0]), "files: outside the matte bit-exact")
    check(np.allclose(got[..., 3], m), "files: alpha is the matte")
    g = read(gpattern % F0)
    check(np.abs(g[..., :3] - wgain[0]).max() < 0.01, "files: gain EXR written")
    check(rep["streamed"] and rep["frames_range"] == [F0, F0 + T - 1], "files: report")

    mir = node.run(**{**kw, "relit_path": str(tmp / "rlm" / "rlm.####.exr"), "out_dir": str(tmp / "outm")},
                   moge_model=FakeMoGe(n, depth), relit_mirrored=True)["result"][0]
    check(np.array_equal(read(mir % F0), got[0]), "files: relit_mirrored flips a mirrored take back")
    raises(lambda: node.run(**kw), "files: sh without a MoGe model", "moge_model")

    def peak(frames):
        tracemalloc.start()
        node.run(**{**kw, "mode": "guided", "length": frames, "out_dir": str(tmp / f"mem{frames}")})
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p
    for t in range(T, 20):
        exr_io.write_rgba(tmp / "el" / f"el.{F0 + t}.exr", np.dstack([element, matte]))
        exr_io.write_rgba(tmp / "rl" / f"rl.{F0 + t}.exr",
                          np.dstack([relit_low, np.ones(relit_low.shape[:2], np.float32)]))
    p4, p20 = peak(4), peak(20)
    check(p20 < 1.3 * p4, "files: RAM does not grow with the length", f"peak {p4 / 2**20:.0f} MB for 4 frames, "
          f"{p20 / 2**20:.0f} MB for 20 (one full frame {element.nbytes / 2**20:.1f} MB)")


def main():
    test_files()
    test_guided_filter_identity()
    test_fast_guided()
    test_quality()
    test_halo()
    test_clamp_temporal_fallback()
    test_guides_and_errors()
    test_holdout()
    test_lift_limit()
    test_node()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
