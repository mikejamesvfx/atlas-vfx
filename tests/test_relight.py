"""Relight + depth holdout (layers/relight.py) on synthetic data: a sphere element, a plate with a sun blob, a ground
plane with a strip of grass in front of the element's legs.

Run: <V141 python> tests\\test_relight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import relight as rl  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail is not None and str(detail) != "" else ""))
    if not cond:
        FAILS.append(label)


H, W, T = 120, 160, 6


def sphere():
    """OpenGL normals and matte of a sphere centred in frame, radius 40 px."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = (xx - W / 2) / 40, -(yy - H / 2) / 40
    r2 = x * x + y * y
    al = (r2 < 1).astype(np.float32)
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    n = np.stack([x, y, z], -1) * al[..., None] + np.array([0, 0, 1], np.float32) * (1 - al[..., None])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.repeat(n[None], T, 0), np.repeat(al[None], T, 0)


def plate(sun_uv=(0.15, 0.1)):
    bg = np.full((T, H, W, 3), 0.05, np.float32)
    bg[:, : H // 3] = (0.30, 0.25, 0.18)         # warm sky
    bg[:, 2 * H // 3:] = (0.12, 0.09, 0.04)      # golden ground
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - sun_uv[0] * W) ** 2 + (yy - sun_uv[1] * H) ** 2
    bg += (np.exp(-d2 / 30.0) * 8.0)[None, ..., None] * np.array([1.0, 0.6, 0.3], np.float32)
    return bg


def test_points_to_normals():
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float32)
    pts = np.stack([xx * 0.01, yy * 0.01, np.full_like(xx, 2.0)], -1)[None]   # plane facing the camera (OpenCV)
    n = rl.opencv_to_opengl(rl.normals_from_points(pts))[0, 5:15, 5:15]
    check(np.allclose(n.mean((0, 1)), [0, 0, 1], atol=1e-3), "plane facing camera -> OpenGL +z", n.mean((0, 1)).round(3))


def test_light_from_plate():
    rig = rl.light_from_plate(plate(), fov_x_deg=50, behind=0.6, smooth=2)
    u, v = rig["sun_uv"][0]
    check(abs(u - 0.15) < 0.03 and abs(v - 0.1) < 0.05, "sun found at its blob", (u, v))
    d = np.asarray(rig["sun_dir"][0])
    check(d[0] < 0 and d[1] > 0 and d[2] < 0, "sun up-left and behind the subject", d.round(3))
    c = np.asarray(rig["sun_colour"])
    check(c[0] == 1.0 and c[0] > c[1] > c[2], "sun colour warm, max channel 1", c)
    check(rig["sky"][0][0] > rig["ground"][0][0], "sky brighter than ground")
    man = rl.light_from_plate(plate(), sun_uv=(1.2, 0.5))
    check(man["sun_uv"][0] == [1.2, 0.5] and man["sun_dir"][0][0] > 0, "manual sun off frame right")
    # SDR plate: the sun core clips to white, its hue only survives in the glow around it
    clipped = rl.light_from_plate(np.minimum(plate(), 1.0), smooth=2)
    c = np.asarray(clipped["sun_colour"])
    check(c[0] == 1.0 and c[1] < 0.9 and c[2] < 0.7, "clipped sun core: colour taken from the glow", c)


def test_relight_rim_side():
    n, al = sphere()
    fg = np.full((T, H, W, 3), 0.4, np.float32) * al[..., None]
    bg = plate()
    rig = rl.light_from_plate(bg, smooth=2)
    relit, passes, gains = rl.relight(fg, al, n, rig, bg, sun_gain=4.0, rim_max=0)
    key = passes["key"][0] @ rl.LUM
    left = key[:, : W // 2][al[0][:, : W // 2] > 0].mean()
    right = key[:, W // 2:][al[0][:, W // 2:] > 0].mean()
    centre = key[H // 2 - 3:H // 2 + 3, W // 2 - 3:W // 2 + 3].mean()
    check(left > 5 * max(right, 1e-6), "rim lands on the sun side", (round(float(left), 4), round(float(right), 4)))
    check(centre < 0.1 * left, "sphere centre (facing camera) gets little rim", round(float(centre), 5))
    amb = passes["ambient"][0]
    top = amb[H // 2 - 35:H // 2 - 25, W // 2 - 3:W // 2 + 3].mean((0, 1))
    bot = amb[H // 2 + 25:H // 2 + 35, W // 2 - 3:W // 2 + 3].mean((0, 1))
    check((top @ rl.LUM) > (bot @ rl.LUM), "up-facing surface takes the brighter sky", (top.round(3), bot.round(3)))
    check(np.allclose(gains, gains[0], rtol=1e-3), "static shot -> constant exposure gain", gains[:3])


def test_rim_max_caps_key():
    n, al = sphere()
    fg = np.full((T, H, W, 3), 0.4, np.float32) * al[..., None]
    bg = plate()
    rig = rl.light_from_plate(bg, smooth=2)
    _, p0, _ = rl.relight(fg, al, n, rig, None, sun_gain=40.0, rim_max=0, exposure_match=False)
    _, p1, _ = rl.relight(fg, al, n, rig, None, sun_gain=40.0, rim_max=1.5, exposure_match=False)
    in_ = al[0] > 0.5
    ratio = (p1["key"][0] @ rl.LUM)[in_] / np.maximum((p1["ambient"][0] @ rl.LUM)[in_], 1e-6)
    check(ratio.max() <= 1.5 + 1e-3, "key capped at 1.5 x ambient", round(float(ratio.max()), 4))
    check((p0["key"][0] @ rl.LUM)[in_].max() > (p1["key"][0] @ rl.LUM)[in_].max(), "cap lowers the hot spot")


def test_exposure_smoothing():
    n, al = sphere()
    fg = np.full((T, H, W, 3), 0.4, np.float32) * al[..., None]
    bg = plate()
    bg[3] *= 4.0                                                     # one flash frame in the plate
    rig = rl.light_from_plate(plate(), smooth=2)
    _, _, g1 = rl.relight(fg, al, n, rig, bg, exposure_smooth=1)
    _, _, g5 = rl.relight(fg, al, n, rig, bg, exposure_smooth=5)
    jump = lambda g: max(abs(np.diff(np.log(g))))
    check(jump(g5) < 0.5 * jump(g1), "exposure smoothing damps a one-frame flash", (round(jump(g1), 3), round(jump(g5), 3)))


def test_depth_holdout():
    # ground plane: depth grows toward the top of frame; element standing with feet at row 100
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    d = np.repeat((2.0 + (H - yy) * 0.05)[None], T, 0)              # row 100 -> 3.0 m
    al = np.zeros((T, H, W), np.float32)
    al[:, 40:101, 70:90] = 1.0
    d[:, 85:95, 60:100] = 2.5                                        # grass strip in front of the legs, 0.5 m nearer
    occ, dist = rl.depth_holdout(d, al, margin=0.03, soft=0.03, foot_rows=4, smooth=1)
    check(abs(dist[0] - 2.95) < 0.1, "element distance read under the feet", dist[0])
    check(occ[0, 90, 80] > 0.99, "grass strip in front occludes the legs", occ[0, 90, 80])
    check(occ[0, 60, 80] == 0, "background behind the element does not occlude", occ[0, 60, 80])
    occ2, _ = rl.depth_holdout(d, al, distance_scale=0.8)
    check(occ2[0, 90, 80] < 0.01, "pulling the element forward (distance_scale 0.8) clears the holdout")


def test_registry():
    from atlas_vfx.registry import load_nodes
    classes, names, disabled = load_nodes(("atlas_vfx.layers",))
    for k in ("AtlasVFXNormals", "AtlasVFXLightFromPlate", "AtlasVFXRelight", "AtlasVFXDepthHoldout"):
        check(k in classes and k in names, f"{k} registered", disabled.get(k, ""))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
