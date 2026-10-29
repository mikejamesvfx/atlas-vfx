"""T6: Card Project and Foreground Element (synthetic).

- a card whose texture encodes its own (u, v) is rendered through a synthetic camera at 1024; for every covered pixel
  the 3D point at the sampled (u, v) is projected forward with ShotCamera.project and must land on that pixel's centre
  (max error < 0.5 px at 1024, plan). The forward model is independent of the renderer's ray/plane maths.
- a checkerboard card: rendered checker parity matches the parity computed from the forward-projected (u, v)
- a card the camera walks through renders without NaN, and only its in-front part
- placement: card bottom on the ground, facing the camera, at the asked distance; texture colour goes sRGB -> ACEScg
  (mid grey 0.5 sRGB -> 0.214 linear) and decontamination removes the backdrop from the edge
- RGBA EXR round trip keeps premultiplied values (half) and says ACEScg (colorInteropID lin_ap1_scene)

Run: <V141 python> tests\\test_card_project.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.layers import card_project as cp  # noqa: E402
from atlas_vfx.layers import foreground_element as fe  # noqa: E402
from test_depth_set_proxy import FRAMES, camera  # noqa: E402

SCALE = 0.25                                    # 4096 -> 1024
OCIO = str(ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio")


def uv_texture(tw=257, th=129):
    u = (np.arange(tw) + 0.5) / tw
    v = (np.arange(th) + 0.5) / th
    U, V = np.meshgrid(u, v)
    return np.stack([U, V, np.zeros_like(U), np.ones_like(U)], -1).astype(np.float32)


def test_geometry(cam):
    corners = [[-0.8, 1.6, -4.0], [0.9, 1.7, -4.5], [0.9, 0.2, -4.6], [-0.8, 0.1, -4.1]]   # tilted, not square-on
    tex = uv_texture()
    worst, n = 0.0, 0
    for f in FRAMES[::7]:
        img = cp.render(cam, f, corners, tex, scale=SCALE)
        a = img[..., 3]
        ys, xs = np.nonzero(a > 0.999)                     # fully inside (no edge blend)
        u, v = img[ys, xs, 0], img[ys, xs, 1]               # premult with a = 1
        P0, U, V, _ = cp.card_frame(corners)
        P = P0 + u[:, None] * U + v[:, None] * V
        uv, depth = cam.project(f, P)
        err = np.hypot(uv[:, 0] * SCALE - (xs + 0.5), uv[:, 1] * SCALE - (ys + 0.5))
        # bilinear sampling of a linear ramp is exact except in the half texel at the border: skip 1 px there
        inner = (u > 1.5 / tex.shape[1]) & (u < 1 - 1.5 / tex.shape[1]) & (v > 1.5 / tex.shape[0]) & \
                (v < 1 - 1.5 / tex.shape[0])
        worst = max(worst, float(err[inner].max()))
        n += int(inner.sum())
    assert worst < 0.5, f"card pixels off by {worst:.3f} px at 1024"
    print(f"ok geometry: {n} covered pixels over {len(FRAMES[::7])} frames land within {worst:.4f} px of the forward "
          f"projection at 1024")


def test_checker(cam):
    corners = [[-1.0, 1.8, -5.0], [1.0, 1.8, -5.0], [1.0, 0.0, -5.0], [-1.0, 0.0, -5.0]]
    tw, th, sq = 400, 360, 40
    yy, xx = np.mgrid[0:th, 0:tw]
    chk = (((xx // sq) + (yy // sq)) % 2).astype(np.float32)
    tex = np.stack([chk, chk, chk, np.ones_like(chk)], -1)
    f = FRAMES[0]
    img = cp.render(cam, f, corners, tex, scale=SCALE)
    uvt = cp.render(cam, f, corners, uv_texture(tw, th), scale=SCALE)
    ys, xs = np.nonzero(uvt[..., 3] > 0.999)
    u, v = uvt[ys, xs, 0], uvt[ys, xs, 1]
    want = ((np.floor(u * tw / sq) + np.floor(v * th / sq)) % 2)
    got = img[ys, xs, 0]
    away = (np.abs((u * tw / sq) % 1 - 0.5) < 0.4) & (np.abs((v * th / sq) % 1 - 0.5) < 0.4)   # off square edges
    bad = np.abs(got[away] - want[away]) > 0.01
    assert not bad.any(), f"{bad.sum()} checker pixels wrong"
    print(f"ok checker: {away.sum()} pixels away from square edges all match the analytic checker")


def test_walk_through(cam):
    f = FRAMES[12]
    M = cam.cam_to_world(f)
    C, fwd = M[:3, 3], -M[:3, 2]
    r = M[:3, 0]
    # a big card straddling the camera: half in front, half behind
    corners = [C - r * 2 + [0, 1, 0] - fwd * 1, C + r * 2 + [0, 1, 0] - fwd * 1,
               C + r * 2 + [0, 1, 0] + fwd * 3, C - r * 2 + [0, 1, 0] + fwd * 3]
    img = cp.render(cam, f, corners, uv_texture(), scale=SCALE)
    assert np.all(np.isfinite(img)) and img[..., 3].max() > 0.99
    ys, xs = np.nonzero(img[..., 3] > 0.999)
    v = img[ys, xs, 1]
    assert v.min() > 0.25 - 0.02, f"pixels from behind the camera were drawn (v min {v.min():.3f})"
    print(f"ok walk-through: card straddling the camera renders only its front part (v from {v.min():.3f}), no NaN")


def test_texture_and_place(cam):
    H, W = 64, 48
    img = np.full((H, W, 3), [0.8, 0.8, 0.8], np.float32)          # light grey backdrop
    m = np.zeros((H, W), np.float32)
    m[10:60, 14:34] = 1.0
    m[10:60, 13] = 0.5                                             # a half-covered edge column
    img[10:60, 14:34] = 0.5
    img[10:60, 13] = 0.5 * 0.5 + 0.8 * 0.5                         # mixed with the backdrop
    rgba, rep = fe.make_texture(img, m, OCIO, erode_px=0)
    mid = 0.5 ** 2.2                                               # not exact sRGB, so compare with OCIO below
    import PyOpenColorIO as ocio
    cfg = ocio.Config.CreateFromFile(OCIO)
    ref = np.array([[[0.5, 0.5, 0.5]]], np.float32)
    cfg.getProcessor(fe.GEN_SPACE, "ACEScg").getDefaultCPUProcessor().applyRGB(ref)
    inner = rgba[rgba[..., 3] > 0.99][:, :3]
    assert np.allclose(inner, ref[0, 0], atol=1e-4), (inner[0], ref)
    assert abs(float(ref[0, 0, 0]) - 0.214) < 0.002, ref
    edge = rgba[np.isclose(rgba[..., 3], 0.5)]
    assert len(edge) and np.allclose(edge[:, :3] / 0.5, ref[0, 0], atol=2e-3), "edge not decontaminated"
    assert rep["size"] == [20 + 1 + 8, 50 + 8], rep["size"]
    print(f"ok texture: sRGB 0.5 -> ACEScg {ref[0, 0, 0]:.4f} (gamma-2.2 guess would be {mid:.4f}), backdrop "
          f"{rep['backdrop_srgb'][0]:.2f} removed from a 50 % edge, cropped to {rep['size']}")

    f = FRAMES[0]
    card = fe.place_card(cam, f, distance_m=2.0, lateral_m=0.4, height_m=0.9, aspect=0.6, ground_y=-0.25)
    TL, TR, BR, BL = map(np.array, card["corners"])
    assert abs(BL[1] + 0.25) < 1e-9 and abs(BR[1] + 0.25) < 1e-9 and abs(TL[1] - BL[1] - 0.9) < 1e-9
    C = cam.position(f)
    base = (BL + BR) / 2
    assert abs(math.hypot(*(base - C)[[0, 2]]) - math.hypot(2.0, 0.4)) < 1e-9
    n = np.cross(TR - TL, BL - TL)
    to_cam = np.array([C[0] - base[0], 0, C[2] - base[2]])
    assert np.cross(n, to_cam)[1] ** 2 < 1e-12 or abs(abs(n @ to_cam) / (np.linalg.norm(n) * np.linalg.norm(to_cam)) - 1) < 0.05
    uv, d = cam.project(f, base)
    assert d[0] > 0 and uv[0, 0] > 2048, "card should be right of centre"
    print(f"ok placement: bottom on y -0.25, 0.9 m tall, {math.hypot(2.0, 0.4):.3f} m from the camera, right of "
          f"centre, facing it")

    with tempfile.TemporaryDirectory() as t:
        p = exr_io.write_rgba(Path(t) / "e.exr", rgba, attrs={"card": card})
        back, attrs = exr_io.read_rgba(p)
        assert np.allclose(back, rgba, atol=2e-3) and attrs["colorInteropID"] == "lin_ap1_scene"
        assert attrs["ocio_colorspace"] == "ACEScg" and attrs["premultiplied"] == 1
    print("ok exr: RGBA half round trip, colorInteropID lin_ap1_scene, premultiplied flagged")


def main() -> int:
    cam = camera()
    test_geometry(cam)
    test_checker(cam)
    test_walk_through(cam)
    test_texture_and_place(cam)
    return 0


if __name__ == "__main__":
    sys.exit(main())
