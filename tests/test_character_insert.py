"""T7: Character Insert (synthetic, geometry part).

- a figure clip on a graded grey backdrop (vignette, darker top, lighter floor) is keyed: alpha 1 inside, 0 on the
  backdrop, the backdrop plate comes back, the figure's
  height and feet row in texture pixels match the drawing within 1 px, edges are decontaminated
- the deck pick: the ray through the projection of a known deck point comes back to that point
- a 2.13 m figure card stands on the deck plane (y = 0.0) with billboard facing; rendered through a 100-frame synthetic
  camera walking at it and climbing (like S2-S3), the bottom edge of the rendered alpha in every leg column lies on the
  projected feet line within 1 px at 4096 (plan), and the figure's height in the world is 2.13 m
- time map: hold and ping-pong; card_at picks per-frame corners and texture paths

Run: <V141 python> tests\\test_character_insert.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import card_project as cp  # noqa: E402
from atlas_vfx.layers import character_insert as ci  # noqa: E402
from test_depth_set_proxy import PLATE, SENSOR, rot  # noqa: E402

OCIO = str(ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio")
FRAMES = list(range(1130, 1230))                       # 100 frames
SCALE = 0.5                                             # render at 2048, errors reported in 4096 px
GREY = np.array([0.52, 0.52, 0.53], np.float32)
FIG = np.array([0.24, 0.17, 0.11], np.float32)
CW, CH = 384, 672                                       # clip size (portrait, like the H3 generation)
HEAD, FEET = 60, 620                                    # figure rows: head top, one past the feet


def camera():
    mats = []
    for i, f in enumerate(FRAMES):
        a = i / (len(FRAMES) - 1)
        R = rot((0, 1, 0), 10 * math.sin(1.5 * a) - 5 * a) @ rot((1, 0, 0), -8 + 3 * a) @ rot((0, 0, 1), 0.8 * math.cos(3 * a))
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [0.3 - 0.9 * a, 0.65 + 0.8 * max(0.0, a - 0.5), -4.0 - 3.0 * a]     # walk in, climb the steps
        mats.append(M)
    return sc.build(name="syn_t7", frame_start=FRAMES[0], fps=24.0, plate=PLATE, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=16.47, source={"kind": "synthetic"},
                    scale={"method": "known_lengths", "m_per_source_unit": 1.0})


def backdrop():
    """A generator-style "flat grey": vignetted, darker at the top, a lighter floor below a soft seam."""
    yy, xx = np.mgrid[0:CH, 0:CW]
    xn, yn = (xx + 0.5) / CW * 2 - 1, (yy + 0.5) / CH * 2 - 1
    g = 0.42 + 0.06 * yn - 0.07 * (xn ** 2 + 0.5 * yn ** 2) + 0.05 / (1 + np.exp(-(yn - 0.6) * 12))
    return (GREY / GREY.mean() * g[..., None]).astype(np.float32)


def figure_clip(n=6, ss=4):
    """n frames of a blocky figure (head, body, two legs) on a graded grey backdrop, anti-aliased by supersampling,
    swaying its head."""
    frames = []
    H, W = CH * ss, CW * ss
    BG = backdrop()
    yy, xx = np.mgrid[0:H, 0:W] / ss
    for i in range(n):
        sway = 6 * math.sin(i)
        m = np.zeros((H, W), bool)
        m |= ((xx - (CW / 2 + sway)) / 34) ** 2 + ((yy - (HEAD + 40)) / 40) ** 2 <= 1      # head, top at HEAD
        m |= (abs(xx - CW / 2) <= 80) & (yy >= HEAD + 70) & (yy < 380)                       # body
        m |= (abs(xx - (CW / 2 - 42)) <= 28) & (yy >= 380) & (yy < FEET)                     # legs, feet at FEET
        m |= (abs(xx - (CW / 2 + 42)) <= 28) & (yy >= 380) & (yy < FEET)
        cov = m.reshape(CH, ss, CW, ss).mean((1, 3)).astype(np.float32)[..., None]
        frames.append(FIG * cov + BG * (1 - cov))
    return np.stack(frames).astype(np.float32)


def test_key():
    clip = figure_clip()
    alpha, B, info = ci.grey_key(clip, erode_px=0)
    assert np.abs(B - backdrop()).max() < 0.02, np.abs(B - backdrop()).max()
    assert alpha[:, 5:40, 5:40].max() == 0 and alpha[:, 200:300, CW // 2 - 40:CW // 2 + 40].min() == 1
    rgba, rep = ci.make_sequence(clip, alpha, B, OCIO)
    y0 = rep["crop"][1]
    assert abs(rep["figure_px"] - (FEET - HEAD)) <= 1, rep["figure_px"]
    assert abs(rep["feet_row"] + y0 - FEET) <= 1, (rep["feet_row"], y0)
    # decontamination: a half-covered edge pixel holds the figure colour, not grey
    import PyOpenColorIO as ocio
    ref = FIG[None, None].copy()
    ocio.Config.CreateFromFile(OCIO).getProcessor(ci.GEN_SPACE, "ACEScg").getDefaultCPUProcessor().applyRGB(ref)
    a = rgba[..., 3]
    edge = (a > 0.3) & (a < 0.7)
    col = rgba[..., :3][edge] / a[edge][:, None]
    assert edge.sum() > 50 and np.abs(col - ref[0, 0]).max() < 0.03, np.abs(col - ref[0, 0]).max()
    print(f"ok key: graded backdrop recovered within {np.abs(B - backdrop()).max():.4f} (fit rms "
          f"{info['residual_rms']:.4f}), figure {rep['figure_px']:.0f} px (drawn {FEET - HEAD}), feet row "
          f"{rep['feet_row'] + y0:.0f} (drawn {FEET}), {edge.sum()} edge pixels decontaminated to the figure colour")
    return rgba, rep


def test_mask_key():
    clip = figure_clip(n=3)
    cov = np.clip(np.abs(clip - backdrop()).sum(-1) / np.abs(FIG - backdrop()).sum(-1), 0, 1)   # exact coverage
    alpha, B, info = ci.mask_key(clip, cov, erode_px=0)
    rgba, rep = ci.make_sequence(clip, alpha, B, OCIO, method=ci.MASK_METHOD)
    assert np.allclose(alpha, cov) and abs(rep["figure_px"] - (FEET - HEAD)) <= 1 and rep["method"] == ci.MASK_METHOD
    print(f"ok mask key: a given matte is used as is, figure {rep['figure_px']:.0f} px, backdrop plate still fitted "
          f"(rms {info['residual_rms']:.4f}) for decontamination")


def test_pick(cam):
    P = np.array([-0.7, 0.0, -9.5])
    for f in (FRAMES[0], FRAMES[50]):
        uv, _ = cam.project(f, P[None])
        back = ci.deck_point_from_pick(cam, f, uv[0, 0] - 0.5, uv[0, 1] - 0.5, 0.0)
        assert np.linalg.norm(back - P) < 1e-6, (back, P)
    print("ok pick: deck point recovered from its projection exactly")
    return P


def test_feet(cam, rgba, rep, foot):
    th, tw = rgba.shape[1:3]
    card = ci.place_character(cam, FRAMES, foot, (tw, th), rep["figure_px"], rep["feet_v"], 2.13, deck_y=0.0)
    card_h = card["size_m"][1]
    assert abs(card_h * rep["figure_px"] / th - 2.13) < 1e-9
    tex = rgba[0]
    worst, cols, frames_seen = 0.0, 0, 0
    for n in FRAMES:
        corners = card["corners_by_frame"][str(n)]
        TL, TR, BR, BL = map(np.array, corners)
        feet_y = BL[1] + (1 - rep["feet_v"]) * card_h
        assert abs(feet_y) < 1e-9, feet_y                              # the feet line is on the deck
        img = cp.render(cam, n, corners, tex, scale=SCALE)
        a = img[..., 3]
        # the feet line in the image: two points on it, projected
        L = BL + (TL - BL) * (1 - rep["feet_v"])
        Rr = BR + (TR - BR) * (1 - rep["feet_v"])
        uv, d = cam.project(n, np.array([L, Rr]))
        if d.min() <= 0:
            continue
        frames_seen += 1
        xs = np.nonzero((a > 0.99).sum(0) > 4)[0]
        for x in xs:
            col = a[:, x]
            rows = np.nonzero(col > 0.5)[0]
            if len(rows) == 0 or rows[-1] + 1 >= len(col):
                continue
            y = rows[-1]
            a0, a1 = col[y], col[y + 1]
            if not (a0 > 0.95 and a1 < 0.5):                            # only crisp bottoms (feet, not hands)
                continue
            edge_v = (y + 0.5 + (a0 - 0.5) / (a0 - a1)) / SCALE
            u = (x + 0.5) / SCALE
            t = (u - uv[0, 0]) / (uv[1, 0] - uv[0, 0])
            line_v = uv[0, 1] + t * (uv[1, 1] - uv[0, 1])
            if abs(line_v - edge_v) > 40:                               # a bottom that is not the feet (body, arm)
                continue
            worst = max(worst, abs(line_v - edge_v))
            cols += 1
    assert frames_seen == len(FRAMES), frames_seen
    assert cols > 1000 and worst < 1.0, f"feet off the deck line by {worst:.3f} px at 4096 ({cols} columns)"
    print(f"ok feet: {cols} foot columns over {frames_seen} frames sit on the deck line within {worst:.3f} px at 4096; "
          f"card {card['size_m'][0]:.2f} x {card_h:.2f} m for a 2.13 m figure")


def test_time_and_card_at():
    pf = list(range(10, 20))
    assert list(ci.time_map(4, pf).values()) == [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
    assert list(ci.time_map(4, pf, end_mode="pingpong").values()) == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]
    card = {"corners": [[0, 1, 0]] * 4, "corners_by_frame": {"10": "A", "11": "B"},
            "texture_pattern": "t.%04d.exr", "texture_by_frame": {"10": 0, "11": 5}}
    assert cp.card_at(card, 11) == ("B", "t.0005.exr") and cp.card_at(card, 99) == ("B", "t.0005.exr")
    assert cp.card_at({"corners": "C", "texture": "x.exr"}, 3) == ("C", "x.exr")
    print("ok time map (hold, ping-pong) and card_at (per-frame corners and textures, ends hold)")


def main() -> int:
    cam = camera()
    rgba, rep = test_key()
    test_mask_key()
    foot = test_pick(cam)
    test_feet(cam, rgba, rep, foot)
    test_time_and_card_at()
    return 0


if __name__ == "__main__":
    sys.exit(main())
