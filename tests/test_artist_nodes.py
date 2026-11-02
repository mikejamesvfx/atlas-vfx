"""The artist nodes (catalogue 55-61): Shot Range, Sun Direction, Light Ball, Layer Picker, Grade, Viewer, Notes.

Each against numbers worked out by hand: the 8k+1 snap and plate index, the exact prompt of the left-sun take that
worked on DSCF3000, where the ball lands, Nuke Grade values, the viewer's wipe column, f-stop and false-colour ends.

Run: <V141 python> tests\\test_artist_nodes.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS = []


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


def test_shot_range():
    from atlas_vfx.layers.node_shot_range import AtlasVFXShotRange, snap_length

    check([snap_length(101, s) for s in ("as is", "8k+1 shorter", "8k+1 longer")] == [101, 97, 105],
          "shot range: 101 frames snap to 97 / 105")
    check(snap_length(97, "8k+1 longer") == 97 and snap_length(5, "8k+1 shorter") == 9, "shot range: 97 stays, min 9")
    r = AtlasVFXShotRange().run(1130, 1230, "8k+1 shorter", 1001, 4096, 2160)["result"]
    check(r[:5] == (1130, 97, 1226, 129, 225), "shot range: first, length, last, plate index", str(r[:5]))
    info = json.loads(r[5])
    check(abs(info["link_gb"] - 9.59) < 0.01 and "Relight Transfer Files" in info["text"],
          "shot range: RAM in words", info["text"].splitlines()[-1])
    raises(lambda: AtlasVFXShotRange().run(1000, 1100), "shot range: before the plate", "before the plate")
    raises(lambda: AtlasVFXShotRange().run(1200, 1100), "shot range: last before first", "before first")


def test_sun():
    from atlas_vfx.layers.node_sun_direction import AtlasVFXSunDirection, sun

    worked = ("relight the video to match the light-direction ball. hard warm golden sunset sunlight entering from "
              "frame right, long shadows falling to the left from the right, low sun")       # 2026-10-30 take
    mirror, prompt, s = AtlasVFXSunDirection().run("the left", "low", "hard warm golden sunset sunlight")["result"]
    check(mirror and s["direction"] == "the right" and prompt == worked,
          "sun: a left sun is the mirrored right, with the prompt of the take that worked")
    s2 = sun("the left", "mid", "soft light", "off")
    check(not s2["mirror"] and s2["direction"] == "the left" and "frame left" in s2["prompt"], "sun: mirror off")
    s3 = sun("the back-left", "high", "x")
    check(s3["direction"] == "the back-right" and s3["mirror"], "sun: back-left mirrors to back-right")
    s4 = sun("behind", "low", "x")
    check(not s4["mirror"] and "frame" not in s4["prompt"], "sun: behind needs no mirror or side words")
    raises(lambda: sun("../x", "low", "x"), "sun: unknown direction refused", "direction")


def test_light_ball():
    from atlas_vfx.layers import node_light_ball as lb
    import torch

    check(lb.place(1280) == (1115, 22, 143), "ball: 143 px, 22 px from the top-right at 1280", str(lb.place(1280)))
    check(lb.place(1920) == (1920 - 33 - 214, 33, 214), "ball: scales with the width", str(lb.place(1920)))
    frames = torch.zeros((2, 704, 1280, 3))
    out = lb.AtlasVFXLightBall().run(frames, {"direction": "the right", "height": "low"})[0].numpy()
    ys, xs = np.nonzero(out[1].max(-1) > 0.05)
    check((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (1115, 22, 1258, 165) and out[0].max() > 0.3,
          "ball: pasted on every frame in its box", f"{xs.min()},{ys.min()}-{xs.max() + 1},{ys.max() + 1}")
    check(float(out[:, 200:, :1100].max()) == 0.0, "ball: nothing else touched")
    raises(lambda: lb.ball_image("../../secret", "low"), "ball: no free paths", "no ball")
    check(len(list(lb.BALLS.glob("*__*.png"))) == 24, "ball: the 24 balls ship in assets/relight_balls")


def test_layer_picker(tmp):
    from atlas_vfx.camera_side import exr_io
    from atlas_vfx.camera_side import plate as pl
    from atlas_vfx.layers import node_layer_picker as lp

    root = tmp / "work"
    for n in (1130, 1131, 1133):
        exr_io.write_rgba(root / "SHOT" / "layers" / "card" / f"card.{n}.exr", np.zeros((8, 16, 4), np.float32))
    (root / "SHOT" / "layers" / "card" / "card.render.json").write_text("{}")
    exr_io.write_rgba(root / "loose" / "x.1001.exr", np.zeros((8, 16, 4), np.float32))      # no sidecar: not listed
    check(lp.find_layers(root) == ["SHOT/layers/card"], "picker: lists folders with EXRs and a sidecar",
          str(lp.find_layers(root)))
    old = pl.WORK_ROOT
    pl.WORK_ROOT = root
    try:
        check(lp.AtlasVFXLayerPicker.INPUT_TYPES()["required"]["layer"][0] == ["SHOT/layers/card"],
              "picker: the list is relative to the work root")
        folder, pattern, first, last, info = lp.AtlasVFXLayerPicker().run("SHOT/layers/card")["result"]
        d = json.loads(info)
        check(pattern.endswith("card.####.exr") and (first, last) == (1130, 1133) and d["missing"] == 1
              and (d["width"], d["height"]) == (16, 8), "picker: pattern, range, missing frames, size", d["text"])
        raises(lambda: lp.AtlasVFXLayerPicker().run("../../Windows"), "picker: outside the work root refused",
               "outside")
    finally:
        pl.WORK_ROOT = old


def test_grade():
    import torch
    from atlas_vfx.layers.node_grade import AtlasVFXGrade, grade

    x = np.array([[0.0, 0.18, 0.5, 1.0, 2.0]], np.float32).T * np.ones(3, np.float32)
    # Nuke: blackpoint 0.1, whitepoint 0.9, lift 0.05, gain 1.2, multiply 1.5, offset 0.01, gamma 1
    A = 1.5 * (1.2 - 0.05) / 0.8
    B = 0.01 + 0.05 - A * 0.1
    want = np.maximum(A * x + B, 0)
    got = grade(x, 0.1, 0.9, 0.05, 1.2, 1.5, 0.01, 1.0)
    check(np.allclose(got, want, atol=1e-6), "grade: Nuke's A*x+B", f"{got[:, 0]}")
    g2 = grade(x, gamma=2.0)
    check(np.allclose(g2[:, 0], np.sqrt(x[:, 0]), atol=1e-6), "grade: gamma 2 is a square root above 0")
    check(np.allclose(grade(x, exposure=1.0), x * 2), "grade: exposure +1 doubles")
    fwd = grade(x, 0.1, 0.9, 0.05, 1.2, 1.5, 0.01, 1.7, exposure=0.5, clamp_black=False)
    back = grade(fwd, 0.1, 0.9, 0.05, 1.2, 1.5, 0.01, 1.7, exposure=0.5, reverse=True, clamp_black=False)
    check(np.allclose(back[1:], x[1:], atol=1e-5), "grade: reverse undoes it")
    img = torch.from_numpy(np.ones((1, 4, 4, 4), np.float32) * 0.5)
    mask = torch.zeros((1, 4, 4))
    mask[:, :, 2:] = 1
    out = AtlasVFXGrade().run(img, 0, 1, 0, 2.0, 1, 0, 1, 0, 1, 1, 1, False, True, False, 0.5, mask)[0].numpy()
    check(np.allclose(out[0, :, :2, :3], 0.5) and np.allclose(out[0, :, 2:, :3], 0.75) and np.allclose(out[..., 3], 0.5),
          "grade: mask and mix blend, alpha passes")


def test_viewer(tmp):
    from atlas_vfx.camera_side import exr_io
    from atlas_vfx.layers import node_viewer as vw

    H, W = 40, 80
    a = np.zeros((H, W, 4), np.float32)
    a[..., :3] = 0.18
    a[10:30, 20:60, 3] = 1.0
    b = a.copy()
    b[..., :3] = 0.5
    img, st = vw.render(a, b, "wipe", wipe=0.25, width=0)
    check(np.allclose(img[:, :19], 0.18) and np.allclose(img[:, 21:], 0.5), "viewer: wipe splits at 25%")
    check(st["alpha_px"] == 800 and abs(st["mean"] - 0.18) < 1e-4, "viewer: stats inside A's alpha", str(st))
    e1, _ = vw.render(a, None, "a", exposure=1.0, width=0)
    check(np.allclose(e1, 0.36), "viewer: +1 stop doubles the linear value (no view)")
    z, _ = vw.render(a, None, "a", zoom_to_alpha=True, width=0)
    check(z.shape[0] < H and z.shape[1] < W and z.shape[1] >= 40, "viewer: zoom to alpha crops to the element",
          str(z.shape))
    sbs, _ = vw.render(a, b, "side by side", width=0)
    check(sbs.shape == (H, 2 * W, 3), "viewer: side by side")
    g = np.zeros((H, W, 4), np.float32)
    g[..., 0:3] = np.linspace(-4, 4, W, dtype=np.float32)[None, :, None]
    g[..., 3] = 1
    fc, st = vw.render(g, None, "gain false colour", width=0)
    check(np.allclose(fc[0, 0], (0.15, 0.35, 1.0)) and np.allclose(fc[0, -1], (1.0, 0.45, 0.05))
          and fc.shape[0] == H + 28, "viewer: false colour ends at -+3 stops, legend strip", str(fc[0, 0]))
    proc, _ = vw.li.view_processors("")
    v, _ = vw.render(a, None, "a", view_proc=proc, width=0)
    check(0.3 < float(v[0, 0, 0]) < 0.6, "viewer: OCIO view applied (0.18 shows as mid grey)", f"{v[0, 0, 0]:.3f}")
    raises(lambda: vw.render(a, None, "wipe"), "viewer: wipe without B", "needs B")
    for n in (1001, 1002):
        exr_io.write_rgba(tmp / "seq" / f"s.{n}.exr", a)
    f = vw.read_frame(str(tmp / "seq" / "s.####.exr"), 1002)
    check(f.shape == (H, W, 4) and abs(float(f[0, 0, 0]) - 0.18) < 1e-3, "viewer: reads a frame of a sequence")
    raises(lambda: vw.read_frame(str(tmp / "seq" / "s.####.exr"), 1500), "viewer: missing frame named", "1001-1002")
    views = vw.views()
    check(views[0].startswith("sRGB - Display / ") and len(views) > 3, "viewer: views from the shipped config")
    out = vw.AtlasVFXViewer().run(str(tmp / "seq" / "s.####.exr"), "", 1001, "a", 0.5, 0.0, 1.0, views[0], False, 64)
    check(tuple(out["result"][0].shape) == (1, 32, 64, 3), "viewer: node gives the view as an IMAGE")


def test_notes():
    from atlas_vfx.layers.node_notes import notes

    t = notes(json.dumps({"mode": "sh", "frames": 97, "clamped_fraction_max": 0.5, "fallback_frames": list(range(48)),
                          "seconds": 700.4, "sh_r2": [0.5] * 97}))
    lines = t.splitlines()
    check(lines[0].startswith("! the gain hit max_stops on up to 50%") and lines[1].startswith("! 48 of 97"),
          "notes: warnings first", lines[0][:60])
    check("mode: sh" in t and "items]" in t, "notes: labels and long lists cut")
    check(notes("not json") == "not json", "notes: plain text passes through")


def test_alpha():
    import torch
    from atlas_vfx.layers.node_alpha_prep import AtlasVFXAlphaPrep, plan
    from atlas_vfx.layers.node_alpha_return import AtlasVFXAlphaReturn

    p = plan(1, 4096, 2160)
    check(p["held"] == 9 and p["work"] == [1920, 1012] and p["padded"] == [1920, 1024],
          "alpha prep: a 4K still holds 9 frames, fits 1920 wide, pads to /32", str(p))
    check(plan(100, 1280, 704)["held"] == 105 and plan(97, 1280, 704)["held"] == 97, "alpha prep: 8n+1 hold")
    raises(lambda: plan(146, 1280, 704), "alpha prep: over 145 frames refused", "145")
    still = np.zeros((1, 216, 410, 3), np.float32)
    still[:, 50:150, 100:300] = 0.8
    img, n, w, h, prep, info = AtlasVFXAlphaPrep().run(torch.from_numpy(still))
    check(tuple(img.shape) == (9, 224, 416, 3) and (n, w, h) == (9, 416, 224)
          and float(img[3, 220, 0, 0]) == 0.5 and float(img[8, 100, 200, 0]) > 0.7,
          "alpha prep: still held, padded grey, never stretched", f"{tuple(img.shape)} {n} {w}x{h}")
    # a fake decode: green = the true matte plus per-frame noise that averaging should remove
    rng = np.random.default_rng(0)
    truth = np.zeros((224, 416), np.float32)
    truth[50:150, 100:300] = 1.0
    dec = np.zeros((9, 224, 416, 3), np.float32)
    dec[..., 1] = truth + rng.normal(0, 0.15, (9, 224, 416))
    dec[..., 0] = 0.3
    mask, mimg, info = AtlasVFXAlphaReturn().run(torch.from_numpy(dec), prep, "green", True)
    m = mask.numpy()
    t_src = np.zeros((216, 410), np.float32)
    t_src[50:150, 100:300] = 1.0
    single = np.clip(dec[0, :216, :410, 1], 0, 1)
    err_avg, err_one = float(np.abs(m[0] - t_src).mean()), float(np.abs(single - t_src).mean())
    check(m.shape == (1, 216, 410) and err_avg < err_one * 0.6 and json.loads(info)["still_averaged"],
          "alpha return: a still's held mattes average to a cleaner one", f"error {err_avg:.3f} vs one frame {err_one:.3f}")
    clip = np.zeros((3, 216, 410, 3), np.float32)
    p3 = AtlasVFXAlphaPrep().run(torch.from_numpy(clip))[4]
    d3 = np.zeros((9, 224, 416, 3), np.float32)
    d3[..., 1] = np.arange(9, dtype=np.float32)[:, None, None] / 10
    m3 = AtlasVFXAlphaReturn().run(torch.from_numpy(d3), p3)[0].numpy()
    check(m3.shape == (3, 216, 410) and np.allclose(m3[:, 100, 100], [0.0, 0.1, 0.2], atol=1e-5),
          "alpha return: a clip keeps its own frames, held ones trimmed")


def test_tiny_vae_hook():
    from atlas_vfx.layers import ltx_live_preview as llp

    check(llp.install() == "not inside ComfyUI", "tiny vae: the live-preview hook stays off outside ComfyUI")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="artist_"))
    test_alpha()
    test_tiny_vae_hook()
    os.environ["ATLAS_VFX_OUT_ROOTS"] = str(tmp)
    test_shot_range()
    test_sun()
    test_light_ball()
    test_layer_picker(tmp)
    test_grade()
    test_viewer(tmp)
    test_notes()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
