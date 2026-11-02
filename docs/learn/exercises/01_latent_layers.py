"""Exercise 01: the latent encode layer system, end to end on synthetic data.

Run from the repo root with the ComfyUI venv's python (it has torch, safetensors, OpenColorIO and OpenCV):

    python docs/learn/exercises/01_latent_layers.py

No plate, no model weights, no network, no GPU. The LTX video VAE is replaced by a small fake with the same shapes
(8x in time, 32x in space, 128 channels), and "LTX" itself is replaced by an identity: it hands back exactly the
reference it was shown. That is enough to see every piece of plumbing atlas-vfx owns around the model:

  1. LTX's frame rule (8k+1) and the latent shape it implies
  2. Latent Pre-render: encode once at float32, cache as safetensors, reuse on a hash match
  3. The colour hop into LTX (ACEScg -> named ACES 2.0 SDR view) and back (the exact inverse)
  4. Why generated elements come in through the inverse view, not as an sRGB texture
  5. The inpaint mask: grows down for a contact shadow, then dilates
  6. LTX Return: outside the mask the plate is bit-exact, so the camera lock can't move
  7. Chunk Plan and Chunk Stitch weights for a shot longer than one LTX run

Everything it writes goes into a temp folder it sets as ATLAS_VFX_OUT_ROOTS. Exits non-zero if a check fails.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="atlas_learn01_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"                    # keep the cache out of your real work folder
pl.WORK_ROOT.mkdir()

from atlas_vfx.layers import card_project as cp  # noqa: E402
from atlas_vfx.layers import chunks as ch  # noqa: E402
from atlas_vfx.layers import element_colour as ec  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402
from atlas_vfx.layers.node_latent_prerender import AtlasVFXLatentPrerender, keep_frames  # noqa: E402

OCIO = str(ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio")
FAILS: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(("   ok   " if cond else "   FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def step(n: int, title: str) -> None:
    print(f"\n{n}. {title}")


class FakeLTXVAE:
    """Stands in for the LTX-2.5 video VAE: same layout, deterministic, counts its encodes.
    Real one: (frames, H, W, 3) pixels -> (1, 128, (frames-1)/8 + 1, H/32, W/32) latent."""

    downscale_index_formula = (8, 32, 32)     # (time, height, width), what the real VAE reports

    def __init__(self):
        self.encodes = 0

    def encode(self, pixels):
        import torch

        self.encodes += 1
        n = pixels.shape[0]
        t = (n - 1) // 8 + 1
        pooled = torch.nn.functional.avg_pool2d(pixels[..., :3].mean(-1, keepdim=True).movedim(-1, 1), 32)
        base = pooled[torch.arange(0, n, 8)[:t], 0]                       # (t, H/32, W/32)
        chans = torch.linspace(0.5, 1.5, 128).view(128, 1, 1, 1)
        return (base.unsqueeze(0) * chans).unsqueeze(0).half()            # half on purpose: Pre-render stores float32


def synthetic_plate(W: int = 512, H: int = 270) -> np.ndarray:
    """A dusk-like ACEScg plate, 0.02 to 0.32, with some texture."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.02 + 0.3 * (xx / W) * (0.5 + 0.5 * yy / H)
    tex = 0.03 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    return np.clip(np.dstack([base + tex, base * 0.95 + tex, base * 0.9 + tex]), 0.001, None).astype(np.float32)


def part1_frame_rule() -> None:
    step(1, "LTX's frame rule: 8k+1 frames in, (k+1) latent frames out")
    for n in (9, 10, 17, 97, 121, 128):
        k = keep_frames(n, 8)
        print(f"   {n:4d} frames given -> {k:4d} kept -> {(k - 1) // 8 + 1:3d} latent frames")
    check(keep_frames(10, 8) == 9 and keep_frames(121, 8) == 121, "10 frames trim to 9, 121 stays 121")
    # the size check from the catalogue: 17 frames at 1280x704 is a (1, 128, 3, 22, 40) latent
    shape = (1, 128, (17 - 1) // 8 + 1, 704 // 32, 1280 // 32)
    check(shape == (1, 128, 3, 22, 40), "17 frames at 1280x704 -> latent (1, 128, 3, 22, 40)", str(shape))
    # try: a 4K reference (4096x2160) is not a multiple of 32 in height. That is why LTX Prep works at a
    # work size (1920x1024 or 1280x704) and Return scales back up.


def part2_prerender() -> None:
    import torch

    step(2, "Latent Pre-render: encode once, float32 on disk, reuse on a hash match")
    vae = FakeLTXVAE()
    node = AtlasVFXLatentPrerender()
    g = torch.Generator().manual_seed(0)
    frames = torch.rand((18, 64, 96, 3), generator=g)                     # 18 frames: 17 kept (8*2+1)
    lat, path, info = node.run(frames, vae, "reference", "")
    meta = json.loads(info)
    print(f"   wrote {Path(path).name} + {Path(path).with_suffix('.json').name}, latent {tuple(lat['samples'].shape)}")
    print(f"   sidecar: frames {meta['frames']}, size {meta['size']}, layout {meta['downscale_index_formula']}, "
          f"dtype {meta['dtype']}, frames_sha256 {meta['frames_sha256'][:12]}...")
    check(vae.encodes == 1 and not meta["cached"], "first run encodes")
    check(lat["samples"].dtype == torch.float32, "stored as float32 even though this VAE returned half")
    check(meta["frames"] == 17, "18 frames trimmed to 17 (8k+1), as LTX's own guide node does")
    t0 = time.perf_counter()
    lat2, _, info2 = node.run(frames, vae, "reference", "")
    print(f"   second run: cached={json.loads(info2)['cached']}, {1000 * (time.perf_counter() - t0):.1f} ms, "
          f"encodes so far {vae.encodes}")
    check(vae.encodes == 1 and torch.equal(lat2["samples"], lat["samples"]), "same frames: file reused, no encode")
    frames[3, 10, 10, 0] += 0.01                                          # one pixel of one frame changes
    node.run(frames, vae, "reference", "")
    check(vae.encodes == 2, "one changed pixel: the hash differs, so it encodes again")
    try:
        node.run(frames, vae, "../../escape", "")
        check(False, "a name with .. is refused")
    except ValueError as e:
        check("name" in str(e), "a name with .. is refused (widgets are untrusted)", str(e)[:60])
    from safetensors.torch import load_file
    keys = sorted(load_file(path))
    check(keys == ["latent_format_version_0", "latent_tensor"], "plain safetensors, no pickle", str(keys))
    # try: change the frame count to 25 and watch the latent grow to 4 frames.


def part3_view(fwd, inv) -> None:
    step(3, "Into LTX and back: ACEScg -> 'ACES 2.0 - SDR 100 nits (Rec.709)' view -> inverse")
    plate = synthetic_plate()
    rep = li.roundtrip_report(plate, fwd, inv, step=1)
    print(f"   dusk plate: median error {100 * rep['median_rel_err']:.4f} %, p99 {100 * rep['p99_rel_err']:.4f} %, "
          f"share over 5 % {100 * rep['share_over_5pct']:.2f} %, max luminance {rep['max_luminance']:.2f}")
    check(rep["median_rel_err"] < 0.01 and not li.roundtrip_warnings(rep, "layer"), "mid-tones survive, no warning")
    for v in (20.0, 100.0, 500.0):                                        # neutral highlights, float path
        x = np.full((1, 1, 3), v, np.float32)
        back = li.from_view(li.to_view(x, fwd), inv)[0, 0, 0]
        print(f"   neutral {v:6.1f} -> view {li.to_view(x, fwd)[0, 0, 0]:.4f} -> back {back:8.2f}")
    warm = np.array([[[6.0, 2.4, 0.8]]], np.float32)                      # a saturated sunset highlight
    print(f"   warm (6.0, 2.4, 0.8) -> view {np.round(li.to_view(warm, fwd)[0, 0], 4)} -> back "
          f"{np.round(li.from_view(li.to_view(warm, fwd), inv)[0, 0], 2)}")
    hot = plate.copy()
    hot[:40, :] = warm[0, 0]                                              # a band of sunset sky over the plate
    rep2 = li.roundtrip_report(hot, fwd, inv, step=1)
    warn = li.roundtrip_warnings(rep2, "layer")
    print(f"   with a warm sky band: share over 5 % = {100 * rep2['share_over_5pct']:.1f} %")
    print(f"   warning LTX Prep would log: {warn[0] if warn else '(none)'}")
    check(bool(warn), "saturated highlights clip in the SDR view (a channel hits 1.0), and Prep says so")
    # note: neutral values come back in float even at 100, but a saturated colour clips one channel at about 2.0.
    # A real LTX take is also 8-bit-ish video, so treat everything above the view's peak as gone.
    # try: mask_mode 'none' (whole-frame replacement) and read how the warning changes:
    #   li.roundtrip_warnings(rep2, "none")


def part4_element_colour(fwd) -> None:
    step(4, "A generated element (display picture) into ACEScg: inverse view vs sRGB texture")
    ramp = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    pic = np.repeat(ramp[:, None], 3, axis=1)[None]                       # (1, 19, 3) grey ramp, display 0-1
    via_view = ec.to_acescg(pic, OCIO, ec.VIEW_INVERSE)
    via_tex = ec.to_acescg(pic, OCIO, ec.SRGB_TEXTURE)
    seen_view = li.to_view(via_view, fwd)                                 # what LTX Prep shows LTX
    seen_tex = li.to_view(via_tex, fwd)
    i = 9                                                                 # display 0.50
    print(f"   display 0.50 -> ACEScg {via_view[0, i, 0]:.4f} (inverse view) vs {via_tex[0, i, 0]:.4f} (sRGB texture)")
    print(f"   ...and what LTX then sees: {seen_view[0, i, 0]:.4f} vs {seen_tex[0, i, 0]:.4f}")
    err_view = float(np.abs(seen_view - pic).max())
    err_tex = float((pic - seen_tex)[0, 5:14, 0].mean())
    check(err_view < 0.01, "inverse view: LTX sees the element as it was made", f"max diff {err_view:.4f}")
    check(err_tex > 0.03, "sRGB texture: tone-mapped twice, so it reaches LTX darker in the mids",
          f"mean {err_tex:.3f} darker")


def part5_mask() -> np.ndarray:
    step(5, "The inpaint mask: alpha > thresh, smeared DOWN for a contact shadow, then dilated")
    alpha = np.zeros((270, 512), np.float32)
    alpha[100:150, 200:240] = 1.0                                         # the element's projected alpha
    m_shadow = li.make_mask(alpha, (512, 270), thresh=0.05, shadow_px=20, dilate_px=0)
    ys, _ = np.nonzero(m_shadow)
    print(f"   element rows 100-149; with shadow_px 20 the mask covers rows {ys.min()}-{ys.max()}")
    check(ys.min() == 100 and ys.max() == 169, "shadow room goes down only (feet), never up (head)")
    m = li.make_mask(alpha, (512, 270), thresh=0.05, shadow_px=20, dilate_px=8)
    print(f"   with dilate_px 8 as well: coverage {100 * m.mean():.2f} % of the frame "
          f"(element itself {100 * alpha.mean():.2f} %)")
    # try: shadow_px 0 and dilate_px 0. LTX then has no room to paint a shadow or blend the edge.
    return alpha


def part6_return(fwd, inv, alpha: np.ndarray) -> None:
    step(6, "LTX Return with an identity 'LTX': outside the mask the plate is bit-exact")
    W, H, ww, wh = 512, 270, 256, 128                                      # plate size and a half-size work size
    plate = synthetic_plate(W, H)
    mask = li.make_mask(alpha, (ww, wh), shadow_px=10, dilate_px=4)
    shown_to_ltx = li.resize(li.to_view(plate, fwd), (ww, wh))           # Prep: view, work size
    ltx_out = shown_to_ltx                                                 # stand-in: LTX changes nothing
    rgba = li.return_frame(ltx_out, mask, (W, H), inv, feather_px=1.5)    # Return: inverse view, plate size, alpha
    comp = cp.over(rgba, plate)
    outside = rgba[..., 3] == 0
    inside = rgba[..., 3] > 0.99
    rel = float(np.abs(comp[inside] - plate[inside]).mean() / plate[inside].mean())
    print(f"   alpha 0 on {100 * outside.mean():.1f} % of pixels; inside, the round trip differs by {100 * rel:.2f} %")
    check(np.array_equal(comp[outside], plate[outside]), "plate untouched outside the mask (bit-exact)")
    check(rel < 0.05, "inside: only resize + view round trip error")
    check(bool(np.all(rgba[..., :3] <= rgba[..., 3:4] * 10 + 1e-6)), "returned layer is premultiplied ACEScg")
    flipped = li.mirror(shown_to_ltx[None])[0]
    check(np.array_equal(li.mirror(flipped[None])[0], shown_to_ltx), "mirror is its own inverse (Prep flips, Return flips back)")
    # try: a real LTX take differs inside the mask. Multiply ltx_out by 1.2 inside the mask and see that the
    # outside stays bit-exact anyway: that is the whole camera-lock guarantee.


def part7_chunks() -> None:
    step(7, "Shots longer than one run: Chunk Plan and Chunk Stitch weights")
    p = ch.plan(1001, 1400, max_len=121, overlap=24)
    print("   " + p.table().replace("\n", "\n   "))
    got = [(c.start, c.end) for c in p.chunks]
    check(got == [(1001, 1121), (1098, 1218), (1195, 1315), (1280, 1400)], "same four chunks as the catalogue")
    seam = p.chunks[1].start                                              # first shared frame of c00 and c01
    for f in (seam - 1, seam, seam + 11, p.chunks[0].end, p.chunks[0].end + 1):
        w = ch.weights(p, f)
        print(f"   frame {f}: " + ", ".join(f"c{k:02d} {v:.3f}" for k, v in w))
    ok = all(abs(sum(v for _, v in ch.weights(p, f)) - 1) < 1e-9 for f in range(p.first, p.last + 1))
    check(ok, "weights sum to 1 on every frame (smoothstep crossfade)")
    check(p.seams()[-1] > 24, "the last chunk slides back to end on 1400, so its seam is longer", str(p.seams()))
    # try: ch.plan(1001, 1400, 97, 16), or ch.plan(1001, 1010, overlap=4) (10 frames: one 9-frame chunk, and
    # the default overlap of 24 is refused because it is longer than the chunk).


def main() -> int:
    t0 = time.perf_counter()
    print("Exercise 01: the latent encode layer system (synthetic, no model)")
    print(f"temp work folder: <system temp>/{TMP.name} (a fresh temp folder, never your real work folder)")
    fwd, inv = li.view_processors(OCIO)
    part1_frame_rule()
    part2_prerender()
    part3_view(fwd, inv)
    part4_element_colour(fwd)
    alpha = part5_mask()
    part6_return(fwd, inv, alpha)
    part7_chunks()
    print(f"\n{len(FAILS)} failed" if FAILS else f"\nall checks passed in {time.perf_counter() - t0:.1f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
