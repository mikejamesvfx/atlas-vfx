"""Latent Pre-render (catalogue 65) and Guide From Latent (66): a reference clip encoded once at float32, cached as
safetensors, reused, and added as an LTX IC-LoRA guide exactly as LTX's own guide node would add the same frames.

A fake VAE stands in for the LTX VAE (deterministic, counts its encodes), so the tests run without the 22B model.

Run: <V141 python> tests\\test_latent_guide.py
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

TMP = Path(tempfile.mkdtemp(prefix="latent_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

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


class FakeVAE:
    """Like the LTX video VAE's shapes: 8x in time (8k+1 frames -> k+1 latent frames), 32x in space, 128 channels.
    The latent is a fixed function of the pixels, so the same frames always give the same latent."""

    downscale_index_formula = (8, 32, 32)

    def __init__(self):
        self.encodes = 0

    def encode(self, pixels):
        import torch

        self.encodes += 1
        n, h, w, _ = pixels.shape
        t = (n - 1) // 8 + 1
        pooled = torch.nn.functional.avg_pool2d(pixels[..., :3].mean(-1, keepdim=True).movedim(-1, 1), 32)
        idx = torch.arange(0, n, 8)[:t]
        base = pooled[idx, 0]                                                    # (t, h/32, w/32)
        ch = torch.linspace(0.5, 1.5, 128).view(128, 1, 1, 1)
        return (base.unsqueeze(0) * ch).unsqueeze(0).to(torch.bfloat16)          # (1, 128, t, h/32, w/32)


def frames(n=10, h=64, w=96, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, h, w, 3), generator=g)


def test_prerender_writes_and_reuses():
    import torch
    from atlas_vfx.layers.node_latent_prerender import AtlasVFXLatentPrerender, keep_frames

    check(keep_frames(10, 8) == 9 and keep_frames(9, 8) == 9 and keep_frames(1, 8) == 1,
          "prerender: keeps LTX's 8k+1 frames")
    vae = FakeVAE()
    node = AtlasVFXLatentPrerender()
    imgs = frames()
    lat, path, info = node.run(imgs, vae, "ref", "")
    d = json.loads(info)
    p = Path(path)
    side = json.loads(p.with_suffix(".json").read_text())
    check(p.is_file() and p.suffix == ".latent" and vae.encodes == 1, "prerender: encodes once and writes .latent",
          str(p.name))
    check(lat["samples"].dtype == torch.float32 and tuple(lat["samples"].shape) == (1, 128, 2, 2, 3),
          "prerender: the latent is float32, 9 frames -> 2 latent frames", str(tuple(lat["samples"].shape)))
    check(side["frames"] == 9 and side["size"] == [96, 64] and side["dtype"] == "float32"
          and side["downscale_index_formula"] == [8, 32, 32], "prerender: sidecar records frames, size, VAE layout",
          str({k: side[k] for k in ("frames", "size", "dtype")}))
    lat2, path2, info2 = node.run(imgs, vae, "ref", "")
    check(vae.encodes == 1 and json.loads(info2)["cached"] and torch.equal(lat2["samples"], lat["samples"]),
          "prerender: the same frames again reuse the file, no second encode")
    node.run(frames(seed=1), vae, "ref", "")
    check(vae.encodes == 2, "prerender: different frames encode again")
    check(d["cached"] is False and len(d["frames_sha256"]) == 64, "prerender: info says encoded, with the frames hash")
    raises(lambda: node.run(imgs, vae, "../evil", ""), "prerender: a hostile name is refused", "name")
    from safetensors.torch import load_file
    t = load_file(str(p))
    check(set(t) == {"latent_tensor"} and t["latent_tensor"].dtype == torch.float32,
          "prerender: the file is plain safetensors with one float32 tensor")


def main():
    for t in (test_prerender_writes_and_reuses,):
        t()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
