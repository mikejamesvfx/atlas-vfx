"""Atlas VFX Latent Pre-render: encode a reference clip once with the VAE in float32 and keep the latent on disk.

The cached latent is what Guide From Latent (catalogue 66) adds as the LTX IC-LoRA guide, so every run on every
machine starts from the same numbers, and a farm or cloud job sends a few MB instead of the frames. The file is plain
safetensors ({"latent_tensor": float32}), beside a JSON sidecar: the frames' hash, count and size, the VAE's layout,
and the file's own hash. The same frames again reuse the file; nothing is encoded twice.

Wire the VAE through LTXVVAEForceFloat32 first for a true float32 encode (the latent is stored as float32 either
way)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import plate as pl

CACHE = "latent_cache"


def keep_frames(n: int, t: int) -> int:
    """The frames LTX keeps of n: t*k+1 (its guide encode drops the rest)."""
    return max(1, (n - 1) // t * t + 1)


def frames_hash(frames: np.ndarray) -> str:
    """sha256 of the frames at 16 bits per channel, plus their shape: identical pictures, identical hash."""
    q = (np.clip(np.asarray(frames, np.float32), 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    h = hashlib.sha256(repr(q.shape).encode())
    h.update(q.tobytes())
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AtlasVFXLatentPrerender:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "path", "info")
    DESCRIPTION = ("Encodes a reference clip once with the VAE (wire it through LTXVVAEForceFloat32) and keeps the "
                   "float32 latent on disk as safetensors with a sidecar. Run it again on the same frames and it "
                   "loads the file instead. Feed latent to Guide From Latent in place of LTX's image guide.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "the reference at the LTX work size (what the guide node would get)"}),
                "vae": ("VAE",),
                "name": ("STRING", {"default": "reference"}),
                "out_dir": ("STRING", {"default": "", "tooltip": "blank = <work>/latent_cache"}),
            }
        }

    def run(self, images, vae, name="reference", out_dir=""):
        import torch
        from safetensors.torch import load_file, save_file

        name = pl.safe_name(name, "name")
        t = int(vae.downscale_index_formula[0])
        clip = images.float().cpu()
        clip = clip[:keep_frames(len(clip), t), ..., :3].contiguous()
        fhash = frames_hash(clip.numpy())
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / CACHE)
        out.mkdir(parents=True, exist_ok=True)
        path, side = out / f"{name}.latent", out / f"{name}.json"
        if path.is_file() and side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except ValueError:
                meta = {}
            if meta.get("frames_sha256") == fhash and meta.get("latent_sha256") == _sha256(path):
                lat = load_file(str(path))["latent_tensor"].float()
                return ({"samples": lat}, str(path), json.dumps({**meta, "cached": True}))
        lat = vae.encode(clip).float().cpu().contiguous()
        save_file({"latent_tensor": lat}, str(path))
        meta = {"frames_sha256": fhash, "frames": int(len(clip)), "size": [int(clip.shape[2]), int(clip.shape[1])],
                "downscale_index_formula": [int(x) for x in vae.downscale_index_formula], "dtype": "float32",
                "latent_shape": [int(x) for x in lat.shape], "latent_sha256": _sha256(path)}
        side.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return ({"samples": lat}, str(path), json.dumps({**meta, "cached": False}))


NODE_CLASS_MAPPINGS = {"AtlasVFXLatentPrerender": AtlasVFXLatentPrerender}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLatentPrerender": "Atlas VFX Latent Pre-render"}
