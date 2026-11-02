"""Work around a comfy-kitchen launch limit that breaks the full LTX-2.5 VAE decode.

comfy-kitchen's Triton `rms_rope_` / `rms_rope` launch one program per (batch, head, token), with the token count on
the grid's third axis. CUDA caps that axis at 65535 blocks. LTX-2.5's diffusion VAE decoder
(comfy/ldm/lightricks/vae/na_diffusion_decoder.py) chunks its attention to about 2**25 / dim tokens, which is more
than that, so every full decode of a 97-frame relight failed with "Triton Error [CUDA]: invalid argument"
(comfy-kitchen 0.2.40, ComfyUI 49abb64b, 2026-11-02), whatever the tile size.

The op is per token: an RMS norm of each head vector, then a rotation by that token's matrix. Splitting the token
axis into pieces of at most 65535 and calling the original on each piece gives the same numbers. Calls under the
limit go straight through, untouched. Installed once from the root __init__; ATLAS_VFX_NO_KITCHEN_FIX=1 turns it off.
Drop it when comfy-kitchen launches with the tokens on the grid's first axis (no limit there).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("atlas_vfx")

GRID_LIMIT = 65535          # CUDA's cap on gridDim.y and gridDim.z


def _token_dim(x, freqs) -> int | None:
    """The axis of x that holds the tokens when it is too long to launch: dim 1 for (b, n, h, d), dim 2 for
    (b, h, n, d). None when nothing is over the limit or the layout is not one we can split safely."""
    for dim in (1, 2):
        n = x.shape[dim]
        if n > GRID_LIMIT and freqs.dim() > dim and freqs.shape[dim] in (1, n):
            return dim
    return None


def _pieces(n: int):
    for a in range(0, n, GRID_LIMIT):
        yield a, min(a + GRID_LIMIT, n)


def _split_inplace(orig):
    def rms_rope_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
        dim = _token_dim(q, freqs_cis)
        if dim is None or k.shape[dim] != q.shape[dim]:
            return orig(q, k, freqs_cis, q_scale, k_scale, epsilon)
        for a, b in _pieces(q.shape[dim]):
            f = freqs_cis if freqs_cis.shape[dim] == 1 else freqs_cis.narrow(dim, a, b - a)
            orig(q.narrow(dim, a, b - a), k.narrow(dim, a, b - a), f, q_scale, k_scale, epsilon)
        return q, k
    rms_rope_._atlas_vfx_split = True
    return rms_rope_


def _split_copy(orig):
    import torch

    def rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
        dim = _token_dim(q, freqs_cis)
        if dim is None or k.shape[dim] != q.shape[dim]:
            return orig(q, k, freqs_cis, q_scale, k_scale, epsilon)
        qs, ks = [], []
        for a, b in _pieces(q.shape[dim]):
            f = freqs_cis if freqs_cis.shape[dim] == 1 else freqs_cis.narrow(dim, a, b - a)
            qo, ko = orig(q.narrow(dim, a, b - a), k.narrow(dim, a, b - a), f, q_scale, k_scale, epsilon)
            qs.append(qo)
            ks.append(ko)
        return torch.cat(qs, dim), torch.cat(ks, dim)
    rms_rope._atlas_vfx_split = True
    return rms_rope


def install() -> str:
    """Wrap comfy_kitchen.rms_rope_ and rms_rope. Returns what happened, for the load log."""
    if os.environ.get("ATLAS_VFX_NO_KITCHEN_FIX"):
        return "off (ATLAS_VFX_NO_KITCHEN_FIX)"
    try:
        import comfy_kitchen as ck
    except Exception:  # noqa: BLE001 (no comfy-kitchen: nothing to fix)
        return "comfy-kitchen not installed"
    done = []
    for name, wrap in (("rms_rope_", _split_inplace), ("rms_rope", _split_copy)):
        fn = getattr(ck, name, None)
        if fn is None or getattr(fn, "_atlas_vfx_split", False):
            continue
        setattr(ck, name, wrap(fn))
        done.append(name)
    return f"split long token runs in comfy_kitchen {', '.join(done)}" if done else "already installed"
