"""light_transfer.guided_filter on CUDA.

At 4K the guided filter is most of Relight Transfer's time (~4 s a frame on the CPU: ~30 box filters over 5 guide
channels and a 5x5 solve per pixel). Here the same steps run in torch: box means by cumulative sums with OpenCV's
BORDER_REFLECT edges, the per-pixel solve batched on the GPU, the coefficients grown with bilinear interpolation
(OpenCV's INTER_LINEAR upscale, half-pixel centres) and applied with the full-res guide. The shrink before the solve
stays OpenCV's (INTER_AREA), so the inputs are the CPU path's own. Results match the CPU filter to float rounding
(tests/test_light_transfer.py checks it).

ATLAS_VFX_GPU_GUIDED=0 keeps the CPU filter.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MIN_PIXELS = 1 << 18                  # below 512x512 the CPU filter is quick enough


@lru_cache(maxsize=1)
def available() -> bool:
    if os.environ.get("ATLAS_VFX_GPU_GUIDED", "1").lower() in ("0", "false", "off"):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _reflect(n: int, r: int, dev):
    """OpenCV BORDER_REFLECT indices for -r .. n+r-1 (edge pixel repeated: fedcba|abcdefgh|hgfedcb)."""
    import torch

    i = torch.arange(-r, n + r, device=dev)
    i = torch.where(i < 0, -i - 1, i)
    i = torch.where(i >= n, 2 * n - i - 1, i)
    return i.clamp(0, n - 1)


def _box(x, r: int):
    """Normalised (2r+1)^2 box mean of (B, H, W), BORDER_REFLECT; sums in float64."""
    import torch
    import torch.nn.functional as F

    _B, H, W = x.shape
    k = 2 * r + 1
    xp = x[:, _reflect(H, r, x.device)][:, :, _reflect(W, r, x.device)].double()
    c = F.pad(torch.cumsum(xp, 1), (0, 0, 1, 0))
    s = c[:, k:] - c[:, :H]
    c = F.pad(torch.cumsum(s, 2), (1, 0))
    return ((c[:, :, k:] - c[:, :, :W]) / (k * k)).float()


def _coeffs(I, p, r: int, eps: float, wt):
    """light_transfer._guided_coeffs. I (k, H, W), p (c, H, W), wt (H, W) or None -> (ma (k, c, H, W), mb (c, H, W))."""
    import torch

    k, H, W = I.shape
    c = p.shape[0]
    if wt is None:
        def mean(x):
            return _box(x, r)
    else:
        den = torch.clamp(_box(wt[None], r)[0], min=1e-6)

        def mean(x):
            return _box(x * wt, r) / den
    mI, mp = mean(I), mean(p)
    if k == 1:
        var = mean(I * I)[0] - mI[0] * mI[0]
        cov = mean(I * p) - mI * mp
        a = cov / (var + eps)
        b = mp - a * mI
        return mean(a)[None], mean(b)
    ii, jj = torch.triu_indices(k, k)
    prod = mean(I[ii] * I[jj]) - mI[ii] * mI[jj]                     # (pairs, H, W)
    cII = torch.empty((H, W, k, k), device=I.device)
    cII[:, :, ii, jj] = prod.permute(1, 2, 0)
    cII[:, :, jj, ii] = prod.permute(1, 2, 0)
    cII += torch.eye(k, device=I.device) * eps
    cIp = (mean((I[:, None] * p[None]).reshape(k * c, H, W)) - (mI[:, None] * mp[None]).reshape(k * c, H, W))
    a = torch.linalg.solve(cII, cIp.view(k, c, H, W).permute(2, 3, 0, 1))         # (H, W, k, c)
    del cII, cIp
    b = mp - torch.einsum("khw,hwkc->chw", mI, a)
    ma = mean(a.permute(2, 3, 0, 1).reshape(k * c, H, W)).view(k, c, H, W)
    return ma, mean(b)


def guided_filter(I_np, p_np, r, eps, weight, subsample, shrink):
    """light_transfer.guided_filter's body. I_np (H, W, k), p_np (H, W, c) float32; shrink is the CPU path's own
    INTER_AREA shrink (weighted when weight is given). Returns (H, W, c) float32."""
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    H, W, k = I_np.shape
    s = int(subsample)

    def up(a):
        return torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.asarray(a, np.float32), -1, 0))).to(dev)

    with torch.no_grad():
        I = up(I_np)
        if s <= 1 or min(H, W) < 4 * s:
            wt = None if weight is None else torch.from_numpy(np.asarray(weight, np.float32)).to(dev)
            ma, mb = _coeffs(I, up(p_np), r, eps, wt)
        else:
            Is, ps, ws = shrink()
            wt = None if ws is None else torch.from_numpy(np.ascontiguousarray(ws, np.float32)).to(dev)
            ma, mb = _coeffs(up(Is), up(ps), max(1, round(r / s)), eps, wt)
            kk, c, h, w = ma.shape
            grow = (lambda x: F.interpolate(x[None], size=(H, W), mode="bilinear", align_corners=False)[0])
            ma = grow(ma.reshape(kk * c, h, w)).view(kk, c, H, W)
            mb = grow(mb)
        out = torch.einsum("khw,kchw->chw", I, ma) + mb
        return np.ascontiguousarray(out.permute(1, 2, 0).cpu().numpy())
