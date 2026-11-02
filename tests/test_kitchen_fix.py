"""atlas_vfx/_kitchen_fix.py: comfy-kitchen's rms_rope split at CUDA's 65535 grid limit gives the same numbers.

1. CPU, any machine: a stand-in kernel that refuses more than 65535 tokens (as the Triton launch does) and applies
   the reference maths. Wrapped, a 70000-token call matches the reference on the whole run, in place and copying,
   for (b, n, h, d) and (b, h, n, d) layouts and for shared (length-1) freqs. Short calls go straight through.
2. GPU with comfy-kitchen (ws): the real Triton rms_rope_ fails on 70000 tokens unwrapped, and wrapped it matches
   the reference within fp32 rounding.
Run: <V141 python> tests/test_kitchen_fix.py
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_vfx import _kitchen_fix as kf  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("ok   " if ok else "FAIL ") + what + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(what)


def reference(x, freqs, scale, eps, dim):
    """RMS norm per head vector, then each interleaved pair rotated by its 2x2 matrix [[f00, f01], [f10, f11]]."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * scale.float()
    x0, x1 = xf[..., 0::2], xf[..., 1::2]
    f = freqs.float()
    if dim == 2:                       # (b, h, n, d): freqs come as (b|1, 1, n, d/2, 2, 2) here
        f = f.reshape(f.shape[0], f.shape[1], f.shape[2], -1, 2, 2)
    o0 = f[..., 0, 0] * x0 + f[..., 0, 1] * x1
    o1 = f[..., 1, 0] * x0 + f[..., 1, 1] * x1
    return torch.stack([o0, o1], -1).flatten(-2).to(x.dtype)


def fake_kernel(dim):
    calls = []

    def rms_rope_(q, k, freqs, q_scale, k_scale=None, eps=1e-6):
        n = q.shape[dim]
        calls.append(n)
        if n > kf.GRID_LIMIT:
            raise RuntimeError("Triton Error [CUDA]: invalid argument")
        q.copy_(reference(q, freqs, q_scale, eps, dim))
        k.copy_(reference(k, freqs, k_scale if k_scale is not None else torch.ones_like(q_scale), eps, dim))
        return q, k

    def rms_rope(q, k, freqs, q_scale, k_scale=None, eps=1e-6):
        return rms_rope_(q.clone(), k.clone(), freqs, q_scale, k_scale, eps)
    return rms_rope_, rms_rope, calls


def inputs(n, dim, shared=False, device="cpu", heads=2, hd=8):
    g = torch.Generator(device="cpu").manual_seed(n + dim)
    shape = (1, n, heads, hd) if dim == 1 else (1, heads, n, hd)
    q = torch.randn(shape, generator=g).to(device)
    k = torch.randn(shape, generator=g).to(device)
    ang = torch.rand(1 if shared else n, hd // 2, generator=g) * 6.28
    m = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(-1, hd // 2, 2, 2)
    freqs = (m.reshape(1, -1, 1, hd // 2, 2, 2) if dim == 1 else m.reshape(1, 1, -1, hd // 2, 2, 2)).to(device)
    qs = (torch.rand(hd, generator=g) + 0.5).to(device)
    ks = (torch.rand(hd, generator=g) + 0.5).to(device)
    return q, k, freqs, qs, ks


def test_cpu():
    n = kf.GRID_LIMIT + 4465                         # 70000
    for dim in (1, 2):
        for shared in (False, True):
            ip, cp, calls = fake_kernel(dim)
            q, k, f, qs, ks = inputs(n, dim, shared)
            want_q, want_k = reference(q, f, qs, 1e-6, dim), reference(k, f, ks, 1e-6, dim)
            try:
                ip(q.clone(), k.clone(), f, qs, ks)
                check(False, f"stand-in kernel refuses {n} tokens")
            except RuntimeError:
                pass
            wq, wk = q.clone(), k.clone()
            kf._split_inplace(ip)(wq, wk, f, qs, ks)
            ok = torch.allclose(wq, want_q, atol=1e-6) and torch.allclose(wk, want_k, atol=1e-6)
            check(ok and max(calls[1:]) <= kf.GRID_LIMIT,
                  f"in place, dim {dim}, {'shared' if shared else 'per-token'} freqs: split matches the whole run",
                  f"pieces {calls[1:]}")
            oq, ok_ = kf._split_copy(cp)(q, k, f, qs, ks)
            check(torch.allclose(oq, want_q, atol=1e-6) and torch.allclose(ok_, want_k, atol=1e-6)
                  and torch.equal(q, inputs(n, dim, shared)[0]), f"copying, dim {dim}: same numbers, inputs untouched")
    ip, _, calls = fake_kernel(1)
    q, k, f, qs, ks = inputs(1000, 1)
    kf._split_inplace(ip)(q, k, f, qs, ks)
    check(calls == [1000], "a call under the limit goes straight through, once", calls)


def test_install():
    fake = types.ModuleType("comfy_kitchen")
    fake.rms_rope_, fake.rms_rope, _ = fake_kernel(1)
    sys.modules["comfy_kitchen"] = fake
    try:
        first, again = kf.install(), kf.install()
        check("rms_rope_" in first and fake.rms_rope_._atlas_vfx_split and again == "already installed",
              "install wraps both once", f"{first}; {again}")
    finally:
        del sys.modules["comfy_kitchen"]


def test_gpu():
    if not torch.cuda.is_available():
        print("skip GPU: no CUDA")
        return
    try:
        import comfy_kitchen as ck
    except ImportError:
        print("skip GPU: comfy-kitchen not installed")
        return
    real = getattr(ck.rms_rope_, "__wrapped_orig__", ck.rms_rope_)
    n = kf.GRID_LIMIT + 4465
    q, k, f, qs, ks = inputs(n, 1, device="cuda", heads=4, hd=64)
    want_q, want_k = reference(q, f, qs, 1e-6, 1), reference(k, f, ks, 1e-6, 1)
    try:
        real(q.clone(), k.clone(), f, qs, ks, 1e-6)
        torch.cuda.synchronize()
        print(f"note: this comfy-kitchen launches {n} tokens unwrapped (fixed upstream?)")
    except RuntimeError as e:
        check("invalid argument" in str(e), "real kernel refuses 70000 tokens unwrapped", str(e).strip()[:60])
    wq, wk = q.clone(), k.clone()
    kf._split_inplace(real)(wq, wk, f, qs, ks, 1e-6)
    torch.cuda.synchronize()
    err = max((wq - want_q).abs().max().item(), (wk - want_k).abs().max().item())
    check(err < 1e-4, "real kernel, wrapped: 70000 tokens match the reference", f"max abs err {err:.2e}")


def main() -> int:
    test_cpu()
    test_install()
    test_gpu()
    print("all passed" if not FAILS else f"{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
