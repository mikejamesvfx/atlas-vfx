"""rs.py's per-pixel maths on CUDA, float64, so the maps match the CPU ones to rounding.

rs_maps on the CPU spends ~25 s per 4K frame (the lens Newton undistort, the per-pixel 3x3 row poses and the
fixed-point ders loop, all numpy on one core). The same steps in torch float64 on the GPU take well under a second.
The row-pose table (RowPoses), the lens coefficients and the formulas are rs.py's own: this file only swaps numpy for
torch, line for line, and tests/test_sensor_rs_gpu.py holds the two paths together.

ATLAS_VFX_GPU_RS=0 keeps everything on the CPU.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

CHUNK_ROWS = 540                      # a quarter of a 4K frame per pass: ~2 GB of float64 temporaries at most


@lru_cache(maxsize=1)
def available() -> bool:
    if os.environ.get("ATLAS_VFX_GPU_RS", "1").lower() in ("0", "false", "off"):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class _Gpu:
    """rs._Setup, the depth and the flow moved to the GPU once per frame."""

    def __init__(self, S, depth, flow):
        import torch

        self.t = torch
        self.dev = torch.device("cuda")
        self.S = S
        f64 = dict(dtype=torch.float64, device=self.dev)
        self.depth = torch.as_tensor(np.ascontiguousarray(depth, np.float64), **f64)
        self.flow = None if flow is None else torch.as_tensor(np.ascontiguousarray(flow, np.float64), **f64)
        P = S.poses
        self.ys0, self.dy, self.ny = float(P.ys[0]), float(P.ys[1] - P.ys[0]), len(P.ys)
        self.ys_last = float(P.ys[-1])
        self.flatM = torch.as_tensor(P.flatM, **f64)
        self.b = torch.as_tensor(P.b, **f64)

    # ------------------------------------------------------------------ rs.py helpers, in torch

    def sample(self, img, x, y):
        """rs.sample: bilinear, clamped, at contract pixels of the S.w x S.h image."""
        t = self.t
        hi, wi = img.shape[:2]
        xs = t.clamp(x * (wi / self.S.w) - 0.5, 0, wi - 1)
        ys = t.clamp(y * (hi / self.S.h) - 0.5, 0, hi - 1)
        x0 = t.clamp(t.floor(xs).long(), max=max(wi - 2, 0))
        y0 = t.clamp(t.floor(ys).long(), max=max(hi - 2, 0))
        x1, y1 = t.clamp(x0 + 1, max=wi - 1), t.clamp(y0 + 1, max=hi - 1)
        fx, fy = xs - x0, ys - y0
        if img.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
        b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
        return a * (1 - fy) + b * fy

    def poses_at(self, y):
        """RowPoses.at."""
        t = self.t
        y = t.clamp(y, self.ys0, self.ys_last)
        f = (y - self.ys0) / self.dy
        i0 = t.clamp(t.floor(f).long(), max=self.ny - 2)
        a = (f - i0)[..., None]
        M = self.flatM[i0] * (1 - a) + self.flatM[i0 + 1] * a
        b = self.b[i0] * (1 - a) + self.b[i0 + 1] * a
        return M.reshape(y.shape + (3, 3)), b

    def row_tau(self, y):
        """rs_time.row_tau, contract pixels, centre anchor."""
        return (self.t.clamp(y, 0.0, float(self.S.h)) / float(self.S.h) - 0.5) * float(self.S.T)

    def _fwd(self, x, y):
        L = self.S.lens
        r2 = x * x + y * y
        rad = 1 + r2 * (L.k1 + r2 * (L.k2 + r2 * L.k3))
        return (x * rad + 2 * L.p1 * x * y + L.p2 * (r2 + 2 * x * x),
                y * rad + L.p1 * (r2 + 2 * y * y) + 2 * L.p2 * x * y)

    def distort(self, u, v):
        L = self.S.lens
        if L.is_identity:
            return u.clone(), v.clone()
        xd, yd = self._fwd((u - L.cx) / L.fx, (v - L.cy) / L.fy)
        return xd * L.fx + L.cx, yd * L.fy + L.cy

    def undistort(self, ud, vd, iters: int = 30, tol: float = 1e-12):
        """BrownConrady.undistort (Newton), same start, same stopping rule."""
        L = self.S.lens
        if L.is_identity:
            return ud.clone(), vd.clone()
        xd = (ud - L.cx) / L.fx
        yd = (vd - L.cy) / L.fy
        r2 = xd * xd + yd * yd
        x = xd / (1 + r2 * (L.k1 + r2 * (L.k2 + r2 * L.k3)))
        y = yd / (1 + r2 * (L.k1 + r2 * (L.k2 + r2 * L.k3)))
        k1, k2, k3, p1, p2 = L.k1, L.k2, L.k3, L.p1, L.p2
        for _ in range(iters):
            r2 = x * x + y * y
            rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            drad = 2 * (k1 + r2 * (2 * k2 + 3 * k3 * r2))
            fx_, fy_ = self._fwd(x, y)
            ex, ey = fx_ - xd, fy_ - yd
            a = rad + x * x * drad + 2 * p1 * y + 6 * p2 * x
            b = x * y * drad + 2 * p1 * x + 2 * p2 * y
            c = b
            d = rad + y * y * drad + 6 * p1 * y + 2 * p2 * x
            det = a * d - b * c
            dx = (d * ex - b * ey) / det
            dy = (a * ey - c * ex) / det
            x = x - dx
            y = y - dy
            if dx.numel() == 0 or float((dx.abs() + dy.abs()).max()) < tol:
                break
        return x * L.fx + L.cx, y * L.fy + L.cy

    def project(self, X):
        S = self.S
        z = -X[..., 2]
        xu = S.cx + S.fx * X[..., 0] / z
        yu = S.cy - S.fy * X[..., 1] / z
        return self.distort(xu, yu)

    # ------------------------------------------------------------------ rs._rs_chunk / rs._ders_chunk

    def rs_chunk(self, X, Y):
        S, t = self.S, self.t
        xu, yu = self.undistort(X, Y)
        D = self.sample(self.depth, X, Y)
        Xt = t.stack([(xu - S.cx) / S.fx * D, -(yu - S.cy) / S.fy * D, -D], -1)
        M, b = self.poses_at(Y)
        X0 = t.einsum("...ij,...j->...i", M, Xt) + b
        qx, qy = self.project(X0)
        if self.flow is not None:
            tau = self.row_tau(Y) + S.time_offset
            f = self.sample(self.flow, X, Y)
            qx, qy = qx - tau * f[..., 0], qy - tau * f[..., 1]
        return qx, qy

    def ders_chunk(self, X, Y, iters):
        S, t = self.S, self.t
        xu, yu = self.undistort(X, Y)
        d = t.stack([(xu - S.cx) / S.fx, -(yu - S.cy) / S.fy, -t.ones_like(xu)], -1)
        px, py = X.clone(), Y.clone()
        for _ in range(max(1, iters)):
            M, b = self.poses_at(py)
            D = self.sample(self.depth, px, py)
            Mtd = t.einsum("...ji,...j->...i", M, d)
            Mtb = t.einsum("...ji,...j->...i", M, b)
            lam = (Mtb[..., 2] - D) / Mtd[..., 2]
            Xt = lam[..., None] * Mtd - Mtb
            nx, ny = self.project(Xt)
            if self.flow is not None:
                tau = self.row_tau(py) + S.time_offset
                f = self.sample(self.flow, px, py)
                nx, ny = nx + tau * f[..., 0], ny + tau * f[..., 1]
            px, py = nx, ny
        return px, py

    def to_st(self, x, y):
        """lens.to_st: (h, w, 2) float32 on the CPU."""
        return self.t.stack([x / self.S.w, 1.0 - y / self.S.h], -1).to(self.t.float32).cpu().numpy()


def rs_maps(S, depth, flow, iters: int, want_ders: bool):
    """rs.rs_maps' loop on the GPU (S is rs._Setup, depth already prepared)."""
    import torch

    g = _Gpu(S, depth, flow)
    w, h = S.w, S.h
    rs = np.empty((h, w, 2), np.float32)
    de = np.empty((h, w, 2), np.float32) if want_ders else None
    xs = torch.arange(w, dtype=torch.float64, device=g.dev) + 0.5
    with torch.no_grad():
        for r0 in range(0, h, CHUNK_ROWS):
            r1 = min(h, r0 + CHUNK_ROWS)
            ys = torch.arange(r0, r1, dtype=torch.float64, device=g.dev) + 0.5
            Y, X = torch.meshgrid(ys, xs, indexing="ij")
            rs[r0:r1] = g.to_st(*g.rs_chunk(X, Y))
            if want_ders:
                de[r0:r1] = g.to_st(*g.ders_chunk(X, Y, iters))
    return rs, de
