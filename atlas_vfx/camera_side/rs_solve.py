"""Rolling-shutter-aware bundle adjustment: refine a global-shutter solve so its poses and points follow a sensor
that reads top to bottom.

Why. A global-shutter (GS) solve of a rolling-shutter (RS) plate is not wrong so much as bent: each frame's pose and
the shared points are pulled to absorb the row-to-row skew. Putting the physical readout back on top of that camera
counts the skew twice (deck reprojection test, NOTES 2026-10-21). This module re-solves poses and points with the
readout in the projection, so the result is a camera that is right at the frame time and expects RS on top.

Model (camera_side/rs_time.py, shared with sensor/rs.py, so its maps fit the camera this writes):
  observed row y (contract pixels, 0 = top edge) was captured at t = f + (y / H - 0.5) * r     (r: readout, frames)
  pose at t: rotation slerped, centre lerped, between whole frames (extrapolated from the inner pair at the ends)
  pinhole projection with the camera's intrinsics (fixed), contract pixels (= COLMAP pixels)
Row time comes from the OBSERVED row, the usual first-order RS model (the error is skew x residual, far below noise).

Unknowns: per frame a small rotation (local rotvec) and centre offset, every point, and optionally r. Gauge: the
first frame is held, and one residual holds the camera path length, so scale stays the measured one. After the
solve the result is moved back onto the input world with a robust similarity fit of the points (Umeyama), so a
deck at y = 0 stays at y = 0 and layers placed in the old world stay put.

Jacobian: finite differences with a sparsity pattern (each observation touches its point, two frames and r), so
scipy needs ~25 residual evaluations per Jacobian. Pure numpy/scipy; no imports from layers/ or sensor/.
"""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from . import rs_time


# ------------------------------------------------------------------------------------------ problem set-up

class Problem:
    """Observations flattened to arrays, plus the initial poses. Frames are the camera's frames inside `frames`."""

    def __init__(self, cam, xyz, observations, *, frames=None, min_track: int = 3, max_points: int | None = None,
                 max_init_px: float = 20.0, seed: int = 0):
        self.cam = cam
        allf = [f for f in cam.frames if f in observations]
        if frames is not None:
            allf = [f for f in allf if frames[0] <= f <= frames[1]]
        if len(allf) < 3:
            raise ValueError("RS solve needs at least 3 frames with observations")
        # frames must be contiguous: the slerp between f and f+1 needs both
        gaps = np.diff(allf)
        if np.any(gaps != 1):
            raise ValueError(f"RS solve needs contiguous frames; gap after {allf[int(np.argmax(gaps != 1))]}")
        self.frames = allf
        self.fidx = {f: i for i, f in enumerate(allf)}
        k = cam.intrinsics(allf[0])
        self.w, self.h = k.width, k.height
        self.fx, self.fy, self.cx, self.cy = k.fx, k.fy, k.cx, k.cy
        M = [cam.cam_to_world(f) for f in allf]
        self.R0 = np.array([m[:3, :3] for m in M])
        self.C0 = np.array([m[:3, 3] for m in M])

        fi, pi, xy = [], [], []
        for f in allf:
            kk, pp = observations[f]
            fi.append(np.full(len(kk), self.fidx[f]))
            pi.append(np.asarray(kk, np.int64))
            xy.append(np.asarray(pp, float))
        fi, pi, xy = np.concatenate(fi), np.concatenate(pi), np.concatenate(xy)

        # gross outliers against the input camera (GS projection), then track length
        X = np.asarray(xyz, float)[pi]
        u, v = self._project_gs(fi, X)
        e = np.hypot(u - xy[:, 0], v - xy[:, 1])
        ok = np.isfinite(e) & (e < max_init_px)
        fi, pi, xy = fi[ok], pi[ok], xy[ok]
        cnt = np.bincount(pi, minlength=len(xyz))
        keep_pt = np.flatnonzero(cnt >= min_track)
        if max_points is not None and len(keep_pt) > max_points:
            rng = np.random.default_rng(seed)
            keep_pt = np.sort(rng.choice(keep_pt, max_points, replace=False))
        remap = -np.ones(len(xyz), np.int64)
        remap[keep_pt] = np.arange(len(keep_pt))
        sel = remap[pi] >= 0
        self.fi, self.pi, self.xy = fi[sel], remap[pi[sel]], xy[sel]
        self.point_ids = keep_pt                       # index into the input xyz
        self.X0 = np.asarray(xyz, float)[keep_pt]
        self.n_frames, self.n_points, self.n_obs = len(allf), len(keep_pt), len(self.fi)
        self.dropped_init = int((~ok).sum())

        # which neighbour each observation interpolates towards: below the centre row -> f+1, above -> f-1.
        # Fixed for 0 < r < 2 frames, so the sparsity pattern does not change with r.
        self.rowfrac = rs_time.row_frac(self.xy[:, 1], self.h)              # tau = rowfrac * r
        self.i0 = rs_time.pair_index(self.fi, self.rowfrac, 0, self.n_frames - 1)   # extrapolate at the ends

    def _project_gs(self, fi, X):
        Xc = np.einsum("nji,nj->ni", self.R0[fi], X - self.C0[fi])
        z = -Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.cx + self.fx * Xc[:, 0] / z, self.cy - self.fy * Xc[:, 1] / z


# ------------------------------------------------------------------------------------------- the model

def _poses(prob: Problem, w, dC):
    """Per-frame rotations (c2w) and centres after the local updates."""
    R = prob.R0 @ Rotation.from_rotvec(w).as_matrix()
    return R, prob.C0 + dC


def project_rs(prob: Problem, R, C, X, r: float, idx=None):
    """RS projection of points X (n,3) for observations idx (default all): pixels (u, v)."""
    sl = slice(None) if idx is None else idx
    i0, fi, rowfrac = prob.i0[sl], prob.fi[sl], prob.rowfrac[sl]
    a = (fi - i0) + rowfrac * r                     # position between frame i0 (a=0) and i0+1 (a=1)
    Rt = rs_time.interp_rot(R, rs_time.rel_rotvecs(R), i0, a)
    Ct = rs_time.interp_centre(C, i0, a)
    Xc = np.einsum("nji,nj->ni", Rt, X - Ct)
    z = -Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return prob.cx + prob.fx * Xc[:, 0] / z, prob.cy - prob.fy * Xc[:, 1] / z


class _Layout:
    """Parameter vector: [r (if solved)] + 6 per free frame (rotvec, dC) + 3 per point."""

    def __init__(self, prob: Problem, solve_readout: bool, fixed_frames=(0,)):
        self.solve_r = solve_readout
        self.free = [i for i in range(prob.n_frames) if i not in set(fixed_frames)]
        self.slot = -np.ones(prob.n_frames, np.int64)
        self.slot[self.free] = np.arange(len(self.free))
        self.o_pose = 1 if solve_readout else 0
        self.o_pts = self.o_pose + 6 * len(self.free)
        self.n = self.o_pts + 3 * prob.n_points

    def unpack(self, x, prob: Problem, r_fixed: float):
        r = float(x[0]) if self.solve_r else r_fixed
        w = np.zeros((prob.n_frames, 3))
        dC = np.zeros((prob.n_frames, 3))
        p = x[self.o_pose:self.o_pts].reshape(-1, 6)
        w[self.free], dC[self.free] = p[:, :3], p[:, 3:]
        X = x[self.o_pts:].reshape(-1, 3)
        return r, w, dC, X


def _path_length(C):
    return float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())


def _residual_fn(prob: Problem, lay: _Layout, r_fixed: float, scale_weight: float, L0: float, r_prior=None):
    def fun(x):
        r, w, dC, X = lay.unpack(x, prob, r_fixed)
        R, C = _poses(prob, w, dC)
        u, v = project_rs(prob, R, C, X[prob.pi], r)
        res = [u - prob.xy[:, 0], v - prob.xy[:, 1]]
        res.append(np.array([scale_weight * (_path_length(C) / L0 - 1.0)]))
        if r_prior is not None:
            res.append(np.array([(r - r_prior[0]) / r_prior[1]]))
        out = np.concatenate(res)
        return np.where(np.isfinite(out), out, 1e3)
    return fun


def _skew(v):
    """(n,3) -> (n,3,3) cross-product matrices: _skew(a) @ b = a x b."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def _jacobian_fn(prob: Problem, lay: _Layout, r_fixed: float, scale_weight: float, L0: float, r_prior=None):
    """Analytic sparse Jacobian of _residual_fn's residuals.

    Pose updates are right-multiplied (R = R0 Exp(w)). For the slerp between frames A and B (Q = R_A^T R_B,
    P = Exp(a log Q)) the row camera's rotation moves by Rt Exp(phi) with phi = (P^T - a Q^T) dA + a dB: exact up to
    the right Jacobian of SO(3), taken as identity. Inter-frame rotations here are a degree or two, so that is a few
    % off in the step direction only; the residual itself is exact, so the solution is unchanged."""
    m = prob.n_obs
    obs = np.arange(m)
    A, B = prob.i0, prob.i0 + 1
    sA, sB = lay.slot[A], lay.slot[B]
    # fixed column layout: per residual row [pts x3, A x6, B x6, (r)]
    k = 15 + (1 if lay.solve_r else 0)

    def jac(x):
        r, w, dC, X = lay.unpack(x, prob, r_fixed)
        R, C = _poses(prob, w, dC)
        a = (prob.fi - A) + prob.rowfrac * r
        Qp = np.einsum("nji,njk->nik", R[:-1], R[1:])
        relp = Rotation.from_matrix(Qp).as_rotvec()
        rel, Q = relp[A], Qp[A]
        P = Rotation.from_rotvec(a[:, None] * rel).as_matrix()
        Rt = R[A] @ P
        RtT = np.transpose(Rt, (0, 2, 1))
        Ct = C[A] * (1 - a)[:, None] + C[B] * a[:, None]
        Xc = np.einsum("nij,nj->ni", RtT, X[prob.pi] - Ct)
        zp = -Xc[:, 2]
        bad = ~(np.isfinite(zp) & (zp > 1e-6))
        zp = np.where(bad, 1.0, zp)
        Juv = np.zeros((m, 2, 3))
        Juv[:, 0, 0] = prob.fx / zp
        Juv[:, 0, 2] = prob.fx * Xc[:, 0] / zp ** 2
        Juv[:, 1, 1] = -prob.fy / zp
        Juv[:, 1, 2] = -prob.fy * Xc[:, 1] / zp ** 2
        Juv[bad] = 0.0
        JR = Juv @ RtT                                               # d/dX
        Sx = _skew(Xc)
        JS = Juv @ Sx
        PT_aQT = np.transpose(P, (0, 2, 1)) - a[:, None, None] * np.transpose(Q, (0, 2, 1))
        blocks = [JR,                                                # point
                  JS @ PT_aQT, -(1 - a)[:, None, None] * JR,         # A: rot, centre
                  a[:, None, None] * JS, -a[:, None, None] * JR]     # B: rot, centre
        vals = np.concatenate(blocks, axis=2)                        # (m, 2, 15)
        if lay.solve_r:
            dXc_da = -np.cross(rel, Xc) - np.einsum("nij,nj->ni", RtT, C[B] - C[A])
            dr = np.einsum("nij,nj->ni", Juv, dXc_da) * prob.rowfrac[:, None]
            vals = np.concatenate([vals, dr[:, :, None]], axis=2)
        cols = np.empty((m, k), np.int64)
        cols[:, 0:3] = lay.o_pts + 3 * prob.pi[:, None] + np.arange(3)
        cols[:, 3:9] = lay.o_pose + 6 * sA[:, None] + np.arange(6)
        cols[:, 9:15] = lay.o_pose + 6 * sB[:, None] + np.arange(6)
        vals[:, :, 3:9][sA < 0] = 0.0                                # fixed frame: no column
        cols[:, 3:9][sA < 0] = 0                                     # dropped below (value 0)
        if lay.solve_r:
            cols[:, 15] = 0
        rows = np.concatenate([obs, obs + m])
        R_ = np.repeat(rows, k)
        C_ = np.concatenate([cols, cols]).ravel()
        V_ = np.concatenate([vals[:, 0, :], vals[:, 1, :]]).ravel()
        keep = V_ != 0
        R_, C_, V_ = [R_[keep]], [C_[keep]], [V_[keep]]
        # path length: d|C_i - C_{i-1}| / dC
        d = np.diff(C, axis=0)
        u = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
        g = np.zeros_like(C)
        g[1:] += u
        g[:-1] -= u
        g *= scale_weight / L0
        fr = np.array(lay.free)
        R_.append(np.full(3 * len(fr), 2 * m))
        C_.append((lay.o_pose + 6 * np.arange(len(fr))[:, None] + np.arange(3, 6)).ravel())
        V_.append(g[fr].ravel())
        n_res = 2 * m + 1
        if lay.solve_r and r_prior is not None:
            R_.append(np.array([2 * m + 1]))
            C_.append(np.array([0]))
            V_.append(np.array([1.0 / r_prior[1]]))
            n_res += 1
        return coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                          shape=(n_res, lay.n)).tocsr()
    return jac


def _point_blocks(V, n_points: int):
    """3x3 diagonal blocks of a block-diagonal sparse matrix (points), from its diagonals."""
    out = np.zeros((n_points, 3, 3))
    for off in (-2, -1, 0, 1, 2):
        d = V.diagonal(off)
        for a in range(3):
            b = a + off
            if not 0 <= b < 3:
                continue
            i = 3 * np.arange(n_points) + min(a, b)
            out[:, a, b] = d[i]
    return out


def _block_diag(blocks):
    n = len(blocks)
    r = (3 * np.arange(n)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
    c = (3 * np.arange(n)[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
    return coo_matrix((blocks.ravel(), (r.ravel(), c.ravel())), shape=(3 * n, 3 * n)).tocsr()


def lm_schur(fun, jac, x0, n_obs: int, o_pts: int, *, soft_px: float, lower=None, upper=None, max_iter: int = 100,
             rtol: float = 1e-7, verbose: int = 0):
    """Levenberg-Marquardt with the point block eliminated (Schur complement), robust soft_l1 by IRLS on each
    observation's 2D error. Residual rows [0, 2*n_obs) are observations (u rows then v rows), the rest are priors.
    Columns [0, o_pts) are the camera block (poses, readout), the rest are points (3 each)."""
    from scipy.linalg import cho_factor, cho_solve

    s2 = soft_px ** 2

    def robust(res):
        e2 = res[:n_obs] ** 2 + res[n_obs:2 * n_obs] ** 2
        rho = 2 * s2 * (np.sqrt(1 + e2 / s2) - 1)
        w = 1 / np.sqrt(1 + e2 / s2)
        return 0.5 * (rho.sum() + np.sum(res[2 * n_obs:] ** 2)), w

    x = x0.copy()
    res = fun(x)
    cost, w = robust(res)
    lam = 1e-3
    hist = []
    n_pts = (len(x) - o_pts) // 3
    for it in range(max_iter):
        J = jac(x).tocsr()
        sw = np.concatenate([np.sqrt(w), np.sqrt(w), np.ones(len(res) - 2 * n_obs)])
        Jw = J.multiply(sw[:, None]).tocsc()
        rw = res * sw
        Jc, Jp = Jw[:, :o_pts], Jw[:, o_pts:]
        U = (Jc.T @ Jc).toarray()
        Wm = (Jc.T @ Jp).tocsr()
        Vb = _point_blocks((Jp.T @ Jp).tocsr(), n_pts)
        gc = Jc.T @ rw
        gp = Jp.T @ rw
        while True:
            Ud = U + lam * np.diag(np.diag(U) + 1e-9)
            Vd = Vb + lam * (np.einsum("nii->ni", Vb)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3))
            Vi = np.linalg.inv(Vd)
            WVi = (Wm @ _block_diag(Vi)).tocsr()
            S = Ud - (WVi @ Wm.T).toarray()
            b = gc - WVi @ gp
            try:
                dc = -cho_solve(cho_factor(S), b)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            dp = -np.einsum("nij,nj->ni", Vi, (gp + Wm.T @ dc).reshape(-1, 3)).ravel()
            xn = x + np.concatenate([dc, dp])
            if lower is not None:
                xn = np.clip(xn, lower, upper)
            resn = fun(xn)
            costn, wn = robust(resn)
            if costn < cost:
                break
            lam *= 5
            if lam > 1e10:
                break
        if lam > 1e10:
            hist.append((it, cost, lam, "stalled"))
            break
        rel = (cost - costn) / cost
        x, res, cost, w = xn, resn, costn, wn
        lam = max(lam / 3, 1e-9)
        hist.append((it, cost, lam))
        if verbose:
            print(f"  LM {it:3d} cost {cost:.6e} rel {rel:.2e} lambda {lam:.1e}", flush=True)
        if rel < rtol:
            break
    return x, cost, hist


def _sparsity(prob: Problem, lay: _Layout, n_res: int):
    m = prob.n_obs
    rows, cols = [], []
    obs = np.arange(m)
    for rr in (obs, obs + m):
        for pair in (prob.i0, prob.i0 + 1):
            s = lay.slot[pair]
            ok = s >= 0
            for c in range(6):
                rows.append(rr[ok])
                cols.append(lay.o_pose + 6 * s[ok] + c)
        for c in range(3):
            rows.append(rr)
            cols.append(lay.o_pts + 3 * prob.pi + c)
        if lay.solve_r:
            rows.append(rr)
            cols.append(np.zeros(m, np.int64))
    # path-length residual touches every free frame's centre
    c = (lay.o_pose + 6 * np.arange(len(lay.free))[:, None] + np.arange(3, 6)[None]).ravel()
    rows.append(np.full(len(c), 2 * m))
    cols.append(c)
    if lay.solve_r and n_res > 2 * m + 1:
        rows.append(np.array([2 * m + 1]))
        cols.append(np.array([0]))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    J = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_res, lay.n)).tocsr()
    J.data[:] = 1                                   # duplicates (i0 == i0+1 never; safety) collapse to 1
    return J


# ------------------------------------------------------------------------------------------- similarity

def umeyama(src, dst, weights=None):
    """Similarity (s, R, t) with dst ~ s R src + t (least squares, optionally weighted)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    wts = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    wts = wts / wts.sum()
    ms, md = wts @ src, wts @ dst
    a, b = src - ms, dst - md
    S = (b * wts[:, None]).T @ a
    U, d, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = float(wts @ (a ** 2).sum(1))
    s = float(np.trace(np.diag(d) @ D) / var)
    return s, R, md - s * R @ ms


def robust_umeyama(src, dst, iters: int = 5):
    w = np.ones(len(src))
    for _ in range(iters):
        s, R, t = umeyama(src, dst, w)
        e = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        mad = np.median(e) + 1e-12
        w = 1.0 / np.maximum(1.0, e / (3.0 * mad))
    return s, R, t, e


# ------------------------------------------------------------------------------------------------- solve

def rms_px(prob: Problem, R, C, X, r):
    u, v = project_rs(prob, R, C, X[prob.pi], r)
    e = np.hypot(u - prob.xy[:, 0], v - prob.xy[:, 1])
    e = e[np.isfinite(e)]
    return {"median_px": float(np.median(e)), "rms_px": float(np.sqrt(np.mean(np.minimum(e, 20) ** 2))),
            "p95_px": float(np.percentile(e, 95))}


def solve(cam, xyz, observations, *, readout_frames: float = 0.0, solve_readout: bool = False,
          readout_prior=None, frames=None, min_track: int = 3, max_points: int | None = None, soft_px: float = 3.0,
          max_nfev: int = 200, scale_weight: float = 1e4, verbose: int = 0, seed: int = 0,
          method: str = "lm", max_iter: int = 100, ftol: float = 1e-7) -> dict:
    """RS bundle adjustment from a GS solve. Returns poses/points in the INPUT world plus a report.

    readout_frames: the readout (frames) used when solve_readout is False, the start value when True.
    readout_prior: optional (mean, sigma) in frames when solving it (a weak spec prior).
    Points are xyz (N,3) in cam's world and observations {frame: (indices, xy contract px)} as camera_io gives.
    method: "lm" (Levenberg-Marquardt + Schur complement, analytic Jacobian; default), "trf" (scipy, analytic),
    "fd" (scipy, finite differences: the first version, ~2 h on DSCF3000)."""
    t0 = time.time()
    prob = Problem(cam, xyz, observations, frames=frames, min_track=min_track, max_points=max_points, seed=seed)
    lay = _Layout(prob, solve_readout)
    x0 = np.zeros(lay.n)
    if solve_readout:
        x0[0] = readout_frames
    x0[lay.o_pts:] = prob.X0.ravel()
    L0 = _path_length(prob.C0)
    fun = _residual_fn(prob, lay, readout_frames, scale_weight, L0, readout_prior if solve_readout else None)
    n_res = len(fun(x0))
    before_gs = rms_px(prob, prob.R0, prob.C0, prob.X0, 0.0)
    before_rs = rms_px(prob, prob.R0, prob.C0, prob.X0, readout_frames)
    xs = np.full(lay.n, 1e-2)
    xs[lay.o_pose:lay.o_pts] = np.tile([1e-3] * 3 + [1e-2] * 3, len(lay.free))
    if solve_readout:
        xs[0] = 0.1
    bounds = (-np.inf * np.ones(lay.n), np.inf * np.ones(lay.n))
    if solve_readout:
        bounds[0][0], bounds[1][0] = 0.0, 1.9
    jfun = _jacobian_fn(prob, lay, readout_frames, scale_weight, L0, readout_prior if solve_readout else None)
    if method == "lm":                             # default: LM + Schur complement, analytic Jacobian
        xs_, cost, hist = lm_schur(fun, jfun, x0, prob.n_obs, lay.o_pts, soft_px=soft_px, lower=bounds[0],
                                   upper=bounds[1], max_iter=max_iter, rtol=ftol, verbose=verbose)
        sol = type("Sol", (), {"x": xs_, "cost": cost, "nfev": len(hist), "status": 1,
                               "message": f"LM-Schur {len(hist)} iterations" + (
                                   " (stalled)" if hist and hist[-1][-1] == "stalled" else
                                   "" if len(hist) < max_iter else " (iteration cap)")})
    else:                                          # scipy trf: analytic or finite-difference Jacobian (slow)
        jkw = {"jac": jfun} if method == "trf" else {"jac_sparsity": _sparsity(prob, lay, n_res)}
        sol = least_squares(fun, x0, loss="soft_l1", f_scale=soft_px, x_scale=xs, bounds=bounds, method="trf",
                            tr_solver="lsmr", max_nfev=max_nfev, ftol=ftol, xtol=1e-10, verbose=verbose, **jkw)
    r, w, dC, X = lay.unpack(sol.x, prob, readout_frames)
    R, C = _poses(prob, w, dC)
    after = rms_px(prob, R, C, X, r)

    # back onto the input world: robust similarity of the solved points onto the input points
    s, Rs, ts, perr = robust_umeyama(X, prob.X0)
    Xw = s * X @ Rs.T + ts
    Rw = np.einsum("ij,njk->nik", Rs, R)
    Cw = s * C @ Rs.T + ts
    after_w = rms_px(prob, Rw, Cw, Xw, r)
    fps = float(cam.data["shot"]["fps"])
    report = {
        "readout_frames": r, "readout_ms": r / fps * 1000.0, "readout_solved": bool(solve_readout),
        "readout_start_frames": float(readout_frames), "readout_prior_frames": readout_prior,
        "frames": [prob.frames[0], prob.frames[-1]], "n_frames": prob.n_frames, "n_points": prob.n_points,
        "n_obs": prob.n_obs, "dropped_init_outliers": prob.dropped_init,
        "reproj_input_gs": before_gs, "reproj_input_with_rs": before_rs, "reproj_solved": after,
        "reproj_solved_in_input_world": after_w,
        "align_to_input": {"scale": s, "rotation_deg": float(np.degrees(Rotation.from_matrix(Rs).magnitude())),
                           "translation_m": float(np.linalg.norm(ts)),
                           "point_shift_median_m": float(np.median(perr))},
        "pose_change": {"rot_deg_median": float(np.degrees(np.median(np.linalg.norm(
                            Rotation.from_matrix(np.einsum("nji,njk->nik", prob.R0, Rw)).as_rotvec(), axis=1)))),
                        "centre_mm_median": float(1000 * np.median(np.linalg.norm(Cw - prob.C0, axis=1)))},
        "cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status), "message": sol.message,
        "seconds": round(time.time() - t0, 1), "model": "rs_anchor centre, row time from observed row, "
        "slerp/lerp between frames, pinhole, robust soft_l1 %.1f px" % soft_px,
    }
    return {"report": report, "frames": prob.frames, "R": Rw, "C": Cw, "xyz": Xw, "point_ids": prob.point_ids,
            "problem": prob}


WORLD_MOVE_WARN_DEG = 5.0


def world_warnings(report: dict, max_deg: float = WORLD_MOVE_WARN_DEG) -> list[str]:
    """Warnings when the solve could not keep the input world: the similarity back onto the input points, or the
    median pose change, is over max_deg. On a good shot both are small (DSCF3000 S2-S3: 0.29 deg, 0.23 deg; the
    synthetic fast pan in test_rs_solve: 1.45 deg, its GS points are bent by the skew). Large values mean the RS
    solve changed the structure, not just bent it (a fast pan with little depth range or baseline: DSCF2973 moved
    15.6 deg, poses 15.3 deg, points 10 % of their depth), so the points no longer fix one similarity, and layers or
    a deck placed in the input camera's world will not line up with this camera."""
    a, p = report.get("align_to_input") or {}, report.get("pose_change") or {}
    rot, pose = float(a.get("rotation_deg", 0.0)), float(p.get("rot_deg_median", 0.0))
    if max(rot, pose) <= max_deg:
        return []
    return [f"the RS solve did not keep the input world: moved {rot:.1f} deg, {a.get('translation_m', 0):.2f} m, "
            f"scale {a.get('scale', 1):.3f} onto it, poses changed {pose:.1f} deg / "
            f"{p.get('centre_mm_median', 0) / 1000:.2f} m median, points {a.get('point_shift_median_m', 0):.2f} m "
            f"median (the structure changed, e.g. a pan with little depth range); anything placed in the input "
            f"camera's world will not line up with this camera"]


def readout_profile(cam, pts: dict, readouts_ms, *, frames=None, max_points: int | None = 8000, max_iter: int = 100,
                    log=None) -> list[dict]:
    """Re-solve at each fixed readout (ms) and report the cost above the best: how deep the valley is. A flat
    profile means the move is too gentle to measure the readout."""
    fps = float(cam.data["shot"]["fps"])
    rows = []
    for ms in readouts_ms:
        r = solve(cam, pts["xyz"], pts["observations"], readout_frames=ms / 1000.0 * fps, frames=frames,
                  max_points=max_points, max_iter=max_iter)["report"]
        rows.append({"readout_ms": float(ms), "cost": r["cost"], "median_px": r["reproj_solved"]["median_px"],
                     "rms_px": r["reproj_solved"]["rms_px"], "iters": r["nfev"], "seconds": r["seconds"]})
        if log:
            log(rows[-1])
    if rows:
        c0 = min(r["cost"] for r in rows)
        for r in rows:
            r["cost_above_best_pct"] = 100.0 * (r["cost"] / c0 - 1.0) if c0 > 0 else 0.0
        rows.sort(key=lambda r: r["readout_ms"])
    return rows


def solve_shot(cam, pts: dict, *, readout_ms: float, solve_readout: bool, frames=None, max_points: int | None = None,
               max_iter: int = 100, profile_ms=(), profile_points: int | None = 8000, name: str | None = None,
               verbose: int = 0, log=None) -> dict:
    """What the RS Solve node and tools/rs_clip_solve.py run: one RS solve of a GS shot camera (readout free from
    readout_ms, or held at it), the RS shot camera it gives, and an optional readout profile (the free readout is
    added to the profile when it was solved). pts: {"xyz", "observations"} in cam's world (measure.points_for_camera).
    Returns {"camera": ShotCamera, "report": solve report, "profile": rows}."""
    fps = float(cam.data["shot"]["fps"])
    res = solve(cam, pts["xyz"], pts["observations"], readout_frames=readout_ms / 1000.0 * fps,
                solve_readout=solve_readout, frames=frames, max_points=max_points, max_iter=max_iter, verbose=verbose)
    rep = res["report"]
    out = to_shot_camera(cam, res, name=name)
    ms_list = [float(v) for v in profile_ms]
    if ms_list and solve_readout:
        ms_list.append(round(rep["readout_ms"], 2))
    rows = readout_profile(cam, pts, ms_list, frames=frames, max_points=profile_points, max_iter=max_iter, log=log)
    return {"camera": out, "report": rep, "profile": rows}


def to_shot_camera(cam, result: dict, *, name: str | None = None):
    """A new shot camera (contract dict) with the solved poses; frames outside the solve keep the input pose."""
    from atlas_vfx.contracts import shot_camera as sc

    data = copy.deepcopy(cam.data)
    by = {f: i for i, f in enumerate(result["frames"])}
    for fr in data["frames"]:
        i = by.get(fr["frame"])
        if i is None:
            continue
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = result["R"][i], result["C"][i]
        fr["cam_to_world"] = M.tolist()
    rep = result["report"]
    data["shot"]["name"] = name or (cam.name + "_rs")
    src = data.setdefault("source", {})
    src["solver"] = f"{src.get('solver', '?')} + atlas-vfx rs_solve (readout {rep['readout_ms']:.1f} ms)"
    src["rmse_px"] = rep["reproj_solved_in_input_world"]["rms_px"]
    src["rmse_px_at_width"] = int(data["plate"]["width"])
    data.setdefault("extensions", {})["atlas_vfx_rs_solve"] = {
        "readout_ms": rep["readout_ms"], "readout_frames": rep["readout_frames"], "rs_anchor": "centre",
        "readout_solved": rep["readout_solved"], "input_camera": cam.name,
        "note": "poses are at the frame time of the centre row; apply this readout on top (Sensor RS Maps)",
        "report": {k: v for k, v in rep.items() if k not in ("message",)}}
    return sc.ShotCamera(sc.validate(data))
