"""Per-person temporal lock.

Generated video runs at its own pace, and two performers can drift
independently. For each person we compare the plate's and the AI's mask
motion (position, size, shape), find the best monotonic alignment with DTW,
then smooth it into a curve tau(plate_frame) -> fractional AI frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def mask_features(masks: np.ndarray) -> np.ndarray:
    """(N,H,W) mattes -> (N,6): centroid x/y, sqrt(area), second moments."""
    masks = np.asarray(masks, np.float64)
    n, h, w = masks.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.zeros((n, 6))
    for i in range(n):
        m = masks[i]
        a = m.sum()
        if a < 1e-6:
            f[i] = np.nan
            continue
        cx = (m * xs).sum() / a
        cy = (m * ys).sum() / a
        dx, dy = xs - cx, ys - cy
        f[i] = [cx, cy, np.sqrt(a), np.sqrt((m * dx * dx).sum() / a),
                np.sqrt((m * dy * dy).sum() / a), (m * dx * dy).sum() / a / max(np.sqrt(a), 1)]
    # fill empty frames by interpolation
    for c in range(f.shape[1]):
        col = f[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            f[:, c] = 0
        elif bad.any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    return f


def _standardise(a: np.ndarray, b: np.ndarray):
    both = np.concatenate([a, b], 0)
    mu = both.mean(0)
    sd = both.std(0)
    sd[sd < 1e-9] = 1
    return (a - mu) / sd, (b - mu) / sd


def dtw_path(cost: np.ndarray, band: int | None = None, open_ends: bool = True):
    """DTW with (1,0),(0,1),(1,1) steps. Returns list of (i, j).

    Every plate frame (rows) must be matched, but with open_ends the AI take
    (columns) may start and stop anywhere: generated takes usually run long
    or start late, and forcing the corners together bends the whole curve.
    """
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    if open_ends:
        D[0, :] = 0
    else:
        D[0, 0] = 0
    for i in range(1, n + 1):
        jlo, jhi = 1, m
        if band is not None:
            c = int(round(i * m / n))
            jlo, jhi = max(1, c - band), min(m, c + band)
        for j in range(jlo, jhi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i = n
    j = int(np.argmin(D[n, 1:])) + 1 if open_ends else m
    path = [(i - 1, j - 1)]
    while i > 1:
        opts = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j)]
        if j > 1:
            opts.append((D[i, j - 1], i, j - 1))
        _, i, j = min(opts, key=lambda o: o[0])
        if j < 1:
            break
        path.append((i - 1, j - 1))
    return path[::-1]


def isotonic(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: closest non-decreasing sequence."""
    y = np.asarray(y, float)
    vals, wts, lens = [], [], []
    for v in y:
        vals.append(v); wts.append(1.0); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            l2 = lens[-2] + lens[-1]
            vals[-2:] = [v2]; wts[-2:] = [w]; lens[-2:] = [l2]
    return np.concatenate([np.full(l, v) for v, l in zip(vals, lens)])


@dataclass
class RetimeCurve:
    tau: np.ndarray          # (N_plate,) fractional AI frame for each plate frame
    cost: float              # mean matched feature distance after alignment
    path: list

    def ai_frame(self, i: int) -> float:
        return float(self.tau[i])

    def nearest(self) -> np.ndarray:
        return np.rint(self.tau).astype(int)

    def offset_stats(self) -> dict:
        d = self.tau - np.arange(len(self.tau))
        return {"mean_offset": float(d.mean()), "max_abs_offset": float(np.abs(d).max()),
                "drift": float(d[-1] - d[0])}


def solve_retime(plate_masks: np.ndarray, ai_masks: np.ndarray, *, smooth_sigma: float = 1.5,
                 velocity_weight: float = 1.0, band: int | None = None, refine_iters: int = 12,
                 plate_features: np.ndarray | None = None, ai_features: np.ndarray | None = None) -> RetimeCurve:
    """Solve tau for one performer from their mattes in the plate and the AI take.

    Pass *_features instead (e.g. body-solve joints flattened per frame) to
    retime on pose rather than silhouette.
    """
    fp0 = mask_features(plate_masks) if plate_features is None else np.asarray(plate_features, float)
    fa0 = mask_features(ai_masks) if ai_features is None else np.asarray(ai_features, float)

    def run(fp, fa):
        if velocity_weight > 0:
            fp = np.concatenate([fp, np.gradient(fp, axis=0) * velocity_weight], 1)
            fa = np.concatenate([fa, np.gradient(fa, axis=0) * velocity_weight], 1)
        fp, fa = _standardise(fp, fa)
        c = np.linalg.norm(fp[:, None, :] - fa[None, :, :], axis=2)
        return c, dtw_path(c, band)

    # pass 1 on motion only (offset-free), pass 2 after removing the constant
    # offset/size difference between the takes that pass 1 measured
    cost, path = run(np.gradient(fp0, axis=0), np.gradient(fa0, axis=0))
    off = np.median(np.array([fa0[j] - fp0[i] for i, j in path]), axis=0)
    cost, path = run(fp0, fa0 - off)

    n, m = cost.shape
    raw = np.zeros(n)
    cnt = np.zeros(n)
    for i, j in path:
        raw[i] += j
        cnt[i] += 1
    raw /= np.maximum(cnt, 1)

    # continuous refinement: search fractional AI time around the DTW answer,
    # interpolating AI features, weighted toward position (the most reliable cue)
    # A constant offset and a time lag look alike on smooth motion, so solve
    # them jointly: alternate (offset | tau) and (tau | offset) until settled.
    grid = np.arange(m, dtype=float)
    sd = np.concatenate([fp0, fa0], 0).std(0)
    sd[sd < 1e-9] = 1
    wts = np.ones(fp0.shape[1])
    wts[:2] = 3.0

    def interp_feats(t):
        return np.stack([np.interp(t, grid, fa0[:, c]) for c in range(fa0.shape[1])], -1)

    ref = raw.copy()
    for _ in range(refine_iters):
        off = np.median(interp_feats(ref) - fp0, axis=0)
        new = ref.copy()
        for i in range(n):
            cand = np.arange(max(0.0, ref[i] - 2), min(m - 1.0, ref[i] + 2) + 1e-9, 0.02)
            if cand.size == 0:
                continue
            d = ((((interp_feats(cand) - off) - fp0[i]) / sd) ** 2 * wts).sum(1)
            new[i] = cand[int(np.argmin(d))]
        new = isotonic(gaussian_filter1d(new, 0.75, mode="nearest"))
        if np.abs(new - ref).max() < 0.01:
            ref = new
            break
        ref = new

    tau = isotonic(ref)
    if smooth_sigma > 0:
        tau = isotonic(gaussian_filter1d(tau, smooth_sigma, mode="nearest"))
    tau = np.clip(tau, 0, m - 1)
    mean_cost = float(np.mean([cost[i, j] for i, j in path]))
    return RetimeCurve(tau, mean_cost, path)


def sample_sequence(seq: np.ndarray, tau: np.ndarray, mode: str = "nearest") -> np.ndarray:
    """Resample a frame sequence (N,…) at fractional indices."""
    seq = np.asarray(seq)
    tau = np.clip(np.asarray(tau, float), 0, seq.shape[0] - 1)
    if mode == "nearest":
        return seq[np.rint(tau).astype(int)]
    i0 = np.floor(tau).astype(int)
    i1 = np.minimum(i0 + 1, seq.shape[0] - 1)
    a = (tau - i0).reshape((-1,) + (1,) * (seq.ndim - 1))
    return (seq[i0] * (1 - a) + seq[i1] * a).astype(seq.dtype)
