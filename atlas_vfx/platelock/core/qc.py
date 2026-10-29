"""Measure before fixing.

QC runs on the raw AI take against the plate and says which stages the shot
actually needs, then runs again after the lock as the pass/fail. Every metric
is lighting-tolerant because the AI take is relit by design.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, sobel

from .retime import mask_features


# ----------------------------------------------------------- frame metrics

def matte_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    A, B = np.asarray(a) > thr, np.asarray(b) > thr
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 1.0


def edge_distance(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    """Symmetric mean distance between silhouette edges, in pixels."""
    A, B = np.asarray(a) > thr, np.asarray(b) > thr
    if not A.any() or not B.any():
        return float("nan")
    ea, eb = A & ~binary_erosion(A), B & ~binary_erosion(B)
    da = distance_transform_edt(~ea)
    db = distance_transform_edt(~eb)
    return float(0.5 * (db[ea].mean() + da[eb].mean()))


def _logluma(img):
    im = np.asarray(img, np.float32)
    y = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2] if im.ndim == 3 else im
    return np.log(np.maximum(y, 1e-4))


def structure_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """NCC of log-luminance gradient magnitude.

    Log turns a relight gain into an offset, the gradient removes the offset,
    so this tracks structure alignment and mostly ignores the relight.
    """
    la, lb = _logluma(a), _logluma(b)
    ga = np.hypot(sobel(la, 0), sobel(la, 1))
    gb = np.hypot(sobel(lb, 0), sobel(lb, 1))
    m = np.ones(ga.shape, bool) if mask is None else np.asarray(mask) > 0.5
    if m.sum() < 16:
        return float("nan")
    x, y = ga[m] - ga[m].mean(), gb[m] - gb[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def rgb_error(a, b, mask=None) -> float:
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean(-1)
    if mask is not None:
        m = np.asarray(mask) > 0.5
        return float(d[m].mean()) if m.any() else float("nan")
    return float(d.mean())


def temporal_lag(plate_masks: np.ndarray, ai_masks: np.ndarray, max_lag: int = 12) -> float:
    """Best constant lag (frames) between plate and AI motion of one performer."""
    fp = mask_features(plate_masks)[:, :2]
    fa = mask_features(ai_masks)[:, :2]
    vp = np.gradient(fp, axis=0).ravel()
    va = np.gradient(fa, axis=0)
    n = min(len(fp), len(fa))
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        i0, i1 = max(0, -lag), min(n, n - lag)
        if i1 - i0 < 4:
            continue
        x = np.gradient(fp, axis=0)[i0:i1].ravel()
        y = va[i0 + lag:i1 + lag].ravel()
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        c = (x * y).sum() / den if den > 0 else -np.inf
        if c > best:
            best, best_lag = c, lag
    return float(best_lag)


# ------------------------------------------------------------ shot report

DEFAULTS = {
    "retime_lag_frames": 0.5,
    "camera_rotation_deg": 0.05,
    "edge_px_warp": 1.5,
    "edge_px_snap": 0.5,
    "iou_regen": 0.85,
}


def person_report(plate_masks, ai_masks, plate_frames=None, ai_frames=None) -> dict:
    n = min(len(plate_masks), len(ai_masks))
    iou = np.array([matte_iou(ai_masks[i], plate_masks[i]) for i in range(n)])
    edge = np.array([edge_distance(ai_masks[i], plate_masks[i]) for i in range(n)])
    rep = {"iou": iou.tolist(), "edge_px": edge.tolist(),
           "mean_iou": float(np.nanmean(iou)), "min_iou": float(np.nanmin(iou)),
           "mean_edge_px": float(np.nanmean(edge)), "max_edge_px": float(np.nanmax(edge)),
           "lag_frames": temporal_lag(plate_masks[:n], ai_masks[:n])}
    if plate_frames is not None and ai_frames is not None:
        s = np.array([structure_ncc(ai_frames[i], plate_frames[i], plate_masks[i]) for i in range(n)])
        rep["structure_ncc"] = s.tolist()
        rep["mean_structure_ncc"] = float(np.nanmean(s))
    return rep


def shot_report(people: dict, camera_delta: dict | None = None, background: dict | None = None,
                thresholds: dict | None = None) -> dict:
    """Combine per-person reports (+ optional camera delta / bg error) and diagnose.

    people: {"name": person_report(...)}
    """
    th = {**DEFAULTS, **(thresholds or {})}
    needs, reasons = [], []
    for name, r in people.items():
        if abs(r["lag_frames"]) >= th["retime_lag_frames"]:
            needs.append("retime")
            reasons.append(f"{name}: motion lags plate by {r['lag_frames']:+.0f} frames")
        if r["mean_edge_px"] > th["edge_px_warp"]:
            needs.append("people_warp")
            reasons.append(f"{name}: silhouette off by {r['mean_edge_px']:.1f}px on average")
        if r["mean_edge_px"] > th["edge_px_snap"]:
            needs.append("key_snap")
        bad = [i for i, v in enumerate(r["iou"]) if v < th["iou_regen"]]
        if bad:
            reasons.append(f"{name}: {len(bad)} frames below IoU {th['iou_regen']} (regen candidates if they survive the warp)")
    if camera_delta is not None and camera_delta["max_rotation_deg"] > th["camera_rotation_deg"]:
        needs.append("camera_reshoot")
        reasons.append(f"camera drifts up to {camera_delta['max_rotation_deg']:.2f} deg from the plate")
    order = ["retime", "camera_reshoot", "people_warp", "key_snap"]
    needs = [s for s in order if s in set(needs)]
    out = {"people": people, "needs": needs, "reasons": reasons, "thresholds": th}
    if camera_delta is not None:
        out["camera"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in camera_delta.items()}
    if background is not None:
        out["background"] = background
    return out


def regen_frames(report: dict, iou_floor: float | None = None) -> dict:
    """After the lock: frames where a warp couldn't get there (shape changed)."""
    floor = iou_floor if iou_floor is not None else report.get("thresholds", DEFAULTS)["iou_regen"]
    return {name: [i for i, v in enumerate(r["iou"]) if v < floor] for name, r in report["people"].items()}
