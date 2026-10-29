"""Cross-view warp: lock an AI frame onto its plate frame, one ID at a time.

Frame to frame (plate N <-> AI frame N), no camera, no retime. For each ID:

  1. match images: both frames in a log space (ACEScct by default), luma, then
     local contrast normalisation so a relight (gain, tint, soft falloff) drops
     out and only structure is left
  2. coarse: edge snap of the AI ID matte onto the plate ID matte (moves the
     interior with its edge), when the AI matte is given
  3. fine: dense DIS optical flow between the matching images inside the ID,
     iterated on the pre-warped AI so each pass only solves the residual
  4. trust: forward-backward check plus a texture test; low-confidence vectors
     are replaced by a confidence-weighted smooth fill
  5. edge: a banded snap of the warped AI matte onto the plate matte

The result is a backward map in the plate frame (out = AI(map)), solved at a
processing width (2K by default) and scaled to full resolution for delivery.
Pixels are never touched here; warping happens on the original linear frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from . import key, warp
from .qc import edge_distance, matte_iou


# ------------------------------------------------------------------ colour

def acescct_from_acescg(lin: np.ndarray) -> np.ndarray:
    """ACEScct encoding of ACEScg linear (S-2016-001). Fallback when OCIO isn't wired in."""
    x = np.asarray(lin, np.float32)
    lo = 10.5402377416545 * x + 0.0729055341958355
    hi = (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52
    return np.where(x <= 0.0078125, lo, hi).astype(np.float32)


def match_image(log_rgb: np.ndarray, sigma: float = 6.0, clip: float = 3.0) -> np.ndarray:
    """Log-encoded RGB -> uint8 structure image, invariant to local gain/offset."""
    im = np.asarray(log_rgb, np.float32)
    y = 0.2722 * im[..., 0] + 0.6741 * im[..., 1] + 0.0537 * im[..., 2] if im.ndim == 3 else im  # AP1 weights
    mu = cv2.GaussianBlur(y, (0, 0), sigma)
    d = y - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma)) + 1e-3
    n = np.clip(d / sd, -clip, clip)
    return np.round((n / clip * 0.5 + 0.5) * 255).astype(np.uint8)


# ------------------------------------------------------------------ flow

_PRESETS = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}


def dis_flow(a: np.ndarray, b: np.ndarray, preset: str = "medium") -> np.ndarray:
    """Flow f with a(x) ~ b(x + f). a, b uint8 single channel."""
    d = cv2.DISOpticalFlow_create(_PRESETS[preset])
    d.setUseSpatialPropagation(True)
    return d.calc(np.ascontiguousarray(a), np.ascontiguousarray(b), None)


def _flow_maps(f: np.ndarray):
    return warp.from_displacement(f[..., 0], f[..., 1])


def _norm_conv(field2: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = np.stack([gaussian_filter(field2[..., c] * w, sigma) for c in range(2)], -1)
    den = gaussian_filter(w, sigma)[..., None]
    return num / np.maximum(den, 1e-6)


# ------------------------------------------------------------------ per ID

@dataclass
class CVConfig:
    proc_width: int = 2048          # solve resolution (maps are scaled to full res after)
    norm_sigma: float = 6.0         # local-normalisation radius, px at proc res
    preset: str = "medium"
    passes: int = 2                 # DIS passes on the progressively warped AI
    fb_px: float = 1.0              # forward-backward tolerance, px
    min_texture: float = 6.0        # match-image gradient below this is untrusted
    fill_sigma: float = 6.0         # smooth fill for untrusted vectors
    edge_trust_px: float = 0.0      # flow trust ramps from 0 at the ID edge to 1 this far inside; 0 = auto
    edge_trust_frac: float = 0.33   # auto: this fraction of the ID's equivalent radius...
    edge_trust_range: tuple = (8.0, 48.0)  # ...clamped to this range (px at proc res)
    tps_step: int = 6               # sample spacing for the thin-plate fill of untrusted flow
    tps_max_points: int = 1500
    tps_smoothing: float = 0.02
    blend_power: float = 2.0        # trusted flow vs spline fill: conf ** power
    smooth_sigma: float = 1.0       # final light smoothing of the flow
    region_px: int = 24             # how far outside the plate ID the flow is solved and eased out
    core_thr: float = 0.95          # match images keep only the solid core of each matte...
    core_erode_px: int = 2          # ...eroded this far, so the mixed rim (AI rim = invented bg) is filled
    coarse: str = "moments"         # "moments" (per-ID scale + offset from the mattes), "snap", "none"
    edge_snap: bool = False         # geometric snap of the warped AI matte; off: edges are coloured by edge extend
    snap_band: float = 4.0
    snap_max_px: float = 12.0
    snap_min_px: float = 0.25
    max_flow_px: float = 64.0


@dataclass
class IDResult:
    maps: tuple                     # (map_x, map_y) at proc res, into the AI frame
    confidence: np.ndarray          # 0..1 at proc res
    stats: dict = field(default_factory=dict)


def _bbox(mask: np.ndarray, pad: int, shape):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def crossview_id(plate_m8: np.ndarray, ai_m8: np.ndarray, plate_matte: np.ndarray,
                 ai_matte: np.ndarray | None, cfg: CVConfig) -> IDResult:
    """Warp for one ID at processing resolution. Inputs are match images (uint8) and mattes (0..1)."""
    H, W = plate_m8.shape
    ident = warp.identity_maps(H, W)
    pm = np.asarray(plate_matte, np.float32)
    region = binary_dilation(pm > 0.02, iterations=max(1, cfg.region_px))
    if not region.any():
        return IDResult(ident, np.zeros((H, W), np.float32), {"empty": True})

    # 1. coarse: per-ID scale and offset from matte moments (an edge snap here would move the
    #    interior by the edge's displacement, which is wrong whenever the sizes differ)
    total = ident
    stats = {}
    if ai_matte is not None:
        am = np.asarray(ai_matte, np.float32)
        stats["edge_px_raw"] = edge_distance(am, pm)
        stats["iou_raw"] = matte_iou(am, pm)
        if cfg.coarse == "moments":
            total = moments_map(pm, am, (H, W))
        elif cfg.coarse == "snap" and np.isfinite(stats["edge_px_raw"]) and stats["edge_px_raw"] > cfg.snap_min_px:
            total = key.edge_snap_iterative(am, pm, iters=2, max_shift=cfg.snap_max_px, sigma=3.0, region=region)

    # outside the ID the plate is screen and the AI is invented background: replace both with a
    # smooth fill of the ID's own interior so nothing out there drags the flow and no fake edge forms
    from .fill import pushpull_fill
    from scipy.ndimage import binary_erosion

    def _core(m):
        c = np.asarray(m) > cfg.core_thr
        return binary_erosion(c, iterations=cfg.core_erode_px) if cfg.core_erode_px > 0 else c
    pin = _core(pm)
    plate_m8 = np.clip(pushpull_fill(plate_m8.astype(np.float32), ~pin), 0, 255).astype(np.uint8)
    if ai_matte is not None:
        ain = _core(ai_matte)
        ai_m8 = np.clip(pushpull_fill(ai_m8.astype(np.float32), ~ain), 0, 255).astype(np.uint8)

    # 2. fine: DIS on the crop around the ID, iterated on the pre-warped AI
    bb = _bbox(region, 8, (H, W))
    y0, y1, x0, x1 = bb
    P = plate_m8[y0:y1, x0:x1]
    reg = region[y0:y1, x0:x1]
    acc = np.zeros((y1 - y0, x1 - x0, 2), np.float32)
    for _ in range(max(1, cfg.passes)):
        cur = warp.compose(total, _flow_maps_full(acc, bb, (H, W)))
        A = warp.remap(ai_m8.astype(np.float32), *cur, border="replicate")[y0:y1, x0:x1]
        A = np.clip(np.round(A), 0, 255).astype(np.uint8)
        f = dis_flow(P, A, cfg.preset)
        f[~reg] = 0
        # compose residual into the accumulated flow: new(x) = acc(x + f) + f
        mx, my = warp.from_displacement(f[..., 0], f[..., 1])
        acc = np.stack([warp.remap(acc[..., c], mx, my, border="replicate") for c in range(2)], -1) + f

    # 3. trust: forward-backward + texture
    cur = warp.compose(total, _flow_maps_full(acc, bb, (H, W)))
    A = np.clip(np.round(warp.remap(ai_m8.astype(np.float32), *cur, border="replicate")[y0:y1, x0:x1]), 0, 255)
    A = A.astype(np.uint8)
    res = dis_flow(P, A, "fast")
    back = dis_flow(A, P, "fast")
    bx, by = warp.from_displacement(res[..., 0], res[..., 1])
    fb = res + np.stack([warp.remap(back[..., c], bx, by, border="replicate") for c in range(2)], -1)
    fb_err = np.hypot(fb[..., 0], fb[..., 1])
    gx = cv2.Sobel(P.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(P.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    tex = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0) / 8.0
    conf = np.exp(-(fb_err / cfg.fb_px) ** 2) * np.clip(tex / cfg.min_texture, 0, 1)
    conf *= np.clip(np.hypot(acc[..., 0], acc[..., 1]) < cfg.max_flow_px, 0, 1)
    # near the ID edge the two frames disagree about what is person and what is background
    # (silhouettes differ, AI edge pixels carry invented background), and the flow locks onto
    # that consistently in both directions, so the FB check can't catch it: ramp trust in from
    # the edge and let the interior correction extend outward instead
    from scipy.ndimage import distance_transform_edt as _edt
    din = _edt(pm[y0:y1, x0:x1] > 0.5)
    trust = cfg.edge_trust_px
    if trust <= 0:
        r_eq = np.sqrt(max((pm > 0.5).sum(), 1) / np.pi)
        trust = float(np.clip(cfg.edge_trust_frac * r_eq, *cfg.edge_trust_range))
    stats["edge_trust_px"] = trust
    ramp = np.clip((din - trust * 0.25) / max(trust * 0.75, 1e-6), 0, 1)
    conf *= (ramp * ramp * (3 - 2 * ramp)).astype(np.float32)
    conf = (conf * reg).astype(np.float32)

    raw_acc = acc.copy()
    sample_ok = conf * (din >= trust)       # fit only where the flow is fully trusted
    filled = _tps_fill(acc, sample_ok, cfg)
    wb = (conf ** cfg.blend_power)[..., None]
    acc = acc * wb + filled * (1 - wb)
    if getattr(cfg, "debug", False):
        stats["_debug"] = {"raw": raw_acc, "filled": filled, "conf": conf, "bb": bb, "coarse": total}
    if cfg.smooth_sigma > 0:
        acc = np.stack([gaussian_filter(acc[..., c], cfg.smooth_sigma) for c in range(2)], -1)
    # ease the flow out at the region edge
    from scipy.ndimage import distance_transform_edt
    inside = pm[y0:y1, x0:x1] > 0.02
    dist = distance_transform_edt(~inside)
    ease = np.clip(1 - dist / max(cfg.region_px, 1), 0, 1).astype(np.float32)
    acc *= ease[..., None]

    total = warp.compose(total, _flow_maps_full(acc, bb, (H, W)))
    conf_full = np.zeros((H, W), np.float32)
    conf_full[y0:y1, x0:x1] = conf
    stats["residual_px_median"] = float(np.median(np.hypot(res[..., 0], res[..., 1])[inside[...]])) if inside.any() else 0.0
    stats["confident_fraction"] = float((conf[inside] > 0.5).mean()) if inside.any() else 0.0

    # 4. edge: banded snap of the warped AI matte onto the plate matte
    if ai_matte is not None and cfg.edge_snap:
        wm = warp.remap(np.asarray(ai_matte, np.float32), *total)
        ed = edge_distance(wm, pm)
        if np.isfinite(ed) and ed > cfg.snap_min_px:
            s = key.edge_snap_iterative(wm, pm, iters=2, min_px=cfg.snap_min_px, max_shift=cfg.snap_max_px,
                                        sigma=2.0, region=region, band=cfg.snap_band)
            total = warp.compose(total, s)
        wm = warp.remap(np.asarray(ai_matte, np.float32), *total)
        stats["edge_px"] = edge_distance(wm, pm)
        stats["iou"] = matte_iou(wm, pm)
    return IDResult(total, conf_full, stats)


def _tps_fill(acc: np.ndarray, conf: np.ndarray, cfg) -> np.ndarray:
    """Fill untrusted flow from the trusted interior with a smoothed thin-plate spline.

    A TPS carries the interior's gradient out to the edge band (a constant fill would lose
    any bend or scale change right where silhouettes need it most).
    """
    from scipy.interpolate import RBFInterpolator
    h, w = conf.shape
    step = max(2, int(cfg.tps_step))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    c = conf[ys, xs]
    good = c > 0.6
    if good.sum() < 12:
        return _norm_conv(acc, conf, cfg.fill_sigma)
    pts = np.stack([xs[good], ys[good]], 1).astype(np.float64)
    val = acc[ys[good], xs[good]].astype(np.float64)
    if pts.shape[0] > cfg.tps_max_points:
        keep = np.random.default_rng(0).choice(pts.shape[0], cfg.tps_max_points, replace=False)
        pts, val = pts[keep], val[keep]
    rbf = RBFInterpolator(pts, val, kernel="thin_plate_spline", smoothing=cfg.tps_smoothing * pts.shape[0], degree=1)
    gy = np.arange(0, h + step, step, dtype=np.float64)
    gx = np.arange(0, w + step, step, dtype=np.float64)
    GX, GY = np.meshgrid(gx, gy)
    g = rbf(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GY.shape + (2,)).astype(np.float32)
    mx = np.broadcast_to((np.arange(w, dtype=np.float32) / step)[None, :], (h, w)).astype(np.float32)
    my = np.broadcast_to((np.arange(h, dtype=np.float32) / step)[:, None], (h, w)).astype(np.float32)
    return np.stack([cv2.remap(g[..., k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                     for k in range(2)], -1)


def moments_map(plate_matte: np.ndarray, ai_matte: np.ndarray, shape):
    """Backward map plate -> AI from matte centroids and axis spreads (scale + offset per axis)."""
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    def mom(m):
        m = np.asarray(m, np.float64)
        a = m.sum()
        if a < 1:
            return None
        cx, cy = (m * xs).sum() / a, (m * ys).sum() / a
        sx = np.sqrt((m * (xs - cx) ** 2).sum() / a)
        sy = np.sqrt((m * (ys - cy) ** 2).sum() / a)
        return cx, cy, max(sx, 1e-3), max(sy, 1e-3)
    p, a = mom(plate_matte), mom(ai_matte)
    if p is None or a is None:
        return warp.identity_maps(H, W)
    mx = a[0] + (xs - p[0]) * (a[2] / p[2])
    my = a[1] + (ys - p[1]) * (a[3] / p[3])
    return mx.astype(np.float32), my.astype(np.float32)


def _flow_maps_full(acc: np.ndarray, bb, shape):
    """Crop flow -> full-frame backward map (identity outside the crop)."""
    H, W = shape
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)
    y0, y1, x0, x1 = bb
    dx[y0:y1, x0:x1] = acc[..., 0]
    dy[y0:y1, x0:x1] = acc[..., 1]
    return warp.from_displacement(dx, dy)


# ------------------------------------------------------------------ frame

def proc_size(width: int, height: int, proc_width: int) -> tuple[int, int]:
    if proc_width <= 0 or proc_width >= width:
        return width, height
    return proc_width, max(2, int(round(height * proc_width / width)))


def resize(a: np.ndarray, w: int, h: int, area: bool = True) -> np.ndarray:
    if a.shape[1] == w and a.shape[0] == h:
        return a
    interp = cv2.INTER_AREA if (area and a.shape[1] > w) else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(a, np.float32), (w, h), interpolation=interp)


def scale_maps(maps, src_wh: tuple[int, int], dst_wh: tuple[int, int], ai_src_wh: tuple[int, int] | None = None):
    """Backward maps solved at src size -> dst (plate) size, pointing into the AI at ai_src_wh.

    Works on displacement (map minus identity), so frame borders scale exactly too.
    Assumes plate and AI share an aspect ratio.
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    aw, ah = ai_src_wh or dst_wh
    mx, my = maps
    ix, iy = warp.identity_maps(sh, sw)
    dx = cv2.resize((mx - ix).astype(np.float32), (dw, dh), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize((my - iy).astype(np.float32), (dw, dh), interpolation=cv2.INTER_LINEAR)
    X, Y = warp.identity_maps(dh, dw)
    # plate full-res pixel -> plate proc coord -> add proc displacement -> AI full-res coord
    u = (X + 0.5) * sw / dw - 0.5 + dx
    v = (Y + 0.5) * sh / dh - 0.5 + dy
    return ((u + 0.5) * aw / sw - 0.5).astype(np.float32), ((v + 0.5) * ah / sh - 0.5).astype(np.float32)


@dataclass
class FrameResult:
    ids: dict                       # name -> IDResult (proc res)
    proc_wh: tuple
    stats: dict


def crossview_frame(plate_log: np.ndarray, ai_log: np.ndarray, plate_mattes: dict, ai_mattes: dict | None,
                    cfg: CVConfig | None = None) -> FrameResult:
    """plate_log / ai_log: log-encoded RGB at any (matching aspect) size; mattes at plate size."""
    cfg = cfg or CVConfig()
    H, W = plate_log.shape[:2]
    pw, ph = proc_size(W, H, cfg.proc_width)
    P8 = match_image(resize(plate_log, pw, ph), cfg.norm_sigma)
    A8 = match_image(resize(ai_log, pw, ph), cfg.norm_sigma)
    out, stats = {}, {}
    for name, pm in plate_mattes.items():
        am = None if not ai_mattes or name not in ai_mattes else resize(ai_mattes[name], pw, ph, area=True)
        r = crossview_id(P8, A8, resize(pm, pw, ph, area=True), am, cfg)
        out[name] = r
        stats[name] = r.stats
    return FrameResult(out, (pw, ph), stats)


def edge_extend(rgb: np.ndarray, alpha: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Colour outside the AI performer replaced by a fill of their own colour (Nuke EdgeExtend style).

    Used where the plate matte reaches past the warped AI matte, so the plate's edge never
    samples the AI's invented background.
    """
    from .fill import pushpull_fill
    return pushpull_fill(np.asarray(rgb, np.float32), np.asarray(alpha) < thr)


def comp_by_plate_mattes(ai_rgb: np.ndarray, warped: dict, plate_mattes: dict, base: np.ndarray | None = None,
                         warped_alpha: dict | None = None):
    """Plate ID mattes are the alphas: out = sum_k m_k * warped_k + (1 - sum m) * base.

    Edges, motion blur and occlusion all come from the plate's own mattes. Where a warped AI
    alpha is given, its colour is edge extended first so plate edges don't pick up AI background.
    """
    base = ai_rgb if base is None else base
    acc = np.zeros_like(ai_rgb, np.float32)
    cover = np.zeros(ai_rgb.shape[:2], np.float32)
    for name, rgb in warped.items():
        if warped_alpha is not None and name in warped_alpha:
            rgb = edge_extend(rgb, warped_alpha[name])
        m = np.clip(np.asarray(plate_mattes[name], np.float32), 0, 1)
        room = np.clip(1 - cover, 0, 1)
        a = np.minimum(m, room)
        acc += rgb * a[..., None]
        cover += a
    return acc + base * (1 - np.clip(cover, 0, 1))[..., None], cover
