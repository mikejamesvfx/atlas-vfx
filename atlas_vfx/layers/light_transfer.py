"""Relight Transfer: the light from a low-res relit take, the pixels from the full-res element.

LTX relight IC-LoRAs (2.3 Relight at 1280x704, 2.5 Day-to-Night / Golden-Hour at FullHD) run single-stage, so the
relit element is soft, and LTX's upscalers either drop the LoRA or invent texture. This module keeps every element
pixel and moves only the change of light:

- gain: at an analysis size (the LoRA's native width) gain = log2(relit / element) per channel, clamped. It is blurred
  inside the matte only (normalised convolution, so the background never pulls on the edge: the halo of the old
  tools/light_transfer.py) and filled outward past the matte so nothing bleeds back in.
- guided (route A): the gain is upsampled to full res and run through a guided filter (He et al.) whose guide is the
  element's log luminance, plus MoGe depth and normals when given. Gain edges then follow the element's own edges.
- sh (route B): per frame, 2nd-order spherical harmonics fitted from normal to gain (9 coefficients per channel) at
  the analysis size, with normals turned to world by the shot camera when there is one so the fit holds still while
  the camera moves. Full-res gain = SH(full-res normals) + the guided upsample of what the SH could not explain (cast
  shadows, colour changes). Form shading comes out at the normals' sharpness. A frame whose fit explains less than
  min_r2 of the gain falls back to guided.

out = element * 2**gain inside the matte (weighted by it); pixels where the matte is 0 are passed through untouched.

Conventions: images (T, H, W, 3) float32 scene-linear (ACEScg), normals OpenGL camera space (x right, y up, z toward
the camera; Normals node 32 output decoded with n*2-1), shot camera cam_to_world in the same axes. Pure numpy / cv2.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - V141 and CI have opencv
    cv2 = None

FLOOR = 1e-4                                                        # linear floor before log2
LUMA_AP1 = np.array([0.2722287, 0.6740818, 0.0536895], np.float32)  # ACEScg luminance
CONF_LUM = 2e-3                                                     # below this a ratio is mostly noise
MODES = ("guided", "sh")


# ----------------------------------------------------------------------------------------------- helpers
def _f32(x):
    return np.ascontiguousarray(np.asarray(x, np.float32))


def _resize(img, w, h, down=True):
    if img.shape[1] == w and img.shape[0] == h:
        return _f32(img)
    interp = cv2.INTER_AREA if (down and w < img.shape[1]) else cv2.INTER_LINEAR
    out = cv2.resize(_f32(img), (int(w), int(h)), interpolation=interp)
    if img.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def _blur(x, sigma):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), float(sigma))


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def _box_n(x, r):
    """Box mean of an (H, W, C) array of any C (OpenCV filters at most 4 channels at once)."""
    if x.ndim == 2:
        return _box(x, r)
    return np.stack([_box(np.ascontiguousarray(x[..., c]), r) for c in range(x.shape[-1])], -1)


def _unit(n):
    return (n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def analysis_size(width, height, analysis_width, relit_width):
    aw = int(analysis_width) if analysis_width > 0 else int(relit_width)
    aw = max(8, min(aw, int(width)))
    return aw, max(8, round(height * aw / width))


def decode_normals(img):
    """Normal map encoded 0-1 (Normals node, MoGe Render normal_opengl) to unit vectors."""
    return _unit(np.asarray(img, np.float32)[..., :3] * 2.0 - 1.0)


# ----------------------------------------------------------------------------------------------- gain at low res
def masked_blur(x, w, sigma, fill_sigma=None):
    """Normalised convolution: blur(x*w)/blur(w). Where w is ~0 nearby, a wider pass fills the value outward so the
    result is defined everywhere and has no step at the matte edge."""
    w = _f32(w)
    x = _f32(x)
    wx = x * w[..., None] if x.ndim == 3 else x * w
    num, den = _blur(wx, sigma), _blur(w, sigma)
    den_ = den[..., None] if x.ndim == 3 else den
    out = num / np.maximum(den_, 1e-6)
    fs = fill_sigma if fill_sigma is not None else max(4.0 * sigma, 0.05 * max(w.shape))
    numf, denf = _blur(wx, fs), _blur(w, fs)
    denf_ = denf[..., None] if x.ndim == 3 else denf
    fill = numf / np.maximum(denf_, 1e-8)
    t = np.clip(den_ / 0.05, 0.0, 1.0)                    # trust the tight blur where enough matte is near
    out = out * t + fill * (1 - t)
    return np.where(denf_ > 1e-8, out, 0.0).astype(np.float32)


def log_gain(element_low, relit_low, max_stops):
    """Per-channel log2(relit / element), clamped to +-max_stops; also the fraction of pixels at the clamp."""
    g = np.log2(np.maximum(relit_low, FLOOR)) - np.log2(np.maximum(element_low, FLOOR))
    clamped = np.abs(g) >= max_stops
    return np.clip(g, -max_stops, max_stops).astype(np.float32), clamped


def confidence(element_low, relit_low, matte_low, full=0.98):
    """Weight of each analysis pixel's gain. Only pixels fully inside the matte count: an edge pixel of the shrunk
    image is part background, and its ratio is the background's (a bright sky would pull the edge gain to 0)."""
    lum = np.minimum(element_low @ LUMA_AP1, relit_low @ LUMA_AP1)
    inside = (matte_low >= full).astype(np.float32)
    return (inside * np.clip(lum / CONF_LUM, 0.0, 1.0)).astype(np.float32)


def temporal_smooth(seq, weights, k):
    """Weighted box over +-k frames (weights (T, h, w)); k 0 returns seq."""
    if k <= 0:
        return seq
    T = len(seq)
    out = np.empty_like(seq)
    for t in range(T):
        lo, hi = max(0, t - k), min(T, t + k + 1)
        w = weights[lo:hi][..., None]
        out[t] = (seq[lo:hi] * w).sum(0) / np.maximum(w.sum(0), 1e-6)
        none = weights[lo:hi].sum(0) < 1e-6
        out[t][none] = seq[t][none]
    return out


# ----------------------------------------------------------------------------------------------- guided filter
def _guided_coeffs(I, p, r, eps, weight):
    """The guided filter's window-averaged linear coefficients: (mean_a (H, W, k, c), mean_b (H, W, c))."""
    k = I.shape[-1]
    if weight is None:
        mean = (lambda x: _box_n(x, r))
    else:
        wt = _f32(weight)
        den = np.maximum(_box(wt, r), 1e-6)
        mean = (lambda x: (_box_n(x * (wt[..., None] if x.ndim == 3 else wt), r)
                           / (den[..., None] if x.ndim == 3 else den)))
    mI, mp = mean(I), mean(p)
    H, W = I.shape[:2]
    if k == 1:
        I1, mI1 = I[..., 0], mI[..., 0]
        var = mean(np.ascontiguousarray(I1 * I1)) - mI1 * mI1
        cov = mean(I1[..., None] * p) - mI1[..., None] * mp
        a = cov / (var[..., None] + eps)
        b = mp - a * mI1[..., None]
        return mean(a)[:, :, None, :], mean(b)
    cII = np.empty((H, W, k, k), np.float32)
    for i in range(k):
        for j in range(i, k):
            v = mean(np.ascontiguousarray(I[..., i] * I[..., j])) - mI[..., i] * mI[..., j]
            cII[..., i, j] = v
            cII[..., j, i] = v
    cII += np.eye(k, dtype=np.float32) * eps
    cIp = np.empty((H, W, k, p.shape[-1]), np.float32)
    for i in range(k):
        cIp[..., i, :] = mean(I[..., i:i + 1] * p) - mI[..., i:i + 1] * mp
    a = np.linalg.solve(cII, cIp)                         # (H, W, k, c)
    del cII, cIp
    b = mp - np.einsum("hwk,hwkc->hwc", mI, a)
    ma = np.stack([mean(np.ascontiguousarray(a[..., i, :])) for i in range(k)], 2)
    return ma, mean(b)


def guided_filter(guide, src, r, eps, weight=None, subsample=1):
    """He et al. guided filter. guide (H, W, k), src (H, W, c); every output channel is locally a linear function of
    the k guide channels within radius r. eps regularises (units of the guide squared). weight (H, W) makes every
    window statistic a weighted mean, so pixels with weight 0 (outside the matte) never shape the result: without
    it a bright background beside the element bends the gain at its edge.

    subsample s > 1 is the fast guided filter (He & Sun 2015): the coefficients a, b are window averages, so they are
    smooth and are solved on the guide and source shrunk by s (radius r/s), then upsampled and applied with the
    FULL-res guide. Edges still come from the full-res guide; the heavy part costs about 1/s^2."""
    I = _f32(guide if guide.ndim == 3 else guide[..., None])
    p = _f32(src if src.ndim == 3 else src[..., None])
    H, W, k = I.shape
    s = int(subsample)
    if s <= 1 or min(H, W) < 4 * s:
        ma, mb = _guided_coeffs(I, p, r, eps, weight)
        return (np.einsum("hwk,hwkc->hwc", I, ma) + mb).astype(np.float32)
    h, w = max(1, round(H / s)), max(1, round(W / s))
    def shrink(x):                                        # per channel: OpenCV resizes at most 4 at once
        x = _f32(x)
        if x.ndim == 2:
            return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        return np.stack([cv2.resize(np.ascontiguousarray(x[..., c]), (w, h), interpolation=cv2.INTER_AREA)
                         for c in range(x.shape[-1])], -1)
    if weight is None:
        Is, ps, ws = shrink(I), shrink(p), None
    else:                                                 # weighted shrink: an edge pixel keeps only element values
        wt = _f32(weight)
        ws = shrink(wt)
        dn = np.maximum(ws, 1e-6)[..., None]
        Is, ps = shrink(I * wt[..., None]) / dn, shrink(p * wt[..., None]) / dn
    ma, mb = _guided_coeffs(Is, ps, max(1, round(r / s)), eps, ws)
    grow = (lambda x: cv2.resize(_f32(x), (W, H), interpolation=cv2.INTER_LINEAR))
    out = np.empty((H, W, p.shape[-1]), np.float32)
    for c in range(p.shape[-1]):
        acc = grow(mb[..., c])
        for i in range(k):
            acc += grow(np.ascontiguousarray(ma[..., i, c])) * I[..., i]
        out[..., c] = acc
    return out


# ----------------------------------------------------------------------------------------------- spherical harmonics
def sh_basis(n):
    """Real SH up to order 2 (9 terms, unnormalised constants dropped: the fit absorbs them). n (..., 3) unit."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1], -1).astype(np.float32)


def fit_sh(n, g, w, ridge=1e-3):
    """Weighted least squares of g (N, 3) on SH(n) (N, 9). Returns (coeffs (9, 3), r2 averaged over channels)."""
    B = sh_basis(n)
    sw = np.sqrt(np.maximum(w, 0))[:, None]
    A = B * sw
    AtA = A.T @ A + ridge * np.eye(9, dtype=np.float32) * max(float(w.sum()), 1.0) * 1e-3
    coef = np.linalg.solve(AtA, A.T @ (g * sw))
    ws = max(float(w.sum()), 1e-6)
    mean = (g * w[:, None]).sum(0) / ws
    tot = ((g - mean) ** 2 * w[:, None]).sum(0)
    res = ((g - B @ coef) ** 2 * w[:, None]).sum(0)
    r2 = float(np.mean(np.where(tot > 1e-9, 1 - res / np.maximum(tot, 1e-12), 1.0)))
    return coef.astype(np.float32), r2


def to_world(n, R_c2w):
    return (n @ np.asarray(R_c2w, np.float32).T).astype(np.float32)


# ----------------------------------------------------------------------------------------------- the transfer
def _bbox(m, pad):
    ys, xs = np.nonzero(m > 1e-4)
    if len(ys) == 0:
        return None
    H, W = m.shape
    return (max(0, ys.min() - pad), min(H, ys.max() + 1 + pad), max(0, xs.min() - pad), min(W, xs.max() + 1 + pad))


def _guide(element, depth, normals, geo_weight):
    """Guide channels: element log2 luminance, then (when given) log depth normalised and normals scaled by
    geo_weight."""
    ch = [np.log2(np.maximum(element @ LUMA_AP1, FLOOR))[..., None]]
    if depth is not None:
        ld = np.log2(np.maximum(depth, 1e-3))
        s = float(np.std(ld)) or 1.0
        ch.append((ld / s * geo_weight)[..., None])
    if normals is not None:
        ch.append(normals * geo_weight)
    return np.concatenate(ch, -1).astype(np.float32)


def transfer(element, relit, matte, *, mode="guided", analysis_width=1280, blur_px=2.0, max_stops=3.0,
             temporal=0, eps=0.01, radius=0, subsample=2, normals=None, depth=None, cam_rotations=None, geo_weight=0.5,
             min_r2=0.3, progress=None):
    """element (T, H, W, 3) full res; relit (T, h, w, 3) same framing, any res; matte (T, H', W') any res.
    normals (T, Hn, Wn, 3) unit OpenGL camera space, depth (T, Hd, Wd) metric, any res; cam_rotations (T, 3, 3)
    cam_to_world. Returns (out (T, H, W, 3), gain (T, H, W, 3) log2, report dict)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    element = np.asarray(element, np.float32)[..., :3]
    relit = np.asarray(relit, np.float32)[..., :3]
    matte = np.asarray(matte, np.float32)
    T, H, W = element.shape[:3]
    for name, arr in (("relit", relit), ("matte", matte), ("normals", normals), ("depth", depth),
                      ("cam_rotations", cam_rotations)):
        if arr is not None and len(arr) not in (T, 1):
            raise ValueError(f"{name} has {len(arr)} frames, element {T}")
    rh, rw = relit.shape[1:3]
    if abs(rw / rh - W / H) > 0.01 * (W / H):
        raise ValueError(f"relit is {rw}x{rh}, element {W}x{H}: the framing must match (aspect differs)")
    if mode == "sh" and normals is None:
        raise ValueError("mode sh needs normals (Atlas VFX Normals from MoGe)")
    pick = (lambda a, t: None if a is None else a[t if len(a) > 1 else 0])
    aw, ah = analysis_size(W, H, analysis_width, rw)
    scale = W / aw
    r = int(radius) if radius > 0 else max(2, round(2 * scale))

    # 1. gain and its weights at the analysis size
    g_low = np.empty((T, ah, aw, 3), np.float32)
    w_low = np.empty((T, ah, aw), np.float32)
    clamp_frac = []
    for t in range(T):
        el = _resize(element[t], aw, ah)
        rl = _resize(pick(relit, t), aw, ah)
        m = np.clip(_resize(pick(matte, t), aw, ah), 0.0, 1.0)
        g, cl = log_gain(el, rl, max_stops)
        w = confidence(el, rl, m)
        g_low[t], w_low[t] = g, w
        clamp_frac.append(float((cl.any(-1) & (m > 0.5)).sum() / max(1, (m > 0.5).sum())))
    g_low = temporal_smooth(g_low, w_low, int(temporal))

    # 2. SH fit per frame (route B)
    coefs, r2s, used_sh = [], [], []
    if mode == "sh":
        for t in range(T):
            n = _resize(pick(normals, t), aw, ah)
            n = _unit(n)
            if cam_rotations is not None:
                n = to_world(n, pick(cam_rotations, t))
            sel = w_low[t] > 0.05
            if sel.sum() < 32:
                coefs.append(np.zeros((9, 3), np.float32))
                r2s.append(0.0)
                continue
            c, r2 = fit_sh(n[sel], g_low[t][sel], w_low[t][sel])
            coefs.append(c)
            r2s.append(r2)
        coefs = np.stack(coefs)
        if temporal > 0:
            k = int(temporal)
            coefs = np.stack([coefs[max(0, t - k):t + k + 1].mean(0) for t in range(T)])
        used_sh = [r >= min_r2 for r in r2s]

    # 3. full res, per frame
    out = element.copy()
    gain = np.zeros_like(element)
    for t in range(T):
        m_full = np.clip(_resize(pick(matte, t), W, H, down=False), 0.0, 1.0)
        box = _bbox(m_full, pad=4 * r + 2)
        if box is None:
            if progress:
                progress(t)
            continue
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1))
        n_full = None
        if normals is not None:
            n_full = _unit(_resize(pick(normals, t), W, H, down=False)[sl])
        d_full = None
        if depth is not None:
            d_full = _resize(pick(depth, t), W, H, down=False)[sl]
        g_t = g_low[t]
        sh_full = None
        if mode == "sh" and used_sh[t]:
            n_low = _unit(_resize(pick(normals, t), aw, ah))
            if cam_rotations is not None:
                n_low = to_world(n_low, pick(cam_rotations, t))
            g_t = g_t - sh_basis(n_low) @ coefs[t]                 # residual: what the SH does not explain
            nw = n_full if cam_rotations is None else to_world(n_full, pick(cam_rotations, t))
            sh_full = sh_basis(nw) @ coefs[t]
        g_t = masked_blur(g_t, w_low[t], blur_px)
        g_up = _resize(g_t, W, H, down=False)[sl]
        guide = _guide(element[t][sl], d_full, n_full, geo_weight)
        g_full = guided_filter(guide, g_up, r, eps, weight=np.maximum(m_full[sl], 1e-3), subsample=subsample)
        if sh_full is not None:
            g_full = g_full + sh_full
        g_full = np.clip(g_full, -max_stops, max_stops)
        mm = m_full[sl][..., None]
        el = element[t][sl]
        res = el * (1.0 + mm * (np.exp2(g_full) - 1.0))
        out[t][sl] = np.where(mm > 0, res, el)
        gain[t][sl] = g_full * (mm > 0)
        if progress:
            progress(t)

    rep = {"mode": mode, "frames": T, "size": [W, H], "analysis_size": [aw, ah], "relit_size": [rw, rh],
           "scale": round(scale, 4), "radius_px": r, "subsample": int(subsample), "eps": eps, "blur_px": blur_px,
           "max_stops": max_stops,
           "temporal": temporal, "guides": ["luma"] + (["depth"] if depth is not None else [])
           + (["normals"] if normals is not None else []),
           "camera": "world (shot camera)" if cam_rotations is not None else "camera space",
           "clamped_fraction_max": round(max(clamp_frac) if clamp_frac else 0.0, 5)}
    if mode == "sh":
        rep.update(sh_r2=[round(x, 4) for x in r2s], sh_frames=int(sum(used_sh)),
                   fallback_frames=[t for t, u in enumerate(used_sh) if not u], min_r2=min_r2)
    return out, gain, rep
