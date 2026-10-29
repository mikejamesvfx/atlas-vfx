"""Normal-based relight of a keyed element (actor on a flat studio plate) into a background plate.

2.5D comp relight, not a render. MoGe gives camera-space normals and depth per frame. The element's albedo is
estimated by dividing out a flat studio key; new shading is a sky / ground hemisphere ambient taken from the
background plus a sun found as the background's brightest blob. A sun behind the subject lands as a rim: a normal
term (surfaces turned toward the sun side, at glancing angles) plus a matte-edge term (alpha minus alpha shifted away
from the sun) that catches hair the normals miss. Exposure is matched to the background around the element and
smoothed over time. Depth holdout cuts the element where the background is nearer the camera than the element's
feet (grass, props in front of the legs).

Conventions: images (T, H, W, 3) float32, linear light (callers convert from sRGB). Normals are OpenGL camera space:
x right, y up, z toward the camera. MoGe is OpenCV (x right, y down, z into the scene), so normals convert with
(x, -y, -z). Light directions point from the surface toward the light.

Proved in tools/relight_proto.py of the ltx25_hdr workflows (2026-10-26, Nino run plate over a sunset BG).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - V141 always has opencv
    cv2 = None

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def _blur(x, sigma):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), float(sigma))


# ----------------------------------------------------------------------------------------------- normals
def normals_from_points(points):
    """OpenCV-space outward normals from a (T, H, W, 3) point map (same construction as ComfyUI's MoGe fallback)."""
    p = np.where(np.isfinite(points), points, 0.0).astype(np.float32)
    dx = np.zeros_like(p)
    dy = np.zeros_like(p)
    dx[:, :, 1:-1] = p[:, :, 2:] - p[:, :, :-2]
    dy[:, 1:-1] = p[:, 2:] - p[:, :-2]
    n = np.cross(dy, dx)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)


def opencv_to_opengl(n):
    return (np.asarray(n, np.float32) * np.array([1.0, -1.0, -1.0], np.float32)).astype(np.float32)


def smooth_normals(n, alpha=None, temporal=1, sigma=1.5):
    """Temporal box (+-temporal frames) then spatial gaussian, alpha-weighted when alpha (T, H, W) is given so the
    background does not bleed into the element's edge. Renormalised."""
    n = np.asarray(n, np.float32)
    T = n.shape[0]
    if temporal > 0:
        acc = np.zeros_like(n)
        for t in range(T):
            lo, hi = max(0, t - temporal), min(T, t + temporal + 1)
            acc[t] = n[lo:hi].mean(0)
        n = acc
    out = np.empty_like(n)
    for t in range(T):
        if alpha is not None:
            w = np.asarray(alpha[t], np.float32) + 1e-3
            out[t] = _blur(n[t] * w[..., None], sigma) / _blur(w, sigma)[..., None]
        else:
            out[t] = _blur(n[t], sigma)
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)


# ----------------------------------------------------------------------------------------------- light from plate
def _median_smooth(x, radius):
    x = np.asarray(x, np.float64)
    if radius <= 0:
        return x
    return np.array([np.median(x[max(0, i - radius):i + radius + 1], 0) for i in range(len(x))])


def sun_direction(u, v, fov_x_deg, aspect, behind=0.6):
    """Direction toward a sun seen at normalised image position (u, v) (0..1, top-left origin; may lie outside the
    frame). behind > 0 puts it behind the subject (back light / rim), < 0 in front of it (key)."""
    tx = np.tan(np.radians(fov_x_deg) / 2)
    d = np.array([(u - 0.5) * 2 * tx, -(v - 0.5) * 2 * tx / aspect, -float(behind)], np.float64)
    return d / np.linalg.norm(d)


def light_from_plate(bg_lin, fov_x_deg=50.0, behind=0.6, smooth=5, sun_uv=None, sun_colour=None):
    """Light rig from a background plate (T, H, W, 3) linear.

    Sun: argmax of the heavily blurred luminance per frame, median-smoothed over +-smooth frames, unless sun_uv
    (u, v) is given. Sun colour (max channel = 1) unless given: luminance-weighted hue of the unclipped halo 4-12 % of
    the width around the sun (an SDR plate clips the core to white).
    Ambient: mean colour of the top third (sky) and bottom third (ground) per frame."""
    bg = np.asarray(bg_lin, np.float32)
    T, H, W, _ = bg.shape
    sig = 0.013 * W
    uvs, cols, sky, ground = [], [], [], []
    for t in range(T):
        y = _blur(bg[t] @ LUM, sig)
        v, u = np.unravel_index(int(np.argmax(y)), y.shape)
        uvs.append((u / W, v / H))
        # sun hue from the halo 4-12 % of the width around the sun, unclipped pixels, luminance-weighted: an SDR plate
        # clips the core and often the sky next to it to white
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - u, yy - v) / W
        halo = (d >= 0.04) & (d < 0.12) & (bg[t].max(-1) < 0.95 * float(bg[t].max()))
        if halo.sum() >= 16:
            w = (bg[t][halo] @ LUM)[:, None]
            c = (bg[t][halo] * w).sum(0) / max(float(w.sum()), 1e-6)
        else:
            c = bg[t][y >= np.quantile(y, 0.995)].mean(0)
        cols.append(c / max(float(c.max()), 1e-6))
        sky.append(bg[t, : H // 3].reshape(-1, 3).mean(0))
        ground.append(bg[t, 2 * H // 3:].reshape(-1, 3).mean(0))
    uvs = _median_smooth(uvs, smooth) if sun_uv is None else np.tile(np.asarray(sun_uv, np.float64), (T, 1))
    col = np.median(np.array(cols), 0) if sun_colour is None else np.asarray(sun_colour, np.float64)
    dirs = [sun_direction(u, v, fov_x_deg, W / H, behind) for u, v in uvs]
    return {"frames": T, "width": W, "height": H, "fov_x": float(fov_x_deg), "behind": float(behind),
            "sun_uv": np.asarray(uvs).round(4).tolist(), "sun_dir": np.asarray(dirs).round(5).tolist(),
            "sun_colour": np.asarray(col).round(4).tolist(),
            "sky": _median_smooth(sky, smooth).round(5).tolist(), "ground": _median_smooth(ground, smooth).round(5).tolist(),
            "source": "plate" if sun_uv is None else "manual sun position"}


def override_sun(rig, sun_rig):
    """`rig` (from the plate: ambient, maybe its own sun estimate) with the sun direction and colour taken from
    `sun_rig` (e.g. Physical Sky). Ambient stays the plate's. A one-frame side is held over the other's frames."""
    F, Fs = int(rig["frames"]), int(sun_rig["frames"])
    if F != Fs and 1 not in (F, Fs):
        raise ValueError(f"sun rig has {Fs} frames, light rig {F}")
    n = max(F, Fs)
    hold = (lambda x, k: [x[0]] * n if k == 1 else list(x))
    out = dict(rig)
    out.update(frames=n, sky=hold(rig["sky"], F), ground=hold(rig["ground"], F),
               sun_dir=hold(sun_rig["sun_dir"], Fs), sun_colour=list(sun_rig["sun_colour"]),
               source=f"{rig.get('source', 'plate')} + sun from {sun_rig.get('source', 'sun rig')}")
    if "sun_uv" in sun_rig:
        out["sun_uv"] = hold(sun_rig["sun_uv"], Fs)
    return out


def physical_sun_gain(rig, sun_rig, trim=1.0):
    """Relight's sun_gain that puts the sun at its physical strength against THIS rig's sky: sun_rig's
    sun_strength is the key against a sky of luminance 1, so scale by the plate sky's (median over frames)
    luminance. trim 1.0 = physical."""
    if "sun_strength" not in sun_rig:
        raise ValueError("sun rig has no sun_strength (connect Atlas VFX Physical Sky's light_rig)")
    sky = float(np.median(np.asarray(rig["sky"], np.float64).reshape(-1, 3) @ LUM))
    return float(trim) * float(sun_rig["sun_strength"]) * sky


# ----------------------------------------------------------------------------------------------- relight
def relight(fg_lin, alpha, normals, rig, bg_lin=None, sun_gain=4.0, ambient_gain=1.0, studio_dir=(0.0, 0.35, 1.0),
            studio_flat=0.45, facing_power=2.0, fresnel_power=4.0, edge_rim=1.5, edge_px=8.0, edge_soft=2.0,
            rim_max=2.0, exposure_match=True, exposure_bias=0.9, exposure_smooth=25):
    """Relit element (T, H, W, 3) linear (straight, not premultiplied) plus passes and the per-frame exposure gains.

    rim_max caps the key at rim_max x the ambient luminance per pixel: stops hot spots on faces turned to the sun
    while leaving the rim on hair and cloth."""
    fg = np.asarray(fg_lin, np.float32)
    al = np.asarray(alpha, np.float32)
    n = np.asarray(normals, np.float32)
    T, H, W, _ = fg.shape
    if rig["frames"] != T and rig["frames"] != 1:
        raise ValueError(f"light rig has {rig['frames']} frames, element has {T}")
    pick = (lambda key, t: np.asarray(rig[key][t if rig["frames"] > 1 else 0], np.float32))
    Ls = np.asarray(studio_dir, np.float32)
    Ls = Ls / np.linalg.norm(Ls)
    relit = np.empty_like(fg)
    amb_p, key_p, src_p = np.empty_like(fg), np.empty_like(fg), np.empty((T, H, W), np.float32)
    raw_gain = []
    for t in range(T):
        nt = n[t]
        s_src = studio_flat + (1 - studio_flat) * np.clip(nt @ Ls, 0, 1)
        albedo = fg[t] / np.maximum(s_src, 0.3)[..., None]
        tt = (nt[..., 1:2] + 1) / 2
        amb = (pick("ground", t) * (1 - tt) + pick("sky", t) * tt) * ambient_gain
        Ld = pick("sun_dir", t)
        lxy = Ld[:2] / (np.linalg.norm(Ld[:2]) + 1e-6)
        nxy = nt[..., :2] / (np.linalg.norm(nt[..., :2], axis=-1, keepdims=True) + 1e-6)
        facing = np.clip(nxy @ lxy, 0, 1) ** facing_power
        fres = (1 - np.clip(nt[..., 2], 0, 1)) ** fresnel_power
        rim = facing * fres
        if edge_rim > 0:
            M = np.float32([[1, 0, -lxy[0] * edge_px], [0, 1, lxy[1] * edge_px]])  # image y is down
            al_s = cv2.warpAffine(al[t], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            rim = rim + edge_rim * _blur(np.clip(al[t] - al_s, 0, 1), edge_soft)
        key = np.asarray(rig["sun_colour"], np.float32) * sun_gain * np.clip(rim, 0, 1.5)[..., None]
        if rim_max > 0:
            ya, yk = amb @ LUM, key @ LUM
            key *= np.minimum(1.0, rim_max * ya / np.maximum(yk, 1e-6))[..., None]
        relit[t] = albedo * (amb + key)
        amb_p[t], key_p[t], src_p[t] = albedo * amb, albedo * key, s_src
        if exposure_match and bg_lin is not None:
            core = al[t] > 0.95
            ring = (cv2.dilate((al[t] > 0.5).astype(np.uint8), np.ones((81, 81), np.uint8)) > 0) & (al[t] < 0.05)
            if core.sum() > 50 and ring.sum() > 50:
                raw_gain.append(exposure_bias * np.median(np.asarray(bg_lin[t], np.float32)[ring] @ LUM)
                                / max(float(np.median((albedo * amb)[core] @ LUM)), 1e-5))
                continue
        raw_gain.append(1.0)
    g = np.asarray(raw_gain, np.float64)
    if exposure_match and exposure_smooth > 1 and T > 1:
        k = int(exposure_smooth)
        g = np.exp(np.convolve(np.pad(np.log(g), k // 2, mode="edge"), np.ones(k) / k, "valid"))[:T]
    gf = g.astype(np.float32)[:, None, None, None]
    return relit * gf, {"ambient": amb_p * gf, "key": key_p * gf, "studio": src_p}, g.round(4).tolist()


# ----------------------------------------------------------------------------------------------- depth holdout
def depth_holdout(bg_depth, alpha, bg_valid=None, margin=0.03, soft=0.03, foot_rows=6, smooth=5, distance_scale=1.0):
    """Occluder mask (T, H, W): 1 where the background is nearer the camera than the element.

    The element's distance is the background depth under its feet (the lowest foot_rows rows of the matte), median-
    smoothed over +-smooth frames and scaled by distance_scale. A background pixel occludes when it is nearer than
    distance x (1 - margin), ramping over soft x distance. Only where the background depth is valid."""
    d = np.asarray(bg_depth, np.float32)
    al = np.asarray(alpha, np.float32)
    T, H, W = al.shape
    if d.shape[1:] != (H, W):
        d = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) for x in d])
    if bg_valid is not None:
        v = np.asarray(bg_valid, np.float32)
        if v.shape[1:] != (H, W):
            v = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in v])
    else:
        v = np.isfinite(d).astype(np.float32)
    d = np.where(np.isfinite(d), d, np.inf)
    dist = []
    for t in range(T):
        ys, xs = np.nonzero(al[t] > 0.5)
        if len(ys) == 0:
            dist.append(np.nan)
            continue
        sel = ys >= ys.max() - foot_rows
        fy = np.minimum(ys[sel] + 2, H - 1)                    # just under the feet
        vals = d[t, fy, xs[sel]]
        vals = vals[np.isfinite(vals)]
        dist.append(float(np.median(vals)) if len(vals) else np.nan)
    dist = np.asarray(dist)
    if np.all(np.isnan(dist)):
        return np.zeros_like(al), dist.tolist()
    dist = np.where(np.isnan(dist), np.nanmedian(dist), dist)
    dist = _median_smooth(dist, smooth) * distance_scale
    occ = np.empty_like(al)
    for t in range(T):
        near = dist[t] * (1 - margin) - d[t]
        occ[t] = np.where(v[t] > 0.5, np.clip(near / max(soft * dist[t], 1e-6), 0, 1), 0)
    return occ.astype(np.float32), np.asarray(dist).round(4).tolist()
