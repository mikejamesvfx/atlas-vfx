"""Plate Depth: MoGe depth per frame, scaled to the shot camera against the solve's points, saved as float EXR.

MoGe's depth is camera z (distance along the optical axis) with its own scale per frame. For each frame the solve
points the camera saw are projected; scale = robust median of (point depth in the shot camera) / (MoGe depth at that
pixel). Frames with too few points take the scale interpolated from their neighbours. The result is z depth in the
shot camera's units (metres after Measure & Scale).

EXR: channels Z (depth, NOT 1/z; 0 where invalid) and A (1 valid / 0 invalid), float32, zip, colorInteropID data
(OCIO Raw),
plus atlas_vfx:* attributes (camera, frame, units, scale). Pure numpy here; MoGe runs in node_plate_depth.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

MIN_POINTS = 20


def hfov_deg(cam: sc.ShotCamera, frame: int) -> float:
    k = cam.intrinsics(frame)
    return math.degrees(2 * math.atan(k.width / (2 * k.fx)))


def sample(depth: np.ndarray, uv_plate: np.ndarray, plate_w: int) -> np.ndarray:
    """Bilinear depth at plate-pixel positions (top-left origin) from a proxy-sized map; NaN outside / invalid."""
    H, W = depth.shape
    s = W / plate_w
    x = uv_plate[:, 0] * s - 0.5
    y = uv_plate[:, 1] * s - 0.5
    ok = (x >= 0) & (y >= 0) & (x <= W - 1) & (y <= H - 1)
    out = np.full(len(uv_plate), np.nan)
    x0 = np.floor(x[ok]).astype(int)
    y0 = np.floor(y[ok]).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = x[ok] - x0, y[ok] - y0
    d = depth.astype(float)
    corners = np.stack([d[y0, x0], d[y0, x1], d[y1, x0], d[y1, x1]], 1)
    val = (corners[:, 0] * (1 - fx) * (1 - fy) + corners[:, 1] * fx * (1 - fy)
           + corners[:, 2] * (1 - fx) * fy + corners[:, 3] * fx * fy)
    bad = ~np.all(np.isfinite(corners) & (corners > 0), axis=1)
    val[bad] = np.nan
    out[ok] = val
    return out


def frame_scale(cam: sc.ShotCamera, frame: int, depth: np.ndarray, points: dict) -> dict:
    """Robust scale for one frame: median ratio after a 3-MAD trim. Uses the solve's observations of this frame
    when present (no occluded points), else every point projected in front of the camera."""
    obs = points.get("observations", {}).get(frame)
    if obs is not None and len(obs[0]):
        idx = obs[0]
    else:
        idx = np.arange(len(points["xyz"]))
    uv, z = cam.project(frame, points["xyz"][idx], ignore_distortion=True)
    m = sample(depth, uv, cam.data["plate"]["width"])
    good = np.isfinite(m) & (z > 0)
    r = z[good] / m[good]
    if len(r) < MIN_POINTS:
        return {"frame": frame, "scale": None, "points": int(len(r))}
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    keep = np.abs(r - med) <= 3 * max(mad, 1e-9 * med)
    s = float(np.median(r[keep]))
    return {"frame": frame, "scale": s, "points": int(keep.sum()), "of": int(len(r)),
            "spread_pct": float(np.median(np.abs(r[keep] / s - 1)) * 100)}


def fill_scales(stats: list[dict]) -> list[dict]:
    """Interpolate scales for frames that had too few points (marked filled)."""
    known = [(s["frame"], s["scale"]) for s in stats if s["scale"] is not None]
    if not known:
        raise ValueError("no frame had enough solve points to scale the depth")
    f, v = zip(*known)
    for s in stats:
        if s["scale"] is None:
            s["scale"] = float(np.interp(s["frame"], f, v))
            s["filled"] = True
    return stats


def unproject(cam: sc.ShotCamera, frame: int, depth: np.ndarray, step: int = 1, mask=None) -> np.ndarray:
    """World points (N,3) from a z-depth map at proxy size (every `step` pixels, valid only)."""
    H, W = depth.shape
    k = cam.intrinsics(frame)
    s = W / k.width
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = depth[v, u].astype(float)
    ok = np.isfinite(z) & (z > 0)
    if mask is not None:
        ok &= mask[v, u] > 0.5
    up, vp = (u[ok] + 0.5) / s, (v[ok] + 0.5) / s
    zc = z[ok]
    Xc = np.stack([(up - k.cx) / k.fx * zc, -(vp - k.cy) / k.fy * zc, -zc], 1)
    M = cam.cam_to_world(frame)
    return Xc @ M[:3, :3].T + M[:3, 3]


def write_exr(path, depth: np.ndarray, valid: np.ndarray, attrs: dict) -> Path:
    import OpenImageIO as oiio

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    H, W = depth.shape
    z = np.where(valid, depth, 0.0).astype(np.float32)
    data = np.stack([z, valid.astype(np.float32)], axis=-1)
    spec = oiio.ImageSpec(W, H, 2, oiio.FLOAT)
    spec.channelnames = ("Z", "A")
    spec.alpha_channel = 1
    spec.z_channel = 0
    spec.attribute("compression", "zip")
    spec.attribute("oiio:ColorSpace", "data")        # OIIO 3 writes the EXR standard colorInteropID = data
    spec.attribute("atlas_vfx:ocio_colorspace", "Raw")
    for k, v in attrs.items():
        spec.attribute(f"atlas_vfx:{k}", v if isinstance(v, (int, float, str)) else json.dumps(v))
    out = oiio.ImageOutput.create(str(p))
    if out is None or not out.open(str(p), spec):
        raise RuntimeError(f"cannot write {p}: {oiio.geterror()}")
    out.write_image(data)
    out.close()
    return p


def read_exr(path) -> tuple[np.ndarray, np.ndarray, dict]:
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    spec = inp.spec()
    data = inp.read_image(oiio.FLOAT)
    attrs = {a.name.split(":", 1)[1]: a.value for a in spec.extra_attribs if a.name.startswith("atlas_vfx:")}
    attrs["colorspace"] = spec.get_string_attribute("oiio:ColorSpace")
    inp.close()
    names = list(spec.channelnames)
    return data[..., names.index("Z")], data[..., names.index("A")], attrs
