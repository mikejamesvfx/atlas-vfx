"""Foreground Element: a generated element on a plain backdrop -> premultiplied ACEScg RGBA texture -> a card standing
on the ground at a distance from the camera at a chosen frame.

Colour: the generator's output is display sRGB; it goes to ACEScg through the named OCIO transform
'sRGB Encoded Rec.709 (sRGB)' -> 'ACEScg' (a texture-style conversion; the element is then graded in comp).
Matte: a background-removal mask (BiRefNet), optionally eroded, then edge colour decontamination: with backdrop colour
B (median of the clearly-background pixels) the foreground colour is F = (I - (1 - a) B) / a, clipped. Then cropped to
the alpha's bounding box.
"""
from __future__ import annotations

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

GEN_SPACE = "sRGB Encoded Rec.709 (sRGB)"


def make_texture(image_srgb: np.ndarray, mask: np.ndarray, ocio_config: str, erode_px: int = 1,
                 feather_px: float = 0.0, pad: int = 4) -> tuple[np.ndarray, dict]:
    """image (H,W,3) display sRGB 0-1, mask (H,W) 1 = element. Returns premultiplied ACEScg RGBA and a report."""
    import cv2

    from atlas_vfx.camera_side.plate import ocio_config as load_cfg

    img = np.clip(image_srgb.astype(np.float32), 0, 1)
    a = np.clip(mask.astype(np.float32), 0, 1)
    if erode_px > 0:
        a = cv2.erode(a, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    if feather_px > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather_px)
    bg_px = a < 0.02
    B = np.median(img[bg_px], axis=0) if bg_px.sum() > 100 else np.zeros(3, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (img - (1 - a[..., None]) * B) / a[..., None]
    F = np.where(a[..., None] > 1e-3, np.clip(F, 0, 1), 0).astype(np.float32)

    cfg = load_cfg(ocio_config)
    lin = np.ascontiguousarray(F.copy())
    cfg.getProcessor(GEN_SPACE, "ACEScg").getDefaultCPUProcessor().applyRGB(lin)
    rgba = np.concatenate([lin * a[..., None], a[..., None]], -1)

    ys, xs = np.nonzero(a > 0.01)
    if len(ys) == 0:
        raise ValueError("matte is empty: nothing to place")
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    rgba = rgba[y0:y1, x0:x1]
    report = {"backdrop_srgb": B.tolist(), "erode_px": erode_px, "feather_px": feather_px,
              "crop": [int(x0), int(y0), int(x1), int(y1)], "size": [int(x1 - x0), int(y1 - y0)],
              "alpha_coverage": float((a > 0.5).mean()), "colour": f"OCIO {GEN_SPACE} -> ACEScg",
              "decontamination": "F = (I - (1 - a) * backdrop) / a, backdrop = median of a < 0.02"}
    return rgba, report


def ground_height(points_xyz: np.ndarray | None, x: float, z: float, radius: float = 0.75,
                  default: float | None = None) -> tuple[float, str]:
    """Median y of solve points within `radius` metres (horizontally) of (x, z), using the lowest half (ground, not
    whatever stands on it)."""
    if points_xyz is not None and len(points_xyz):
        d = np.hypot(points_xyz[:, 0] - x, points_xyz[:, 2] - z)
        near = points_xyz[d < radius, 1]
        if len(near) >= 10:
            low = np.sort(near)[: max(5, len(near) // 2)]
            return float(np.median(low)), f"median of the lower half of {len(near)} solve points within {radius} m"
    if default is None:
        raise ValueError(f"no solve points near ({x:.2f}, {z:.2f}) to find the ground; give ground_y")
    return float(default), "given"


def place_card(cam: sc.ShotCamera, frame: int, distance_m: float, lateral_m: float, height_m: float, aspect: float,
               ground_y: float, yaw_deg: float = 0.0) -> dict:
    """An upright card facing the camera at `frame`, its bottom edge on the ground at ground_y, `distance_m` ahead
    (horizontally) and `lateral_m` to the right. aspect = texture width / height."""
    M = cam.cam_to_world(frame)
    C = M[:3, 3]
    f = -M[:3, 2]
    f = np.array([f[0], 0.0, f[2]])
    f /= np.linalg.norm(f)
    up = np.array([0, 1.0, 0])
    r = np.cross(f, up)
    if yaw_deg:
        a = np.radians(yaw_deg)
        r = r * np.cos(a) + f * np.sin(a)
    base = C + f * distance_m + np.cross(f, up) * lateral_m
    base[1] = ground_y
    w = height_m * aspect
    TL = base - r * w / 2 + up * height_m
    TR = base + r * w / 2 + up * height_m
    BR = base + r * w / 2
    BL = base - r * w / 2
    return {"corners": [TL.tolist(), TR.tolist(), BR.tolist(), BL.tolist()], "base": base.tolist(),
            "size_m": [float(w), float(height_m)], "placed_at_frame": frame, "distance_m": distance_m,
            "lateral_m": lateral_m, "ground_y": ground_y}
