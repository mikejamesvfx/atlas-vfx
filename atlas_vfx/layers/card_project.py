"""Card Project: an RGBA texture on a flat card in the shot camera world, rendered through the shot camera.

Per output pixel: the camera ray is intersected with the card's plane (exact for a planar card, the same mapping a
homography gives, but it stays right when the camera is close to or passes through the card, where a homography
breaks). The hit gives (u, v) on the card; the premultiplied texture is sampled bilinearly (cv2.remap). Output is
premultiplied RGBA float in the texture's colourspace (ACEScg for atlas-vfx elements), at plate resolution x scale.

A card is {"name", "corners": [TL, TR, BR, BL] metres, "texture": path to an RGBA EXR, ...}; texture row 0 is the
TL-TR edge. No motion blur, no lighting: a 2.5D element.
"""
from __future__ import annotations

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

NEAR = 0.02


def card_at(card: dict, frame: int) -> tuple[list, str]:
    """(corners, texture path) of a card at a plate frame. A still card has "corners" and "texture"; an animated one
    (Character Insert) has "corners_by_frame" and/or "texture_pattern" (printf, texture index) + "texture_by_frame"
    (plate frame -> index); frames outside the maps hold the nearest end."""
    def pick(m):
        if str(frame) in m:
            return m[str(frame)]
        keys = sorted(int(k) for k in m)
        return m[str(keys[0] if frame < keys[0] else keys[-1])]

    corners = pick(card["corners_by_frame"]) if card.get("corners_by_frame") else card["corners"]
    if card.get("texture_pattern"):
        tex = card["texture_pattern"] % pick(card["texture_by_frame"])
    else:
        tex = card["texture"]
    return corners, tex


def card_frame(corners):
    c = np.asarray(corners, float)
    P0, U, V = c[0], c[1] - c[0], c[3] - c[0]
    n = np.cross(U, V)
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("degenerate card")
    if np.linalg.norm(c[0] + U + V - c[2]) > 1e-6 * max(np.linalg.norm(U), np.linalg.norm(V)):
        raise ValueError("card corners are not a parallelogram (TL, TR, BR, BL)")
    return P0, U, V, n


def render(cam: sc.ShotCamera, frame: int, corners, texture: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Premultiplied RGBA (H*scale, W*scale, C) of the card seen from `frame`. Pixels the card does not cover are 0."""
    import cv2

    k = cam.intrinsics(frame)
    W, H = int(round(k.width * scale)), int(round(k.height * scale))
    th, tw, C = texture.shape
    out = np.zeros((H, W, C), np.float32)
    P0, U, V, n = card_frame(corners)
    M = cam.cam_to_world(frame)
    R, Cw = M[:3, :3], M[:3, 3]

    # bounding box of the card on screen when it is fully in front; else the whole frame
    uv, depth = cam.project(frame, np.asarray(corners, float))
    if np.all(depth > NEAR):
        x0, y0 = np.floor(uv.min(0) * scale).astype(int) - 2
        x1, y1 = np.ceil(uv.max(0) * scale).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            return out
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    px = (xs + 0.5) / scale
    py = (ys + 0.5) / scale
    d_cam = np.stack([(px - k.cx) / k.fx, -(py - k.cy) / k.fy, -np.ones_like(px)], -1)
    d = d_cam @ R.T.astype(np.float32)
    den = d @ n.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((P0 - Cw) @ n) / den
    hit = Cw.astype(np.float32) + d * t[..., None]
    rel = hit - P0.astype(np.float32)
    # rel = u U + v V, solved with the Gram matrix (U and V need not be perpendicular)
    G = np.array([[U @ U, U @ V], [U @ V, V @ V]])
    Ginv = np.linalg.inv(G)
    a_u, a_v = (Ginv[0, 0] * U + Ginv[0, 1] * V), (Ginv[1, 0] * U + Ginv[1, 1] * V)
    u = rel @ a_u.astype(np.float32)
    v = rel @ a_v.astype(np.float32)
    depth_ok = t * (-(d @ R[:, 2].astype(np.float32))) > NEAR      # in front of the camera
    ok = np.isfinite(t) & depth_ok & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    mapx = np.where(ok, u * tw - 0.5, -10).astype(np.float32)
    mapy = np.where(ok, v * th - 0.5, -10).astype(np.float32)
    samp = cv2.remap(texture.astype(np.float32), mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if samp.ndim == 2:
        samp = samp[..., None]
    out[y0:y1, x0:x1] = np.where(ok[..., None], samp, 0)
    return out


def over(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Premultiplied A over B (RGBA or RGB background)."""
    a = fg[..., 3:4]
    rgb = fg[..., :3] + bg[..., :3] * (1 - a)
    if bg.shape[-1] == 4:
        return np.concatenate([rgb, a + bg[..., 3:4] * (1 - a)], -1)
    return rgb
