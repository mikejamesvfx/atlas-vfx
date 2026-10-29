"""Depth holdout: cut a card out wherever the plate is nearer the camera than the card.

Plate depth comes from Plate Depth (MoGe z depth in the shot camera's metres, EXR channels Z and A). The card's own
depth per pixel is where each camera ray meets the card's plane. Where plate z is smaller than card z by more than
`margin_m`, the card is hidden, ramping over `soft_m` so the edge is not a hard step. Pixels with no valid plate depth
are left alone. The result multiplies the premultiplied RGBA, so it lands in the layer EXRs and every consumer
(Comp Preview, the Nuke build) gets it.

2.5D limits: the plate depth is a 1024-wide estimate, so holdout edges are only as sharp as that; thin things (a
railing, a post) hold out well, hair-fine detail does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc


def card_depth(cam: sc.ShotCamera, frame: int, corners, w: int, h: int, box=None) -> np.ndarray:
    """Camera z (metres, positive in front) of the card's plane through every pixel of a w x h image of the plate
    frame (or only box = (y0, y1, x0, x1) of it). inf where the ray misses the plane in front of the camera."""
    y0, y1, x0, x1 = box if box is not None else (0, h, 0, w)
    W2C = cam.world_to_cam(frame)
    P = np.asarray(corners, float)[:3] @ W2C[:3, :3].T + W2C[:3, 3]      # three corners in camera space
    n = np.cross(P[1] - P[0], P[2] - P[0])
    k = cam.intrinsics(frame)
    s = w / cam.data["plate"]["width"]
    u = (np.arange(x0, x1) + 0.5) / s
    v = (np.arange(y0, y1) + 0.5) / s
    dx = (u - k.cx) / k.fx
    dy = -(v - k.cy) / k.fy
    D = np.stack(np.broadcast_arrays(dx[None, :], dy[:, None], np.full((y1 - y0, x1 - x0), -1.0)), -1)
    den = D @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (P[0] @ n) / den                                              # depth along -z (ray z component is -1)
    t = np.where(np.isfinite(t) & (t > 0), t, np.inf)
    return t.astype(np.float32)


def keep_mask(plate_z: np.ndarray, plate_valid: np.ndarray, card_z: np.ndarray, margin_m: float = 0.6,
              soft_m: float = 0.1) -> np.ndarray:
    """1 where the card shows, 0 where the plate is in front of it (by more than margin + soft), ramp between."""
    behind = card_z - plate_z - margin_m                                  # > 0: plate nearer than the card
    occ = np.clip(behind / max(soft_m, 1e-6), 0.0, 1.0)
    occ = np.where((plate_valid > 0.5) & (plate_z > 0) & np.isfinite(card_z), occ, 0.0)
    return (1.0 - occ).astype(np.float32)


def depth_frames(folder) -> dict[int, Path]:
    """{nuke frame: depth EXR} for a Plate Depth folder (depth.<frame>.exr)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"holdout depth folder not found: {folder}")
    out = {}
    for p in folder.glob("depth.*.exr"):
        try:
            out[int(p.name.split(".")[1])] = p
        except ValueError:
            continue
    if not out:
        raise FileNotFoundError(f"no depth.<frame>.exr in {folder}")
    return out


def apply(img: np.ndarray, cam: sc.ShotCamera, frame: int, corners, depth_path, margin_m: float = 0.6,
          soft_m: float = 0.1) -> tuple[np.ndarray, float]:
    """Hold out a rendered card image (h, w, 4 premultiplied). Returns (image, fraction of the card's alpha removed)."""
    import cv2

    from atlas_vfx.camera_side import plate_depth as pdp

    h, w = img.shape[:2]
    a = img[..., 3]
    if not (a > 0).any():
        return img, 0.0
    z, valid, _ = pdp.read_exr(depth_path)
    z = cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(a > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1        # only where the card is
    cz = card_depth(cam, frame, corners, w, h, (y0, y1, x0, x1))
    keep = keep_mask(z[y0:y1, x0:x1], valid[y0:y1, x0:x1], cz, margin_m, soft_m)
    out = img.copy()
    out[y0:y1, x0:x1] *= keep[..., None]
    removed = float(1.0 - out[..., 3].sum() / max(a.sum(), 1e-9))
    return out, removed
