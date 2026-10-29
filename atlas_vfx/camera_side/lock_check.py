"""Lock Check: a gridded card (and any measured points) projected through every frame onto the display proxy -> mp4.
If the camera is right, the card stays glued to the surface it sits on. Ported from prototypes/vfx_probe/card_check.py.

Card: centre (metres, shot camera world), orientation 'ground' (horizontal, normal +Y) or 'vertical' (upright,
facing the first frame's camera), size in metres, grid divisions.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import plate as pl

NEAR = 0.05   # metres in front of the camera


def card_lines(cam: sc.ShotCamera, centre, orientation: str = "ground", width_m: float = 1.0,
               height_m: float = 1.0, divisions: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    c = np.asarray(centre, float)
    if orientation == "ground":
        u, v = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
        f0 = -cam.cam_to_world(cam.frames[0])[:3, 2]
        flat = np.array([f0[0], 0.0, f0[2]])
        if np.linalg.norm(flat) > 1e-6:                     # square to the first view
            v = flat / np.linalg.norm(flat)
            u = np.cross(v, [0, 1.0, 0])
    elif orientation == "vertical":
        to_cam = cam.position(cam.frames[0]) - c
        n = np.array([to_cam[0], 0.0, to_cam[2]])
        n /= np.linalg.norm(n)
        u, v = np.cross([0, 1.0, 0], n), np.array([0, 1.0, 0])
    else:
        raise ValueError(f"orientation must be ground or vertical, not {orientation!r}")
    o = c - u * width_m / 2 - v * height_m / 2
    U, V = u * width_m, v * height_m
    lines = []
    for k in range(divisions + 1):
        a = k / divisions
        lines.append((o + U * a, o + U * a + V))
        lines.append((o + V * a, o + V * a + U))
    return lines


def _clip_segment(pa_c, pb_c):
    """Clip a camera-space segment (Nuke axes, visible z < -NEAR) to the near plane. None if fully behind."""
    za, zb = -pa_c[2], -pb_c[2]
    if za < NEAR and zb < NEAR:
        return None
    if za < NEAR or zb < NEAR:
        t = (NEAR - za) / (zb - za)
        p = pa_c + (pb_c - pa_c) * t
        if za < NEAR:
            pa_c = p
        else:
            pb_c = p
    return pa_c, pb_c


def project_segments(cam: sc.ShotCamera, frame: int, lines, scale: float):
    """Pixel segments at proxy scale (plate px * scale) for one frame, near-clipped."""
    W2C = cam.world_to_cam(frame)
    k = cam.intrinsics(frame)
    out = []
    for a, b in lines:
        seg = _clip_segment(W2C[:3, :3] @ a + W2C[:3, 3], W2C[:3, :3] @ b + W2C[:3, 3])
        if seg is None:
            continue
        pts = []
        for p in seg:
            d = -p[2]
            pts.append(((k.cx + k.fx * p[0] / d) * scale, (k.cy - k.fy * p[1] / d) * scale))
        out.append(pts)
    return out


def render(plate_ref: dict, shot_camera: dict | sc.ShotCamera, out_mp4, *, lines, marks: dict | None = None,
           width: int = 1024, keep_every: int = 0):
    """Draw the card (green) and marks (red crosses with labels) on display proxies for the camera's frames that
    fall in the plate_ref range; encode to mp4. Returns (path, kept preview frames as float32 HxWx3)."""
    import cv2

    cam = shot_camera if isinstance(shot_camera, sc.ShotCamera) else sc.ShotCamera(shot_camera)
    off = plate_ref["frame_offset"]
    f0 = max(plate_ref["first"], cam.frames[0] - off)
    f1 = min(plate_ref["last"], cam.frames[-1] - off)
    if f1 < f0:
        raise ValueError(f"camera frames {cam.frames[0]}-{cam.frames[-1]} do not overlap the plate "
                         f"{plate_ref['nuke_first']}-{plate_ref['nuke_last']}")
    W, H = pl.proxy_size(plate_ref, width)
    cw, ch = cam.data["plate"]["width"], cam.data["plate"]["height"]
    if abs(cw / ch - plate_ref["width"] / plate_ref["height"]) > 1e-3:
        raise ValueError(f"camera plate {cw}x{ch} and plate {plate_ref['width']}x{plate_ref['height']} differ in aspect")
    scale = W / cw
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(plate_ref["fps"][0] / plate_ref["fps"][1]), "-i", "-", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    kept = []
    lw = max(1, round(W / 600))
    try:
        for i, img in pl.decode_rgb(plate_ref, f0, f1, width=width):
            n = i + off
            d8 = np.ascontiguousarray((pl.to_display(img, plate_ref) * 255 + 0.5).astype(np.uint8))
            for (xa, ya), (xb, yb) in project_segments(cam, n, lines, scale):
                cv2.line(d8, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (0, 255, 120),
                         lw, cv2.LINE_AA)
            for label, X in (marks or {}).items():
                uv, depth = cam.project(n, X)
                if depth[0] > NEAR:
                    x, y = int(round(uv[0][0] * scale)), int(round(uv[0][1] * scale))
                    cv2.drawMarker(d8, (x, y), (255, 60, 60), cv2.MARKER_CROSS, 6 * lw + 6, lw, cv2.LINE_AA)
                    cv2.putText(d8, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * lw, (255, 60, 60), lw,
                                cv2.LINE_AA)
            cv2.putText(d8, f"{cam.name}  {n}", (10, 20 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, (255, 230, 0), lw,
                        cv2.LINE_AA)
            enc.stdin.write(d8.tobytes())
            if keep_every and (i - f0) % keep_every == 0:
                kept.append(d8.astype(np.float32) / 255.0)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed ({enc.returncode})")
    return out, kept
