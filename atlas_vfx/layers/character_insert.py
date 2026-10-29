"""Character Insert: a generated character clip on flat grey -> keyed premultiplied ACEScg RGBA sequence -> an upright
card whose feet line sits on the deck plane at a picked point, scaled so the figure is a given height.

Key: a distance key against the grey backdrop. Generators' "flat grey" is a vignetted gradient with a floor, so the
backdrop B is a smooth clean plate, a degree-4 polynomial per channel fitted to the backdrop pixels of the clip's
median frame (seeded from the side and top borders, refitted on every pixel near the fit). Per pixel d = |I - B| in
display sRGB. A floor shadow under the feet is darker than the fitted floor, so it keys as a soft contact shadow.
A plain clip((d - lo) / (hi - lo)) saturates long before full coverage, leaving grey in every edge, so d is divided by
the largest d within 3 px (the solid foreground beside an edge pixel; never less than hi): an edge pixel then gets its
coverage. Below lo the alpha ramps to 0 (backdrop noise). Then eroded by erode_px. Edges are decontaminated as in Foreground Element
(F = (I - (1 - a) B) / a), converted 'sRGB Encoded Rec.709 (sRGB)' -> ACEScg by OCIO, premultiplied, and every frame is
cropped to the same box (the union over the clip) so the card keeps one size.

Scale: the figure's height in texture pixels is the median over frames of (feet row - head row) of the alpha > 0.5
rows. The feet row is a low percentile of those per-frame bottoms, so an odd frame cannot lift the feet off the deck.
card height = height_m * texture_height / figure_px; the card is placed so the feet row lands on the deck plane.

Placement: 'billboard' turns the card about the vertical through the feet point to face the camera on every frame
(keeps a fixed-view clip from going thin as the camera walks round it); 'fixed' faces the camera at one frame. Either
way the feet row stays on the plane, so the feet hold to the deck. This is 2.5D: no parallax inside the figure, no
contact shadow beyond what the key keeps, and the view of him never changes. Fine for a camera that mostly walks
forward at him, wrong for an orbit.
"""
from __future__ import annotations

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from .foreground_element import GEN_SPACE

LOCAL_PX = 3
POLY_DEG = 4
KEY_METHOD = ("distance key: d = |I - B| in display sRGB, B = smooth backdrop plate (degree-4 polynomial fitted to "
              "the backdrop pixels of the median frame, refitted); alpha = d / max(local max of d within 3 px, hi), ramped to 0 below lo; then erode; edge decontamination F = (I - (1 - a) B) / a; OCIO "
              f"'{GEN_SPACE}' -> ACEScg; premultiplied; one crop for the whole clip")


def backdrop_colour(frames: np.ndarray, border: float = 0.04) -> np.ndarray:
    """Median colour of the top and side borders of every frame (N,H,W,3)."""
    n, h, w, _ = frames.shape
    b = max(2, int(round(border * min(h, w))))
    px = [frames[:, :b].reshape(-1, 3), frames[:, :int(h * 0.7), :b].reshape(-1, 3),
          frames[:, :int(h * 0.7), -b:].reshape(-1, 3)]
    return np.median(np.concatenate(px)[:: max(1, n // 8)], axis=0).astype(np.float32)


def _poly_terms(x, y, deg):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def backdrop_plate(frames: np.ndarray, deg: int = POLY_DEG, tol: float = 0.035, iters: int = 6,
                   step: int = 8) -> tuple[np.ndarray, dict]:
    """A smooth clean plate of the backdrop (H,W,3): a degree-`deg` 2D polynomial per channel, fitted to the pixels of
    the clip's median frame that look like backdrop. Starts from the side and top borders, then each pass refits on
    every pixel within tol of the current fit (a generator's "flat grey" is a vignetted gradient with a floor, so one
    colour cannot key it). The camera is locked off, so one plate serves the whole clip."""
    n, H, W, _ = frames.shape
    med = np.median(frames[np.linspace(0, n - 1, min(n, 9)).astype(int)], axis=0).astype(np.float32)
    small = med[::step, ::step]
    h, w = small.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    xn, yn = (xx * step + step / 2) / W * 2 - 1, (yy * step + step / 2) / H * 2 - 1
    A = _poly_terms(xn.ravel(), yn.ravel(), deg)
    bg = np.zeros((h, w), bool)
    bg[:, : int(w * 0.12)] = bg[:, -int(w * 0.12):] = True
    bg[: int(h * 0.06)] = True
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[bg.ravel()], small.reshape(-1, 3)[bg.ravel()], rcond=None)
        fit = (A @ coef).reshape(h, w, 3)
        d = np.linalg.norm(small - fit, axis=-1)
        bg = d < tol
    Y, X = np.mgrid[0:H, 0:W]
    plate = (_poly_terms((X + 0.5) / W * 2 - 1, (Y + 0.5) / H * 2 - 1, deg) @ coef).astype(np.float32)
    res = d[bg]
    return plate, {"model": f"degree {deg} polynomial per channel, {iters} refits, tol {tol}",
                   "backdrop_share": float(bg.mean()), "residual_rms": float(np.sqrt((res ** 2).mean())),
                   "plate_min": plate.reshape(-1, 3).min(0).tolist(), "plate_max": plate.reshape(-1, 3).max(0).tolist()}


def grey_key(frames: np.ndarray, lo: float = 0.06, hi: float = 0.16, erode_px: int = 1,
             backdrop: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """frames (N,H,W,3) display sRGB 0-1 -> alpha (N,H,W), backdrop plate (H,W,3) or colour (3,), report."""
    import cv2

    if backdrop is None:
        B, info = backdrop_plate(frames)
    else:
        B, info = np.asarray(backdrop, np.float32), {"model": "given"}
    alpha = np.empty(frames.shape[:3], np.float32)
    kern = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8) if erode_px > 0 else None
    near = np.ones((2 * LOCAL_PX + 1, 2 * LOCAL_PX + 1), np.uint8)
    for i, f in enumerate(frames):
        d = np.linalg.norm(f.astype(np.float32) - B, axis=-1)
        # an edge pixel's distance over the distance of the solid foreground next to it is its coverage; hi is the
        # floor for that divisor (so faint noise does not become solid), lo ramps the noise out
        den = np.maximum(cv2.dilate(d, near), hi)
        a = (np.clip(d / den, 0, 1) * np.clip((d - lo / 2) / (lo / 2), 0, 1)).astype(np.float32)
        a[a > 0.98] = 1.0                       # solid: d varies a little inside as the backdrop behind it grades
        if kern is not None:
            a = cv2.erode(a, kern)
        alpha[i] = a
    return alpha, B, info


MASK_METHOD = ("matte from a background-removal model (BiRefNet), eroded; backdrop = smooth plate (degree-4 polynomial "
               "of the backdrop pixels) used only for edge decontamination F = (I - (1 - a) B) / a; OCIO "
               f"'{GEN_SPACE}' -> ACEScg; premultiplied; one crop for the whole clip")


def mask_key(frames: np.ndarray, masks: np.ndarray, erode_px: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use a per-frame matte (N,H,W, 1 = figure) instead of the distance key; the backdrop plate is still fitted for
    decontamination. A generator's grey floor, its seam and the floor shadows defeat a distance key; a matting model
    does not care."""
    import cv2

    B, info = backdrop_plate(frames)
    alpha = np.empty(frames.shape[:3], np.float32)
    kern = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8) if erode_px > 0 else None
    for i in range(len(frames)):
        m = masks[min(i, len(masks) - 1)].astype(np.float32)
        if m.shape != frames.shape[1:3]:
            m = cv2.resize(m, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0, 1)
        alpha[i] = cv2.erode(m, kern) if kern is not None else m
    info = dict(info, matte="given mask", erode_px=erode_px)
    return alpha, B, info


def figure_rows(alpha: np.ndarray, thresh: float = 0.5, min_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: head row (first row with >= min_px alpha pixels) and feet row (one past the last such row)."""
    heads, feet = [], []
    for a in alpha:
        rows = np.nonzero((a > thresh).sum(1) >= min_px)[0]
        if len(rows) == 0:
            heads.append(np.nan)
            feet.append(np.nan)
        else:
            heads.append(rows[0])
            feet.append(rows[-1] + 1)
    return np.array(heads, float), np.array(feet, float)


def make_sequence(frames: np.ndarray, alpha: np.ndarray, B: np.ndarray, ocio_config: str,
                  pad: int = 4, method: str = KEY_METHOD) -> tuple[np.ndarray, dict]:
    """Keyed clip -> premultiplied ACEScg RGBA (N,h,w,4) cropped to one box, and the figure metrics in that box."""
    from atlas_vfx.camera_side.plate import ocio_config as load_cfg

    union = (alpha > 0.01).any(0)
    ys, xs = np.nonzero(union)
    if len(ys) == 0:
        raise ValueError("key is empty: nothing to place (check lo/hi against the backdrop)")
    H, W = union.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    proc = load_cfg(ocio_config).getProcessor(GEN_SPACE, "ACEScg").getDefaultCPUProcessor()
    out = np.empty((len(frames), y1 - y0, x1 - x0, 4), np.float32)
    for i in range(len(frames)):
        img = np.clip(frames[i, y0:y1, x0:x1].astype(np.float32), 0, 1)
        a = alpha[i, y0:y1, x0:x1, None]
        Bc = B[y0:y1, x0:x1] if B.ndim == 3 else B
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (img - (1 - a) * Bc) / a
        F = np.ascontiguousarray(np.where(a > 1e-3, np.clip(F, 0, 1), 0).astype(np.float32))
        proc.applyRGB(F)
        out[i, ..., :3] = F * a
        out[i, ..., 3:] = a
    heads, feet = figure_rows(out[..., 3])
    ok = np.isfinite(heads)
    if not ok.any():
        raise ValueError("no frame has a figure after keying")
    feet_row = float(np.percentile(feet[ok], 90))
    fig_px = float(np.median(feet[ok] - heads[ok]))
    th = out.shape[1]
    report = {"method": method, "backdrop_srgb": (B.reshape(-1, 3).mean(0) if B.ndim == 3 else B).tolist(), "crop": [int(x0), int(y0), int(x1), int(y1)],
              "size": [int(x1 - x0), int(y1 - y0)], "frames": len(frames), "figure_px": fig_px,
              "feet_row": feet_row, "feet_v": feet_row / th,
              "feet_row_spread_px": float(np.ptp(feet[ok])), "head_row_spread_px": float(np.ptp(heads[ok])),
              "empty_frames": int((~ok).sum())}
    return out, report


def deck_point_from_pick(cam: sc.ShotCamera, frame: int, u: float, v: float, deck_y: float = 0.0) -> np.ndarray:
    """Hit of the camera ray through pixel (u, v) (top-left origin, plate pixels) with the plane y = deck_y."""
    k = cam.intrinsics(frame)
    M = cam.cam_to_world(frame)
    d = M[:3, :3] @ np.array([(u + 0.5 - k.cx) / k.fx, -(v + 0.5 - k.cy) / k.fy, -1.0])
    C = M[:3, 3]
    if abs(d[1]) < 1e-9:
        raise ValueError("pick ray is parallel to the deck")
    t = (deck_y - C[1]) / d[1]
    if t <= 0:
        raise ValueError(f"pick ({u}, {v}) at frame {frame} looks above the horizon: it never meets the deck")
    return C + t * d


def card_corners(foot: np.ndarray, facing_from: np.ndarray, width_m: float, height_m: float, feet_v: float,
                 deck_y: float) -> list:
    """Upright card [TL, TR, BR, BL], centred on `foot`, facing the point `facing_from` (horizontally), with texture
    row fraction feet_v on y = deck_y."""
    f = np.array([facing_from[0] - foot[0], 0.0, facing_from[2] - foot[2]])
    nrm = np.linalg.norm(f)
    f = f / nrm if nrm > 1e-6 else np.array([0, 0, 1.0])
    up = np.array([0, 1.0, 0])
    r = np.cross(up, f)                                   # right as seen from the camera
    top = deck_y + feet_v * height_m
    base = np.array([foot[0], top - height_m, foot[2]])
    TL = base - r * width_m / 2 + up * height_m
    TR = base + r * width_m / 2 + up * height_m
    BR = base + r * width_m / 2
    BL = base - r * width_m / 2
    return [TL.tolist(), TR.tolist(), BR.tolist(), BL.tolist()]


def place_character(cam: sc.ShotCamera, frames, foot, tex_size, figure_px: float, feet_v: float, height_m: float,
                    deck_y: float = 0.0, mode: str = "billboard", face_frame: int | None = None) -> dict:
    """Card dict with corners for every frame. tex_size = (w, h) texture pixels."""
    tw, th = tex_size
    card_h = height_m * th / figure_px
    card_w = card_h * tw / th
    foot = np.asarray(foot, float)
    by_frame = {}
    for n in frames:
        src = n if mode == "billboard" else (face_frame if face_frame is not None else frames[0])
        by_frame[str(n)] = card_corners(foot, cam.position(src), card_w, card_h, feet_v, deck_y)
    return {"corners_by_frame": by_frame, "corners": by_frame[str(frames[0])], "foot": foot.tolist(),
            "size_m": [float(card_w), float(card_h)], "figure_height_m": height_m, "feet_v": feet_v,
            "deck_y": deck_y, "mode": mode}


def time_map(n_tex: int, plate_frames: list, start: int = 0, end_mode: str = "hold") -> dict:
    """Plate frame -> texture index. The clip starts at `start` on the first plate frame; past its end it holds the
    last frame, ping-pongs, or loops back to `start` (loop)."""
    out = {}
    start = max(0, min(int(start), n_tex - 1))
    for i, n in enumerate(plate_frames):
        j = start + i
        if j >= n_tex:
            if end_mode == "loop" and n_tex - start > 1:
                j = start + (j - start) % (n_tex - start)
            elif end_mode == "pingpong" and n_tex > 1:
                period = 2 * (n_tex - 1)
                j = j % period
                j = j if j < n_tex else period - j
            else:
                j = n_tex - 1
        out[str(n)] = int(j)
    return out
