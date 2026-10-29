"""LTX integration pass: a generated layer (e.g. a character on its card) is re-rendered into the real plate by LTX-2.5 with
the in/outpainting IC-LoRA, so it picks up the plate's light, blur, grain and a contact shadow. The camera lock stays
ours: LTX only paints inside a mask built from the layer's own projected alpha, and everything outside the mask is
the original plate, untouched, in the delivered layer.

Prep (to LTX):
- frames: plate (input space -> ACEScg) with the layer comped over -> display-referred through the named OCIO view
  (ACES 2.0 SDR, sRGB display): LTX expects ordinary video. Resized to the work size (default 1920x1024, a
  non-uniform scale of about 1 % that Return undoes exactly) and to half that for LTX's first stage.
- reference video for the IC-LoRA: the plate WITHOUT the layer (LTX greens the masked region and paints it).
- start image: the first frame WITH the layer, so the element's look enters through LTX's image conditioning.
- mask: the layer's alpha > thresh, smeared down by shadow_px (room for a contact shadow under the feet) and dilated
  by dilate_px, at the work size.

Return (from LTX):
- LTX's display-referred frames -> inverse of the same named view -> ACEScg, upscaled to plate size;
  alpha = the mask, upscaled and feathered. Delivered as a premultiplied ACEScg RGBA EXR layer that goes over the
  plate instead of the original layer. Outside the mask alpha is 0, so the plate shows through exactly.
- The inverse view is only trusted where the forward/inverse round trip is close (ACES 2.0 SDR compresses
  highlights); `roundtrip_report` measures it on the plate itself and Return records it in the sidecar.
"""
from __future__ import annotations

import numpy as np

from atlas_vfx.camera_side import plate as pl


def view_processors(ocio_config: str, working: str = "ACEScg", display: str = pl.DEFAULT_DISPLAY,
                    view: str = pl.DEFAULT_VIEW):
    """(forward, inverse) CPU processors: working <-> display through the named display/view."""
    import PyOpenColorIO as ocio

    cfg = pl.ocio_config(ocio_config)
    fwd = cfg.getProcessor(ocio.DisplayViewTransform(src=working, display=display, view=view))
    inv = cfg.getProcessor(ocio.DisplayViewTransform(src=working, display=display, view=view,
                                                     direction=ocio.TRANSFORM_DIR_INVERSE))
    return fwd.getDefaultCPUProcessor(), inv.getDefaultCPUProcessor()


def to_view(lin: np.ndarray, fwd) -> np.ndarray:
    out = np.ascontiguousarray(lin, dtype=np.float32).copy()
    fwd.applyRGB(out)
    return np.clip(out, 0.0, 1.0)


def from_view(disp: np.ndarray, inv) -> np.ndarray:
    out = np.ascontiguousarray(np.clip(disp, 0.0, 1.0), dtype=np.float32).copy()
    inv.applyRGB(out)
    return out


def roundtrip_report(lin: np.ndarray, fwd, inv, step: int = 4) -> dict:
    """ACEScg -> view -> inverse on real pixels: relative error, and the share of pixels the view clips."""
    s = lin[::step, ::step].reshape(-1, 3).astype(np.float32)
    back = from_view(to_view(s, fwd), inv)
    lum = lambda x: x @ np.array([0.2722287, 0.6740818, 0.0536895], np.float32)   # AP1 luminance weights
    L, Lb = lum(s), lum(back)
    ok = L > 1e-4
    rel = np.abs(Lb[ok] - L[ok]) / L[ok]
    return {"median_rel_err": float(np.median(rel)), "p99_rel_err": float(np.percentile(rel, 99)),
            "share_over_5pct": float((rel > 0.05).mean()), "max_luminance": float(L.max())}


def make_mask(alpha: np.ndarray, size: tuple[int, int], thresh: float = 0.05, shadow_px: int = 24,
              dilate_px: int = 12) -> np.ndarray:
    """Layer alpha (plate res) -> inpaint mask (work res): alpha > thresh, smeared down shadow_px, dilated."""
    import cv2

    W, H = size
    a = cv2.resize(alpha.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    m = (a > thresh).astype(np.uint8)
    if shadow_px > 0:
        k = np.zeros((2 * shadow_px + 1, 1), np.uint8)
        k[: shadow_px + 1] = 1                     # dilate with the upper half: grows the mask downward only
        m = cv2.dilate(m, k)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    return m.astype(np.float32)


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    interp = cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def bbox(alpha: np.ndarray, thresh: float = 0.05):
    ys, xs = np.nonzero(alpha > thresh)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_windows(boxes: list, work_size: tuple[int, int], plate_size: tuple[int, int], margin: int = 96,
                 smooth: int = 8) -> list[list[int]]:
    """Per-frame crop windows [x0, y0, w, h] (plate px) with the work aspect, never smaller than the work size (so the
    element is at native resolution whenever it fits), growing only as much as needed to hold its box + margin,
    smoothed over +-smooth frames (a jittery crop reads as camera shake to LTX) and kept inside the plate. Frames
    with no element reuse the nearest window."""
    ww, wh = work_size
    W, H = plate_size
    aspect = ww / wh
    n = len(boxes)
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        raise ValueError("element never on screen")
    filled = [boxes[i] if boxes[i] is not None else boxes[min(idx, key=lambda j: abs(j - i))] for i in range(n)]
    b = np.array(filled, float)
    b[:, :2] -= margin
    b[:, 2:] += margin
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    need_h = np.maximum(b[:, 3] - b[:, 1], (b[:, 2] - b[:, 0]) / aspect)
    scale = np.maximum(need_h / wh, 1.0)
    max_scale = min(H / wh, W / ww)
    k = smooth
    pad = lambda v: np.concatenate([np.full(k, v[0]), v, np.full(k, v[-1])])
    smax = np.array([pad(scale)[i:i + 2 * k + 1].max() for i in range(n)])          # grow early, never late
    s = np.convolve(pad(smax), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    s = np.maximum(np.minimum(np.maximum(s, scale), max_scale), 1.0)
    cx = np.convolve(pad((b[:, 0] + b[:, 2]) / 2), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    cy = np.convolve(pad((b[:, 1] + b[:, 3]) / 2), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    out = []
    for i in range(n):
        w, h = int(round(ww * s[i])), int(round(wh * s[i]))
        w, h = min(w, W), min(h, H)
        x0 = int(round(cx[i] - w / 2))
        y0 = int(round(cy[i] - h / 2))
        # keep the element's box inside (smoothing may lag it), then keep the window inside the plate
        x0 = min(max(x0, int(b[i, 2]) - w), int(b[i, 0]))
        y0 = min(max(y0, int(b[i, 3]) - h), int(b[i, 1]))
        x0 = min(max(x0, 0), W - w)
        y0 = min(max(y0, 0), H - h)
        out.append([x0, y0, w, h])
    return out


def crop(img: np.ndarray, win: list[int], size: tuple[int, int]) -> np.ndarray:
    x0, y0, w, h = win
    return resize(img[y0:y0 + h, x0:x0 + w], size)


def return_window(ltx_disp: np.ndarray, mask_work: np.ndarray, win: list[int], plate_size: tuple[int, int], inv,
                  feather_px: float = 3.0) -> np.ndarray:
    """A cropped LTX frame back into its window on a plate-size premultiplied ACEScg RGBA (zero elsewhere)."""
    import cv2

    W, H = plate_size
    x0, y0, w, h = win
    out = np.zeros((H, W, 4), np.float32)
    lin = from_view(resize(ltx_disp.astype(np.float32), (w, h)), inv)
    a = cv2.resize(mask_work.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if feather_px > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather_px)
    edge = max(2, int(3 * feather_px))                 # fade at window sides that are inside the plate
    ramp = np.ones((h, w), np.float32)
    for side, inside in (("l", x0 > 0), ("r", x0 + w < W), ("t", y0 > 0), ("b", y0 + h < H)):
        if not inside:
            continue
        r = np.linspace(0, 1, edge, dtype=np.float32)
        if side == "l":
            ramp[:, :edge] *= r[None]
        elif side == "r":
            ramp[:, -edge:] *= r[::-1][None]
        elif side == "t":
            ramp[:edge] *= r[:, None]
        else:
            ramp[-edge:] *= r[::-1][:, None]
    a = np.clip(a * ramp, 0, 1)
    out[y0:y0 + h, x0:x0 + w] = np.dstack([lin * a[..., None], a])
    return out


def return_frame(ltx_disp: np.ndarray, mask_work: np.ndarray, plate_size: tuple[int, int], inv,
                 feather_px: float = 3.0) -> np.ndarray:
    """One LTX frame (work res, display) + its mask -> premultiplied ACEScg RGBA at plate res."""
    import cv2

    W, H = plate_size
    lin = from_view(resize(ltx_disp.astype(np.float32), (W, H)), inv)
    a = cv2.resize(mask_work.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if feather_px > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather_px)
    a = np.clip(a, 0, 1)
    return np.dstack([lin * a[..., None], a]).astype(np.float32)


# ---------------------------------------------------------------- sensor pass (clean-space loop)
# With Sensor Pass maps, Prep takes the plate's camera OFF before anything else (de-RS, undistort, devignette, in
# ACEScg), so LTX sees clean, global-shutter, rectilinear frames that match our pinhole card layers. Return puts the
# camera back ON the returned layer (vignette, CA + distortion, RS) so it sits on the real plate.

def sensor_maps(rs_folder: str = "", lens_folder: str = ""):
    """Sensor Pass maps (atlas_vfx.sensor.jobs.Maps), or None when both folders are blank."""
    rs = pl.clean_path(rs_folder) or None
    lens = pl.clean_path(lens_folder) or None
    if not rs and not lens:
        return None
    from atlas_vfx.sensor import jobs
    return jobs.Maps(rs, lens)


def sensor_record(maps) -> dict | None:
    if maps is None:
        return None
    return {"rs_folder": str(maps.rs_folder) if maps.rs_folder else None,
            "lens_folder": str(maps.lens_folder) if maps.lens_folder else None}


def remove_camera(lin: np.ndarray, maps, frame: int) -> np.ndarray:
    """Plate frame (ACEScg, whole plate) -> clean space."""
    from atlas_vfx.sensor import jobs
    out, _ = jobs.apply_batch(lin[None].astype(np.float32), maps, direction="remove camera", first_frame=frame,
                              colorspace="ACEScg")
    return out[0]


def apply_camera(rgba: np.ndarray, maps, frame: int) -> np.ndarray:
    """Clean-space premultiplied ACEScg RGBA layer (whole plate) -> back on the plate's camera."""
    from atlas_vfx.sensor import jobs
    out, _ = jobs.apply_batch(rgba[None].astype(np.float32), maps, direction="apply camera", first_frame=frame,
                              colorspace="ACEScg")
    return out[0]
