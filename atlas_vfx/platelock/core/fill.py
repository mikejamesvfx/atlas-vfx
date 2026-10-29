"""Push-pull hole fill for float images (smooth, no ringing, any channel count)."""
from __future__ import annotations

import cv2
import numpy as np


def pushpull_fill(img: np.ndarray, holes: np.ndarray, levels: int = 8) -> np.ndarray:
    """Fill `holes` (True = missing) from surrounding pixels, coarse to fine."""
    im = np.asarray(img, np.float32)
    squeeze = im.ndim == 2
    if squeeze:
        im = im[..., None]
    w = (~np.asarray(holes, bool)).astype(np.float32)
    if w.all():
        return img
    pyr = [(im * w[..., None], w)]
    for _ in range(levels):
        pim, pw = pyr[-1]
        if min(pw.shape) < 2:
            break
        h2, w2 = (pw.shape[0] + 1) // 2, (pw.shape[1] + 1) // 2
        pyr.append((cv2.resize(pim, (w2, h2), interpolation=cv2.INTER_AREA).reshape(h2, w2, -1),
                    cv2.resize(pw, (w2, h2), interpolation=cv2.INTER_AREA)))
    cur_im, cur_w = pyr[-1]
    cur = cur_im / np.maximum(cur_w, 1e-6)[..., None]
    for pim, pw in reversed(pyr[:-1]):
        up = cv2.resize(cur, (pw.shape[1], pw.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(pw.shape + (-1,))
        a = np.clip(pw, 0, 1)[..., None]
        cur = pim / np.maximum(pw, 1e-6)[..., None] * a + up * (1 - a)
    out = np.where(np.asarray(holes, bool)[..., None], cur, im)
    return out[..., 0] if squeeze else out
