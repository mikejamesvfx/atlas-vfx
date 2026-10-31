"""Atlas VFX Grade: Nuke's Grade node, on linear images.

    A = multiply * (gain - lift) / (whitepoint - blackpoint)
    B = offset + lift - A * blackpoint
    out = A * in + B, then gamma on values above 0 (out ** (1 / gamma)), then the clamps

exposure (stops) scales the input first. reverse runs the inverse, so a grade can be undone. mask / mix blend with
the input as Nuke's mask and mix do. RGB only; alpha passes through."""
from __future__ import annotations

import numpy as np


def grade(rgb: np.ndarray, blackpoint=0.0, whitepoint=1.0, lift=0.0, gain=1.0, multiply=1.0, offset=0.0, gamma=1.0,
          exposure=0.0, tint=(1.0, 1.0, 1.0), reverse=False, clamp_black=True, clamp_white=False) -> np.ndarray:
    if whitepoint == blackpoint:
        raise ValueError("whitepoint equals blackpoint")
    if gamma <= 0:
        raise ValueError("gamma must be above 0")
    rgb = np.asarray(rgb, np.float32)
    A = multiply * np.asarray(tint, np.float32) * (gain - lift) / (whitepoint - blackpoint)
    B = offset + lift - A * blackpoint
    e = np.float32(2.0 ** exposure)
    if not reverse:
        y = (rgb * e) * A + B
        if gamma != 1.0:
            y = np.where(y > 0, np.power(np.maximum(y, 0), 1.0 / gamma), y)
    else:
        y = rgb
        if gamma != 1.0:
            y = np.where(y > 0, np.power(np.maximum(y, 0), gamma), y)
        if np.any(A == 0):
            raise ValueError("reverse: this grade maps everything to one value (gain = lift or multiply 0)")
        y = (y - B) / A / e
    if clamp_black:
        y = np.maximum(y, 0.0)
    if clamp_white:
        y = np.minimum(y, 1.0)
    return y.astype(np.float32)


class AtlasVFXGrade:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = ("Nuke's Grade on linear images: blackpoint, whitepoint, lift, gain, multiply, offset, gamma, with "
                   "the same maths and the same black clamp default. exposure is in stops; the r/g/b tint multiplies "
                   "gain per channel. reverse undoes the grade. mask and mix as in Nuke.")

    @classmethod
    def INPUT_TYPES(cls):
        f = (lambda d, lo=-100.0, hi=100.0, step=0.01, tip="": ("FLOAT", {"default": d, "min": lo, "max": hi,
                                                                         "step": step, "tooltip": tip}))
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "linear (scene-referred) image"}),
                "blackpoint": f(0.0, tip="this input value becomes black"),
                "whitepoint": f(1.0, tip="this input value becomes white"),
                "lift": f(0.0, tip="what black becomes"),
                "gain": f(1.0, tip="what white becomes"),
                "multiply": f(1.0),
                "offset": f(0.0),
                "gamma": f(1.0, 0.01, 10.0),
                "exposure": f(0.0, -20.0, 20.0, 0.1, "stops, applied to the input first"),
                "tint_r": f(1.0, 0.0, 10.0),
                "tint_g": f(1.0, 0.0, 10.0),
                "tint_b": f(1.0, 0.0, 10.0),
                "reverse": ("BOOLEAN", {"default": False}),
                "clamp_black": ("BOOLEAN", {"default": True}),
                "clamp_white": ("BOOLEAN", {"default": False}),
                "mix": f(1.0, 0.0, 1.0),
            },
            "optional": {"mask": ("MASK",)},
        }

    def run(self, image, blackpoint, whitepoint, lift, gain, multiply, offset, gamma, exposure, tint_r, tint_g,
            tint_b, reverse, clamp_black, clamp_white, mix, mask=None):
        import torch

        img = image.float().cpu().numpy()
        out = img.copy()
        out[..., :3] = grade(img[..., :3], blackpoint, whitepoint, lift, gain, multiply, offset, gamma, exposure,
                             (tint_r, tint_g, tint_b), reverse, clamp_black, clamp_white)
        w = np.float32(mix)
        if mask is not None:
            m = mask.float().cpu().numpy()
            if m.ndim == 2:
                m = m[None]
            if m.shape[1:] != img.shape[1:3]:
                raise ValueError(f"mask is {m.shape[2]}x{m.shape[1]}, image {img.shape[2]}x{img.shape[1]}")
            w = w * m[np.minimum(np.arange(len(img)), len(m) - 1)][..., None]
        out[..., :3] = img[..., :3] + w * (out[..., :3] - img[..., :3])
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {"AtlasVFXGrade": AtlasVFXGrade}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXGrade": "Atlas VFX Grade"}
