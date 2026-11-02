"""How a generated (display-referred) image comes into ACEScg, and goes back out.

A generator (Z-Image, LTX, any video model) hands back display pictures. LTX Prep shows the plate to LTX through the
ACES 2.0 SDR view, so an element that is to survive the LTX round trip unchanged must come in through the exact
inverse of that view: what it looked like on screen is what it looks like after LTX. The old way, the texture
transform 'sRGB Encoded Rec.709 (sRGB)' -> ACEScg, treats the picture as a flat sRGB texture: same look on a texture,
but LTX Prep then tone-maps it again on the way out, so elements came back dimmer and flatter than they were made
(flow review 2026-11-01). Both are named OCIO transforms of the shipped config.
"""
from __future__ import annotations

import numpy as np

from atlas_vfx.camera_side import plate as pl

VIEW_INVERSE = "inverse display view (ACES 2.0 SDR, as LTX)"
SRGB_TEXTURE = "sRGB texture"
SOURCES = (VIEW_INVERSE, SRGB_TEXTURE)
GEN_SPACE = "sRGB Encoded Rec.709 (sRGB)"


def to_acescg_processor(ocio_config: str, source: str = VIEW_INVERSE):
    """CPU processor, display picture (0-1) -> ACEScg, for `source` (one of SOURCES)."""
    if source == VIEW_INVERSE:
        import PyOpenColorIO as ocio

        cfg = pl.ocio_config(pl.ocio_path(ocio_config))
        t = ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW,
                                      direction=ocio.TRANSFORM_DIR_INVERSE)
        return pl.fast_cpu(cfg.getProcessor(t))
    if source == SRGB_TEXTURE:
        return pl.fast_cpu(pl.ocio_config(pl.ocio_path(ocio_config)).getProcessor(GEN_SPACE, "ACEScg"))
    raise ValueError(f"source colour {source!r}: one of {SOURCES}")


def from_acescg_processor(ocio_config: str, source: str = VIEW_INVERSE):
    """The way back: ACEScg -> display picture, the forward of `source`."""
    import PyOpenColorIO as ocio

    cfg = pl.ocio_config(pl.ocio_path(ocio_config))
    if source == VIEW_INVERSE:
        return pl.fast_cpu(cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY,
                                                                      view=pl.DEFAULT_VIEW)))
    if source == SRGB_TEXTURE:
        return pl.fast_cpu(cfg.getProcessor("ACEScg", GEN_SPACE))
    raise ValueError(f"source colour {source!r}: one of {SOURCES}")


def to_acescg(rgb: np.ndarray, ocio_config: str, source: str = VIEW_INVERSE) -> np.ndarray:
    out = np.ascontiguousarray(np.clip(rgb[..., :3], 0.0, 1.0), dtype=np.float32).copy()
    to_acescg_processor(ocio_config, source).applyRGB(out)
    return out


def describe(source: str) -> str:
    return ("OCIO inverse of sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709) -> ACEScg" if source == VIEW_INVERSE
            else f"OCIO {GEN_SPACE} -> ACEScg")
