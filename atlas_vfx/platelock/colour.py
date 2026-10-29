"""OCIO for Plate Lock: frames go to a log space for matching only; pixels are warped in their own space.

The match space is ACEScct (the config's compositing_log role). Every conversion is a named OCIO
transform from the repo config; nothing is inferred from a file. If PyOpenColorIO is missing and the
source is already ACEScg, the built-in ACEScct curve is used and that is logged.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np

from .core.crossview import acescct_from_acescg

log = logging.getLogger("atlas_vfx")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OCIO = REPO / "config" / "atlas_aces2.0_studio_flog2.ocio"
MATCH_SPACE = "ACEScct"


@functools.lru_cache(maxsize=16)
def _processor(config_path: str, src: str, dst: str):
    import PyOpenColorIO as OCIO
    cfg = OCIO.Config.CreateFromFile(config_path)
    names = {cs.getName() for cs in cfg.getColorSpaces()}
    for n in (src, dst):
        if n not in names and cfg.getColorSpace(n) is None:
            raise ValueError(f"OCIO colorspace {n!r} is not in {config_path}")
    return cfg.getProcessor(src, dst).getDefaultCPUProcessor()


def convert(rgb: np.ndarray, src: str, dst: str, config_path: str | None = None) -> np.ndarray:
    if src == dst:
        return np.asarray(rgb, np.float32)
    cfgp = str(config_path or DEFAULT_OCIO)
    try:
        proc = _processor(cfgp, src, dst)
    except ImportError:
        if src == "ACEScg" and dst == MATCH_SPACE:
            log.warning("platelock: PyOpenColorIO missing, using the built-in ACEScct curve")
            return acescct_from_acescg(rgb)
        raise
    out = np.ascontiguousarray(rgb[..., :3], np.float32).copy()
    proc.applyRGB(out)
    return out


def to_match(rgb: np.ndarray, src: str, config_path: str | None = None) -> np.ndarray:
    """Any input space -> the match space (ACEScct)."""
    return convert(rgb, src, MATCH_SPACE, config_path)
