"""ComfyUI entry point for atlas-vfx.

ComfyUI loads this file as the package for custom_nodes/atlas-vfx (a junction to the repo). It puts the repo on
sys.path so `import atlas_vfx` resolves to this checkout, then gathers the node mappings from camera_side, layers
and deliver. A node whose optional dependency is missing is left out and the reason is logged.
"""
from __future__ import annotations

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from atlas_vfx import __version__, _atlas_camera  # noqa: E402
from atlas_vfx.registry import load_nodes  # noqa: E402

log = logging.getLogger("atlas_vfx")

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, DISABLED = load_nodes()

# Frontend: web/atlas_vfx_sun_globe.js (Physical Sky's Sun Globe) and its three.js bundle in web/lib. Display only;
# every node runs the same without it.
WEB_DIRECTORY = "./web"

_ac = _atlas_camera.report()
log.info(
    "atlas-vfx %s: %d nodes loaded, %d disabled; Atlas Camera: %s",
    __version__,
    len(NODE_CLASS_MAPPINGS),
    len(DISABLED),
    f"{_ac['source']} {_ac['version'] or '?'} at {_ac['root']}" if _ac["root"] else "not found",
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
