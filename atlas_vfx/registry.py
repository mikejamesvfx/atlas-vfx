"""Gathers node mappings from the subpackages, disabling only the nodes whose dependencies are missing.

Each subpackage lists its node modules in NODE_MODULES (module names relative to the subpackage). Convention: one
node per module, so an ImportError in a module disables just that node. A node class can also declare
REQUIRES = ("OpenImageIO", ...) (import names); if any is missing the class is skipped and the reason logged.

No ComfyUI imports here, so the tests can run this outside ComfyUI.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging

log = logging.getLogger("atlas_vfx")

SUBPACKAGES = ("atlas_vfx.camera_side", "atlas_vfx.layers", "atlas_vfx.platelock", "atlas_vfx.deliver",
               "atlas_vfx.sensor")


def _missing(requires) -> list[str]:
    out = []
    for name in requires or ():
        try:
            if importlib.util.find_spec(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


def load_nodes(subpackages=SUBPACKAGES):
    """Returns (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, disabled) where disabled maps a node module or
    class name to the reason it was left out."""
    classes: dict = {}
    names: dict = {}
    disabled: dict = {}
    for pkg_name in subpackages:
        pkg = importlib.import_module(pkg_name)
        for mod_name in getattr(pkg, "NODE_MODULES", ()):
            full = f"{pkg_name}.{mod_name}"
            try:
                mod = importlib.import_module(full)
            except ImportError as e:
                reason = f"missing dependency {e.name or e}"
                disabled[full] = reason
                log.warning("atlas-vfx: disabled %s (%s)", full, reason)
                continue
            mod_names = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
            for key, cls in getattr(mod, "NODE_CLASS_MAPPINGS", {}).items():
                miss = _missing(getattr(cls, "REQUIRES", ()))
                if miss:
                    reason = "missing dependency " + ", ".join(miss)
                    disabled[key] = reason
                    log.warning("atlas-vfx: disabled %s (%s)", key, reason)
                    continue
                if key in classes:
                    raise RuntimeError(f"atlas-vfx: node {key} registered twice ({full})")
                classes[key] = cls
                if key in mod_names:
                    names[key] = mod_names[key]
    return classes, names, disabled
