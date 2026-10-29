"""The only place atlas-vfx imports Atlas Camera. Read-only: never write into its trees from here.

Lookup order:
  1. atlas_camera already imported (ComfyUI loaded the atlas-camera pack, which puts itself on sys.path)
  2. ATLAS_CAMERA_ROOT env var
  3. the pack installed next to this one, <custom_nodes>/atlas-camera (this repo is a junction there)
  4. ~/ComfyUI_V141/ComfyUI/custom_nodes/atlas-camera (the test instance, when running outside ComfyUI)
  5. the dev tree, ~/Desktop/AtlasCamera_Claude (Mike's work in flight; read only)

Nothing is imported until a module is asked for. If Atlas Camera lacks something, write it in camera_side/ and note
it in docs/dev/UPSTREAM_TODO.md.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

# abspath, not resolve(): under ComfyUI this file is reached through the custom_nodes junction, and that is the
# folder the sibling pack lives in
SIBLING_ROOT = Path(os.path.abspath(__file__)).parents[1].parent / "atlas-camera"
INSTALLED_ROOT = Path.home() / "ComfyUI_V141" / "ComfyUI" / "custom_nodes" / "atlas-camera"
DEV_ROOT = Path.home() / "Desktop" / "AtlasCamera_Claude"


def _candidates() -> list[tuple[str, Path]]:
    out = []
    env = os.environ.get("ATLAS_CAMERA_ROOT")
    if env:
        out.append(("env", Path(env)))
    out.append(("sibling", SIBLING_ROOT))
    out.append(("installed", INSTALLED_ROOT))
    out.append(("dev", DEV_ROOT))
    return out


def _is_root(p: Path) -> bool:
    return (p / "atlas_camera" / "__init__.py").is_file()


def _version(root: Path) -> str | None:
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def find() -> tuple[str, Path] | tuple[None, None]:
    """(source, root) of the Atlas Camera checkout to use, without importing it."""
    mod = sys.modules.get("atlas_camera")
    if mod is not None and getattr(mod, "__file__", None):
        return "imported", Path(mod.__file__).resolve().parents[1]
    for source, p in _candidates():
        if _is_root(p):
            return source, p
    return None, None


def report() -> dict:
    """What was found, for logs and sidecars. Does not import Atlas Camera."""
    source, root = find()
    return {
        "source": source,
        "root": str(root) if root else None,
        "version": _version(root) if root else None,
        "candidates": {s: {"path": str(p), "ok": _is_root(p)} for s, p in _candidates()},
    }


def available() -> bool:
    return find()[1] is not None


def ensure_on_path() -> Path:
    """Make `import atlas_camera` work. Appends to sys.path so nothing of ours is shadowed. Returns the root."""
    source, root = find()
    if root is None:
        raise RuntimeError(
            "Atlas Camera not found; looked at "
            + ", ".join(str(p) for _, p in _candidates())
            + " (set ATLAS_CAMERA_ROOT)"
        )
    if source != "imported" and str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def import_module(name: str):
    """Import atlas_camera.<name> on demand, e.g. import_module("plate.oiio_io")."""
    ensure_on_path()
    return importlib.import_module(f"atlas_camera.{name}")


def oiio_io():
    return import_module("plate.oiio_io")


def nuke_exporter():
    return import_module("exporters.nuke_exporter")


def real_plate_nuke():
    return import_module("exporters.real_plate_nuke")


def dcc_transform():
    return import_module("exporters.dcc_transform")
