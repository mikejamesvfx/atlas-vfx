"""The only place atlas-vfx imports Atlas Scene. Read-only: never write into its trees from here.

Atlas Scene OWNS the physical sky (its `environment` package plus the `atlas_scene_prague` extension, which wraps
the official Prague Sky Model). atlas-vfx never copies that code; Physical Sky asks for it here, lazily, and only
when it renders a sky. Nothing is imported until a module is asked for, so every node loads without Atlas Scene.

Atlas Scene lookup order:
  1. atlas_scene already imported
  2. ATLAS_SCENE_ROOT env var (the checkout: the folder holding atlas_scene/)
  3. installed into this Python (pip install -e "<atlas-scene>[environment]")
  4. the dev tree, ~/atlas-scene-DEV/atlas-scene (Mike's checkout; read only)

The Prague extension (`atlas_scene_prague`): installed into this Python, else <root>/integrations/prague/src. Its
compiled part (`_native<abi suffix>`) is looked for in: ATLAS_SCENE_PRAGUE_NATIVE (a folder), the package folder
itself, then the base interpreter's own site-packages (a venv made from the Python that pip-installed the
extension, as V141's venv is). Only a file with this interpreter's extension suffix is ever loaded.

The dataset is not found here: the node's `dataset` widget, else ATLAS_PRAGUE_DATASET (Atlas Scene's own default).
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

DEV_ROOT = Path.home() / "atlas-scene-DEV" / "atlas-scene"
PRAGUE_SRC = Path("integrations") / "prague" / "src"


class AtlasSceneMissing(RuntimeError):
    """Atlas Scene (or its Prague extension) is not reachable from this Python."""


def _candidates() -> list[tuple[str, Path]]:
    out = []
    env = os.environ.get("ATLAS_SCENE_ROOT")
    if env:
        out.append(("env", Path(os.path.expandvars(os.path.expanduser(env)))))
    out.append(("dev", DEV_ROOT))
    return out


def _is_root(p: Path) -> bool:
    return (p / "atlas_scene" / "__init__.py").is_file()


def _installed_root() -> Path | None:
    try:
        spec = importlib.util.find_spec("atlas_scene")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parents[1]


def find() -> tuple[str, Path] | tuple[None, None]:
    """(source, root) of the Atlas Scene checkout to use, without importing it."""
    mod = sys.modules.get("atlas_scene")
    if mod is not None and getattr(mod, "__file__", None):
        return "imported", Path(mod.__file__).resolve().parents[1]
    cands = _candidates()
    if cands[0][0] == "env" and _is_root(cands[0][1]):
        return "env", cands[0][1]
    inst = _installed_root()
    if inst is not None:
        return "installed", inst
    for source, p in cands:
        if _is_root(p):
            return source, p
    return None, None


def available() -> bool:
    return find()[1] is not None


def _version(root: Path) -> str | None:
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _prague_pkg_dir(root: Path | None) -> tuple[str | None, Path | None]:
    """(source, folder) of the atlas_scene_prague package, without importing it."""
    mod = sys.modules.get("atlas_scene_prague")
    if mod is not None and getattr(mod, "__file__", None):
        return "imported", Path(mod.__file__).resolve().parent
    try:
        spec = importlib.util.find_spec("atlas_scene_prague")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return "installed", Path(spec.origin).resolve().parent
    if root is not None and (root / PRAGUE_SRC / "atlas_scene_prague" / "__init__.py").is_file():
        return "root", root / PRAGUE_SRC / "atlas_scene_prague"
    return None, None


def _native_in(folder: Path) -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        p = folder / f"_native{suffix}"
        if p.is_file():
            return p
    return None


def _native_candidates(pkg_dir: Path | None) -> list[tuple[str, Path]]:
    out = []
    env = os.environ.get("ATLAS_SCENE_PRAGUE_NATIVE")
    if env:
        out.append(("env", Path(os.path.expandvars(os.path.expanduser(env)))))
    if pkg_dir is not None:
        out.append(("package", pkg_dir))
    base = sysconfig.get_paths(vars={"base": sys.base_prefix, "platbase": sys.base_prefix})
    out.append(("base interpreter", Path(base["platlib"]) / "atlas_scene_prague"))
    return out


def find_native(pkg_dir: Path | None) -> tuple[str, Path] | tuple[None, None]:
    loaded = sys.modules.get("atlas_scene_prague._native")
    if loaded is not None and getattr(loaded, "__file__", None):
        return "imported", Path(loaded.__file__)
    for source, folder in _native_candidates(pkg_dir):
        p = _native_in(folder)
        if p is not None:
            return source, p
    return None, None


def report() -> dict:
    """What would be (or was) used, for logs, info and sidecars. Does not import Atlas Scene."""
    source, root = find()
    psrc, pdir = _prague_pkg_dir(root)
    nsrc, npath = find_native(pdir)
    rep = {
        "source": source,
        "root": str(root) if root else None,
        "version": _version(root) if root else None,
        "commit": _git(root, "rev-parse", "HEAD") if root else None,
        "dirty": None,
        "prague": {"source": psrc, "package": str(pdir) if pdir else None,
                   "native_source": nsrc, "native": str(npath) if npath else None,
                   "upstream_commit": None},
        "candidates": {s: {"path": str(p), "ok": _is_root(p)} for s, p in _candidates()},
    }
    if rep["commit"]:
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        rep["dirty"] = None if status is None else bool(status)
    mod = sys.modules.get("atlas_scene_prague")
    if mod is not None:
        rep["prague"]["upstream_commit"] = getattr(mod, "UPSTREAM_COMMIT", None)
    return rep


def install_message(what: str) -> str:
    looked = ", ".join(f"{s}: {p}" for s, p in _candidates())
    return (
        f"Atlas Scene {what} not found; Physical Sky renders the sky through Atlas Scene's Prague integration "
        f"(Atlas Scene owns it, atlas-vfx does not copy it). Looked at {looked} and this Python's site-packages. "
        "Fix one of: set ATLAS_SCENE_ROOT to your atlas-scene checkout (the folder holding atlas_scene/); or "
        'pip install -e "<atlas-scene>[environment]" into this Python. The Prague extension: '
        "pip install -e <atlas-scene>/integrations/prague --no-deps (needs a C++17 compiler and CMake), or set "
        "ATLAS_SCENE_PRAGUE_NATIVE to the folder holding its compiled _native module built for this Python "
        f"({sys.version.split()[0]}). The dataset: the node's dataset widget or ATLAS_PRAGUE_DATASET."
    )


def ensure_on_path() -> Path:
    """Make `import atlas_scene` work. Appends to sys.path so nothing of ours or ComfyUI's is shadowed."""
    source, root = find()
    if root is None:
        raise AtlasSceneMissing(install_message("(the atlas_scene package)"))
    if source in ("env", "dev") and str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def import_module(name: str):
    """Import atlas_scene.<name> on demand, e.g. import_module("environment.render")."""
    ensure_on_path()
    return importlib.import_module(f"atlas_scene.{name}")


def prague():
    """The atlas_scene_prague module, with its compiled extension loaded. Raises AtlasSceneMissing with the fix."""
    if "atlas_scene_prague" in sys.modules:
        return sys.modules["atlas_scene_prague"]
    root = ensure_on_path()
    psrc, pdir = _prague_pkg_dir(root)
    if pdir is None:
        raise AtlasSceneMissing(install_message("Prague extension (atlas_scene_prague)"))
    nsrc, npath = find_native(pdir)
    if npath is None:
        raise AtlasSceneMissing(install_message("Prague compiled module (atlas_scene_prague._native)"))
    if psrc == "root" and str(pdir.parent) not in sys.path:
        sys.path.append(str(pdir.parent))
    if nsrc not in ("package", "imported"):
        # the compiled module lives outside the package folder (editable install made by another interpreter):
        # load it under its package name first, so the package's `from . import _native` finds it
        spec = importlib.util.spec_from_file_location("atlas_scene_prague._native", npath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001 - wrong ABI, missing DLL: say what and where
            del sys.modules[spec.name]
            raise AtlasSceneMissing(f"could not load {npath} ({e}); " + install_message("Prague compiled module"))
    return importlib.import_module("atlas_scene_prague")
