"""T0: the package imports without ComfyUI, the registry gathers nodes and disables only what is missing, the
Atlas Camera adapter is lazy, and every package imports only what the import graph in AGENTS.md allows
(camera_side never imports layers/ or deliver/, and so on for the other five).

Run: <V141 python> tests\\test_import.py   (plain script, exits non-zero on failure; no pytest)
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = ("contracts", "camera_side", "platelock", "sensor", "layers", "deliver")
# The import graph AGENTS.md allows ("Shape of the repo"): package -> the atlas_vfx packages it may import.
# `_atlas_camera` is the Atlas Camera adapter; `registry` is the root's and nobody else's. `from atlas_vfx import
# __version__` is not an edge (the root package itself; absent after the split, callers guard it).
ALLOWED: dict[str, set[str]] = {
    "contracts": set(),
    "camera_side": {"contracts", "_atlas_camera"},
    "platelock": {"contracts", "camera_side"},
    "sensor": {"contracts", "camera_side", "platelock"},
    "layers": {"contracts", "camera_side", "sensor", "_atlas_scene"},
    "deliver": {"contracts", "camera_side", "platelock", "sensor", "layers", "_atlas_camera"},
}
# `_atlas_scene` is the Atlas Scene adapter (Prague sky, 2026-10-28): only layers (Physical Sky) uses it.
ROOT_MODULES = ("_atlas_camera", "_atlas_scene", "registry")


def atlas_imports(source: str, package: str) -> list[str]:
    """Every atlas_vfx module one source file imports, fully qualified: absolute imports, relative ones resolved
    from `package` (the file's own package, e.g. 'atlas_vfx.platelock.core'), and import_module-style strings."""
    pkg = package.split(".")
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(pkg[: len(pkg) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            out.append(base)
            out += [f"{base}.{a.name}" for a in node.names]       # `from .. import layers`, `from atlas_vfx import deliver`
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # import_module("atlas_vfx.layers...") style strings; prose (docstrings) is not a module path
            if re.fullmatch(r"[\w.]+", node.value) and node.value.startswith("atlas_vfx."):
                out.append(node.value)
    return [t for t in out if t == "atlas_vfx" or t.startswith("atlas_vfx.")]


def edges(targets: list[str], own: str) -> set[str]:
    """The atlas_vfx packages (or root modules) the targets touch, other than the file's own package."""
    got = set()
    for t in targets:
        parts = t.split(".")
        top = parts[1] if len(parts) > 1 else ""
        if top != own and (top in PACKAGES or top in ROOT_MODULES):
            got.add(top)
    return got


def check_scanner() -> None:
    cases = [
        ("from atlas_vfx.layers import card_project", "atlas_vfx.camera_side", {"layers"}),
        ("import atlas_vfx.deliver.nuke_builder", "atlas_vfx.camera_side", {"deliver"}),
        ("from ..layers import x", "atlas_vfx.camera_side", {"layers"}),
        ("from .. import deliver", "atlas_vfx.camera_side", {"deliver"}),
        ("from ...camera_side import rs_time", "atlas_vfx.platelock.core", {"camera_side"}),
        ("from ...camera_side.plate import safe_name", "atlas_vfx.platelock.core", {"camera_side"}),
        ("from atlas_vfx import layers", "atlas_vfx.camera_side", {"layers"}),
        ("import importlib; importlib.import_module('atlas_vfx.layers.card_project')", "atlas_vfx.camera_side", {"layers"}),
        ("from atlas_vfx.registry import load_nodes", "atlas_vfx.layers", {"registry"}),
        ("from atlas_vfx.contracts import shot_camera", "atlas_vfx.camera_side", {"contracts"}),
        ("from .._atlas_camera import oiio_io", "atlas_vfx.camera_side", {"_atlas_camera"}),
        ("from .. import _atlas_scene", "atlas_vfx.layers", {"_atlas_scene"}),
        ("from atlas_vfx._atlas_scene import prague", "atlas_vfx.camera_side", {"_atlas_scene"}),
        ("def f():\n    from atlas_vfx.sensor import jobs\n", "atlas_vfx.layers", {"sensor"}),
        ("from . import plate", "atlas_vfx.camera_side", set()),
        ("from .core import camera", "atlas_vfx.platelock", set()),
        ("from ..core.camera import CameraTrack", "atlas_vfx.platelock.core", set()),
        ("from atlas_vfx import __version__", "atlas_vfx.platelock", set()),
        ("import numpy as np", "atlas_vfx.camera_side", set()),
        ("x = 'player layers in a comment-like string'", "atlas_vfx.camera_side", set()),
        ('"""Rule: no imports from atlas_vfx.layers or atlas_vfx.deliver."""', "atlas_vfx.camera_side", set()),
    ]
    for src, package, want in cases:
        got = edges(atlas_imports(src, package), package.split(".")[1])
        assert got == want, f"scanner: {src!r} in {package}: got {sorted(got)}, want {sorted(want)}"


def check_graph() -> tuple[int, dict[str, set[str]]]:
    """Every .py under the six packages imports only what ALLOWED says. Returns (files scanned, edges seen)."""
    seen: dict[str, set[str]] = {p: set() for p in PACKAGES}
    bad, n = [], 0
    for pkg in PACKAGES:
        for py in sorted((ROOT / "atlas_vfx" / pkg).rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            parts = list(py.relative_to(ROOT / "atlas_vfx").with_suffix("").parts)
            package = ".".join(["atlas_vfx", *(parts if parts[-1] == "__init__" else parts)[:-1]])
            n += 1
            for e in sorted(edges(atlas_imports(py.read_text(encoding="utf-8"), package), pkg)):
                seen[pkg].add(e)
                if e not in ALLOWED[pkg]:
                    may = ", ".join(sorted(ALLOWED[pkg])) or "nothing"
                    bad.append(f"{py.relative_to(ROOT)}: imports atlas_vfx.{e} ({pkg} may import {may})")
    assert not bad, "import graph (AGENTS.md, Shape of the repo) broken:\n  " + "\n  ".join(bad)
    assert sorted(seen) == sorted(ALLOWED)
    return n, seen


def check_registry_disables_only_missing() -> None:
    """A fake subpackage with one good node, one module that imports a missing package and one class that
    REQUIRES a missing package: only the good node survives, both others are reported."""
    from atlas_vfx.registry import load_nodes

    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "fakepack_t0"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("NODE_MODULES = ('good', 'broken', 'needs')\n")
        (pkg / "good.py").write_text(textwrap.dedent("""
            class Good:
                pass
            NODE_CLASS_MAPPINGS = {"FakeGood": Good}
            NODE_DISPLAY_NAME_MAPPINGS = {"FakeGood": "Fake Good"}
        """))
        (pkg / "broken.py").write_text("import not_a_real_package_t0\nNODE_CLASS_MAPPINGS = {'FakeBroken': object}\n")
        (pkg / "needs.py").write_text(textwrap.dedent("""
            class Needs:
                REQUIRES = ("numpy", "also_not_real_t0")
            NODE_CLASS_MAPPINGS = {"FakeNeeds": Needs}
        """))
        sys.path.insert(0, td)
        try:
            classes, names, disabled = load_nodes(("fakepack_t0",))
        finally:
            sys.path.remove(td)
            for k in [k for k in sys.modules if k.startswith("fakepack_t0")]:
                del sys.modules[k]
    assert list(classes) == ["FakeGood"], classes
    assert names == {"FakeGood": "Fake Good"}, names
    assert "not_a_real_package_t0" in disabled.get("fakepack_t0.broken", ""), disabled
    assert "also_not_real_t0" in disabled.get("FakeNeeds", ""), disabled
    assert "numpy" not in disabled["FakeNeeds"], disabled


def load_comfy_entry():
    """Load the repo's root __init__.py the way ComfyUI does (as package 'atlas-vfx')."""
    spec = importlib.util.spec_from_file_location(
        "atlas-vfx", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atlas-vfx"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import atlas_vfx
    from atlas_vfx import _atlas_camera, _atlas_scene, contracts

    schema = json.loads(contracts.SHOT_CAMERA_SCHEMA.read_text(encoding="utf-8"))
    assert schema["properties"]["contract"]["const"] == "shot-camera@0.1"

    check_scanner()
    n_files, seen = check_graph()
    check_registry_disables_only_missing()

    entry = load_comfy_entry()
    assert isinstance(entry.NODE_CLASS_MAPPINGS, dict)
    assert set(entry.NODE_DISPLAY_NAME_MAPPINGS) <= set(entry.NODE_CLASS_MAPPINGS)
    assert all(k.startswith("AtlasVFX") for k in entry.NODE_CLASS_MAPPINGS), list(entry.NODE_CLASS_MAPPINGS)

    leaked = [m for m in ("comfy", "folder_paths", "nodes", "server", "atlas_camera", "atlas_scene",
                          "atlas_scene_prague") if m in sys.modules]
    assert not leaked, f"imported at load time but should not be: {leaked}"

    rep = _atlas_camera.report()
    print(f"ok atlas_vfx {atlas_vfx.__version__}")
    print(f"ok import graph: {n_files} files in {len(PACKAGES)} packages, every edge allowed (scanner self-check passed)")
    for pkg in PACKAGES:
        print(f"    {pkg} -> {', '.join(sorted(seen[pkg])) or '(nothing)'}")
    print("ok registry: missing module and missing REQUIRES each disable only their own node")
    print(f"ok entry point: {len(entry.NODE_CLASS_MAPPINGS)} nodes, {len(entry.DISABLED)} disabled, no ComfyUI imports")
    print(f"atlas camera: {json.dumps(rep, indent=2)}")
    srep = _atlas_scene.report()
    assert "atlas_scene" not in sys.modules, "_atlas_scene.report() must not import Atlas Scene"
    print(f"atlas scene (not imported): {json.dumps(srep, indent=2)}")

    if rep["root"]:
        mod = _atlas_camera.dcc_transform()
        print(f"ok atlas camera lazy import: {mod.__name__} from {mod.__file__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
