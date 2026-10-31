"""Shots on disk. A shot is a folder under <work root>/comp/<shot>/ holding shot.json and, per layer, its renders:

    comp/<shot>/shot.json
    comp/<shot>/layers/<layer>/v001/<files>.####.exr   (one folder per render; the layer points at one)

shot.json is the only state; versions are read from the folders. Every name and path comes from HTTP and is
untrusted: names go through plate.safe_name, folders through plate.resolve_out_dir, and a sequence layer's source
must sit under the work root."""
from __future__ import annotations

import json
import re
from pathlib import Path

from atlas_vfx.camera_side import plate as pl

COMP = "comp"
KINDS = ("sequence", "recipe")
BLENDS = ("over", "plus", "multiply", "screen")
CHANGEABLE = ("visible", "opacity", "blend", "version", "params", "order")
_VERSION = re.compile(r"^v\d{3}$")
_FRAME = re.compile(r"^(.*?)(\d+)(\.exr)$")


def root() -> Path:
    return pl.WORK_ROOT / COMP


def shot_dir(name: str) -> Path:
    return pl.resolve_out_dir("", root() / pl.safe_name(name, "shot name"))


def layer_dir(name: str, layer: str) -> Path:
    return shot_dir(name) / "layers" / pl.safe_name(layer, "layer name")


def _file(name: str) -> Path:
    return shot_dir(name) / "shot.json"


def save_shot(shot: dict) -> None:
    clean = {**shot, "layers": [{k: v for k, v in lay.items() if k != "versions"} for lay in shot["layers"]]}
    p = _file(shot["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, indent=1), encoding="utf-8")
    tmp.replace(p)


def create_shot(name: str, plate: str, first: int, last: int, frame_offset: int = pl.NUKE_FRAME_OFFSET,
                ocio_config: str = "") -> dict:
    name = pl.safe_name(name, "shot name")
    first, last, frame_offset = int(first), int(last), int(frame_offset)
    if last < first:
        raise ValueError(f"last {last} is before first {first}")
    if _file(name).exists():
        raise ValueError(f"shot {name!r} exists")
    plate = pl.clean_path(plate)
    if plate and not Path(plate).is_file():
        raise ValueError(f"plate {Path(plate).name!r} not found")
    shot = {"name": name, "plate": plate, "first": first, "last": last, "frame_offset": frame_offset,
            "ocio_config": str(ocio_config or ""), "layers": []}
    save_shot(shot)
    return load_shot(name)


def list_shots() -> list[str]:
    r = root()
    if not r.is_dir():
        return []
    return sorted(d.name for d in r.iterdir() if (d / "shot.json").is_file())


def versions(name: str, layer: str) -> list[str]:
    d = layer_dir(name, layer)
    if not d.is_dir():
        return []
    return sorted(v.name for v in d.iterdir() if v.is_dir() and _VERSION.match(v.name))


def load_shot(name: str) -> dict:
    p = _file(name)
    if not p.is_file():
        raise ValueError(f"no shot {pl.safe_name(name, 'shot name')!r}")
    shot = json.loads(p.read_text(encoding="utf-8"))
    for lay in shot["layers"]:
        lay["versions"] = versions(shot["name"], lay["name"])
    return shot


def _layer(shot: dict, layer: str) -> dict:
    for lay in shot["layers"]:
        if lay["name"] == layer:
            return lay
    raise ValueError(f"no layer {layer!r} in {shot['name']}")


def _check_source(source: str) -> str:
    p = Path(pl.clean_path(source))
    work = pl.WORK_ROOT.resolve()
    rp = p.resolve()
    if work != rp and work not in rp.parents:
        raise ValueError(f"source {p.name!r} is outside the work root")
    return str(p)


def add_layer(name: str, layer: str, kind: str, source: str = "", recipe: str = "", params: dict | None = None) -> dict:
    shot = load_shot(name)
    layer = pl.safe_name(layer, "layer name")
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r}: one of {KINDS}")
    if any(x["name"] == layer for x in shot["layers"]):
        raise ValueError(f"layer {layer!r} exists")
    lay = {"name": layer, "kind": kind, "source": "", "recipe": "", "params": dict(params or {}), "blend": "over",
           "opacity": 1.0, "visible": True, "version": None, "order": len(shot["layers"])}
    if kind == "sequence":
        lay["source"] = _check_source(source)
    else:
        lay["recipe"] = pl.safe_name(recipe, "recipe")
    shot["layers"].append(lay)
    save_shot(shot)
    return {**lay, "versions": []}


def update_layer(name: str, layer: str, **changes) -> dict:
    shot = load_shot(name)
    lay = _layer(shot, layer)
    for k, v in changes.items():
        if k not in CHANGEABLE:
            raise ValueError(f"{k}: not a layer setting ({', '.join(CHANGEABLE)})")
        if k == "visible":
            v = bool(v)
        elif k == "opacity":
            v = float(v)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"opacity {v}: 0 to 1")
        elif k == "blend" and v not in BLENDS:
            raise ValueError(f"blend {v!r}: one of {BLENDS}")
        elif k == "version" and v is not None and v not in lay["versions"]:
            raise ValueError(f"version {v!r}: not rendered ({', '.join(lay['versions']) or 'none yet'})")
        elif k == "params" and not isinstance(v, dict):
            raise ValueError("params: a JSON object")
        elif k == "order":
            v = int(v)
        lay[k] = v
    save_shot(shot)
    return lay


def next_version(name: str, layer: str) -> str:
    """Make and return the next version folder, so two renders in a row never share one."""
    have = versions(name, layer)
    v = f"v{(int(have[-1][1:]) + 1 if have else 1):03d}"
    (layer_dir(name, layer) / v).mkdir(parents=True, exist_ok=False)
    return v


def _pattern_in(folder: Path) -> str | None:
    for f in sorted(folder.rglob("*.exr")):
        m = _FRAME.match(f.name)
        if m:
            return str(f.parent / f"{m.group(1)}{'#' * len(m.group(2))}{m.group(3)}")
    return None


def layer_pattern(shot: dict, layer: dict) -> str | None:
    """The #### pattern a layer shows: a sequence layer's source, or the EXRs of a recipe layer's chosen version (its
    newest when none is chosen); None when nothing is rendered."""
    if layer["kind"] == "sequence":
        return layer["source"] or None
    vs = layer.get("versions") or versions(shot["name"], layer["name"])
    v = layer.get("version") or (vs[-1] if vs else None)
    if not v:
        return None
    return _pattern_in(layer_dir(shot["name"], layer["name"]) / v)
