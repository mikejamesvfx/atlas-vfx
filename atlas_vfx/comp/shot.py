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
from atlas_vfx.layers import shot_mode as sm

COMP = "comp"
PREVIEW = "preview"
SHOW_FIRST = ("", "beauty")        # a render shows the EXRs at the version root, else in beauty/ (Relight Transfer)
KINDS = ("sequence", "recipe")
BLENDS = ("over", "plus", "multiply", "screen")
CHANGEABLE = ("visible", "opacity", "blend", "version", "params", "order", "mix")
MIX_MAX = 2.0           # a light mix weight: 0 off, 1 as rendered, up to 2 (a light pushed brighter)
_VERSION = re.compile(r"^v\d{3}$")
_FRAME = re.compile(r"^(.*?)(\d+)(\.exr)$")


def root() -> Path:
    return pl.WORK_ROOT / COMP


def shot_dir(name: str) -> Path:
    return pl.resolve_out_dir("", root() / pl.safe_name(name, "shot name"))


def layer_dir(name: str, layer: str) -> Path:
    return shot_dir(name) / "layers" / pl.safe_name(layer, "layer name")


def preview_dir(name: str, layer: str) -> Path:
    """Where Preview writes: one folder per layer, overwritten by the next preview, never a version."""
    return layer_dir(name, layer) / PREVIEW


def _file(name: str) -> Path:
    return shot_dir(name) / "shot.json"


def save_shot(shot: dict) -> None:
    clean = {**shot, "layers": [{k: v for k, v in lay.items() if k not in ("versions", "empty_versions", "has_preview")}
                                for lay in shot["layers"]]}
    p = _file(shot["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, indent=1), encoding="utf-8")
    tmp.replace(p)


def create_shot(name: str, plate: str, first: int, last: int, frame_offset: int = pl.NUKE_FRAME_OFFSET,
                ocio_config: str = "", mode: str = "plate", screen: str = "green", width: int = 0, height: int = 0,
                fps: str = "24") -> dict:
    """mode (layers/shot_mode.py): plate (a background plate; blank = no plate yet, as before), foreground (an
    actor plate shot on a green or blue screen; the plate is required) or generate (no plate: width, height and fps
    describe the virtual plate the background is made for)."""
    name = pl.safe_name(name, "shot name")
    first, last, frame_offset = int(first), int(last), int(frame_offset)
    mode = sm.check_mode(mode)
    if last < first:
        raise ValueError(f"last {last} is before first {first}")
    if _file(name).exists():
        raise ValueError(f"shot {name!r} exists")
    plate = pl.clean_path(plate)
    if mode == "generate":
        if plate:
            raise ValueError("a generate shot has no plate: its background is made (leave the plate blank)")
        # a shot's first/last are Nuke frames here: the virtual plate is just those frames (checks size and fps)
        ref = pl.virtual_plate_ref(width, height, fps, last - first + 1, 0, None, ocio_config, first, name)
    else:
        if mode == "foreground" and not plate:
            raise ValueError("a foreground shot needs its screen plate")
        if plate and not Path(plate).is_file():
            raise ValueError(f"plate {Path(plate).name!r} not found")
    shot = {"name": name, "mode": mode, "plate": plate, "first": first, "last": last, "frame_offset": frame_offset,
            "ocio_config": str(ocio_config or ""), "layers": []}
    if mode == "foreground":
        shot["screen"] = sm.check_screen(screen)
    if mode == "generate":
        shot["size"] = [ref["width"], ref["height"]]
        shot["fps"] = list(ref["fps"])
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
    shot.setdefault("mode", "plate")                        # shots made before the mode existed
    for lay in shot["layers"]:
        vs = versions(shot["name"], lay["name"])
        full = [v for v in vs if _pattern_in(layer_dir(shot["name"], lay["name"]) / v)]
        lay["versions"] = full                              # renders with frames: what the artist can pick
        lay["empty_versions"] = [v for v in vs if v not in full]   # failed, cancelled or still running
        lay["has_preview"] = _pattern_in(preview_dir(shot["name"], lay["name"])) is not None
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
        elif k == "version" and v == PREVIEW and not lay["has_preview"]:
            raise ValueError("version 'preview': no preview yet")
        elif k == "version" and v not in (None, PREVIEW) and v not in lay["versions"]:
            raise ValueError(f"version {v!r}: not rendered ({', '.join(lay['versions']) or 'none yet'})")
        elif k == "params" and not isinstance(v, dict):
            raise ValueError("params: a JSON object")
        elif k == "order":
            v = int(v)
        elif k == "mix":
            if not isinstance(v, dict):
                raise ValueError("mix: an object of version -> weight")
            clean = {}
            for ver, w in v.items():
                if ver not in lay["versions"]:
                    raise ValueError(f"mix: version {ver!r} is not rendered ({', '.join(lay['versions']) or 'none'})")
                if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0.0 <= float(w) <= MIX_MAX:
                    raise ValueError(f"mix: weight for {ver} must be 0-{MIX_MAX:g}")
                clean[ver] = float(w)
            v = clean
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
    if not folder.is_dir():
        return None
    for sub in SHOW_FIRST:
        d = folder / sub if sub else folder
        for f in sorted(d.glob("*.exr")) if d.is_dir() else []:
            m = _FRAME.match(f.name)
            if m:
                return str(f.parent / f"{m.group(1)}{'#' * len(m.group(2))}{m.group(3)}")
    return None


def version_pattern(shot: dict, layer: dict, version: str) -> str | None:
    """The #### pattern of one version of a recipe layer (None when it has no frames)."""
    if version == PREVIEW:
        return _pattern_in(preview_dir(shot["name"], layer["name"]))
    return _pattern_in(layer_dir(shot["name"], layer["name"]) / version)


def layer_pattern(shot: dict, layer: dict) -> str | None:
    """The #### pattern a layer shows: a sequence layer's source, or the EXRs of a recipe layer's chosen version (its
    newest when none is chosen); None when nothing is rendered."""
    if layer["kind"] == "sequence":
        return layer["source"] or None
    if "versions" in layer:
        vs = layer["versions"]
    else:
        d = layer_dir(shot["name"], layer["name"])
        vs = [v for v in versions(shot["name"], layer["name"]) if _pattern_in(d / v)]
    v = layer.get("version") or (vs[-1] if vs else None)   # latest = the newest render that has frames
    if not v:
        return None
    if v == PREVIEW:
        return _pattern_in(preview_dir(shot["name"], layer["name"]))
    return _pattern_in(layer_dir(shot["name"], layer["name"]) / v)
