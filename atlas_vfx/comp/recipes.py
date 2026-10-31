"""Recipes: a generative operation as data. recipes/<name>.json holds an API graph and says which graph inputs the
artist's fields, the shot and the output folder set, and how the quick Preview differs from the full Render.
tools/comp_recipes.py writes them from the graph builders; nothing here runs code from a recipe.

    {"name", "label", "description", "graph": {API graph},
     "fields": [{"key", "label", "type": choice|text|int|float|bool|layer, "default", "choices"?, "min"?, "max"?,
                 "targets": [[node, input] or [node, input, "pattern"|"folder"] for a layer field]}],
     "context": {"plate"|"first"|"last"|"frame_offset"|"first_frame"|"length"|"plate_first"|"plate_last":
                 [[node, input], ...]},
     "outputs": [[node, input, subfolder], ...],      # out_dir inputs, set to <version folder>/<subfolder>
     "names": [[node, input], ...],                   # set to the layer name
     "preview": {"frames": N?, "set": [[node, input, value]], "nodes": {id: node}, "drop": [node, ...],
                 "outputs": [[node, input, subfolder]]?}}   # preview's own out_dir map (default: outputs)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from atlas_vfx.camera_side import plate as pl

RECIPES = Path(__file__).resolve().parents[2] / "recipes"
TYPES = ("choice", "text", "int", "float", "bool", "layer")
CONTEXT_KEYS = ("plate", "first", "last", "frame_offset", "first_frame", "length", "plate_first", "plate_last",
                "cache_dir")
TEXT_MAX = 2000


def _names() -> list[str]:
    return sorted(p.stem for p in RECIPES.glob("*.json"))


def load_recipe(name: str) -> dict:
    if name not in _names():
        raise ValueError(f"recipe {name!r}: one of {', '.join(_names())}")
    return json.loads((RECIPES / f"{name}.json").read_text(encoding="utf-8"))


def list_recipes() -> list[dict]:
    return [{k: v for k, v in load_recipe(n).items() if k != "graph"} for n in _names()]


def list_presets() -> list[dict]:
    """The Presets gallery (recipes/presets/presets.json): id, group, name, what, recipe, params. A preset whose
    recipe is missing is left out."""
    p = RECIPES / "presets" / "presets.json"
    if not p.is_file():
        return []
    names = set(_names())
    return [x for x in json.loads(p.read_text(encoding="utf-8"))["presets"] if x.get("recipe") in names]


def _number(key: str, v, kind):
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"{key}: a number")
    try:
        n = kind(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: {v!r} is not a number") from None
    if kind is int and isinstance(v, float) and v != int(v):
        raise ValueError(f"{key}: a whole number")
    return n


def validate(recipe: dict, params: dict) -> dict:
    """The params with defaults filled in, each checked against its field. Anything else is a ValueError naming it."""
    if not isinstance(params, dict):
        raise ValueError("params: a JSON object")
    fields = {f["key"]: f for f in recipe["fields"]}
    for k in params:
        if k not in fields:
            raise ValueError(f"{k}: not a field of {recipe['name']}")
    out = {}
    for k, f in fields.items():
        v = params.get(k, f.get("default"))
        t = f["type"]
        if t == "choice":
            if v not in f["choices"]:
                raise ValueError(f"{k}: {v!r} is not one of {f['choices']}")
        elif t == "text":
            v = str(v if v is not None else "")
            if len(v) > TEXT_MAX:
                raise ValueError(f"{k}: longer than {TEXT_MAX} characters")
        elif t in ("int", "float"):
            v = _number(k, v, int if t == "int" else float)
            if "min" in f and v < f["min"] or "max" in f and v > f["max"]:
                raise ValueError(f"{k}: {v} outside {f.get('min')}-{f.get('max')}")
        elif t == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{k}: true or false")
        elif t == "layer":
            v = pl.safe_name(v, k) if v else ""
        else:
            raise ValueError(f"{k}: unknown field type {t!r}")
        out[k] = v
    return out


def context_for(shot: dict, layer: dict) -> dict:
    """What a recipe needs from the shot: the plate, the frame range (LTX's 8k+1 length from the shot's first frame),
    the layer's name and every other layer's files."""
    from atlas_vfx.layers.node_shot_range import snap_length

    from .shot import layer_dir, layer_pattern

    first, off = int(shot["first"]), int(shot["frame_offset"])
    n = snap_length(int(shot["last"]) - first + 1, "8k+1 shorter")
    return {"plate": shot["plate"], "first": first, "last": first + n - 1, "frame_offset": off, "first_frame": first,
            "length": n, "plate_first": first - off, "plate_last": first + n - 1 - off, "layer": layer["name"],
            "cache_dir": str(layer_dir(shot["name"], layer["name"]) / "cache"),
            "layer_patterns": {x["name"]: layer_pattern(shot, x) for x in shot["layers"] if x["name"] != layer["name"]}}


def _set(g: dict, node, inp: str, value):
    g[str(node)]["inputs"][inp] = value


def _drop(g: dict, drop) -> None:
    gone = {str(n) for n in drop}
    for n in gone:
        g.pop(n, None)
    while True:
        dead = [nid for nid, node in g.items()
                if any(isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) and str(v[0]) not in g
                       for v in node["inputs"].values())]
        if not dead:
            return
        for nid in dead:
            del g[nid]


def fill(recipe: dict, params: dict, context: dict, out_dir: str, mode: str = "render") -> dict:
    """A ready API graph (a copy; the recipe is not changed) for this layer, writing into out_dir."""
    if mode not in ("render", "preview"):
        raise ValueError(f"mode {mode!r}: render or preview")
    params = validate(recipe, params)
    ctx = dict(context)
    pv = recipe.get("preview", {})
    if mode == "preview" and pv.get("frames"):
        n = min(int(pv["frames"]), int(ctx["length"]))
        ctx.update(length=n, last=ctx["first_frame"] + n - 1, plate_last=ctx["plate_first"] + n - 1)
    g = copy.deepcopy(recipe["graph"])
    for f in recipe["fields"]:
        v = params[f["key"]]
        if f["type"] == "layer":
            pattern = ctx["layer_patterns"].get(v) if v else None
            if not pattern:
                raise ValueError(f"{f['key']}: pick a layer that has frames" if not v else
                                 f"{f['key']}: layer {v!r} has nothing rendered")
            for t in f["targets"]:
                _set(g, t[0], t[1], str(Path(pattern).parent) if (len(t) > 2 and t[2] == "folder") else pattern)
        else:
            for t in f["targets"]:
                _set(g, t[0], t[1], v)
    for key, targets in recipe.get("context", {}).items():
        if key not in CONTEXT_KEYS:
            raise ValueError(f"recipe {recipe['name']}: unknown context {key!r}")
        for t in targets:
            _set(g, t[0], t[1], ctx[key])
    outputs = pv.get("outputs", recipe["outputs"]) if mode == "preview" else recipe["outputs"]
    for node, inp, sub in outputs:
        _set(g, node, inp, str(Path(out_dir) / sub) if sub else str(out_dir))
    for node, inp in recipe.get("names", []):
        _set(g, node, inp, ctx["layer"])
    if mode == "preview":
        for node, inp, value in pv.get("set", []):
            _set(g, node, inp, value)
        for nid, node in pv.get("nodes", {}).items():
            g[str(nid)] = copy.deepcopy(node)
        _drop(g, pv.get("drop", []))
    return g
