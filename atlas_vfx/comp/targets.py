"""Where Atlas Comp can send a job, and the preflight that decides whether it may.

Targets: ws (this ComfyUI) always, plus those in <work root>/atlas_comp_targets.local.json, which lives outside
the repo because it names machines on Mike's LAN:

    {"targets": [{"name": "render", "kind": "comfyui", "url": "http://<address>:8188", "note": "3090 24 GB"},
                 {"name": "cloud", "kind": "comfy_cloud"}]}

Preflight (ADR-021 Amendment 4 draft, rule 1): before anything is sent, the target must have every node the job
names and every model file its loaders name. One miss and the job is refused, the miss named. Nothing falls back
to another machine on its own."""
from __future__ import annotations

import json
import urllib.request
from urllib.parse import urlparse

from atlas_vfx.camera_side import plate as pl

TARGETS_FILE = "atlas_comp_targets.local.json"
KINDS = ("comfyui", "comfy_cloud")
WS = {"name": "ws", "kind": "comfyui", "url": "http://127.0.0.1:8188", "note": "this machine", "local": True}
# inputs that name a model file: a value not in the target's list means the file is not on that machine
MODEL_INPUTS = ("ckpt_name", "unet_name", "lora_name", "vae_name", "clip_name", "clip_name1", "clip_name2",
                "model_name", "tae_name", "upscale_model_name", "control_net_name", "style_model_name")


def load_targets() -> list[dict]:
    targets = [dict(WS)]
    p = pl.WORK_ROOT / TARGETS_FILE
    if not p.is_file():
        return targets
    for t in json.loads(p.read_text(encoding="utf-8")).get("targets", []):
        name = pl.safe_name(t.get("name", ""), "target name")
        kind = t.get("kind")
        if kind not in KINDS:
            raise ValueError(f"target {name}: kind {kind!r} is not one of {KINDS}")
        clean = {"name": name, "kind": kind, "note": str(t.get("note", ""))[:80]}
        if kind == "comfyui":
            url = str(t.get("url", "")).rstrip("/")
            u = urlparse(url)
            if u.scheme not in ("http", "https") or not u.netloc:
                raise ValueError(f"target {name}: url must be http(s)://host:port")
            clean["url"] = url
        if name in [x["name"] for x in targets]:
            raise ValueError(f"target {name}: listed twice")
        targets.append(clean)
    return targets


def public(t: dict) -> dict:
    """What the app may show: never the address (it names Mike's LAN)."""
    return {k: t[k] for k in ("name", "kind", "note") if k in t}


def _fetch(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _check(graph: dict, has_node, model_options) -> tuple[list[str], list[str]]:
    missing_nodes, missing_models = [], []
    for node in graph.values():
        cls = node["class_type"]
        if not has_node(cls):
            if cls not in missing_nodes:
                missing_nodes.append(cls)
            continue
        for key, val in node.get("inputs", {}).items():
            if key in MODEL_INPUTS and isinstance(val, str):
                opts = model_options(cls, key)
                if opts is not None and val not in opts and val not in missing_models:
                    missing_models.append(val)
    return missing_nodes, missing_models


def preflight(target: dict, graph: dict, fetch=None, cloud=None) -> dict:
    """{"ok", "missing_nodes", "missing_models", "notes"} for running `graph` on `target`. comfyui targets are asked
    once for /object_info; comfy_cloud targets through `cloud` (an object with has_node(name) and
    model_options(class_type, input) -> list | None)."""
    notes: list[str] = []
    if target["kind"] == "comfyui":
        try:
            info = (fetch or _fetch)(target["url"] + "/object_info", timeout=10)
        except (OSError, ValueError) as e:
            return {"ok": False, "missing_nodes": [], "missing_models": [],
                    "notes": [f"{target['name']} is unreachable: {type(e).__name__}"]}

        def options(cls, key):
            spec = info.get(cls, {}).get("input", {})
            for group in ("required", "optional"):
                s = spec.get(group, {}).get(key)
                if s and isinstance(s[0], list):
                    return s[0]
            return None
        nodes, models = _check(graph, lambda c: c in info, options)
    elif target["kind"] == "comfy_cloud":
        if cloud is None:
            return {"ok": False, "missing_nodes": [], "missing_models": [], "notes": ["Comfy Cloud is not set up"]}
        nodes, models = _check(graph, cloud.has_node, cloud.model_options)
    else:
        raise ValueError(f"target kind {target['kind']!r}")
    if nodes:
        notes.append(f"{target['name']} is missing nodes: {', '.join(nodes)}")
    if models:
        notes.append(f"{target['name']} is missing models: {', '.join(models)}")
    return {"ok": not nodes and not models, "missing_nodes": nodes, "missing_models": models, "notes": notes}
