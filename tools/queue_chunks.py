"""Queue one ComfyUI prompt per chunk of a Chunk Plan: the "index widget, queue N times" way to run a long shot.

Takes an API-format workflow (ComfyUI: Export (API)) that holds exactly one Atlas VFX Chunk Plan node, and for each
chunk k sets that node's `chunk` to k and replaces {chunk} (00, 01, ...) and {tag} (c00, c01, ...) in every string
widget of the other nodes, so output names such as LTX Return's `name` or SaveImage's `filename_prefix` differ per
chunk. Chunk Stitch nodes are left out of the per-chunk prompts (their pattern keeps its {tag}); queue the stitch
graph once all chunks have rendered.

Run:  <V141 python> tools\\queue_chunks.py my_shot_api.json [--only 0,2] [--dry-run] [--api http://127.0.0.1:8188]
Prints the plan and one prompt id per chunk. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.layers import chunks as ch  # noqa: E402

PLAN, STITCH = "AtlasVFXChunkPlan", "AtlasVFXChunkStitch"


def find_plan(graph: dict) -> str:
    ids = [nid for nid, n in graph.items() if n.get("class_type") == PLAN]
    if len(ids) != 1:
        raise SystemExit(f"need exactly one {PLAN} node in the workflow, found {len(ids)}")
    return ids[0]


def plan_of(graph: dict, nid: str) -> ch.ChunkPlan:
    inp = graph[nid]["inputs"]
    for k in ("first", "last", "max_len", "overlap"):
        if isinstance(inp.get(k), list):
            raise SystemExit(f"{PLAN} '{k}' is wired from another node; set it as a widget to queue by chunk")
    return ch.plan(inp["first"], inp["last"], inp["max_len"], inp["overlap"])


def per_chunk(graph: dict, nid: str, c: ch.Chunk) -> tuple[dict, list[str]]:
    """The prompt for chunk c, and the 'node.input = value' lines it changed."""
    g = copy.deepcopy(graph)
    for k in [k for k, n in g.items() if n.get("class_type") == STITCH]:
        del g[k]
    dangling = [k for k, n in g.items() for v in n.get("inputs", {}).values()
                if isinstance(v, list) and len(v) == 2 and str(v[0]) not in g]
    if dangling:
        raise SystemExit(f"nodes {sorted(set(dangling))} take input from a Chunk Stitch; keep the stitch in its own "
                         "graph")
    g[nid]["inputs"]["chunk"] = c.index
    changed = [f"{nid}.chunk = {c.index}"]
    for k, n in g.items():
        if k == nid:
            continue
        for name, v in n.get("inputs", {}).items():
            if isinstance(v, str) and ("{chunk}" in v or "{tag}" in v):
                n["inputs"][name] = v.replace("{chunk}", f"{c.index:02d}").replace("{tag}", c.tag)
                changed.append(f"{k}.{name} = {n['inputs'][name]}")
    return g, changed


def post(api: str, graph: dict) -> str:
    req = urllib.request.Request(f"{api}/prompt", data=json.dumps({"prompt": graph, "client_id": "atlas-vfx-chunks"})
                                 .encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_id"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("workflow", help="API-format workflow JSON")
    ap.add_argument("--only", default="", help="comma list of chunk numbers (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the per-chunk changes, queue nothing")
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    a = ap.parse_args(argv)

    graph = json.loads(Path(a.workflow).read_text(encoding="utf-8"))
    if "nodes" in graph:
        raise SystemExit("that is a GUI workflow; export it with Export (API)")
    nid = find_plan(graph)
    p = plan_of(graph, nid)
    print(p.table())
    todo = [int(s) for s in a.only.split(",") if s.strip()] or list(range(len(p.chunks)))
    bad = [k for k in todo if not 0 <= k < len(p.chunks)]
    if bad:
        raise SystemExit(f"--only {bad}: the plan has chunks 0-{len(p.chunks) - 1}")
    for k in todo:
        g, changed = per_chunk(graph, nid, p.chunks[k])
        if a.dry_run:
            print(f"{p.chunks[k].tag}: " + "; ".join(changed))
        else:
            print(f"{p.chunks[k].tag}: queued {post(a.api, g)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
