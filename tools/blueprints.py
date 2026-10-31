"""Build the atlas-vfx subgraph blueprints: whole jobs as one box in ComfyUI's node library, showing only the few
settings an artist changes. Written to subgraphs/ at the repo root; ComfyUI lists custom_nodes/*/subgraphs/*.json
(app/subgraph_manager.py), so the pack ships them. No Python runs inside a blueprint: it is the existing nodes.

  Relight Element 4K   the LTX-2.3 relight with mirror, LTX Return and Relight Transfer Files, from
                       tools/ltx_relight_workflow.graph_4k(): layer, frames, sun, look, seed, transfer mode, name
  LTX Clean Loop       workflows/ltx_element_clean_loop_s2s3.json: plate, layer, first frame, prompt, seeds, name
                       (121 frames; the frame counts inside are fixed, change the full graph for another length)

Run: <V141 python> tools\\blueprints.py [--api http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ui_workflow  # noqa: E402

OUT = HERE.parent / "subgraphs"
NS = uuid.UUID("6f1c2a52-6d4e-4a8f-9a57-1f0e6c3b9a01")      # fixed: the same blueprint keeps its id across builds


def to_blueprint(wf: dict, name: str, description: str, expose: list[tuple[int, str]],
                 category: str = "Atlas VFX") -> dict:
    """A GUI workflow as a ComfyUI subgraph blueprint whose box shows the widgets in `expose` ((node id, widget))."""
    nodes = copy.deepcopy(wf["nodes"])
    ids = {n["id"]: n for n in nodes}
    for nid, widget in expose:
        if nid not in ids:
            raise ValueError(f"{name}: no node {nid} to expose {widget}")
    links = [{"id": l[0], "origin_id": l[1], "origin_slot": l[2], "target_id": l[3], "target_slot": l[4],
              "type": l[5]} for l in wf["links"]]
    sid = str(uuid.uuid5(NS, name))
    sg = {"id": sid, "version": 1,
          "state": {"lastGroupId": len(wf.get("groups", [])), "lastNodeId": max(ids),
                    "lastLinkId": max([l["id"] for l in links] or [0]), "lastRerouteId": 0},
          "revision": 0, "config": {}, "name": name,
          "inputNode": {"id": -10, "bounding": [-400, 0, 120, 60]},
          "outputNode": {"id": -20, "bounding": [4000, 0, 120, 60]},
          "inputs": [], "outputs": [], "widgets": [], "nodes": nodes, "groups": copy.deepcopy(wf.get("groups", [])),
          "links": links, "extra": {"workflowRendererVersion": "LG"}, "category": category,
          "description": description}
    box = {"id": max(ids) + 1, "type": sid, "pos": [0, 0], "size": [380, 60 + 26 * len(expose)], "flags": {},
           "order": 0, "mode": 0, "inputs": [], "outputs": [],
           "properties": {"proxyWidgets": [[str(n), w] for n, w in expose]}, "widgets_values": [], "title": name}
    return {"revision": 0, "last_node_id": box["id"], "last_link_id": 0, "nodes": [box], "links": [],
            "version": 0.4, "definitions": {"subgraphs": [sg]}}


def write(bp: dict, file: str) -> Path:
    OUT.mkdir(exist_ok=True)
    p = OUT / file
    p.write_text(ui_workflow.portable(json.dumps(bp, indent=1)), encoding="utf-8")
    return p


def relight_element_4k() -> dict:
    import ltx_relight_workflow as lr
    from physical_sky_workflow import local_info
    from atlas_vfx.registry import load_nodes

    g = lr.graph_4k()
    for n in g.values():
        cls = load_nodes()[0].get(n["class_type"])
        if cls is not None:
            ui_workflow._INFO[n["class_type"]] = local_info(cls)
    wf = ui_workflow.convert(g, "")
    expose = [(71, "layer"), (70, "first"), (70, "last"), (72, "direction"), (72, "height"), (72, "look"),
              (49, "noise_seed"), (57, "mode"), (57, "name"), (59, "mode"), (59, "exposure")]
    return to_blueprint(wf, "Atlas VFX Relight Element 4K",
                        "Relights a rendered layer with the LTX-2.3 Relight IC-LoRA and delivers it at plate res: pick "
                        "the layer, the frames (Nuke numbers), where the sun is and the look. A sun on the left is done "
                        "mirrored. Writes the LTX layer (LTX Return) and the 4K EXRs (Relight Transfer Files, MoGe-3 "
                        "inside), then shows Notes and a wipe over the element.", expose)


def clean_loop() -> dict:
    wf = json.loads((HERE.parent / "workflows" / "ltx_element_clean_loop_s2s3.json").read_text(encoding="utf-8"))
    expose = [(1, "path"), (100, "layer_folder"), (100, "first_frame"), (106, "text"), (120, "noise_seed"),
              (136, "noise_seed"), (150, "name")]
    return to_blueprint(wf, "Atlas VFX LTX Clean Loop",
                        "Integrates a rendered layer into the plate with the LTX-2.3 in/outpainting IC-LoRA, two "
                        "stages, 121 frames from first_frame: pick the plate, the layer and describe it. Writes an "
                        "ACEScg layer for the Nuke Builder (LTX Return). The frame counts inside are fixed at 121; for "
                        "another length use the full workflow.", expose)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    print("wrote", write(relight_element_4k(), "relight_element_4k.json"))
    print("wrote", write(clean_loop(), "ltx_clean_loop.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
