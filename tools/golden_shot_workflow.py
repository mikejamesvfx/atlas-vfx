"""Save the golden graph: the whole shot in one workflow, the way the pieces are meant to connect (flow review
2026-11-01). DSCF3000 S2-S3.

  Plate Read (0-296, for the solve) -> Solve Mask -> Solve Plate -> Measure & Scale (S2-S3 picks: metres, up)
      -> Solve View, Lock Check, Camera Export
      -> Sensor Lens Maps (the solved lens, auto overscan)  = the clean frame
  Plate Read (S2-S3, the shot)
      -> Z-Image grass -> Foreground Element x2 -> Card Project (clean + plate versions)
      -> element clip -> Character Insert -> Card Project (clean + plate)   [no depth holdout: Plate Depth is of the
         distorted plate and is refused with a lens folder for now]
      -> LTX Prep (clean space: lens folder from the camera, the element card linked) -> LTX element clean loop
         (two stage, LTX-2.5 + in/outpainting IC-LoRA) -> LTX Return (clean/ + redistorted)
      -> Comp Preview (plate versions over the plate)
      -> Nuke Builder, lens_delivery clean + STMaps (clean layers -> one STMap redistort -> over the plate)

Built from the committed DEMO_all_in_one_s2s3 and ltx_element_clean_loop_s2s3 workflows (tools/ui_workflow.to_api)
with the camera, lens and LTX links rewired, so their tuned settings stay. Nothing is queued: pick the element clip
in LoadVideo first. Also re-saves the t8 workflow from tools/run_t8_nuke.py's GRAPH (its saved widgets were stale).

Atlas VFX node definitions come from this checkout; core and LTX nodes from V141's /object_info.
Run: <V141 python> tools\\golden_shot_workflow.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ui_workflow  # noqa: E402
from atlas_vfx import registry  # noqa: E402

ui_workflow.API = "http://127.0.0.1:8188"
WF = ROOT / "workflows"
FOOTAGE = "~/Videos/atlas-vfx_footage"
PICKS = "~/atlas-vfx-work/DSCF3000/picks/DSCF3000_S2S3_picks_GUESS.json"
LOOP_OFFSET = 1000                           # clean-loop node ids move up by this


def seed(nodes):
    classes, _, _ = registry.load_nodes()
    for cls in {n["type"] if "type" in n else n["class_type"] for n in nodes}:
        c = classes.get(cls)
        if c is not None:
            ui_workflow._INFO[cls] = {"input": c.INPUT_TYPES(), "output": list(c.RETURN_TYPES),
                                      "output_name": list(c.RETURN_NAMES)}


def load_api(name):
    wf = json.loads((WF / f"{name}.json").read_text(encoding="utf-8"))
    seed(wf["nodes"])
    return ui_workflow.to_api(wf)


def relink(graph, old, new):
    """Every input linked to old [node, slot] now comes from new."""
    for n in graph.values():
        for k, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and [str(v[0]), v[1]] == old:
                n["inputs"][k] = list(new)


def build() -> dict:
    demo = load_api("DEMO_all_in_one_s2s3")
    loop = load_api("ltx_element_clean_loop_s2s3")
    g = copy.deepcopy(demo)

    # --- camera: the tracker instead of an imported .abc
    g["200"] = {"class_type": "AtlasVFXPlateRead", "inputs": {
        "path": f"{FOOTAGE}/DSCF3000.MOV", "first": 0, "last": 296, "ocio_config": "", "input_space": "auto",
        "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 96}}
    g["201"] = {"class_type": "AtlasVFXSolveMask", "inputs": {
        "plate_ref": ["200", 0], "classes": "person. car. bicycle. dog. tree. bush. flag.", "window": 16,
        "box_threshold": 0.35, "text_threshold": 0.25, "dilate_px": 6, "proxy_width": 1024}}
    g["202"] = {"class_type": "AtlasVFXSolvePlate", "inputs": {
        "plate_ref": ["200", 0], "sensor_preset": "auto", "sensor_width_mm": 23.5, "sensor_height_mm": 12.39,
        "focal_guess_mm": 16.0, "solve_focal": True, "solve_distortion": True, "lens_model": "k1 k2 (RADIAL)",
        "proxy_width": 1024, "max_features": 8192, "overlap": 15, "reject_px": 2.0, "name": "DSCF3000_golden_solve",
        "out_dir": "", "masks": ["201", 0], "mask_dilate_px": 4, "seed": 0}}
    g["203"] = {"class_type": "AtlasVFXMeasureScale", "inputs": {
        "shot_camera": ["202", 0], "picks_path": PICKS, "solve_points": "", "max_spread_pct": 5.0, "force": False,
        "name": "DSCF3000_golden_metric", "out_dir": ""}}
    g["204"] = {"class_type": "AtlasVFXSensorLensMaps", "inputs": {
        "lens_source": "camera", "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "ca_r": 0.0, "ca_b": 0.0,
        "vig_1": 0.0, "vig_2": 0.0, "vig_3": 0.0, "map_scale": 1.0, "out_dir": "", "shot_camera": ["203", 0],
        "overscan": "auto"}}
    g["205"] = {"class_type": "AtlasVFXSolveView", "inputs": {
        "shot_camera": ["203", 0], "max_points": 20000, "plate_frames": 8, "seeds": "[]"}}
    g["206"] = {"class_type": "PreviewAny", "inputs": {"source": ["202", 2]}}
    relink(g, ["4", 0], ["203", 0])
    del g["4"], g["5"]                                            # Camera Import and its summary

    # --- every card is clean + plate (the lens folder of the solved camera)
    for cid in ("40", "41", "72"):
        g[cid]["inputs"]["sensor_lens_folder"] = ["204", 1]
    g["72"]["inputs"]["holdout_depth"] = ""                       # refused with a lens folder (distorted depth)

    # --- the LTX clean loop, its Plate Read replaced by the shot's, the element card and lens linked
    for nid, n in loop.items():
        if n["class_type"] == "AtlasVFXPlateRead":
            continue
        m = copy.deepcopy(n)
        for k, v in m["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                src = str(v[0])
                m["inputs"][k] = ["2", v[1]] if loop[src]["class_type"] == "AtlasVFXPlateRead" else \
                    [str(int(src) + LOOP_OFFSET), v[1]]
        g[str(int(nid) + LOOP_OFFSET)] = m
    prep = next(k for k, n in g.items() if n["class_type"] == "AtlasVFXLTXPrep")
    ret = next(k for k, n in g.items() if n["class_type"] == "AtlasVFXLTXReturn")
    g[prep]["inputs"].update({"layer_folder": ["72", 0], "sensor_lens_folder": ["204", 1], "sensor_rs_folder": ""})

    # --- delivery: the LTX take replaces the element card; layers clean + STMaps in Nuke
    relink(g, ["72", 0], [ret, 0])
    g[prep]["inputs"]["layer_folder"] = ["72", 0]                 # (the relink above touched Prep too)
    g["91"]["inputs"].update({"lens_delivery": "clean + STMaps", "sensor_lens_folder": ["204", 1],
                              "shot": "DSCF3000_S2S3_golden", "set_proxy_obj": ""})
    g["90"]["inputs"]["name"] = "DSCF3000_golden"
    return g


def refresh_t8():
    import run_t8_nuke

    seed(run_t8_nuke.GRAPH.values())
    return ui_workflow.save(run_t8_nuke.GRAPH, "t8_camera_export_nuke_builder_s2s3",
                            "T8 Camera Export + Nuke Builder (S2-S3, grass)")


if __name__ == "__main__":
    g = build()
    seed(g.values())
    print("saved", ui_workflow.save(g, "golden_dscf3000_s2s3",
                                    "Golden shot: tracker -> metres -> lens (overscan) -> cards -> LTX clean loop -> "
                                    "Nuke (clean + STMaps)"))
    print("saved", refresh_t8())
