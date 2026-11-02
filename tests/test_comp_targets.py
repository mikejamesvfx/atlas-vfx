"""Atlas Comp render targets and preflight (atlas_vfx/comp/targets.py): a job is checked against a target before
anything is sent, and refused with the missing node or model named (ADR-021 Amendment 4 draft, rule 1).

Run: <V141 python> tests\\test_comp_targets.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="targets_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def raises(fn, label, needle=""):
    try:
        fn()
    except ValueError as e:
        check(needle in str(e), label, str(e)[:100])
        return
    check(False, label, "no error")


OBJECT_INFO = {
    "UNETLoader": {"input": {"required": {"unet_name": [["ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"]],
                                          "weight_dtype": [["default", "fp8"]]}}},
    "LTXICLoRALoaderModelOnly": {"input": {"required": {"model": ["MODEL"],
                                                        "lora_name": [["ltx25\\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"]],
                                                        "strength_model": ["FLOAT", {}]}}},
    "KSamplerSelect": {"input": {"required": {"sampler_name": [["euler", "euler_ancestral"]]}}},
}

GRAPH = {
    "12": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "weight_dtype": "default"}},
    "17": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
        "model": ["12", 0], "lora_name": "ltx23\\ltx-2.3-22b-ic-lora-relight-1.0.safetensors", "strength_model": 1.0}},
    "48": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
    "43": {"class_type": "AtlasVFXGuideFromLatent", "inputs": {}},
}


def fake_fetch(info):
    calls = []

    def fetch(url, timeout=10):
        calls.append(url)
        if url.endswith("/object_info"):
            return info
        raise OSError("not found")
    fetch.calls = calls
    return fetch


def test_targets_file():
    from atlas_vfx.comp import targets as tg

    ts = tg.load_targets()
    check([t["name"] for t in ts] == ["ws"] and ts[0]["kind"] == "comfyui", "targets: without the file, ws only")
    (pl.WORK_ROOT / tg.TARGETS_FILE).write_text(json.dumps({"targets": [
        {"name": "render", "kind": "comfyui", "url": "http://render.example:8188", "note": "3090 24 GB"},
        {"name": "cloud", "kind": "comfy_cloud"}]}))
    ts = tg.load_targets()
    check([t["name"] for t in ts] == ["ws", "render", "cloud"], "targets: the local file adds farm and cloud",
          str([t["name"] for t in ts]))
    (pl.WORK_ROOT / tg.TARGETS_FILE).write_text(json.dumps({"targets": [
        {"name": "bad", "kind": "comfyui", "url": "file:///etc/passwd"}]}))
    raises(tg.load_targets, "targets: only http(s) addresses", "http")
    (pl.WORK_ROOT / tg.TARGETS_FILE).write_text(json.dumps({"targets": [{"name": "../x", "kind": "comfyui",
                                                                          "url": "http://a:1"}]}))
    raises(tg.load_targets, "targets: names are safe names", "name")
    (pl.WORK_ROOT / tg.TARGETS_FILE).write_text(json.dumps({"targets": [{"name": "x", "kind": "ssh"}]}))
    raises(tg.load_targets, "targets: known kinds only", "kind")
    (pl.WORK_ROOT / tg.TARGETS_FILE).unlink()


def test_preflight():
    from atlas_vfx.comp import targets as tg

    t = {"name": "hub", "kind": "comfyui", "url": "http://hub.example:8188"}
    fetch = fake_fetch(OBJECT_INFO)
    r = tg.preflight(t, GRAPH, fetch=fetch)
    check(not r["ok"] and r["missing_nodes"] == ["AtlasVFXGuideFromLatent"],
          "preflight: a node the target lacks is named", str(r["missing_nodes"]))
    check(r["missing_models"] == ["ltx23\\ltx-2.3-22b-ic-lora-relight-1.0.safetensors"],
          "preflight: a model file the target lacks is named", str(r["missing_models"]))
    check(fetch.calls == ["http://hub.example:8188/object_info"], "preflight: one request to the target")
    info = dict(OBJECT_INFO, AtlasVFXGuideFromLatent={"input": {"required": {}}})
    info["LTXICLoRALoaderModelOnly"] = {"input": {"required": {"model": ["MODEL"], "lora_name": [[
        "ltx23\\ltx-2.3-22b-ic-lora-relight-1.0.safetensors"]], "strength_model": ["FLOAT", {}]}}}
    r = tg.preflight(t, GRAPH, fetch=fake_fetch(info))
    check(r["ok"] and not r["missing_nodes"] and not r["missing_models"], "preflight: a ready target passes")
    g = json.loads(json.dumps(GRAPH))
    g["48"]["inputs"]["sampler_name"] = "dpm_9000"
    r = tg.preflight(t, g, fetch=fake_fetch(info))
    check(r["ok"], "preflight: non-model choices are left to ComfyUI's own validation", str(r))

    def down(url, timeout=10):
        raise OSError("connection refused")
    r = tg.preflight(t, GRAPH, fetch=down)
    check(not r["ok"] and "unreachable" in " ".join(r["notes"]), "preflight: an unreachable target is refused",
          str(r["notes"]))


def test_combo_and_separators():
    """A Linux farm box spells folders with / and newer loaders list options as ["COMBO", {"options": [...]}]; ws
    writes ltx25\\name. Preflight must read both formats, match across separators, and localize() must rewrite the
    graph to the target's own spelling (live on render 2026-10-31: the LoRA was "not found")."""
    from atlas_vfx.comp import targets as tg

    info = {"LTXICLoRALoaderModelOnly": {"input": {"required": {"lora_name": ["COMBO", {"multiselect": False,
                                                                                         "options": ["ltx25/alpha.safetensors"]}]}}},
            "UNETLoader": {"input": {"required": {"unet_name": [["ltx.safetensors"]]}}}}
    g = {"1": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"lora_name": "ltx25\\alpha.safetensors"}},
         "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx.safetensors"}}}
    t = {"name": "render", "kind": "comfyui", "url": "http://render.example:8188"}
    r = tg.preflight(t, g, fetch=lambda url, timeout=10: info)
    check(r["ok"], "combo: a / spelling on the target matches ws's \\ spelling", str(r))
    tg.localize(g, info)
    check(g["1"]["inputs"]["lora_name"] == "ltx25/alpha.safetensors", "combo: the graph is rewritten to the target's "
          "spelling before it is sent", g["1"]["inputs"]["lora_name"])
    g["1"]["inputs"]["lora_name"] = "ltx25\\missing.safetensors"
    r = tg.preflight(t, g, fetch=lambda url, timeout=10: info)
    check(not r["ok"] and r["missing_models"] == ["ltx25\\missing.safetensors"],
          "combo: a model missing from a COMBO list is still refused", str(r["missing_models"]))


def main():
    for f in (test_targets_file, test_preflight, test_combo_and_separators):
        f()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
