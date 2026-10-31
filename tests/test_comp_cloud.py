"""Atlas Comp's Comfy Cloud target (atlas_vfx/comp/cloud.py): generate runs on Cloud with standard and LTX nodes only
(the image guide, frames uploaded one by one), credits are estimated first and nothing leaves ws until the artist
confirms, a model Cloud lacks is refused at preflight, and the frame count is checked after. A fake Cloud stands in.

Run: <V141 python> tests\\test_comp_cloud.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TMP = Path(tempfile.mkdtemp(prefix="cloud_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

import test_comp_queue as tq  # noqa: E402  (its graphs and fake ComfyUI)

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


class FakeWs(tq.FakeClient):
    """ws: pre writes the reference frames (SaveImage) as well as the latent."""

    def wait(self, pid, cancelled=lambda: False):
        h = super().wait(pid, cancelled)
        g = self.submitted[int(pid.rsplit("-", 1)[1]) - 1]
        for nid, node in g.items():
            if node["class_type"] == "SaveImage":
                h["outputs"][nid] = {"images": [{"filename": f"ref_{i:05d}_.png", "subfolder": "atlas_comp",
                                                 "type": "output"} for i in range(17)]}
        return h

    def download(self, info):
        return b"png" if info["filename"].endswith(".png") else b"sampled"


class FakeCloud(tq.FakeClient):
    kind = "comfy_cloud"

    def __init__(self, info):
        super().__init__("cloud", info=info)


def cloud_info(lora=True):
    info = tq.all_info()
    info.pop("AtlasVFXGuideFromLatent", None)
    for k in [k for k in info if k.startswith("AtlasVFX")]:
        info.pop(k)
    info.update({"LoadImage": {"input": {"required": {}}}, "ImageBatch": {"input": {"required": {}}},
                 "LTXAddVideoICLoRAGuide": {"input": {"required": {}}}})
    info["LTXICLoRALoaderModelOnly"] = {"input": {"required": {"lora_name": [[
        "ltx-2.3-22b-ic-lora-relight-1.0.safetensors"] if lora else ["something-else.safetensors"]]}}}
    return info


MAP = {"ltx23\\ltx-2.3-22b-ic-lora-relight-1.0.safetensors": "ltx-2.3-22b-ic-lora-relight-1.0.safetensors"}


def test_graphs():
    from atlas_vfx.comp import cloud

    r, g = tq.relight_graph(TMP / "out_g")
    pre, gen, post, leaks = cloud.cloud_graphs(g, r, [f"f{i}.png" for i in range(17)], "res.latent", MAP)
    kinds = {n["class_type"] for n in gen.values()}
    check(not [k for k in kinds if k.startswith("AtlasVFX")] and "LTXAddVideoICLoRAGuide" in kinds and tq.links_ok(gen),
          "graphs: Cloud's generate has no atlas node; the guide is LTX's own image guide", str(sorted(kinds))[:140])
    loads = sorted(n["inputs"]["image"] for n in gen.values() if n["class_type"] == "LoadImage")
    check(len(loads) == 17 and sum(n["class_type"] == "ImageBatch" for n in gen.values()) == 16,
          "graphs: the 17 uploaded frames are loaded and batched with core nodes", f"{len(loads)} loads")
    loras = [n["inputs"]["lora_name"] for n in gen.values() if "lora_name" in n["inputs"]]
    check(loras == ["ltx-2.3-22b-ic-lora-relight-1.0.safetensors"], "graphs: model names follow the target's map",
          str(loras))
    saves = [n for n in pre.values() if n["class_type"] == "SaveImage"]
    check(len(saves) == 1 and tq.links_ok(pre), "graphs: pre saves the reference frames on ws")


def test_estimate():
    from atlas_vfx.comp import cloud

    e9, e97 = cloud.estimate(9), cloud.estimate(97)
    check(0 < e9["credits_low"] < e9["credits_high"] < e97["credits_low"] * 4 and e97["credits_high"] > e9["credits_high"],
          "estimate: a range in credits that grows with the frames", f"{e9} {e97}")
    check("rough" in e97["note"], "estimate: says it is rough (Cloud has no estimate endpoint)")


def run(cloud_client, confirm=None, frames=17):
    from atlas_vfx.comp import cloud
    from atlas_vfx.comp import queue as qu

    r, g = tq.relight_graph(TMP / f"out_{time.time_ns()}")
    out = [n for n in g.values() if n["class_type"] == "AtlasVFXLTXReturn"][0]["inputs"]["out_dir"]
    local = FakeWs("ws", frames=frames)
    job = qu.new_job("S", "rl", "preview", "cloud", expected=17, out_dir=out)
    t = threading.Thread(target=cloud.run_cloud_job, args=(job, g, r, local, cloud_client, MAP), daemon=True)
    t.start()
    t0 = time.time()
    while time.time() - t0 < 5 and job["status"] not in ("confirm", "done", "failed", "cancelled"):
        time.sleep(0.02)
    seen = dict(job), list(cloud_client.uploads), list(local.submitted)
    if confirm is True:
        job["confirmed"] = True
    elif confirm is False:
        job["cancel"] = True
    t.join(5)
    return job, local, seen


def test_confirm_first():
    job, local, (at_confirm, uploads, submitted) = run(FakeCloud(cloud_info()), confirm=True)
    check(at_confirm["status"] == "confirm" and at_confirm.get("estimate") and not uploads and not submitted,
          "confirm: the job waits with its estimate; nothing uploaded or run before the yes",
          f"{at_confirm['status']} {uploads} {len(submitted)}")
    check(job["status"] == "done" and job["frames_back"] == 17, "confirm: after the yes it runs to done",
          f"{job['status']} {job['message']}")
    job, local, _ = run(FakeCloud(cloud_info()), confirm=False)
    check(job["status"] == "cancelled" and not local.submitted, "confirm: a no cancels it and nothing ran",
          job["status"])


def test_refused_and_short():
    c = FakeCloud(cloud_info(lora=False))
    job, local, _ = run(c)
    check(job["status"] == "failed" and "relight" in job["message"] and not c.uploads,
          "preflight: a LoRA Cloud lacks is refused before any estimate or upload", job["message"][:120])
    job, _, _ = run(FakeCloud(cloud_info()), confirm=True, frames=12)
    check(job["status"] == "failed" and "12 of 17" in job["message"], "check: a short result fails the job",
          job["message"])


def test_client_redirect():
    """Cloud's /api/view answers 302 to a signed storage URL: the key goes to Cloud only, never to storage."""
    from atlas_vfx.comp import cloud

    seen = []

    def opener(req, timeout=0):
        seen.append((req.full_url, req.get_header("X-api-key")))
        if "/api/view" in req.full_url:
            raise cloud.Redirect("https://storage.example/signed?x=1")
        return b"bytes"
    c = cloud.CloudClient("k123", open_fn=opener)
    data = c.download({"filename": "a.latent", "subfolder": "atlas_comp", "type": "output"})
    check(data == b"bytes" and seen[0][1] == "k123" and seen[1] == ("https://storage.example/signed?x=1", None),
          "client: the download follows the redirect without the key", str(seen))
    check("k123" not in repr(c) and "k123" not in str(cloud.public_state(c)), "client: the key is never shown")


def main():
    for f in (test_graphs, test_estimate, test_confirm_first, test_refused_and_short, test_client_redirect):
        f()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
