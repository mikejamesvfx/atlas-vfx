"""Atlas Comp render queue (atlas_vfx/comp/queue.py): a recipe job split into pre (local), generate (the target) and
post (local); preflight before anything is sent; the frame count checked after; one target down never stops the
queue. Fake ComfyUI clients stand in for ws and the farm.

Run: <V141 python> tests\\test_comp_queue.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="queue_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def links_ok(g):
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) and str(v[0]) not in g:
                return False
    return True


def relight_graph(out_dir):
    from atlas_vfx.comp import recipes as rc

    r = rc.load_recipe("relight")
    ctx = {"plate": "", "first": 1130, "last": 1146, "frame_offset": 1001, "first_frame": 1130, "length": 17,
           "plate_first": 129, "plate_last": 145, "layer": "rl", "cache_dir": str(TMP / "cache"),
           "layer_patterns": {"el": str(TMP / "el" / "el.####.exr")}}
    return r, rc.fill(r, rc.validate(r, {"element": "el"}), ctx, str(out_dir), "preview")


def matte_graph(out_dir):
    from atlas_vfx.comp import recipes as rc

    r = rc.load_recipe("matte")
    ctx = {"plate": "", "first": 1130, "last": 1146, "frame_offset": 1001, "first_frame": 1130, "length": 17,
           "plate_first": 129, "plate_last": 145, "layer": "mt", "cache_dir": str(TMP / "cache_m"),
           "layer_patterns": {"el": str(TMP / "el" / "el.####.exr")}}
    return r, rc.fill(r, rc.validate(r, {"element": "el"}), ctx, str(out_dir), "preview")


def test_split():
    from atlas_vfx.comp import queue as qu

    r, g = relight_graph(TMP / "out")
    pre, gen, post, leaks = qu.split_graph(g, r["stages"], "job1_guide.latent", "job1_result.latent")
    kinds = lambda x: {n["class_type"] for n in x.values()}  # noqa: E731
    check("AtlasVFXLatentPrerender" in kinds(pre) and "SamplerCustomAdvanced" not in kinds(pre) and links_ok(pre),
          "split: pre ends at the cached latent, no sampling", str(sorted(kinds(pre)))[:120])
    check({"LoadLatent", "SaveLatent", "SamplerCustomAdvanced", "AtlasVFXGuideFromLatent"} <= kinds(gen)
          and "AtlasVFXLatentPrerender" not in kinds(gen) and "AtlasVFXLTXPrep" not in kinds(gen) and links_ok(gen),
          "split: generate loads the guide, samples, saves the latent; no local-only nodes", str(sorted(kinds(gen)))[:160])
    atlas = sorted(k for k in kinds(gen) if k.startswith("AtlasVFX"))
    check(atlas == ["AtlasVFXGuideFromLatent"], "split: Shot Range and Sun Direction are resolved on ws, so generate "
          "needs only Guide From Latent from atlas-vfx", str(atlas))
    texts = [n["inputs"]["text"] for n in gen.values() if n["class_type"] == "CLIPTextEncode"]
    check(any("relight the video to match the light-direction ball" in t for t in texts if isinstance(t, str)),
          "split: the prompt reaches generate as plain text", str(texts)[:100])
    loads = [n["inputs"]["latent"] for n in gen.values() if n["class_type"] == "LoadLatent"]
    check(loads == ["job1_guide.latent"], "split: generate's guide is the uploaded latent", str(loads))
    check("SamplerCustomAdvanced" not in kinds(post) and "LoadLatent" in kinds(post) and "AtlasVFXLTXReturn" in kinds(post)
          and links_ok(post), "split: post decodes the returned latent and runs Return, no sampling",
          str(sorted(kinds(post)))[:160])
    check(leaks == {}, "split: relight's generate needs no value from ws's own nodes", str(leaks))


def test_split_leaks():
    """Matte's empty latent takes length, width and height from Alpha Prep, a ws node reading ws files. Pre reports
    them (core PreviewAny) and generate gets them as plain numbers, so the target never runs Alpha Prep."""
    from atlas_vfx.comp import queue as qu

    r, g = matte_graph(TMP / "out_m")
    pre, gen, post, leaks = qu.split_graph(g, r["stages"], "m_guide.latent", "m_result.latent")
    atlas = sorted({n["class_type"] for n in gen.values() if n["class_type"].startswith("AtlasVFX")})
    check(atlas == ["AtlasVFXGuideFromLatent"] and links_ok(gen),
          "leaks: matte's generate holds no ws-only node (Alpha Prep, Sequence Read)", str(atlas))
    probes = {k: n["inputs"]["source"] for k, n in pre.items() if n["class_type"] == "PreviewAny"}
    check(sorted(map(tuple, probes.values())) == [("2", 1), ("2", 2), ("2", 3)] and set(probes) == set(leaks),
          "leaks: pre reports Alpha Prep's length, width and height", str(probes))
    hist = {k: {"text": [{1: "9", 2: "1920", 3: "1088"}[v[1]]]} for k, v in probes.items()}
    qu.fill_leaks(gen, leaks, hist)
    e = gen["42"]["inputs"]
    check((e["width"], e["height"], e["length"], gen["44"]["inputs"]["frames_number"]) == (1920, 1088, 9, 9),
          "leaks: generate gets them as numbers", str((e["width"], e["height"], e["length"])))
    try:
        qu.fill_leaks(gen, leaks, {})
        check(False, "leaks: a value pre never reported fails the job")
    except ValueError as ex:
        check("did not report" in str(ex), "leaks: a value pre never reported fails the job", str(ex)[:80])


class FakeClient:
    """A ComfyUI that runs nothing but remembers what it was given, and writes frames on a post graph."""

    def __init__(self, name, info=None, frames=17, down=False):
        self.name, self.info, self.frames, self.down = name, info, frames, down
        self.submitted, self.uploads = [], []

    def object_info(self):
        if self.down:
            raise OSError("connection refused")
        return self.info

    def upload(self, name, data):
        if self.down:
            raise OSError("connection refused")
        self.uploads.append(name)
        return name

    def submit(self, graph):
        if self.down:
            raise OSError("connection refused")
        self.submitted.append(graph)
        for node in graph.values():
            if node["class_type"] == "AtlasVFXLatentPrerender":
                out = Path(node["inputs"]["out_dir"])
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{node['inputs']['name']}.latent").write_bytes(b"latent")
            if node["class_type"] == "AtlasVFXLTXReturn":
                out = Path(node["inputs"]["out_dir"])
                out.mkdir(parents=True, exist_ok=True)
                for i in range(self.frames):
                    (out / f"rl.{1130 + i}.exr").write_bytes(b"exr")
        return f"{self.name}-{len(self.submitted)}"

    def wait(self, pid, cancelled=lambda: False):
        if self.down:
            raise OSError("connection lost")
        g = self.submitted[int(pid.rsplit("-", 1)[1]) - 1]
        outs = {}
        for nid, node in g.items():
            if node["class_type"] == "PreviewAny":
                outs[nid] = {"text": [{1: "9", 2: "1920", 3: "1088"}.get(node["inputs"]["source"][1], "0")]}
            if node["class_type"] == "SaveLatent":
                outs[nid] = {"latents": [{"filename": "atlas_00001_.latent", "subfolder": "atlas_comp", "type": "output"}]}
        return {"outputs": outs}

    def download(self, info):
        return b"sampled"


def all_info():
    from atlas_vfx.comp import recipes as rc

    g = rc.load_recipe("relight")["graph"]
    info = {n["class_type"]: {"input": {"required": {}}} for n in g.values()}
    info.update({"LoadLatent": {"input": {"required": {}}}, "SaveLatent": {"input": {"required": {}}}})
    return info


def test_remote_job():
    from atlas_vfx.comp import queue as qu

    r, g = relight_graph(TMP / "out_remote")
    local, remote = FakeClient("ws"), FakeClient("render", info=all_info())
    job = qu.new_job("S", "rl", "preview", "render", expected=17, out_dir=str(TMP / "out_remote"))
    qu.run_job(job, g, r, local, remote, remote_is_local=False)
    check(job["status"] == "done" and job["frames_back"] == 17, "remote job: done with all frames back",
          f"{job['status']} {job.get('message')}")
    check(len(local.submitted) == 2 and len(remote.submitted) == 1, "remote job: pre and post on ws, generate on the target")
    check(remote.uploads == [f"{job['id']}_guide.latent"] and local.uploads == [f"{job['id']}_result.latent"],
          "remote job: the guide latent goes out, the sampled latent comes back", f"{remote.uploads} {local.uploads}")
    check([s for s, _ in job["log"]] == ["preflight", "pre", "upload", "generate", "download", "post", "check"],
          "remote job: stages in order", str([s for s, _ in job["log"]]))


def test_refused_and_failed():
    from atlas_vfx.comp import queue as qu

    r, g = relight_graph(TMP / "out_refuse")
    info = all_info()
    del info["AtlasVFXGuideFromLatent"]
    local, remote = FakeClient("ws"), FakeClient("hub", info=info)
    job = qu.new_job("S", "rl", "preview", "hub", expected=17, out_dir=str(TMP / "out_refuse"))
    qu.run_job(job, g, r, local, remote, remote_is_local=False)
    check(job["status"] == "failed" and "AtlasVFXGuideFromLatent" in job["message"] and not remote.uploads
          and not local.submitted, "refused: preflight names the missing node and nothing is sent or run",
          job["message"][:100])
    r, g = relight_graph(TMP / "out_short")
    local, remote = FakeClient("ws", frames=12), FakeClient("render", info=all_info())
    job = qu.new_job("S", "rl", "preview", "render", expected=17, out_dir=str(TMP / "out_short"))
    qu.run_job(job, g, r, local, remote, remote_is_local=False)
    check(job["status"] == "failed" and "12 of 17" in job["message"], "failed: a short result fails the job",
          job["message"])


def test_queue_survives_dead_target():
    from atlas_vfx.comp import queue as qu

    r, g = relight_graph(TMP / "out_q1")
    _, g2 = relight_graph(TMP / "out_q2")
    local = FakeClient("ws")
    clients = {"render": FakeClient("render", info=all_info(), down=True), "hub": FakeClient("hub", info=all_info())}
    q = qu.Queue(local_client=local, client_for=lambda name: clients[name])
    j1 = q.add(qu.new_job("S", "rl", "preview", "render", expected=17, out_dir=str(TMP / "out_q1")), g, r)
    j2 = q.add(qu.new_job("S", "rl", "preview", "hub", expected=17, out_dir=str(TMP / "out_q2")), g2, r)
    t = time.time()
    while time.time() - t < 10 and any(j["status"] not in ("done", "failed", "cancelled") for j in (j1, j2)):
        time.sleep(0.05)
    check(j1["status"] == "failed" and "render" in j1["message"], "queue: a target that is down fails its job",
          j1["message"][:80])
    check(j2["status"] == "done", "queue: the other target's job still runs", j2["status"])
    check([j["id"] for j in q.list()] == [j1["id"], j2["id"]], "queue: jobs listed in order")


def main():
    for f in (test_split, test_split_leaks, test_remote_job, test_refused_and_failed, test_queue_survives_dead_target):
        f()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
