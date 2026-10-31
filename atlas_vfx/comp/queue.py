"""Atlas Comp's render queue: recipe jobs sent to ws, a farm ComfyUI or (later) Comfy Cloud, one job per target at
a time, in the order queued (ADR-021 Amendment 4 draft).

A job on ws runs its whole graph here. A job on another machine is split where the recipe says (`stages`):

  pre       on ws: everything up to Latent Pre-render, which writes the float32 guide latent
  generate  on the target: Load Latent (the uploaded guide) -> the LTX sampling -> Save Latent of the result
  post      on ws: Load Latent (the downloaded result) -> decode -> LTX Return -> the 4K transfer

so only latents travel (a few MB each way) and the plate, the 4K element and the outputs never leave ws. Before
anything is sent the generate graph is preflighted on the target (every node, every model); after post the
expected frames must be on disk, or the job fails and the layer is left as it was."""
from __future__ import annotations

import copy
import itertools
import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import targets as tg

DONE = ("done", "failed", "cancelled")
_ids = itertools.count(1)


def new_job(shot: str, layer: str, mode: str, target: str, expected: int, out_dir: str, version: str = "") -> dict:
    return {"id": f"j{int(time.time())}_{next(_ids)}", "shot": shot, "layer": layer, "mode": mode, "target": target,
            "expected": int(expected), "out_dir": str(out_dir), "version": version, "status": "queued",
            "stage": "queued", "message": "", "frames_back": 0, "log": [], "cancel": False}


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("cancel", "out_dir")} | {"log": [list(x) for x in job["log"]]}


# ---------------------------------------------------------------------------------------------------- splitting
def _refs(node: dict):
    for k, v in node["inputs"].items():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
            yield k, str(v[0]), v[1]


def ancestors(g: dict, roots) -> set[str]:
    seen, todo = set(), [str(r) for r in roots]
    while todo:
        n = todo.pop()
        if n in seen or n not in g:
            continue
        seen.add(n)
        todo += [src for _, src, _ in _refs(g[n])]
    return seen


def sinks(g: dict) -> list[str]:
    used = {src for node in g.values() for _, src, _ in _refs(node)}
    return [n for n in g if n not in used]


def _replace_ref(g: dict, src: str, slot: int, new: list) -> None:
    for node in g.values():
        for k, s, sl in list(_refs(node)):
            if s == src and sl == slot:
                node["inputs"][k] = list(new)


# Atlas nodes that are pure and cheap: worked out on ws and handed to another machine as plain values, so it needs
# no atlas-vfx install for them.
RESOLVABLE = {"AtlasVFXShotRange": ("atlas_vfx.layers.node_shot_range", "AtlasVFXShotRange"),
              "AtlasVFXSunDirection": ("atlas_vfx.layers.node_sun_direction", "AtlasVFXSunDirection")}


def _resolve_pure(g: dict) -> dict:
    """Replace links from RESOLVABLE nodes with their output values, then drop those nodes when nothing still
    links to them. Only literal-input nodes are resolved."""
    import importlib

    for nid, node in list(g.items()):
        spec = RESOLVABLE.get(node["class_type"])
        if spec is None or any(True for _ in _refs(node)):
            continue
        mod = importlib.import_module(spec[0])
        out = getattr(mod, spec[1])().run(**node["inputs"])
        result = out["result"] if isinstance(out, dict) else out
        for other in g.values():
            for k, src, slot in list(_refs(other)):
                if src == nid and isinstance(result[slot], (int, float, str, bool)):
                    other["inputs"][k] = result[slot]
    still = {src for node in g.values() for _, src, _ in _refs(node)}
    return {k: v for k, v in g.items() if not (v["class_type"] in RESOLVABLE and k not in still)}


def split_graph(graph: dict, stages: dict, guide_name: str, result_name: str) -> tuple[dict, dict, dict]:
    """(pre, generate, post) API graphs for a job on another machine. Node ids 9000-9002 are added."""
    pre_id, (gen_id, gen_slot) = str(stages["pre"]), (str(stages["gen_out"][0]), int(stages["gen_out"][1]))
    pre = {k: copy.deepcopy(graph[k]) for k in ancestors(graph, [pre_id])}

    gen = copy.deepcopy(graph)
    gen["9000"] = {"class_type": "LoadLatent", "inputs": {"latent": guide_name}}
    _replace_ref(gen, pre_id, 0, ["9000", 0])
    gen["9001"] = {"class_type": "SaveLatent", "inputs": {"samples": [gen_id, gen_slot],
                                                           "filename_prefix": f"atlas_comp/{Path(result_name).stem}"}}
    gen = _resolve_pure({k: gen[k] for k in ancestors(gen, ["9001"])})

    post = copy.deepcopy(graph)
    post["9002"] = {"class_type": "LoadLatent", "inputs": {"latent": result_name}}
    _replace_ref(post, gen_id, gen_slot, ["9002", 0])
    post = {k: post[k] for k in ancestors(post, [s for s in sinks(graph) if s != gen_id])}
    return pre, gen, post


# ---------------------------------------------------------------------------------------------------- running
def _count_frames(out_dir: str) -> int:
    d = Path(out_dir)
    if not d.is_dir():
        return 0
    for sub in ("", "beauty"):
        n = len(list((d / sub if sub else d).glob("*.exr")))
        if n:
            return n
    return 0


def _log(job: dict, stage: str, text: str = "") -> None:
    job["stage"] = stage
    job["log"].append((stage, text))


def run_job(job: dict, graph: dict, recipe: dict, local, remote, remote_is_local: bool) -> dict:
    """Run one job to done or failed. local/remote: clients with object_info(), upload(name, bytes), submit(graph)
    -> id, wait(id, cancelled) -> history record, download(file info) -> bytes."""
    cancelled = (lambda: job["cancel"])
    job["status"] = "running"
    try:
        if remote_is_local:
            _log(job, "generate", "the whole graph on ws")
            local.wait(local.submit(graph), cancelled)
        else:
            guide, result = f"{job['id']}_guide.latent", f"{job['id']}_result.latent"
            pre, gen, post = split_graph(graph, recipe["stages"], guide, result)
            _log(job, "preflight", job["target"])
            check = tg.preflight({"name": job["target"], "kind": "comfyui", "url": ""}, gen,
                                 fetch=lambda url, timeout=10: remote.object_info())
            if not check["ok"]:
                raise ValueError("; ".join(check["notes"]) + ": not sent")
            _log(job, "pre", "ws")
            local.wait(local.submit(pre), cancelled)
            pre_node = graph[str(recipe["stages"]["pre"])]["inputs"]
            data = (Path(pre_node["out_dir"]) / f"{pre_node['name']}.latent").read_bytes()
            _log(job, "upload", f"{len(data)} bytes")
            gen["9000"]["inputs"]["latent"] = remote.upload(guide, data)
            _log(job, "generate", job["target"])
            hist = remote.wait(remote.submit(gen), cancelled)
            files = hist.get("outputs", {}).get("9001", {}).get("latents", [])
            if not files:
                raise ValueError(f"{job['target']} returned no latent")
            _log(job, "download", files[0].get("filename", ""))
            sampled = remote.download(files[0])
            post["9002"]["inputs"]["latent"] = local.upload(result, sampled)
            _log(job, "post", "ws")
            local.wait(local.submit(post), cancelled)
        _log(job, "check", f"{job['expected']} frames expected")
        job["frames_back"] = _count_frames(job["out_dir"])
        if job["frames_back"] < job["expected"]:
            raise ValueError(f"{job['frames_back']} of {job['expected']} frames came back")
        job["status"], job["message"] = "done", f"{job['frames_back']} frames"
    except Exception as e:  # noqa: BLE001 (a job failing never stops the queue)
        job["status"] = "cancelled" if job["cancel"] else "failed"
        job["message"] = f"{job['target']}: {e}" if isinstance(e, OSError) else str(e)
    return job


class Queue:
    """One worker thread per target; jobs in the order added."""

    def __init__(self, local_client, client_for, on_done=None):
        self.local, self.client_for, self.on_done = local_client, client_for, on_done
        self.jobs: list[dict] = []
        self._pending: dict[str, list] = {}
        self._lock = threading.Lock()
        self._workers: dict[str, threading.Thread] = {}

    def add(self, job: dict, graph: dict, recipe: dict) -> dict:
        with self._lock:
            self.jobs.append(job)
            self._pending.setdefault(job["target"], []).append((job, graph, recipe))
            w = self._workers.get(job["target"])
            if w is None or not w.is_alive():
                w = threading.Thread(target=self._work, args=(job["target"],), daemon=True,
                                     name=f"atlas-comp-queue-{job['target']}")
                self._workers[job["target"]] = w
                w.start()
        return job

    def _work(self, target: str) -> None:
        while True:
            with self._lock:
                todo = self._pending.get(target) or []
                if not todo:
                    return
                job, graph, recipe = todo.pop(0)
            if job["cancel"]:
                job["status"] = "cancelled"
                continue
            remote_is_local = target == "ws"
            try:
                remote = self.local if remote_is_local else self.client_for(target)
            except Exception as e:  # noqa: BLE001
                job["status"], job["message"] = "failed", str(e)
                continue
            run_job(job, graph, recipe, self.local, remote, remote_is_local)
            if self.on_done:
                try:
                    self.on_done(job)
                except Exception as e:  # noqa: BLE001
                    job["message"] += f" (after: {e})"

    def list(self) -> list[dict]:
        return list(self.jobs)

    def cancel(self, job_id: str) -> dict:
        for j in self.jobs:
            if j["id"] == job_id:
                j["cancel"] = True
                if j["status"] == "queued":
                    j["status"] = "cancelled"
                return j
        raise ValueError(f"no job {job_id!r}")


# ---------------------------------------------------------------------------------------------------- HTTP client
class HttpClient:
    """A ComfyUI over HTTP (ws itself or a farm machine)."""

    def __init__(self, url: str, poll: float = 2.0):
        self.url, self.poll, self.client_id = url.rstrip("/"), poll, f"atlas-comp-{uuid.uuid4().hex[:8]}"

    def _get(self, path: str, timeout: float = 30):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return r.read()

    def object_info(self) -> dict:
        return json.loads(self._get("/object_info", 30))

    def upload(self, name: str, data: bytes) -> str:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/upload/image", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["name"]

    def submit(self, graph: dict) -> str:
        req = urllib.request.Request(self.url + "/prompt", json.dumps(
            {"prompt": graph, "client_id": self.client_id}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        if d.get("node_errors"):
            raise ValueError(f"the target refused the graph: {json.dumps(d['node_errors'])[:300]}")
        return d["prompt_id"]

    def wait(self, pid: str, cancelled=lambda: False) -> dict:
        while True:
            if cancelled():
                urllib.request.urlopen(urllib.request.Request(self.url + "/interrupt", b"", method="POST"), timeout=10)
                raise ValueError("cancelled")
            h = json.loads(self._get(f"/history/{pid}"))
            if pid in h:
                st = h[pid]["status"]
                if st.get("status_str") != "success":
                    errs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                    raise ValueError(f"the graph failed: {(errs or ['unknown error'])[0][:300]}")
                return h[pid]
            time.sleep(self.poll)

    def download(self, info: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                    "type": info.get("type", "output")})
        return self._get(f"/view?{q}", 120)
