"""Atlas Comp's Comfy Cloud target. Cloud has LTX and the standard nodes but no atlas-vfx, so its generate stage
uses LTX's own image guide: pre on ws saves the reference frames, each is uploaded, and core LoadImage + ImageBatch
put them back together on Cloud. The sampled latent comes back and post runs on ws, as for a farm job.

Nothing leaves ws before the artist says yes: preflight first (nodes and model names), then a credit estimate, then
the job waits in "confirm" until the app's Confirm (or Cancel). Cloud has no estimate endpoint, so the estimate is a
rough range from GPU time.

Cloud API (docs.comfy.org/development/cloud/api-reference, read 2026-10-31): https://cloud.comfy.org, header
X-API-Key, POST /api/prompt, POST /api/upload/image, GET /api/job/{id}/status, GET /api/jobs/{id} for outputs,
GET /api/view answers 302 to a signed storage URL, fetched WITHOUT the key. The key comes from the environment
(COMFY_CLOUD_API_KEY) and is never written anywhere.

Model names on Cloud differ from ws's folders (an imported LoRA lands where the import put it), so a target can
carry a name map in the local targets file: {"name": "cloud", "kind": "comfy_cloud", "models": {"ws name": "cloud
name"}}."""
from __future__ import annotations

import copy
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import queue as qu
from . import targets as tg

BASE = "https://cloud.comfy.org"
# comfy.org/cloud/pricing (read 2026-10-31): billed by the GPU second; RTX PRO 6000 96 GB is 736.39 credits an hour.
CREDITS_PER_SECOND = 736.39 / 3600
TERMINAL_OK = {"completed", "success"}
TERMINAL_BAD = {"error", "non_retryable_error", "lost", "cancelled", "failed"}
CONFIRM_TIMEOUT = 3600
LOAD0, BATCH0, REF_SAVE = 9200, 9600, "9003"


class Redirect(Exception):
    def __init__(self, location: str):
        super().__init__(location)
        self.location = location


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Redirect(newurl)


def _default_open(req, timeout=60) -> bytes:
    with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------------------------------- graphs
def _map_models(g: dict, model_map: dict) -> None:
    for node in g.values():
        for k, v in node["inputs"].items():
            if k in tg.MODEL_INPUTS and isinstance(v, str) and v in model_map:
                node["inputs"][k] = model_map[v]


def cloud_graphs(graph: dict, recipe: dict, frame_names: list[str], result_name: str, model_map: dict | None = None):
    """(pre, generate, post, leaks) for Cloud: the farm split, plus pre saving the reference frames, and generate
    swapping Guide From Latent for LTX's image guide fed by the uploaded frames."""
    stages = recipe["stages"]
    pre, gen, post, leaks = qu.split_graph(graph, stages, "unused.latent", result_name)
    pre_node = graph[str(stages["pre"])]["inputs"]
    pre[REF_SAVE] = {"class_type": "SaveImage", "inputs": {
        "images": pre_node["images"], "filename_prefix": f"atlas_comp/{Path(result_name).stem}_ref"}}
    pre = {k: pre[k] for k in qu.ancestors(pre, [REF_SAVE, *leaks])}     # Cloud takes frames: no latent encode
    gen = copy.deepcopy(gen)
    gen.pop("9000", None)
    loads = []
    for i, name in enumerate(frame_names):
        gen[str(LOAD0 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        loads.append([str(LOAD0 + i), 0])
    batch = loads[0]
    for i, nxt in enumerate(loads[1:]):
        gen[str(BATCH0 + i)] = {"class_type": "ImageBatch", "inputs": {"image1": batch, "image2": nxt}}
        batch = [str(BATCH0 + i), 0]
    for k, n in list(gen.items()):
        if n["class_type"] == "AtlasVFXGuideFromLatent":
            i = n["inputs"]
            gen[k] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
                "positive": i["positive"], "negative": i["negative"], "vae": i["vae"], "latent": i["latent"],
                "image": batch, "frame_idx": 0, "strength": i["strength"],
                "latent_downscale_factor": i["latent_downscale_factor"], "crop": "disabled",
                "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "use_streaming": False,
                "tile_frames": 97}}
    _map_models(gen, model_map or {})
    return pre, gen, post, leaks


def estimate(frames: int) -> dict:
    lo, hi = 60 + 2 * frames, 120 + 5 * frames
    return {"seconds_low": lo, "seconds_high": hi, "credits_low": round(lo * CREDITS_PER_SECOND, 1),
            "credits_high": round(hi * CREDITS_PER_SECOND, 1), "frames": frames,
            "note": "rough: Cloud has no estimate endpoint; GPU seconds at the RTX PRO 6000 rate"}


# ---------------------------------------------------------------------------------------------------- running
def run_cloud_job(job: dict, graph: dict, recipe: dict, local, cloud, model_map: dict | None = None) -> dict:
    cancelled = (lambda: job["cancel"])
    result = f"{job['id']}_result.latent"
    try:
        _, probe, _, _ = cloud_graphs(graph, recipe, ["probe.png"], result, model_map)
        qu._log(job, "preflight", job["target"])
        check = tg.preflight({"name": job["target"], "kind": "comfyui", "url": ""}, probe,
                             fetch=lambda url, timeout=10: cloud.object_info())
        if not check["ok"]:
            raise ValueError("; ".join(check["notes"]) + ": not sent")
        job["estimate"] = estimate(job["expected"])
        job["status"] = "confirm"
        qu._log(job, "confirm", f"about {job['estimate']['credits_low']}-{job['estimate']['credits_high']} credits")
        t0 = time.time()
        while not job.get("confirmed"):
            if job["cancel"] or time.time() - t0 > CONFIRM_TIMEOUT:
                job["cancel"] = True
                raise ValueError("not confirmed: nothing sent")
            time.sleep(0.05)
        job["status"] = "running"
        t0 = time.time() - 1
        qu._log(job, "pre", "ws")
        pre, _, post, leaks = cloud_graphs(graph, recipe, ["probe.png"], result, model_map)
        outs = local.wait(local.submit(pre), cancelled).get("outputs", {})
        frames = (outs.get(REF_SAVE) or {}).get("images") or []
        if not frames:
            raise ValueError("pre saved no reference frames")
        names = []
        for i, f in enumerate(frames):
            names.append(cloud.upload(f"{job['id']}_ref_{i:04d}.png", local.download(f)))
        qu._log(job, "upload", f"{len(names)} reference frames to Comfy Cloud")
        _, gen, _, _ = cloud_graphs(graph, recipe, names, result, model_map)
        qu.fill_leaks(gen, leaks, outs)
        qu._log(job, "generate", job["target"])
        hist = cloud.wait(cloud.submit(gen), cancelled)
        files = hist.get("outputs", {}).get("9001", {}).get("latents", [])
        if not files:
            raise ValueError(f"{job['target']} returned no latent")
        qu._log(job, "download", files[0].get("filename", ""))
        post["9002"]["inputs"]["latent"] = local.upload(result, cloud.download(files[0]))
        qu._log(job, "post", "ws")
        local.wait(local.submit(post), cancelled)
        qu._log(job, "check", f"{job['expected']} frames expected")
        job["frames_back"] = qu._count_frames(job["out_dir"], t0)
        if job["frames_back"] < job["expected"]:
            raise ValueError(f"{job['frames_back']} of {job['expected']} frames came back")
        job["status"], job["message"] = "done", f"{job['frames_back']} frames"
    except Exception as e:  # noqa: BLE001 (a job failing never stops the queue)
        job["status"] = "cancelled" if job["cancel"] else "failed"
        job["message"] = f"{job['target']}: {e}" if isinstance(e, OSError) else str(e)
    return job


# ---------------------------------------------------------------------------------------------------- HTTP client
class CloudClient:
    kind = "comfy_cloud"

    def __init__(self, key: str, base: str = BASE, poll: float = 3.0, model_map: dict | None = None, open_fn=None):
        self._key, self.base, self.poll = key, base.rstrip("/"), poll
        self.model_map = dict(model_map or {})
        self._open = open_fn or _default_open

    def __repr__(self):
        return f"CloudClient({self.base})"

    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None, method=None):
        return urllib.request.Request(self.base + path, data, {"X-API-Key": self._key, **(headers or {})},
                                      method=method)

    def _json(self, path: str, body=None, timeout=60):
        data = None if body is None else json.dumps(body).encode()
        hdr = {} if body is None else {"Content-Type": "application/json"}
        return json.loads(self._open(self._req(path, data, hdr), timeout=timeout))

    def object_info(self) -> dict:
        return self._json("/api/object_info")

    def upload(self, name: str, data: bytes) -> str:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        r = json.loads(self._open(self._req("/api/upload/image", body,
                                            {"Content-Type": f"multipart/form-data; boundary={boundary}"}), 300))
        return r["name"]

    def submit(self, graph: dict) -> str:
        r = self._json("/api/prompt", {"prompt": graph})
        if r.get("node_errors"):
            raise ValueError(f"Comfy Cloud refused the graph: {json.dumps(r['node_errors'])[:300]}")
        return r["prompt_id"]

    def wait(self, pid: str, cancelled=lambda: False) -> dict:
        q = urllib.parse.quote(pid)
        while True:
            if cancelled():
                self._json("/api/queue", {"delete": [pid]})
                raise ValueError("cancelled")
            st = self._json(f"/api/job/{q}/status")
            s = str(st.get("status", "")).lower()
            if s in TERMINAL_BAD:
                raise ValueError(f"Comfy Cloud: {s}: {str(st.get('error_message') or '')[:300]}")
            if s in TERMINAL_OK:
                return self._json(f"/api/jobs/{q}")
            time.sleep(self.poll)

    def download(self, info: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                    "type": info.get("type", "output")})
        try:
            return self._open(self._req(f"/api/view?{q}"), timeout=300)
        except Redirect as r:          # signed storage URL: fetched without the key
            return self._open(urllib.request.Request(r.location), timeout=300)


def public_state(c: CloudClient) -> dict:
    return {"base": c.base, "key": "set" if c._key else "missing", "models_mapped": len(c.model_map)}
