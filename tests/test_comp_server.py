"""Atlas Comp's routes (atlas_vfx/comp/server.py) on a test aiohttp server: shots, layers, recipes, graphs, frames,
the app's static files, and what a hostile request gets.

Run: <V141 python> tests\\test_comp_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="comp_srv_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import plate as pl  # noqa: E402

pl.WORK_ROOT = TMP / "work"
pl.WORK_ROOT.mkdir()

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def seq(folder: Path, stem: str, frames) -> str:
    from atlas_vfx.camera_side import exr_io

    for n in frames:
        img = np.zeros((16, 32, 4), np.float32)
        img[..., :3], img[..., 3] = 0.3, 1.0
        exr_io.write_rgba(folder / f"{stem}.{n}.exr", img)
    return str(folder / f"{stem}.####.exr")


async def run():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from atlas_vfx.comp import server as srv

    app = web.Application()
    app.add_routes(srv.build_routes())
    async with TestClient(TestServer(app)) as c:
        r = await c.get("/atlas")
        check(r.status == 200 and "text/html" in r.headers["Content-Type"], "app: /atlas serves the page", r.status)
        for bad in ("/atlas/app/..%2Fatlas_vfx%2F__init__.py", "/atlas/app/x.py", "/atlas/app/..\\README.md"):
            r = await c.get(bad)
            check(r.status == 404, f"app: {bad} is not served", r.status)

        r = await c.get("/atlas/api/shots")
        check(r.status == 200 and (await r.json())["shots"] == [], "shots: empty at first")
        r = await c.post("/atlas/api/shots", json={"name": "S1", "plate": "", "first": 1130, "last": 1226})
        check(r.status == 200 and (await r.json())["name"] == "S1", "shots: created", r.status)
        r = await c.post("/atlas/api/shots", json={"name": "../x", "plate": "", "first": 1, "last": 2})
        body = await r.json()
        check(r.status == 400 and "shot name" in body["error"], "shots: a hostile name is a 400", body)
        r = await c.post("/atlas/api/shots", data=b"not json", headers={"Content-Type": "application/json"})
        check(r.status == 400, "shots: a body that is not JSON is a 400", r.status)

        pat = seq(pl.WORK_ROOT / "renders" / "el", "el", [1130])
        r = await c.post("/atlas/api/shots/S1/layers", json={"layer": "el", "kind": "sequence", "source": pat})
        check(r.status == 200, "layers: added", await r.text())
        seq(pl.WORK_ROOT / "renders" / "el", "el", [1131, 1132, 1140])
        r = await c.get("/atlas/api/shots/S1")
        el = [x for x in (await r.json())["layers"] if x["name"] == "el"][0]
        check(el["frames"] == [[1130, 1132], [1140, 1140]], "shots: each layer's frame ranges for the timeline",
              str(el.get("frames")))
        r = await c.patch("/atlas/api/shots/S1/layers/el", json={"opacity": 0.5})
        check(r.status == 200 and (await r.json())["opacity"] == 0.5, "layers: changed")
        r = await c.post("/atlas/api/shots/S1/layers", json={"layer": "rl", "kind": "recipe", "recipe": "relight",
                                                              "params": {"element": "el"}})
        check(r.status == 200, "layers: a recipe layer", await r.text())

        r = await c.get("/atlas/api/recipes")
        names = {x["name"] for x in (await r.json())["recipes"]}
        check(names == {"matte", "relight", "sky"}, "recipes: listed", str(names))

        r = await c.post("/atlas/api/shots/S1/layers/rl/graph", json={"mode": "render",
                                                                       "params": {"direction": "../x"}})
        body = await r.json()
        check(r.status == 400 and "direction" in body["error"], "graph: a bad param is a 400 naming it", body)
        vs = []
        for _ in range(2):
            r = await c.post("/atlas/api/shots/S1/layers/rl/graph", json={"mode": "render"})
            body = await r.json()
            vs.append(body.get("version"))
        check(vs == ["v001", "v002"] and "57" in body["prompt"], "graph: two renders, v001 then v002", str(vs))
        r = await c.post("/atlas/api/shots/S1/layers/rl/graph", json={"mode": "preview"})
        body = await r.json()
        check(r.status == 200 and body["version"] == "preview" and "57" not in body["prompt"],
              "graph: a preview has no 4K transfer and no version", str(body.get("version")))

        r = await c.get("/atlas/api/shots/S1/frame", params={"f": "1130", "w": "640"})
        data = await r.read()
        check(r.status == 200 and r.headers["Content-Type"] == "image/jpeg" and data[:2] == bytes([0xFF, 0xD8]),
              "frame: a JPEG", r.status)
        r = await c.get("/atlas/api/shots/S1/frame", params={"f": "abc"})
        check(r.status == 400, "frame: a bad frame number is a 400", r.status)
        r = await c.get("/atlas/api/presets")
        ps = (await r.json())["presets"]
        check(r.status == 200 and any(p["group"] == "Right sun" for p in ps), "presets: listed with the Right sun group")
        right_low = [p for p in ps if p["id"] == "right-low"][0]
        r = await c.post("/atlas/api/shots/S1/layers", json={"layer": "right_low", "kind": "recipe",
                                                              "recipe": right_low["recipe"],
                                                              "params": {**right_low["params"], "element": "el"}})
        check(r.status == 200 and (await r.json())["params"]["direction"] == "the right",
              "presets: a card adds a relight layer with its settings", r.status)
        r = await c.get("/atlas/api/targets")
        ts = (await r.json())["targets"]
        check(r.status == 200 and ts[0]["name"] == "ws" and all("url" not in t for t in ts),
              "queue: targets listed without their addresses", str(ts))
        r = await c.post("/atlas/api/queue", json={"shot": "S1", "layer": "rl", "mode": "preview", "target": "nope"})
        check(r.status == 400 and "nope" in (await r.json())["error"], "queue: an unknown target is a 400", r.status)
        from atlas_vfx.comp import queue as qu

        class Local:
            def object_info(self): return {}
            def upload(self, name, data): return name
            def submit(self, g): return "p1"
            def wait(self, pid, cancelled=lambda: False): return {"outputs": {}}
            def download(self, info): return b""
        srv.set_queue(qu.Queue(local_client=Local(), client_for=lambda n: Local(), on_done=srv.finish_job))
        r = await c.post("/atlas/api/queue", json={"shot": "S1", "layer": "rl", "mode": "preview", "target": "ws"})
        job = await r.json()
        check(r.status == 200 and job["target"] == "ws" and job["expected"] == 17, "queue: a job is queued",
              str({k: job.get(k) for k in ("target", "expected", "status")}))
        for _ in range(50):
            await asyncio.sleep(0.05)
            js = (await (await c.get("/atlas/api/queue")).json())["jobs"]
            if js and js[-1]["status"] in ("done", "failed"):
                break
        check(js[-1]["status"] == "failed" and "0 of 17" in js[-1]["message"] and "out_dir" not in js[-1],
              "queue: a job whose frames never arrive fails, listed without paths", js[-1]["message"])
        r = await c.post(f"/atlas/api/queue/{job['id']}/cancel")
        check(r.status == 200, "queue: cancel answers", r.status)
        r = await c.post(f"/atlas/api/queue/{job['id']}/confirm")
        check(r.status == 400 and "confirm" in (await r.json())["error"],
              "queue: only a job waiting for its yes can be confirmed", r.status)
        srv.set_queue(None)
        r = await c.get("/atlas/api/queue", headers={"Host": "127.0.0.1:5000"})
        check(r.status == 200 and srv._QUEUE.local.url.endswith(f":{srv._comfy_port()}")
              and not srv._QUEUE.local.url.endswith(":5000"),
              "queue: ws's address is ComfyUI's own port, never the request's Host header", srv._QUEUE.local.url)
        srv.set_queue(qu.Queue(local_client=Local(), client_for=lambda n: Local(), on_done=srv.finish_job))
        (pl.WORK_ROOT / "atlas_comp_targets.local.json").write_text(json.dumps({"targets": [
            {"name": "cloud", "kind": "comfy_cloud", "models": {"a.safetensors": "b.safetensors"}}]}))
        os.environ.pop("COMFY_CLOUD_API_KEY", None)
        try:
            srv._client_for("cloud")
            check(False, "cloud: without COMFY_CLOUD_API_KEY the target says so")
        except ValueError as e:
            check("COMFY_CLOUD_API_KEY" in str(e), "cloud: without COMFY_CLOUD_API_KEY the target says so", str(e))
        os.environ["COMFY_CLOUD_API_KEY"] = "k-test"
        cc = srv._client_for("cloud")
        check(cc.kind == "comfy_cloud" and cc.model_map == {"a.safetensors": "b.safetensors"},
              "cloud: the client carries the target's model map")
        os.environ.pop("COMFY_CLOUD_API_KEY", None)
        r = await c.get("/atlas/api/targets")
        check(all("models" not in t for t in (await r.json())["targets"]), "cloud: the model map is not listed")
        (pl.WORK_ROOT / "atlas_comp_targets.local.json").unlink()
        r = await c.get("/atlas/api/views")
        check(r.status == 200 and len((await r.json())["views"]) > 3, "views: listed")
        bad = pl.WORK_ROOT / "bad.mov"
        bad.write_bytes(b"not a movie")
        r = await c.post("/atlas/api/shots", json={"name": "BAD", "plate": str(bad), "first": 1130, "last": 1131})
        state = None
        for _ in range(100):
            await asyncio.sleep(0.1)
            d = await (await c.get("/atlas/api/shots/BAD")).json()
            state = d["plate_proxy"]
            if state == "error":
                break
        check(state == "error" and d.get("plate_proxy_error"), "proxy: a plate that will not decode shows an error",
              f"{state} {str(d.get('plate_proxy_error'))[:60]}")
        await asyncio.sleep(0.3)
        d = await (await c.get("/atlas/api/shots/BAD")).json()
        check(d["plate_proxy"] == "error", "proxy: a failed build is not retried on its own", d["plate_proxy"])
        r = await c.post("/atlas/api/shots/BAD/plate_proxy")
        check(r.status == 200 and (await r.json())["plate_proxy"] in ("building", "error"),
              "proxy: the retry route starts a new build", r.status)
        r = await c.get("/atlas/api/shots/BAD/frame", params={"f": "1130", "w": "128"})
        check(r.status == 200, "frame: a shot whose plate failed still draws", r.status)
        r = await c.get("/atlas/api/shots/nope")
        check(r.status == 400, "shots: an unknown shot is a 400", r.status)


def main():
    asyncio.run(run())
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
