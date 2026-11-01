"""Atlas Comp's routes on ComfyUI's own server (same process, same port): the app page at /atlas and a small JSON
API under /atlas/api. Every request is untrusted (ComfyUI's API has no auth): names, paths and params are checked in
shot.py / recipes.py, a ValueError becomes a 400 with its message, the app's static files are served only from app/
by plain name. Heavy work (frames, plate proxies) runs off the event loop.

Queueing is the browser's: it asks here for a filled graph, then posts it to ComfyUI's /prompt with its client_id,
so live previews and progress arrive on ComfyUI's websocket."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path

from atlas_vfx.camera_side import plate as pl

from . import bake as bk
from . import cloud
from . import frames as fr
from . import queue as qu
from . import recipes as rc
from . import shot as sh
from . import targets as tg

log = logging.getLogger("atlas_vfx.comp")
APP = Path(__file__).resolve().parents[2] / "app"
STATIC = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_building = fr.BUILDING
_lock = threading.Lock()


def _proxy_state(shot: dict) -> str:
    return fr.proxy_state(shot)[0]


def _start_proxy(shot: dict) -> None:
    """Build the plate proxy in a background thread (once per shot at a time)."""
    with _lock:
        if not shot.get("plate") or shot["name"] in _building or fr.proxy_dir(shot).is_dir():
            return
        _building.add(shot["name"])
        fr.FAILED.pop(shot["name"], None)

    def work():
        try:
            fr.build_plate_proxy(shot)
        except Exception as e:  # noqa: BLE001 (remembered in fr.FAILED and shown in the app; never retried alone)
            log.warning("Atlas Comp: plate proxy for %s failed: %s", shot["name"], str(e)[:300])
        finally:
            _building.discard(shot["name"])
    threading.Thread(target=work, daemon=True, name=f"atlas-comp-proxy-{shot['name']}").start()


def _ranges(frames) -> list[list[int]]:
    out = []
    for n in sorted(frames):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def _shot_json(name: str) -> dict:
    """The shot as the app sees it: plus the plate proxy state and, per layer, the frame ranges it has on disk."""
    from atlas_vfx.layers import chunks as ch

    s = sh.load_shot(name)
    s["plate_proxy"], s["plate_proxy_error"] = fr.proxy_state(s)
    for lay in s["layers"]:
        pattern = sh.layer_pattern(s, lay)
        try:
            lay["frames"] = _ranges(ch.scan(pattern)) if pattern else []
        except ValueError:
            lay["frames"] = []
    return s


def _int(q, key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(q.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key}: a whole number") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key}: {lo}-{hi}")
    return v


def _float(q, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(q.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key}: a number") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key}: {lo}-{hi}")
    return v


def _names(q, key: str):
    v = q.get(key)
    if v is None or v == "":
        return None
    return [pl.safe_name(x, key) if x != "plate" else x for x in v.split(",") if x]


def _job_graph(name: str, layer: str, mode: str, extra_params=None) -> dict:
    """A recipe layer's filled graph for one run: Render takes the next version, Preview the preview folder.
    Params are validated before any folder is made."""
    s = sh.load_shot(name)
    lay = [x for x in s["layers"] if x["name"] == layer]
    if not lay or lay[0]["kind"] != "recipe":
        raise ValueError(f"{layer!r} is not a recipe layer")
    lay = lay[0]
    recipe = rc.load_recipe(lay["recipe"])
    params = rc.validate(recipe, {**lay.get("params", {}), **(extra_params or {})})
    ctx = rc.context_for(s, lay)
    if mode == "preview":
        rc.fill(recipe, params, ctx, str(sh.preview_dir(name, layer)), mode)
        version, out = sh.PREVIEW, sh.preview_dir(name, layer)
        out.mkdir(parents=True, exist_ok=True)
    else:
        rc.fill(recipe, params, ctx, str(sh.layer_dir(name, layer) / "v000"), mode)
        version = sh.next_version(name, layer)
        out = sh.layer_dir(name, layer) / version
    frames = recipe.get("preview", {}).get("frames")
    expected = min(int(frames), ctx["length"]) if (mode == "preview" and frames) else ctx["length"]
    return {"shot": s["name"], "layer": layer, "graph": rc.fill(recipe, params, ctx, str(out), mode),
            "version": version, "out": out, "recipe": recipe, "expected": expected}


_QUEUE = None


def set_queue(q) -> None:
    global _QUEUE
    _QUEUE = q


def finish_job(job: dict) -> None:
    """A finished job switches its layer to what it wrote; a failed one leaves the layer alone."""
    if job["status"] == "done" and job.get("version"):
        try:
            sh.update_layer(job["shot"], job["layer"], version=job["version"])
        except ValueError as e:
            job["message"] += f" (layer not switched: {e})"


def _client_for(name: str):
    t = next((t for t in tg.load_targets() if t["name"] == name), None)
    if t is None:
        raise ValueError(f"no target {name!r}")
    if t["kind"] == "comfy_cloud":
        key = os.environ.get("COMFY_CLOUD_API_KEY", "")
        if not key:
            raise ValueError(f"{name}: set COMFY_CLOUD_API_KEY (your Comfy account's API key) and restart ComfyUI")
        return cloud.CloudClient(key, model_map=t.get("models", {}))
    return qu.HttpClient(t["url"])


def _comfy_port() -> int:
    """This ComfyUI's own port, never one taken from a request's Host header."""
    try:
        from server import PromptServer
        return int(PromptServer.instance.port)
    except Exception:  # noqa: BLE001 (outside ComfyUI: tests)
        try:
            from comfy.cli_args import args
            return int(args.port)
        except Exception:  # noqa: BLE001
            return 8188


def _queue(request):
    global _QUEUE
    if _QUEUE is None:
        port = _comfy_port()
        _QUEUE = qu.Queue(local_client=qu.HttpClient(f"http://127.0.0.1:{port}"), client_for=_client_for,
                          on_done=finish_job)
    return _QUEUE


def build_routes():
    from aiohttp import web

    routes = web.RouteTableDef()

    def guard(fn):
        async def handler(request):
            try:
                return await fn(request)
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            except web.HTTPException:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("Atlas Comp %s %s", request.method, request.path)
                return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        handler.__name__ = fn.__name__
        return handler

    async def body(request) -> dict:
        try:
            data = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("the request body is not JSON") from None
        if not isinstance(data, dict):
            raise ValueError("the request body must be a JSON object")
        return data

    async def off_loop(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    @routes.get("/atlas")
    @guard
    async def page(request):
        return web.FileResponse(APP / "index.html", headers={"Content-Type": "text/html"})

    @routes.get("/atlas/app/{name}")
    @guard
    async def static(request):
        name = request.match_info["name"]
        p = APP / name
        if not _NAME.match(name) or p.suffix not in STATIC or not p.is_file() or p.resolve().parent != APP.resolve():
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Content-Type": STATIC[p.suffix], "Cache-Control": "no-cache"})

    @routes.get("/atlas/api/shots")
    @guard
    async def shots(request):
        return web.json_response({"shots": sh.list_shots()})

    @routes.post("/atlas/api/shots")
    @guard
    async def create(request):
        d = await body(request)
        s = sh.create_shot(d.get("name", ""), d.get("plate", ""), _int(d, "first", 1001, 0, 10 ** 6),
                           _int(d, "last", 1001, 0, 10 ** 6), _int(d, "frame_offset", pl.NUKE_FRAME_OFFSET, 0, 10 ** 6),
                           mode=str(d.get("mode", "plate")), screen=str(d.get("screen", "green")),
                           width=_int(d, "width", 0, 0, 16384), height=_int(d, "height", 0, 0, 16384),
                           fps=str(d.get("fps", "24"))[:16])
        _start_proxy(s)
        return web.json_response(_shot_json(s["name"]))

    @routes.get("/atlas/api/shots/{shot}")
    @guard
    async def get_shot(request):
        s = _shot_json(request.match_info["shot"])
        if s["plate_proxy"] == "missing":
            _start_proxy(s)
            s["plate_proxy"] = _proxy_state(s)
        return web.json_response(s)

    @routes.post("/atlas/api/shots/{shot}/plate_proxy")
    @guard
    async def retry_proxy(request):
        s = sh.load_shot(request.match_info["shot"])
        _start_proxy(s)
        return web.json_response(_shot_json(s["name"]))

    @routes.post("/atlas/api/shots/{shot}/layers")
    @guard
    async def add_layer(request):
        d = await body(request)
        params = d.get("params") or {}
        if d.get("kind") == "recipe":
            rc.validate(rc.load_recipe(d.get("recipe", "")), params)
        lay = sh.add_layer(request.match_info["shot"], d.get("layer", ""), d.get("kind", ""), d.get("source", ""),
                           d.get("recipe", ""), params)
        return web.json_response(lay)

    @routes.patch("/atlas/api/shots/{shot}/layers/{layer}")
    @guard
    async def change_layer(request):
        d = await body(request)
        name, layer = request.match_info["shot"], request.match_info["layer"]
        if "params" in d:
            lay = [x for x in sh.load_shot(name)["layers"] if x["name"] == layer]
            if lay and lay[0]["kind"] == "recipe":
                d["params"] = rc.validate(rc.load_recipe(lay[0]["recipe"]), d["params"])
        return web.json_response(sh.update_layer(name, layer, **d))

    @routes.post("/atlas/api/shots/{shot}/layers/{layer}/bake_mix")
    @guard
    async def bake_mix(request):
        """Write the layer's light mix as its next version (full-size ACEScg EXRs, comp/bake.py)."""
        return web.json_response(bk.start(request.match_info["shot"], request.match_info["layer"]))

    @routes.get("/atlas/api/shots/{shot}/layers/{layer}/bake_mix")
    @guard
    async def bake_status(request):
        return web.json_response(bk.status(request.match_info["shot"], request.match_info["layer"]))

    @routes.get("/atlas/api/recipes")
    @guard
    async def recipes(request):
        mode = request.query.get("mode")                      # a shot's mode: only its recipes
        return web.json_response({"recipes": rc.list_recipes(mode or None)})

    @routes.get("/atlas/api/presets")
    @guard
    async def presets(request):
        return web.json_response({"presets": rc.list_presets()})

    @routes.post("/atlas/api/shots/{shot}/layers/{layer}/graph")
    @guard
    async def graph(request):
        d = await body(request)
        job = _job_graph(request.match_info["shot"], request.match_info["layer"], d.get("mode", "render"),
                         d.get("params"))
        return web.json_response({"prompt": job["graph"], "version": job["version"], "out_dir": str(job["out"])})

    @routes.get("/atlas/api/targets")
    @guard
    async def targets(request):
        return web.json_response({"targets": [tg.public(t) for t in tg.load_targets()]})

    @routes.get("/atlas/api/queue")
    @guard
    async def queue_list(request):
        return web.json_response({"jobs": [qu.public(j) for j in _queue(request).list()]})

    @routes.post("/atlas/api/queue")
    @guard
    async def queue_add(request):
        d = await body(request)
        target = str(d.get("target", "ws"))
        if target not in [t["name"] for t in tg.load_targets()]:
            raise ValueError(f"target {target!r}: not one of the targets")
        mode = d.get("mode", "render")
        jg = _job_graph(d.get("shot", ""), d.get("layer", ""), mode, d.get("params"))
        job = qu.new_job(jg["shot"], jg["layer"], mode, target, jg["expected"], str(jg["out"]), jg["version"])
        _queue(request).add(job, jg["graph"], jg["recipe"])
        return web.json_response(qu.public(job))

    @routes.post("/atlas/api/queue/{job}/confirm")
    @guard
    async def queue_confirm(request):
        return web.json_response(qu.public(_queue(request).confirm(request.match_info["job"])))

    @routes.post("/atlas/api/queue/{job}/cancel")
    @guard
    async def queue_cancel(request):
        return web.json_response(qu.public(_queue(request).cancel(request.match_info["job"])))

    @routes.get("/atlas/api/shots/{shot}/frame")
    @guard
    async def frame(request):
        q = request.query
        s = sh.load_shot(request.match_info["shot"])
        f = _int(q, "f", s["first"], 0, 10 ** 6)
        w = _int(q, "w", 1280, 64, 4096)
        jpg = await off_loop(lambda: fr.display_jpeg(
            s, f, w, _float(q, "exposure", 0.0, -20, 20), q.get("view") or fr.DEFAULT_VIEW, _names(q, "a"),
            _names(q, "b"), _float(q, "wipe", 0.5, 0, 1)))
        return web.Response(body=jpg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @routes.get("/atlas/api/views")
    @guard
    async def views(request):
        from atlas_vfx.layers.node_viewer import views as vw_views
        return web.json_response({"views": vw_views()})

    return routes


def register() -> str:
    """Add the routes to ComfyUI's server; returns what happened (logged by the root __init__)."""
    try:
        from server import PromptServer
    except ImportError:
        return "not inside ComfyUI"
    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return "ComfyUI's server is not up yet"
    for r in build_routes():
        inst.routes.route(r.method, r.path)(r.handler)
    return "Atlas Comp at /atlas"
