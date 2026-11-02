"""The Light room's frame cache: every take, and what sits under and over the mixed layer, as half-float RGBA files the
browser uploads straight to the GPU (WebGL2 RGBA16F), at two levels:

    proxy  PROXY_W wide (HD, 1920, by default; ATLAS_VFX_PLAY_W), for scrubbing and playback: the browser holds every
           frame of the shot on the graphics card (97 frames of five takes + under/over at HD is about 10.5 GB)
    full   the paused frame: the source size capped at FULL_MAX_W wide (HD, 1920, by default: a 4K set of seven
           images is ~500 MB to read and upload; ATLAS_VFX_VIEW_MAX_W=0 keeps the source size)

A frame file is  b"ATLF16\\n" + uint32 header length + header JSON + float16 RGBA rows (top row first, little
endian). The header names its source key: sha1 of (source file, mtime_ns, size, level, width). A changed source no
longer matches and the file is remade on the next request. Read back with numpy frombuffer, never pickle.

Files live in <work>/comp/<shot>/cache/ and are only ever served from there. A full 4K frame is ~71 MB, so the cache
keeps to a size budget across every shot (ATLAS_VFX_CACHE_GB, default 40): past it, the least recently used frames go,
full-size ones first (cheap to remake from the EXR; the proxies are what playback needs). Only this cache's own .f16
files (and stale temp files) inside cache folders are ever removed. clear() empties one shot's cache."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import plate as pl
from atlas_vfx.layers import node_viewer as vw

from . import frames as fr
from . import mix as mx
from . import shot as sh

MAGIC = b"ATLF16\n"
LEVELS = ("proxy", "full")
PROXY_W = int(os.environ.get("ATLAS_VFX_PLAY_W", "1920") or 1920)
FULL_MAX_W = int(os.environ.get("ATLAS_VFX_VIEW_MAX_W", "1920") or 0)
BUDGET_GB_DEFAULT = 40.0
SCAN_EVERY_BYTES = 512 << 20        # check the budget after this much has been written since the last check
STATUS: dict = {}               # (shot, layer, level) -> {"state", "done", "of", "error"}
_lock = threading.Lock()
_evict_lock = threading.Lock()
_written_since_scan = 0
_pool = ThreadPoolExecutor(max(2, min(8, (os.cpu_count() or 4) // 3)), thread_name_prefix="atlas-cache")


def cache_root(name: str) -> Path:
    return sh.shot_dir(name) / "cache"


def _level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"level {level!r}: one of {LEVELS}")
    return level


def size_for(src_w: int, src_h: int, level: str) -> tuple[int, int]:
    w = PROXY_W if _level(level) == "proxy" else FULL_MAX_W
    if not w or src_w <= w:
        return src_w, src_h
    return w, max(2, int(round(src_h * w / src_w / 2)) * 2)


def encode(rgba: np.ndarray, header: dict) -> bytes:
    a = np.ascontiguousarray(rgba, np.float32)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"cache frame must be (H, W, 4), got {a.shape}")
    h = json.dumps({**header, "width": a.shape[1], "height": a.shape[0], "channels": 4,
                    "dtype": "float16le"}).encode()
    return MAGIC + struct.pack("<I", len(h)) + h + a.astype("<f2").tobytes()


def decode(data: bytes) -> tuple[np.ndarray, dict]:
    if not data.startswith(MAGIC):
        raise ValueError("not an Atlas cache frame")
    n = struct.unpack("<I", data[len(MAGIC):len(MAGIC) + 4])[0]
    o = len(MAGIC) + 4
    head = json.loads(data[o:o + n].decode())
    W, H = int(head["width"]), int(head["height"])
    px = np.frombuffer(data, "<f2", count=W * H * 4, offset=o + n).reshape(H, W, 4)
    return px.astype(np.float32), head


def source_key(src: Path, level: str, width: int) -> str:
    st = src.stat()
    return hashlib.sha1(f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{level}|{width}".encode()).hexdigest()


def _inside(path: Path, root: Path) -> Path:
    """path must sit under root. Lexical (abspath + normcase), not resolve(): every part is a safe_name, and
    resolve() raced while warm() created folders in parallel."""
    rp = os.path.normcase(os.path.abspath(path))
    rr = os.path.normcase(os.path.abspath(root))
    if not rp.startswith(rr.rstrip("\\/") + os.sep):
        raise ValueError("cache path outside the cache folder")
    return path


def _read_header(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                return None
            n = struct.unpack("<I", f.read(4))[0]
            return json.loads(f.read(n).decode())
    except (OSError, ValueError, struct.error):
        return None


def _write(path: Path, data: bytes) -> None:
    global _written_since_scan
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{threading.get_ident()}")
    tmp.write_bytes(data)
    tmp.replace(path)
    with _evict_lock:
        _written_since_scan += len(data)
        due = _written_since_scan >= SCAN_EVERY_BYTES
        if due:
            _written_since_scan = 0
    if due:
        evict()


def _read(path: Path) -> bytes:
    """A cache file's bytes; its mtime is bumped so eviction sees it as recently used."""
    data = path.read_bytes()
    try:
        os.utime(path, None)
    except OSError:
        pass
    return data


def budget_bytes() -> int:
    try:
        gb = float(os.environ.get("ATLAS_VFX_CACHE_GB", BUDGET_GB_DEFAULT))
    except ValueError:
        gb = BUDGET_GB_DEFAULT
    return int(max(0.0, gb) * (1 << 30))


def _cache_files() -> list[tuple[float, int, bool, Path]]:
    """(mtime, bytes, is_full, path) of every cache frame file under <work>/comp/*/cache (and stale temp files)."""
    out = []
    comp = sh.root()
    if not comp.is_dir():
        return out
    for shot_dir in comp.iterdir():
        root = shot_dir / "cache"
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if not f.is_file() or not (f.suffix == ".f16" or ".tmp" in f.suffix):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, f.parent.name == "full", f))
    return out


def usage() -> dict:
    files = _cache_files()
    return {"bytes": sum(b for _, b, _, _ in files), "files": len(files), "budget_bytes": budget_bytes(),
            "full_bytes": sum(b for _, b, full, _ in files if full)}


def evict(budget: int | None = None) -> dict:
    """Bring the cache under its budget (down to 90 % of it): least recently used first, full-size frames before
    proxies, stale temp files (older than an hour) always. Returns what went."""
    import time

    budget = budget_bytes() if budget is None else int(budget)
    files = _cache_files()
    total = sum(b for _, b, _, _ in files)
    gone, freed = 0, 0
    now = time.time()
    for m, b, _full, f in files:                      # temp files a crashed write left behind
        if ".tmp" in f.suffix and now - m > 3600:
            try:
                f.unlink()
                total -= b
                freed += b
                gone += 1
            except OSError:
                pass
    if total > budget:
        target = int(budget * 0.9)
        order = sorted((x for x in files if x[3].suffix == ".f16"), key=lambda x: (not x[2], x[0]))
        for _m, b, _full, f in order:
            if total <= target:
                break
            try:
                f.unlink()
                total -= b
                freed += b
                gone += 1
            except OSError:
                pass
    return {"removed_files": gone, "freed_bytes": freed, "bytes": total, "budget_bytes": budget}


def clear(name: str) -> dict:
    """Empty one shot's Light room cache (its .f16 frame files and temp files only)."""
    root = cache_root(name)
    gone, freed = 0, 0
    if root.is_dir():
        for f in root.rglob("*"):
            if f.is_file() and (f.suffix == ".f16" or ".tmp" in f.suffix):
                try:
                    b = f.stat().st_size
                    f.unlink()
                    gone += 1
                    freed += b
                except OSError:
                    pass
    return {"removed_files": gone, "freed_bytes": freed}


# ------------------------------------------------------------------------------------------------ takes

def take_file(name: str, layer: str, version: str, level: str, frame: int) -> Path:
    root = cache_root(name)
    p = root / pl.safe_name(layer, "layer name") / pl.safe_name(version, "version") / _level(level) / f"{int(frame):06d}.f16"
    return _inside(p, root)


def _source(shot: dict, lay: dict, version: str, frame: int) -> Path | None:
    pattern = sh.version_pattern(shot, lay, version)
    if not pattern:
        return None
    files = vw.ch.scan(pattern)
    return Path(files[frame]) if frame in files else None


def take_bytes(name: str, layer: str, version: str, level: str, frame: int) -> bytes | None:
    """The cache file for one take frame, made (or remade, if its source changed) when needed; None when the
    version has no such frame."""
    shot = sh.load_shot(name)
    lay = sh._layer(shot, layer)
    if version not in lay["versions"]:
        raise ValueError(f"version {version!r} is not rendered")
    src = _source(shot, lay, version, int(frame))
    if src is None:
        return None
    path = take_file(name, layer, version, level, frame)
    head = _read_header(path)
    w = head.get("width", 0) if head is not None else 0
    # kept only at the size the current caps give (a file from before the cap changed is remade)
    if head is not None and "src_w" in head and size_for(head["src_w"], head["src_h"], level)[0] == w             and head.get("key") == source_key(src, level, w):
        return _read(path)
    img = vw.read_frame(str(src), int(frame))
    W, H = size_for(img.shape[1], img.shape[0], level)
    data = encode(fr._resize(img, (W, H)), {"key": source_key(src, level, W), "frame": int(frame),
                                              "layer": layer, "version": version, "level": level,
                                              "src_w": int(img.shape[1]), "src_h": int(img.shape[0])})
    _write(path, data)
    return data


def warm(name: str, layer: str, level: str = "proxy") -> dict:
    """Build every frame of the layer's mixed takes (or all its versions) at `level` in the background."""
    level = _level(level)
    shot = sh.load_shot(name)
    lay = sh._layer(shot, layer)
    m = mx.normalize(lay.get("mix"))
    versions = list(m["takes"]) if m else list(lay["versions"])
    jobs = [(v, n) for v in versions for n in range(shot["first"], shot["last"] + 1)]
    key = (shot["name"], lay["name"], level)
    with _lock:
        if STATUS.get(key, {}).get("state") == "running":
            return STATUS[key]
        STATUS[key] = {"state": "running", "done": 0, "of": len(jobs), "error": None}

    def one(job):
        v, n = job
        try:
            take_bytes(name, layer, v, level, n)
        except Exception as e:  # noqa: BLE001 (kept for the app; the rest carry on)
            STATUS[key]["error"] = str(e)[:300]
        with _lock:
            STATUS[key]["done"] += 1
            if STATUS[key]["done"] >= STATUS[key]["of"]:
                STATUS[key]["state"] = "done" if not STATUS[key]["error"] else "error"
    if not jobs:
        STATUS[key]["state"] = "done"
    for j in jobs:
        _pool.submit(one, j)
    return STATUS[key]


def status(name: str, layer: str, level: str = "proxy") -> dict:
    key = (pl.safe_name(name, "shot name"), pl.safe_name(layer, "layer name"), _level(level))
    return STATUS.get(key, {"state": "idle", "done": 0, "of": 0, "error": None})


# ------------------------------------------------------------------------------------------------ under / over

def _layer_at(shot: dict, lay: dict, frame: int, level: str, size: tuple[int, int] | None):
    """A layer's RGBA at a cache level (a mixed layer mixed from its cached takes)."""
    m = mx.normalize(lay.get("mix"))
    if m:
        imgs = {}
        for v in mx.active(m):
            data = take_bytes(shot["name"], lay["name"], v, level, frame)
            if data is not None:
                img = decode(data)[0]
                imgs[v] = img if size is None else fr._resize(img, size)
        return mx.mix_frame(imgs, m)
    v = lay.get("version") or (lay["versions"][-1] if lay.get("versions") else None)
    if lay["kind"] == "recipe" and v and v != sh.PREVIEW:
        if v not in lay.get("versions", []):          # chosen version has no frames (failed render): left out,
            return None                               # as the comp leaves it out
        data = take_bytes(shot["name"], lay["name"], v, level, frame)
        return None if data is None else decode(data)[0]
    return fr._single(shot, lay, frame)              # sequence layers and previews: read directly


def stack_key(shot: dict, layer: str, side: str, level: str) -> str:
    """What under / over depend on: the plate and the other layers' state on that side of `layer`."""
    lay = sh._layer(shot, layer)
    o = lay.get("order", 0)
    keep = ("name", "kind", "source", "version", "visible", "opacity", "blend", "order", "mix", "versions")
    others = [{k: x.get(k) for k in keep} for x in shot["layers"]
              if x["name"] != layer and ((x.get("order", 0) < o) if side == "under" else (x.get("order", 0) > o))]
    plate = shot.get("plate") if side == "under" else None
    return hashlib.sha1(json.dumps([side, level, plate, others], sort_keys=True, default=str).encode()).hexdigest()


def stack_bytes(name: str, layer: str, side: str, level: str, frame: int, size: tuple[int, int]) -> bytes:
    """Everything under (plate + lower layers) or over (higher layers) the mixed layer, at `size` (the takes'
    size at that level), premultiplied linear ACEScg. Cached per stack key."""
    if side not in ("under", "over"):
        raise ValueError("side: under or over")
    level = _level(level)
    shot = sh.load_shot(name)
    lay = sh._layer(shot, layer)
    key = stack_key(shot, layer, side, level)
    root = cache_root(name)
    path = _inside(root / "_stack" / pl.safe_name(layer, "layer name") / side / level / f"{int(frame):06d}.f16", root)
    head = _read_header(path)
    if head is not None and head.get("key") == key and [head["width"], head["height"]] == list(size):
        return _read(path)
    W, H = size
    canvas = np.zeros((H, W, 4), np.float32)
    if side == "under":
        plate = fr._plate_frame(shot, frame, W)
        if plate is not None:
            canvas = fr._resize(plate, (W, H)).copy()
    o = lay.get("order", 0)
    for x in sorted(shot["layers"], key=lambda q: q.get("order", 0)):
        if x["name"] == layer or not x.get("visible", True):
            continue
        if (x.get("order", 0) < o) != (side == "under"):
            continue
        img = _layer_at(shot, x, frame, level, (W, H))
        if img is None:
            continue
        canvas = fr.blend(canvas, fr._resize(img, (W, H)), x.get("blend", "over"), float(x.get("opacity", 1.0)))
    data = encode(canvas, {"key": key, "frame": int(frame), "layer": layer, "side": side, "level": level})
    _write(path, data)
    return data


# ------------------------------------------------------------------------------------------------ view shader

_SHADERS: dict = {}


def view_shader(ocio_config: str, view: str) -> dict:
    """OCIO's own GPU code for ACEScg -> the named display / view, as GLSL ES 3.0 (WebGL2): the shader text with a
    function `vec4 OCIOView(vec4)`, and its lookup textures as float32 lists. Cached per (config, view)."""
    key = (ocio_config, view)
    if key in _SHADERS:
        return _SHADERS[key]
    import PyOpenColorIO as ocio

    display, _, v = str(view).partition(" / ")
    cfg = pl.ocio_config(pl.ocio_path(ocio_config))
    t = ocio.DisplayViewTransform(src=pl.WORKING_SPACE, display=display.strip() or pl.DEFAULT_DISPLAY,
                                  view=v.strip() or pl.DEFAULT_VIEW)
    gp = cfg.getProcessor(t).getDefaultGPUProcessor()
    d = ocio.GpuShaderDesc.CreateShaderDesc(language=ocio.GPU_LANGUAGE_GLSL_ES_3_0, functionName="OCIOView",
                                            resourcePrefix="ocio_")
    gp.extractGpuShaderInfo(d)
    tex = []
    for x in d.getTextures():
        vals = np.asarray(x.getValues(), np.float32)
        ch = 1 if x.channel == ocio.GpuShaderDesc.TEXTURE_RED_CHANNEL else 3
        tex.append({"sampler": x.samplerName, "width": int(x.width), "height": int(x.height), "channels": ch,
                    "linear": x.interpolation != ocio.INTERP_NEAREST, "values": vals.tolist()})
    tex3 = []
    for x in d.get3DTextures():
        tex3.append({"sampler": x.samplerName, "edge": int(x.edgeLen), "linear": x.interpolation != ocio.INTERP_NEAREST,
                     "values": np.asarray(x.getValues(), np.float32).tolist()})
    out = {"view": f"{display.strip()} / {v.strip()}", "function": "OCIOView", "shader": d.getShaderText(),
           "textures": tex, "textures3d": tex3, "uniforms": [u[0] for u in d.getUniforms()]}
    _SHADERS[key] = out
    return out
