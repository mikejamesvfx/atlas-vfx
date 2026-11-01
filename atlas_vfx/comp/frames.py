"""Viewer frames for Atlas Comp: the shot's visible layers composited in linear ACEScg at proxy size (plate at the
bottom when the shot has one), then shown through a named OCIO view as a JPEG. Reuses the Viewer node's reading,
wipe and view code. Layers are premultiplied EXRs; a layer missing the frame is left out, never an error."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from atlas_vfx.camera_side import plate as pl
from atlas_vfx.layers import ltx_integration as li
from atlas_vfx.layers import node_viewer as vw

from .shot import layer_pattern

BLENDS = ("over", "plus", "multiply", "screen")
DEFAULT_VIEW = f"{pl.DEFAULT_DISPLAY} / {pl.DEFAULT_VIEW}"
_PLATE_REFS: dict = {}
_PLATE_FRAMES: OrderedDict = OrderedDict()
_PROCS: dict = {}
PLATE_CACHE = 8
PROXY_W = 1280
BUILDING: set[str] = set()          # shots whose plate proxy is being made (one build at a time)
FAILED: dict[str, str] = {}         # shot -> why its last proxy build failed (cleared by the next build)
_CACHE_LOCK = threading.Lock()


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    interp = cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def _plate_ref(shot: dict) -> dict:
    key = (shot["plate"], shot.get("ocio_config", ""))
    if key not in _PLATE_REFS:
        from atlas_vfx.camera_side.node_plate_read import AtlasVFXPlateRead

        out = AtlasVFXPlateRead().run(shot["plate"], 0, -1, shot.get("ocio_config", ""), "auto", "auto",
                                      int(shot["frame_offset"]), 64, 100000)
        _PLATE_REFS[key] = (out["result"] if isinstance(out, dict) else out)[0]
    return _PLATE_REFS[key]


def proxy_dir(shot: dict, width: int = PROXY_W):
    from .shot import shot_dir

    return shot_dir(shot["name"]) / f"plate_proxy_{int(width)}"


def proxy_state(shot: dict) -> tuple[str, str]:
    """(state, message): none (no plate), building, error (the last build failed, with why), ready, missing."""
    if not shot.get("plate"):
        return "none", ""
    if shot["name"] in BUILDING:
        return "building", ""
    if proxy_dir(shot).is_dir():
        return "ready", ""
    if shot["name"] in FAILED:
        return "error", FAILED[shot["name"]]
    return "missing", ""


def build_plate_proxy(shot: dict, width: int = PROXY_W, progress=None) -> str:
    """Decode the shot's range of the plate once (one pass, a movie decodes fast forwards and slowly by seeking) into
    linear ACEScg half EXRs `width` wide; the viewer reads those. Written into <dir>.tmp and renamed when complete,
    so an interrupted build never looks ready. A failure is remembered (proxy_state) and raised. Returns the ####
    pattern."""
    import shutil

    name = shot["name"]
    BUILDING.add(name)
    FAILED.pop(name, None)
    final = proxy_dir(shot, width)
    tmp = final.with_name(final.name + ".tmp")
    try:
        shutil.rmtree(tmp, ignore_errors=True)       # our own leftovers from an interrupted build
        _build(shot, width, progress, tmp)
        tmp.replace(final)
        return str(final / "plate.####.exr")
    except Exception as e:
        FAILED[name] = f"{type(e).__name__}: {e}"[:600]
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        BUILDING.discard(name)


def _build(shot: dict, width: int, progress, out) -> None:
    from atlas_vfx.camera_side import exr_io

    ref = _plate_ref(shot)
    off = int(ref["frame_offset"])
    first = max(int(shot["first"]), off)
    last = min(int(shot["last"]), off + int(ref["frame_count"]) - 1)
    out.mkdir(parents=True, exist_ok=True)
    for (i, img), n in zip(pl.decode_rgb(ref, first - off, last - off), range(first, last + 1)):
        lin = np.ascontiguousarray(pl.to_working(img, ref), np.float32)
        h = max(1, round(lin.shape[0] * width / lin.shape[1]))
        rgba = np.dstack([_resize(lin, (width, h)), np.ones((h, width), np.float32)])
        exr_io.write_rgba(out / f"plate.{n:04d}.exr", rgba, attrs={"frame": n, "source": "Atlas Comp plate proxy"})
        if progress:
            progress(n - first + 1, last - first + 1)


def _plate_frame(shot: dict, frame: int, width: int) -> np.ndarray | None:
    """The plate at `frame` as linear ACEScg RGBA (alpha 1) at `width`, from the plate proxy; None when the shot has
    no plate, the proxy is not ready (the server builds it in the background) or lacks the frame. Never builds here
    and never raises, so the layers always composite."""
    if not shot.get("plate"):
        return None
    key = (shot["name"], int(frame), int(width))
    with _CACHE_LOCK:
        if key in _PLATE_FRAMES:
            _PLATE_FRAMES.move_to_end(key)
            return _PLATE_FRAMES[key]
    dirs = [d for d in (proxy_dir(shot, width), proxy_dir(shot, PROXY_W)) if d.is_dir()]
    if not dirs:
        return None
    try:
        img = vw.read_frame(str(dirs[0] / "plate.####.exr"), int(frame))
    except (ValueError, RuntimeError, OSError):
        return None
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    rgba = _resize(img, (int(width), h))
    with _CACHE_LOCK:
        _PLATE_FRAMES[key] = rgba
        while len(_PLATE_FRAMES) > PLATE_CACHE:
            _PLATE_FRAMES.popitem(last=False)
    return rgba


def blend(base: np.ndarray, layer: np.ndarray, mode: str, opacity: float) -> np.ndarray:
    """Premultiplied RGBA `layer` onto `base` (both H, W, 4)."""
    if mode not in BLENDS:
        raise ValueError(f"blend {mode!r}: one of {BLENDS}")
    f = layer * np.float32(opacity)
    fa = f[..., 3:4]
    out = base.copy()
    b, ba = base[..., :3], base[..., 3:4]
    if mode == "over":
        out[..., :3] = f[..., :3] + b * (1 - fa)
    elif mode == "plus":
        out[..., :3] = b + f[..., :3]
    elif mode == "multiply":
        out[..., :3] = b * (1 - fa) + b * f[..., :3]
    else:
        out[..., :3] = b + f[..., :3] - b * f[..., :3]
    out[..., 3:4] = np.clip(fa + ba * (1 - fa), 0, 1)
    return out


def _single(shot: dict, lay: dict, frame: int) -> np.ndarray | None:
    pattern = layer_pattern(shot, lay)
    if not pattern:
        return None
    try:
        return vw.read_frame(pattern, int(frame))
    except (ValueError, RuntimeError):
        return None


def _mixed(shot: dict, lay: dict, frame: int) -> np.ndarray | None:
    """A light mix of the layer's versions (comp/mix.py: blend for relit takes, sum for light passes). Versions
    missing the frame are left out; None when none has it."""
    from . import mix as mx
    from .shot import version_pattern

    m = mx.normalize(lay["mix"])
    images, size = {}, None
    for ver in mx.active(m):
        pattern = version_pattern(shot, lay, ver)
        if not pattern:
            continue
        try:
            img = vw.read_frame(pattern, int(frame))
        except (ValueError, RuntimeError):
            continue
        if size is not None and img.shape[:2] != size:
            img = _resize(img, (size[1], size[0]))
        size = img.shape[:2]
        images[ver] = img
    return mx.mix_frame(images, m)


def composite(shot: dict, frame: int, width: int = 1280, only: list[str] | None = None
              ) -> tuple[np.ndarray, list[str]]:
    """Linear ACEScg RGBA at `width` and the names of the layers that went in (bottom first)."""
    width = max(16, int(width))
    canvas = _plate_frame(shot, frame, width) if only is None or "plate" in only else None
    used = []
    layers = sorted(shot["layers"], key=lambda x: x.get("order", 0))
    for lay in layers:
        if not lay.get("visible", True) or (only is not None and lay["name"] not in only):
            continue
        img = _mixed(shot, lay, frame) if lay.get("mix") else _single(shot, lay, frame)
        if img is None:
            continue
        if canvas is None:
            h = max(1, round(img.shape[0] * width / img.shape[1]))
            canvas = np.zeros((h, width, 4), np.float32)
        img = _resize(img, (canvas.shape[1], canvas.shape[0]))
        canvas = blend(canvas, img, lay.get("blend", "over"), float(lay.get("opacity", 1.0)))
        used.append(lay["name"])
    if canvas is None:
        canvas = np.zeros((max(1, round(width * 1080 / 1920)), width, 4), np.float32)
        for lay in layers:                        # the shape of any layer, so an empty frame keeps the shot's aspect
            pattern = layer_pattern(shot, lay)
            if not pattern:
                continue
            try:
                files = vw.ch.scan(pattern)
                if files:
                    a = vw.read_frame(pattern, next(iter(files)))
                    canvas = np.zeros((max(1, round(a.shape[0] * width / a.shape[1])), width, 4), np.float32)
                    break
            except (ValueError, RuntimeError):
                continue
    return canvas, used


def _proc(shot: dict, view: str):
    key = (shot.get("ocio_config", ""), view)
    if key not in _PROCS:
        display, _, v = str(view).partition(" / ")
        _PROCS[key] = li.view_processors(shot.get("ocio_config", ""), display=display.strip(), view=v.strip())[0]
    return _PROCS[key]


def display_jpeg(shot: dict, frame: int, width: int = 1280, exposure: float = 0.0, view: str = DEFAULT_VIEW,
                 a: list[str] | None = None, b: list[str] | None = None, wipe: float = 0.5, quality: int = 88) -> bytes:
    """The viewer frame: `a` (or every visible layer) through the view; with `b`, an A/B wipe at `wipe`."""
    import cv2

    if view not in vw.views(shot.get("ocio_config", "")):
        raise ValueError(f"view {view!r}: not in the OCIO config")
    ia, _ = composite(shot, frame, width, a)
    ib = composite(shot, frame, width, b)[0] if b is not None else None
    img, _ = vw.render(ia, ib, "wipe" if ib is not None else "a", wipe=float(wipe), exposure=float(exposure),
                       view_proc=_proc(shot, view), width=0)
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
