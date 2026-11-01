"""Bake a light mix into a new version of its layer: full-size ACEScg EXRs through comp/mix.py, so the mix becomes an
ordinary layer (Nuke, LTX, anything that reads a sequence). The browser's live mix follows the same maths."""
from __future__ import annotations

import json
import threading
import time

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.layers import node_viewer as vw

from . import mix as mx
from . import shot as sh

STATUS: dict = {}               # (shot, layer) -> {"state", "done", "of", "version", "error"}
_lock = threading.Lock()


def _frames(shot: dict, lay: dict, versions: list[str]) -> list[int]:
    have = set()
    for v in versions:
        pattern = sh.version_pattern(shot, lay, v)
        if pattern:
            have |= set(vw.ch.scan(pattern))
    return sorted(n for n in have if shot["first"] <= n <= shot["last"])


def bake_mix(name: str, layer: str, progress=None) -> dict:
    """Write the layer's current mix as the next version. Returns {version, frames, pattern}."""
    shot = sh.load_shot(name)
    lay = sh._layer(shot, layer)
    m = mx.normalize(lay.get("mix"))
    if m is None or not mx.active(m):
        raise ValueError(f"layer {layer!r} has no light mix with an active take to bake")
    use = mx.active(m)
    frames = _frames(shot, lay, use)
    if not frames:
        raise ValueError(f"layer {layer!r}: the mixed versions have no frames in {shot['first']}-{shot['last']}")
    patterns = {v: sh.version_pattern(shot, lay, v) for v in use}
    ver = sh.next_version(name, layer)
    out = sh.layer_dir(name, layer) / ver
    stem = pl.safe_name(f"{layer}_mix", "layer name")
    t0 = time.time()
    for k, n in enumerate(frames):
        images, size = {}, None
        for v, pattern in patterns.items():
            if not pattern:
                continue
            try:
                img = vw.read_frame(pattern, n)
            except (ValueError, RuntimeError):
                continue
            if size is not None and img.shape[:2] != size:
                import cv2

                img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            size = img.shape[:2]
            images[v] = img
        rgba = mx.mix_frame(images, m)
        if rgba is None:
            continue
        exr_io.write_rgba(out / f"{stem}.{n:04d}.exr", np.ascontiguousarray(rgba, np.float32),
                          attrs={"layer": layer, "frame": n, "source": "Atlas Comp light mix",
                                 "mix_mode": m["mode"]})
        if progress:
            progress(k + 1, len(frames))
    (out / "mix.json").write_text(json.dumps({"mix": m, "versions": use, "frames": [frames[0], frames[-1]],
                                              "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    return {"version": ver, "frames": [frames[0], frames[-1]], "pattern": str(out / f"{stem}.####.exr")}


def start(name: str, layer: str) -> dict:
    """Bake in a background thread (one per layer at a time); STATUS holds the progress."""
    key = (pl.safe_name(name, "shot name"), pl.safe_name(layer, "layer name"))
    with _lock:
        if STATUS.get(key, {}).get("state") == "running":
            return STATUS[key]
        STATUS[key] = {"state": "running", "done": 0, "of": 0, "version": None, "error": None}

    def prog(done, of):
        STATUS[key].update(done=done, of=of)

    def work():
        try:
            res = bake_mix(*key, progress=prog)
            STATUS[key].update(state="done", version=res["version"])
        except Exception as e:  # noqa: BLE001 (shown in the app)
            STATUS[key].update(state="error", error=str(e)[:400])
    threading.Thread(target=work, daemon=True, name=f"atlas-comp-bake-{key[0]}-{key[1]}").start()
    return STATUS[key]


def status(name: str, layer: str) -> dict:
    key = (pl.safe_name(name, "shot name"), pl.safe_name(layer, "layer name"))
    return STATUS.get(key, {"state": "idle", "done": 0, "of": 0, "version": None, "error": None})
