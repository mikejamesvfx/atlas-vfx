"""Light mix: the maths and the record, one place for the server (frames, bake) and the browser's shader to follow.

A layer's versions are takes of the same shot (e.g. Relight presets). shot.json keeps

    v2  {"mode": "blend" | "sum", "takes": {version: {"weight", "stops", "tint": [r, g, b], "mute", "solo"}}}
    v1  {version: weight}                      (before 2026-11-01: read as mode "sum", stops 0, no tint)

In linear ACEScg, premultiplied, per take i: gain g_i = 2^stops_i * tint_i on RGB (alpha untouched).
  active  = the solo takes if any take is solo, else the takes not muted, and only those with weight > 0
  blend   RGB = sum(w_i g_i T_i) / sum(w_i),  A = sum(w_i A_i) / sum(w_i)    (full relit takes: exposure holds)
  sum     RGB = sum(w_i g_i T_i),             A = clip(sum(w_i A_i), 0, 1)   (light passes: light adds)
No active take: None (the layer drops out, as an empty frame does). app/lightmix.js mirrors mix_frame line for line.
"""
from __future__ import annotations

import numpy as np

MODES = ("blend", "sum")
WEIGHT_MAX = {"blend": 1.0, "sum": 2.0}
STOPS = (-8.0, 4.0)
TINT_MAX = 4.0
LUMA_AP1 = np.array([0.2722287, 0.6740818, 0.0536895], np.float32)     # ACEScg luminance weights


def tint_unit(rgb) -> list[float]:
    """An RGB tint scaled to luminance 1 in ACEScg, so a tint changes colour, not exposure."""
    t = np.asarray(rgb, np.float64)
    if t.shape != (3,) or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError(f"tint {rgb!r}: three numbers >= 0")
    y = float(t @ LUMA_AP1)
    if y <= 1e-6:
        raise ValueError(f"tint {rgb!r}: too dark to scale to luminance 1")
    return [round(float(v / y), 6) for v in t]


def normalize(mix) -> dict | None:
    """Any stored mix (v1 or v2, or None) -> v2 with every field filled."""
    if not mix:
        return None
    if "takes" not in mix:                                          # v1: {version: weight}, light adds
        return {"mode": "sum", "takes": {v: {"weight": float(w), "stops": 0.0, "tint": [1.0, 1.0, 1.0],
                                             "mute": False, "solo": False} for v, w in mix.items()}}
    takes = {}
    for v, t in mix["takes"].items():
        takes[v] = {"weight": float(t.get("weight", 1.0)), "stops": float(t.get("stops", 0.0)),
                    "tint": [float(x) for x in t.get("tint", [1.0, 1.0, 1.0])],
                    "mute": bool(t.get("mute", False)), "solo": bool(t.get("solo", False))}
    return {"mode": mix.get("mode", "blend"), "takes": takes}


def validate(mix, versions: list[str]) -> dict:
    """An untrusted mix from the app -> a clean v2 record (or ValueError). v1 dicts are accepted as before."""
    if not isinstance(mix, dict):
        raise ValueError("mix: an object")
    if "takes" in mix and not isinstance(mix["takes"], dict):
        raise ValueError("mix: takes must be an object of version -> settings")
    if "takes" not in mix:
        for w in mix.values():
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise ValueError("mix: an object of version -> weight")
    else:
        for t in mix["takes"].values():
            if not isinstance(t, dict):
                raise ValueError("mix: each take is an object")
            for k in ("weight", "stops"):
                if k in t and (isinstance(t[k], bool) or not isinstance(t[k], (int, float))):
                    raise ValueError(f"mix: {k} must be a number")
            for k in ("mute", "solo"):
                if k in t and not isinstance(t[k], bool):
                    raise ValueError(f"mix: {k} must be true or false")
            if "tint" in t and (not isinstance(t["tint"], list) or len(t["tint"]) != 3
                                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in t["tint"])):
                raise ValueError("mix: tint must be three numbers")
    m = normalize(mix)
    if m["mode"] not in MODES:
        raise ValueError(f"mix mode {m['mode']!r}: one of {MODES}")
    wmax = WEIGHT_MAX[m["mode"]]
    for v, t in m["takes"].items():
        if v not in versions:
            raise ValueError(f"mix: version {v!r} is not rendered ({', '.join(versions) or 'none'})")
        if not 0.0 <= t["weight"] <= wmax:
            raise ValueError(f"mix: weight for {v} must be 0-{wmax:g}")
        if not STOPS[0] <= t["stops"] <= STOPS[1]:
            raise ValueError(f"mix: stops for {v} must be {STOPS[0]:g} to {STOPS[1]:+g}")
        if any(not 0.0 <= x <= TINT_MAX for x in t["tint"]):
            raise ValueError(f"mix: tint for {v} must be 0-{TINT_MAX:g} per channel")
    return m


def active(mix: dict) -> list[str]:
    """The takes that go in: the solo ones if any, else the unmuted ones; weight > 0 either way."""
    takes = mix["takes"]
    solo = [v for v, t in takes.items() if t["solo"]]
    pick = solo if solo else [v for v, t in takes.items() if not t["mute"]]
    return [v for v in pick if takes[v]["weight"] > 0]


def mix_frame(images: dict, mix) -> np.ndarray | None:
    """images {version: premultiplied RGBA float32 (H, W, 4), all one size} -> the mixed RGBA, or None.
    Versions in the mix but missing from images (no such frame) are left out of both sums."""
    m = normalize(mix)
    if m is None:
        return None
    use = [v for v in active(m) if v in images]
    if not use:
        return None
    rgb = alpha = None
    wsum = 0.0
    for v in use:
        t = m["takes"][v]
        w = np.float32(t["weight"])
        g = (np.float32(2.0 ** t["stops"]) * np.asarray(t["tint"], np.float32)) * w
        img = np.asarray(images[v], np.float32)
        c, a = img[..., :3] * g, img[..., 3] * w
        rgb, alpha = (c, a) if rgb is None else (rgb + c, alpha + a)
        wsum += float(w)
    out = np.empty(rgb.shape[:2] + (4,), np.float32)
    if m["mode"] == "blend":
        out[..., :3] = rgb / np.float32(wsum)
        out[..., 3] = alpha / np.float32(wsum)
    else:
        out[..., :3] = rgb
        out[..., 3] = np.clip(alpha, 0.0, 1.0)
    return out
