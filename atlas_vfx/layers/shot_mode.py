"""What a shot starts from, chosen before anything else (owner call 2026-11-01). Shared by Shot Setup (the node) and
Atlas Comp (shot.json), so a graph and the app read one record.

  plate       a real background plate: track it, add layers over it (the flow so far)
  foreground  an actor shot on green or blue screen: key it, track the screen plate, generate the background
              behind them through that camera, relight the actor to it
  generate    no plate: a virtual plate (size, fps, frames) and an authored camera; the background is made by LTX
              (then tracked like a plate), by a set build (sky, cards, set proxy) through the camera, or both

Every mode ends with a plate_ref and a camera, so the nodes after it work the same way.
"""
from __future__ import annotations

MODES = ("plate", "foreground", "generate")
SCREENS = ("green", "blue")
ROUTES = {
    "plate": ("track",),
    "foreground": ("key", "track", "background_set", "background_hybrid", "background_ltx", "relight"),
    "generate": ("ltx_plate", "set_build", "hybrid"),
}
LABELS = {"plate": "Background plate", "foreground": "Foreground on a screen", "generate": "Create both"}


def check_mode(mode: str) -> str:
    m = str(mode or "plate").strip().lower()
    if m not in MODES:
        raise ValueError(f"shot mode {mode!r}: one of {MODES}")
    return m


def check_screen(screen: str) -> str:
    s = str(screen or "green").strip().lower()
    if s not in SCREENS:
        raise ValueError(f"screen colour {screen!r}: one of {SCREENS}")
    return s


def record(mode: str, plate_ref: dict, screen: str | None = None) -> dict:
    """The ATLAS_VFX_SHOT payload Shot Setup hands on (JSON-safe)."""
    mode = check_mode(mode)
    rec = {"kind": "atlas_vfx.shot", "version": 1, "mode": mode, "label": LABELS[mode], "routes": list(ROUTES[mode]),
           "virtual_plate": bool(plate_ref.get("virtual")),
           "size": [plate_ref["width"], plate_ref["height"]], "fps": list(plate_ref["fps"]),
           "frames": [plate_ref["nuke_first"], plate_ref["nuke_last"]]}
    if mode == "foreground":
        rec["screen"] = check_screen(screen)
    return rec
