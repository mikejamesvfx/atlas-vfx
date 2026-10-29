"""Comp preview: the plate with every layer EXR sequence over it, back to front, in ACEScg, through the config's display
view. For watching a shot in ComfyUI (a movie), not for delivery: Nuke builds the real comp.

Layer folders are what Card Project writes: <name>.<frame>.exr premultiplied ACEScg RGBA (and <name>.render.json).
A folder with no file for a frame contributes nothing on that frame.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl


def parse_folders(text: str) -> list[Path]:
    """One folder per line, back to front. Blank lines and # comments ignored; duplicates kept once."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        s = pl.clean_path(line)
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if str(p).lower() not in seen:
            seen.add(str(p).lower())
            out.append(p)
    return out


def layer_frames(folder: Path) -> tuple[str, dict[int, Path]]:
    """(layer name, {nuke frame: exr path}) from render.json's pattern, or from the files present."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"layer folder not found: {folder}")
    reports = sorted(folder.glob("*.render.json"))
    frames = {}
    if reports:
        r = json.loads(reports[0].read_text(encoding="utf-8"))
        name = r.get("card") or reports[0].name.split(".")[0]
    else:
        name = None
    rx = re.compile(r"^(?P<name>.+)\.(?P<f>\d+)\.exr$", re.I)
    for f in folder.glob("*.exr"):
        m = rx.match(f.name)
        if m and (name is None or m.group("name") == name) and ".tex." not in f.name:
            frames[int(m.group("f"))] = f
            name = name or m.group("name")
    if not frames:
        raise FileNotFoundError(f"no <name>.<frame>.exr layer files in {folder}")
    return name, frames


def over(fg: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA over RGB."""
    return fg[..., :3] + bg_rgb * (1.0 - fg[..., 3:4])


def comp_frame(bg_lin: np.ndarray, layers: list[np.ndarray | None]) -> np.ndarray:
    """bg ACEScg RGB; layers back to front, each premultiplied ACEScg RGBA at bg size or None."""
    out = bg_lin
    for L in layers:
        if L is not None:
            out = over(L, out)
    return out


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _burn(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    h = img.shape[0]
    s = max(0.4, h / 900.0)
    org = (int(12 * s), int(32 * s))
    band = min(h, int(48 * s))                       # OpenCV 5 draws text on 8-bit only: just the top strip
    strip = (np.clip(img[:band], 0, 1) * 255.0 * 0.35 + 0.5).astype(np.uint8)     # darkened strip, no outline
    cv2.putText(strip, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 220, 50), max(1, int(round(1.5 * s))),
                cv2.LINE_AA)
    img[:band] = strip.astype(np.float32) / 255.0
    return img


def render(plate_ref: dict, folders: list[Path], width: int = 1280, step: int = 1, burn_in: bool = True,
           label: str = "", progress=None) -> tuple[np.ndarray, dict]:
    """(frames N x H x W x 3 display-referred float32, report). Frames are the plate's (nuke_first..nuke_last)."""
    off = plate_ref["frame_offset"]
    layers = [layer_frames(f) for f in folders]
    disp = pl._cpu(plate_ref["ocio_config"], "ACEScg", None, pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW)
    W, H = pl.proxy_size(plate_ref, width)
    i0, i1 = plate_ref["first"], plate_ref["last"]
    out, used = [], {name: 0 for name, _ in layers}
    total = len(range(i0, i1 + 1, max(1, step)))
    for k, (i, img) in enumerate(pl.decode_rgb(plate_ref, i0, i1, width=width, step=step)):
        n = i + off
        bg = pl.to_working(img, plate_ref)
        Ls = []
        for name, frames in layers:
            p = frames.get(n)
            if p is None:
                Ls.append(None)
                continue
            Ls.append(_fit(exr_io.read_rgba(p)[0].astype(np.float32), W, H))
            used[name] += 1
        c = np.ascontiguousarray(comp_frame(bg, Ls), dtype=np.float32)
        disp.applyRGB(c)
        c = np.clip(c, 0.0, 1.0)
        if burn_in:
            c = _burn(c, f"{label + '  ' if label else ''}{n}")
        out.append(c)
        if progress:
            progress(k + 1, total)
    report = {"frames": [i0 + off, i1 + off], "step": step, "size": [W, H], "count": len(out),
              "layers_back_to_front": [{"name": nm, "folder": str(f), "frames_used": used[nm]}
                                       for (nm, _), f in zip(layers, folders)],
              "view": f"{pl.DEFAULT_DISPLAY} / {pl.DEFAULT_VIEW}", "fps": plate_ref["fps"][0] / plate_ref["fps"][1]}
    return np.stack(out), report
