"""Atlas VFX Layer Picker: choose a rendered layer or EXR sequence from a list instead of typing its path. Lists the
folders under the work root that hold an EXR sequence and a sidecar (Card Project, LTX Return, EXR Write, Relight
Transfer Files all write one), relative to the work root so a saved graph carries no user folder."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from atlas_vfx.camera_side import plate as pl

from . import chunks as ch

NONE = "(nothing rendered yet)"
MAX_DEPTH = 4
_NUM = re.compile(r"(\d+)(\.[A-Za-z0-9]+)$")


def find_layers(root: Path, max_depth: int = MAX_DEPTH) -> list[str]:
    """Folders under root (posix, relative) holding *.exr plus a *.render.json or *.sidecar.json. Hidden folders and
    anything deeper than max_depth are skipped."""
    root = Path(root)
    found = []
    if not root.is_dir():
        return found
    base = len(root.parts)
    for d, dirs, files in os.walk(root):
        depth = len(Path(d).parts) - base
        dirs[:] = sorted(x for x in dirs if not x.startswith(".")) if depth < max_depth else []
        if any(f.endswith(".exr") for f in files) and any(f.endswith((".render.json", ".sidecar.json"))
                                                          for f in files):
            found.append(Path(d).relative_to(root).as_posix())
    return sorted(found)


def inside(root: Path, rel: str) -> Path:
    """root / rel, refusing anything that resolves outside root (the widget value is untrusted)."""
    root = Path(root).resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"{rel!r} is outside the work root")
    return p


def describe(folder: Path) -> dict:
    """Pattern, frame range and header facts of the one EXR sequence in folder."""
    import OpenImageIO as oiio

    exrs = sorted(f for f in folder.iterdir() if f.suffix == ".exr" and _NUM.search(f.name))
    if not exrs:
        raise ValueError(f"{folder.name}: no numbered EXRs")
    m = _NUM.search(exrs[0].name)
    stem = exrs[0].name[:m.start()]
    pattern = folder / f"{stem}{'#' * len(m.group(1))}{m.group(2)}"
    frames = ch.scan(str(pattern))
    inp = oiio.ImageInput.open(str(exrs[0]))
    if inp is None:
        raise ValueError(f"cannot read {exrs[0].name}: {oiio.geterror()}")
    s = inp.spec()
    info = {"width": s.width, "height": s.height, "channels": list(s.channelnames),
            "colorspace": s.get_string_attribute("oiio:ColorSpace") or "unknown"}
    inp.close()
    first, last = min(frames), max(frames)
    missing = (last - first + 1) - len(frames)
    return {"pattern": str(pattern), "first": first, "last": last, "frames": len(frames), "missing": missing, **info}


class AtlasVFXLayerPicker:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("folder", "pattern", "first", "last", "info")
    DESCRIPTION = ("Pick a rendered layer or EXR sequence from the work folder (no typing paths). folder goes to "
                   "LTX Prep's layer_folder, pattern to Relight Transfer Files or Viewer, first / last are its frame "
                   "range. The list is read when ComfyUI loads the nodes: press R (refresh) or reload after "
                   "rendering something new.")

    @classmethod
    def INPUT_TYPES(cls):
        layers = find_layers(pl.WORK_ROOT) or [NONE]
        return {"required": {"layer": (layers, {"tooltip": "folders under the work root with an EXR sequence"})}}

    def run(self, layer):
        if layer == NONE:
            raise ValueError("nothing rendered yet under the work root")
        folder = inside(pl.WORK_ROOT, layer)
        d = describe(folder)
        text = (f"{layer}: {d['frames']} frames {d['first']}-{d['last']}"
                + (f" ({d['missing']} missing)" if d["missing"] else "")
                + f", {d['width']}x{d['height']} {','.join(d['channels'])}, {d['colorspace']}")
        return {"ui": {"text": [text]},
                "result": (str(folder), d["pattern"], d["first"], d["last"], json.dumps({**d, "text": text}))}

    @classmethod
    def VALIDATE_INPUTS(cls, layer):
        return True     # a graph saved on another machine names a layer this one may not list yet; run() checks


NODE_CLASS_MAPPINGS = {"AtlasVFXLayerPicker": AtlasVFXLayerPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLayerPicker": "Atlas VFX Layer Picker"}
