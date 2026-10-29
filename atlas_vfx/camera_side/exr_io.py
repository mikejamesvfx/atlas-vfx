"""EXR read/write through OpenImageIO with the colorspace always in the header.

OIIO 3.2 writes the OpenEXR standard `colorInteropID` from `oiio:ColorSpace` (ACEScg -> lin_ap1_scene). We also stamp
the OCIO name we mean (`atlas_vfx:ocio_colorspace`) so Nuke users and other readers see it in plain words.
Base for EXR Write (catalogue 23, T8).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_rgba(path, rgba: np.ndarray, colorspace: str = "ACEScg", attrs: dict | None = None,
               half: bool = True, premultiplied: bool = True) -> Path:
    import OpenImageIO as oiio

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    H, W, C = rgba.shape
    if C != 4:
        raise ValueError(f"expected RGBA, got {C} channels")
    spec = oiio.ImageSpec(W, H, 4, oiio.HALF if half else oiio.FLOAT)
    spec.channelnames = ("R", "G", "B", "A")
    spec.alpha_channel = 3
    spec.attribute("compression", "zip")
    spec.attribute("oiio:ColorSpace", colorspace)
    spec.attribute("atlas_vfx:ocio_colorspace", colorspace)
    spec.attribute("atlas_vfx:premultiplied", int(premultiplied))
    for k, v in (attrs or {}).items():
        spec.attribute(f"atlas_vfx:{k}", v if isinstance(v, (int, float, str)) else json.dumps(v))
    out = oiio.ImageOutput.create(str(p))
    if out is None or not out.open(str(p), spec):
        raise RuntimeError(f"cannot write {p}: {oiio.geterror()}")
    out.write_image(np.ascontiguousarray(rgba, dtype=np.float32))
    out.close()
    return p


INTEROP = {"ACEScg": "lin_ap1_scene", "ACES2065-1": "lin_ap0_scene", "Raw": "data"}


def header(path) -> dict:
    """The parts of an EXR header a comp cares about, without reading pixels."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    s = inp.spec()
    out = {"width": s.width, "height": s.height, "channels": list(s.channelnames),
           "format": str(s.format), "colorInteropID": s.get_string_attribute("colorInteropID"),
           "ocio_colorspace": s.get_string_attribute("atlas_vfx:ocio_colorspace"),
           "premultiplied": s.get_int_attribute("atlas_vfx:premultiplied", -1)}
    inp.close()
    return out


def check_sequence(pattern: str, first: int, last: int, width: int | None = None, height: int | None = None,
                   colorspace: str = "ACEScg", every: int = 1) -> dict:
    """Checks an EXR sequence (printf pattern, e.g. element.%04d.exr) is complete and every checked frame is RGBA half in
    `colorspace` (by colorInteropID and our ocio name) at the given size. Raises on the first problem."""
    missing = [n for n in range(first, last + 1) if not Path(pattern % n).is_file()]
    if missing:
        raise ValueError(f"{pattern}: {len(missing)} frames missing, first {missing[0]}")
    want_id = INTEROP.get(colorspace)
    for n in list(range(first, last + 1, every)) + [last]:
        h = header(pattern % n)
        problems = []
        if h["channels"] != ["R", "G", "B", "A"]:
            problems.append(f"channels {h['channels']}")
        if h["format"] != "half":
            problems.append(f"format {h['format']}")
        if h["ocio_colorspace"] != colorspace or (want_id and h["colorInteropID"] != want_id):
            problems.append(f"colorspace {h['ocio_colorspace']!r} / {h['colorInteropID']!r}")
        if width and (h["width"], h["height"]) != (width, height):
            problems.append(f"size {h['width']}x{h['height']}")
        if problems:
            raise ValueError(f"{pattern % n}: " + ", ".join(problems))
    return {"pattern": pattern, "frames": [first, last], "colorspace": colorspace, "width": h["width"],
            "height": h["height"]}


def read_rgba(path) -> tuple[np.ndarray, dict]:
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    spec = inp.spec()
    data = inp.read_image(oiio.FLOAT)
    attrs = {a.name.split(":", 1)[1]: a.value for a in spec.extra_attribs if a.name.startswith("atlas_vfx:")}
    attrs["colorInteropID"] = spec.get_string_attribute("colorInteropID")
    inp.close()
    return data, attrs
