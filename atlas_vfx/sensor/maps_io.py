"""Sensor map files: float32 ZIP EXRs tagged OCIO Raw (colorInteropID data), and the Nuke Reads to wire them.

RS folder (Sensor RS Maps):
  rs.####.exr            apply RS: output sensor image, samples the clean image      (STMap, uv in R/G, B = 0)
  ders.####.exr          remove RS: output clean image, samples the plate            (STMap)
  rs_blur<k>.####.exr    per-row motion blur sub-samples (only with shutter angle > 0; average the STMap results)
  sensor_rs.json         settings, where each value came from, per-frame skew + round trip, sidecar
  sensor_rs.nk           Reads for the above (paste / File > Insert; wire each into an STMap's stmap input)
Lens folder (Sensor Lens Maps), one file each (the lens does not change over the shot):
  undistort_{r,g,b}.exr  redistort_{r,g,b}.exr  vignette.exr (RGB = gain)  sensor_lens.json  sensor_lens.nk
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import nuke_text as nt

RAW = "Raw"


def write_exr(path, chans: dict, attrs: dict | None = None) -> Path:
    """chans {name: (H, W)} float -> float32 ZIP EXR, OCIO Raw."""
    import OpenImageIO as oiio

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    names = list(chans)
    data = np.stack([np.asarray(chans[n], np.float32) for n in names], -1)
    H, W = data.shape[:2]
    spec = oiio.ImageSpec(W, H, len(names), oiio.FLOAT)
    spec.channelnames = tuple(names)
    spec.attribute("compression", "zip")
    spec.attribute("oiio:ColorSpace", "data")          # OIIO 3 writes colorInteropID = data
    spec.attribute("atlas_vfx:ocio_colorspace", RAW)
    for k, v in (attrs or {}).items():
        spec.attribute(f"atlas_vfx:{k}", v if isinstance(v, (int, float, str)) else json.dumps(v))
    out = oiio.ImageOutput.create(str(p))
    if out is None or not out.open(str(p), spec):
        raise RuntimeError(f"cannot write {p}: {oiio.geterror()}")
    out.write_image(np.ascontiguousarray(data))
    out.close()
    return p


def write_stmap(path, st: np.ndarray, attrs: dict | None = None) -> Path:
    return write_exr(path, {"R": st[..., 0], "G": st[..., 1], "B": np.zeros(st.shape[:2], np.float32)}, attrs)


def read_exr(path) -> tuple[np.ndarray, list, dict]:
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    spec = inp.spec()
    data = inp.read_image(oiio.FLOAT)
    attrs = {a.name.split(":", 1)[1]: a.value for a in spec.extra_attribs if a.name.startswith("atlas_vfx:")}
    attrs["colorInteropID"] = spec.get_string_attribute("colorInteropID")
    attrs["compression"] = spec.get_string_attribute("compression")
    inp.close()
    return data, list(spec.channelnames), attrs


def read_stmap(path) -> np.ndarray:
    data, names, _ = read_exr(path)
    return np.ascontiguousarray(data[..., [names.index("R"), names.index("G")]])


def frames_in(folder, stem: str) -> dict[int, Path]:
    rx = re.compile(rf"^{re.escape(stem)}\.(\d+)\.exr$", re.I)
    out = {}
    for p in Path(folder).glob(f"{stem}.*.exr"):
        m = rx.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _fwd(p) -> str:
    return str(p).replace("\\", "/")


def _read(path, name, x, first=None, last=None, y=0) -> str:
    """One Read node. The path is a quoted knob value through nuke_text.nk_str, so a folder name can never close
    the knob or the node; the node name goes through nk_name."""
    rng = f"\n first {int(first)}\n last {int(last)}" if first is not None else ""
    return (f"Read {{\n inputs 0\n file {nt.nk_str(_fwd(path))}{rng}\n colorspace {nt.nk_str(RAW)}\n raw true\n"
            f" on_error black\n name {nt.nk_name(name)}\n xpos {x}\n ypos {y}\n}}")


def nuke_rs(folder, first: int, last: int, blur: int = 0) -> str:
    o = Path(folder)
    lines = ["# atlas-vfx Sensor RS maps: wire Read_rs into STMap.stmap (UV channels rgb) over the CLEAN image to put",
             "# the plate's rolling shutter on it; Read_ders over the PLATE takes it off. Raw, no colour transform.",
             f"Root {{\n inputs 0\n first_frame {first}\n last_frame {last}\n}}",
             _read(o / "rs.####.exr", "Read_rs", 0, first, last),
             _read(o / "ders.####.exr", "Read_ders", 150, first, last)]
    for k in range(blur):
        lines.append(_read(o / f"rs_blur{k}.####.exr", f"Read_rs_blur{k}", 300 + 150 * k, first, last))
    return "\n".join(lines) + "\n"


def nuke_lens(folder) -> str:
    o = Path(folder)
    lines = ["# atlas-vfx Sensor Lens maps: STMap per channel (Shuffle each channel through its own STMap, then",
             "# ShuffleCopy back), redistort = clean -> plate look, undistort = plate -> clean. vignette.exr: multiply",
             "# a LINEAR (ACEScg) image by it to add the plate's vignette, divide to remove it. Raw, no colour."]
    x = 0
    for kind in ("redistort", "undistort"):
        for c in "rgb":
            lines.append(_read(o / f"{kind}_{c}.exr", f"Read_{kind}_{c}", x))
            x += 150
    lines.append(_read(o / "vignette.exr", "Read_vignette", x))
    return "\n".join(lines) + "\n"
