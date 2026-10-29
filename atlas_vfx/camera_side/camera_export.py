"""Camera Export (catalogue 25): a shot camera -> Nuke .chan (metres, rot order ZXY), the Nuke camera settings the chan
cannot carry (apertures, window offset, rot order) as JSON, and a time-sampled USD camera (.usda, hand written: the
V141 venv has no pxr, and Atlas Camera's USDExporter takes its own solve types; see UPSTREAM_TODO).

Nuke: haperture = sensor width; vaperture = haperture * height / width / pixel aspect (Nuke ignores vaperture for the
projection and uses the format, but a matching value keeps the knobs honest). The principal point becomes
win_translate, in Nuke's window units (half the frame width = 1, y up).

USD: Y up, metersPerUnit 1, one `xformOp:transform` time sample per frame (USD matrices are row-vector, so the
transpose of cam_to_world), focalLength and apertures in mm as DCCs read them, aperture offsets from the principal
point, timeCodesPerSecond = fps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc


def nuke_settings(cam: sc.ShotCamera) -> dict:
    k = cam.intrinsics(cam.frames[0])
    pa = float(cam.data["plate"].get("pixel_aspect", 1.0))
    hap = float(cam.data["sensor"]["width_mm"])
    return {"rot_order": "ZXY", "xform_order": "SRT", "haperture": hap, "vaperture": hap * k.height / k.width / pa,
            "win_translate": [(k.cx - k.width / 2) / (k.width / 2), -(k.cy - k.height / 2) / (k.width / 2) * pa],
            "format": [k.width, k.height, pa], "frames": [cam.frames[0], cam.frames[-1]],
            "fps": float(cam.data.get("fps", 24.0)), "units": "metres", "camera": cam.name}


def write_chan(cam: sc.ShotCamera, path, units_per_m: float = 1.0) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cam.to_nuke_chan(p, units_per_m=units_per_m)
    return p


def read_chan(path) -> dict:
    """frame -> (tx, ty, tz, rx, ry, rz, focal) from a chan written by write_chan."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = line.split()
            out[int(v[0])] = tuple(float(x) for x in v[1:8])
    return out


def _m(M) -> str:
    return "(" + ", ".join("(" + ", ".join(f"{x:.9g}" for x in row) + ")" for row in M) + ")"


def write_usda(cam: sc.ShotCamera, path, near: float = 0.01, far: float = 10000.0) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    k0 = cam.intrinsics(cam.frames[0])
    hap = float(cam.data["sensor"]["width_mm"])
    pa = float(cam.data["plate"].get("pixel_aspect", 1.0))
    vap = hap * k0.height / k0.width / pa
    hoff = (k0.cx - k0.width / 2) / k0.width * hap
    voff = -(k0.cy - k0.height / 2) / k0.height * vap
    xf = ",\n".join(f"            {n}: {_m(cam.cam_to_world(n).T)}" for n in cam.frames)
    focals = [float(cam.intrinsics(n).focal_mm) for n in cam.frames]
    if max(focals) - min(focals) < 1e-9:
        focal = f"        float focalLength = {focals[0]:.6f}"
    else:
        focal = ("        float focalLength.timeSamples = {\n" +
                 ",\n".join(f"            {n}: {f:.6f}" for n, f in zip(cam.frames, focals)) + "\n        }")
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in cam.name) or "camera"
    if name[0].isdigit():
        name = "c_" + name
    text = f'''#usda 1.0
(
    defaultPrim = "{name}"
    upAxis = "Y"
    metersPerUnit = 1
    timeCodesPerSecond = {float(cam.data.get("fps", 24.0)):g}
    framesPerSecond = {float(cam.data.get("fps", 24.0)):g}
    startTimeCode = {cam.frames[0]}
    endTimeCode = {cam.frames[-1]}
    doc = "atlas-vfx Camera Export: shot-camera@0.1 '{cam.name}', metres, Y up, camera looks down -Z"
)

def Camera "{name}"
{{
        float horizontalAperture = {hap:.6f}
        float verticalAperture = {vap:.6f}
        float horizontalApertureOffset = {hoff:.6f}
        float verticalApertureOffset = {voff:.6f}
{focal}
        float2 clippingRange = ({near:g}, {far:g})
        token projection = "perspective"
        matrix4d xformOp:transform.timeSamples = {{
{xf}
        }}
        uniform token[] xformOpOrder = ["xformOp:transform"]
}}
'''
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_usda_transforms(path) -> dict:
    """frame -> 4x4 cam_to_world (column-vector) from a usda written by write_usda (for tests)."""
    import re

    text = Path(path).read_text(encoding="utf-8")
    block = text.split("xformOp:transform.timeSamples = {", 1)[1].split("\n        }", 1)[0]
    out = {}
    for m in re.finditer(r"(-?\d+): (\(\(.*?\)\))", block):
        rows = json.loads(m.group(2).replace("(", "[").replace(")", "]"))
        out[int(m.group(1))] = np.array(rows).T
    return out


def export(cam: sc.ShotCamera, out_dir, name: str | None = None, units_per_m: float = 1.0) -> dict:
    out = Path(out_dir)
    name = name or cam.name
    chan = write_chan(cam, out / f"{name}.chan", units_per_m)
    usda = write_usda(cam, out / f"{name}.usda")
    settings = dict(nuke_settings(cam), units_per_m=units_per_m, chan=str(chan), usda=str(usda))
    js = out / f"{name}.nuke_camera.json"
    js.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return {"chan": str(chan), "usda": str(usda), "nuke_camera": str(js), "settings": settings}
