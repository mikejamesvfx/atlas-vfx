"""Nuke Builder (catalogue 24): writes build_<shot>.py, a script Mike pastes into the Nuke Script Editor (Indie 17.1:
scripts are encrypted on disk and Python cannot make WriteGeo, so everything arrives as Python).

The script makes, all named with a prefix (default AVFX_) so a re-run deletes and rebuilds only its own nodes:
- Read of the plate (DPX sequence by default, colorspace from the plate, e.g. F-Log F-Gamut, from the shipped OCIO config)
- a Camera from the exported .chan (metres, rot order ZXY), apertures and window offset from the camera JSON
- optionally a ReadGeo of the Set Proxy OBJ (for lining up)
- a Read per layer EXR sequence (ACEScg, premultiplied), merged over the plate back to front
- a Write (EXR, ACEScg) that is not rendered
Optionally it switches the root to OCIO with the shipped config (ACEScg working space) and sets the frame range,
fps and format. Every knob it sets is tried one by one; anything Nuke refuses is reported at the end, not fatal, so
one renamed knob in a new Nuke does not stop the build. It reads nothing but the files listed in DATA["reads"].
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from atlas_vfx.camera_side import nuke_text as nt
from atlas_vfx.camera_side.plate import safe_name

# Node classes that Nuke Indie (and plain Nuke) does not have or cannot make from Python. The test fails if the
# generated script names any of them.
NUKEX_ONLY = ("WriteGeo", "CameraTracker", "PlanarTracker3D", "DepthGenerator", "ModelBuilder", "PointCloudGenerator",
              "LensDistortion", "LensDistortion2", "BlinkScript", "Kronos", "MotionBlur", "VectorGenerator",
              "Denoise2", "CopyCat", "Inference", "BigCat", "SmartVector", "VectorDistort", "ParticleEmitter",
              "ParticleSystem", "F_RigRemoval", "F_Steadiness", "F_WireRemoval", "GridWarpTracker", "Bokeh",
              "ZDefocus2", "Relight", "CameraSolver")


def _fwd(p) -> str:
    return str(p).replace("\\", "/")


def find_layer(folder) -> dict:
    """A Card Project output folder -> {name, pattern, first, last} from its render.json."""
    folder = Path(folder)
    rj = sorted(folder.glob("*.render.json"))
    if not rj:
        raise FileNotFoundError(f"no *.render.json in {folder}")
    r = json.loads(rj[0].read_text(encoding="utf-8"))
    return {"name": r["card"], "pattern": _fwd(folder / r["pattern"]), "first": r["frames"][0], "last": r["frames"][1],
            "colorspace": "ACEScg", "premultiplied": True}


def plate_source(plate_path, kind: str = "dpx", frame_offset: int = 1001) -> dict:
    """Where the plate conversions live (next to the original, as Mike made them)."""
    p = Path(plate_path)
    if kind == "dpx":
        d = p.parent / f"{p.stem}_dpx"
        return {"kind": "dpx", "file": _fwd(d / f"{p.stem}.%04d.dpx"), "probe": _fwd(d / f"{p.stem}.{frame_offset:04d}.dpx"),
                "start_at": None, "range_note": "10-bit DPX, full-range code values as recorded"}
    if kind == "prores":
        f = p.parent / f"{p.stem}_ProRes422HQ.mov"
        return {"kind": "prores", "file": _fwd(f), "probe": _fwd(f), "start_at": frame_offset,
                "range_note": "ProRes 422 HQ, video range (Nuke's reader expands it)"}
    if kind == "original":
        return {"kind": "original", "file": _fwd(p), "probe": _fwd(p), "start_at": frame_offset,
                "range_note": "HEVC full range: check Nuke reads it as full range before trusting it"}
    raise ValueError(f"plate kind {kind!r}: dpx, prores or original")


def build_data(*, shot: str, plate: dict, plate_colorspace: str, plate_frames: tuple[int, int], fmt: tuple,
               ocio_config: str, camera_chan: str, camera_settings: dict, layers: list[dict], write_pattern: str,
               set_proxy_obj: str | None = None, prefix: str = "AVFX_", set_root: bool = True, fps: float = 24.0,
               x_offset: int = 0) -> dict:
    # names that become the script's file name, Nuke node names and the prefix of the nodes a re-run deletes
    shot = safe_name(shot, "shot")
    if not re.fullmatch(r"\w+", prefix or ""):
        raise ValueError(f"prefix {prefix!r}: letters, digits and _ only, not empty (a re-run deletes every node "
                         f"whose name starts with it)")
    layers = [dict(l, name=safe_name(l["name"], "layer name")) for l in layers]
    reads = [plate["probe"], _fwd(camera_chan), _fwd(ocio_config)] + [l["pattern"] % l["first"] for l in layers]
    if set_proxy_obj:
        reads.append(_fwd(set_proxy_obj))
    return {"shot": shot, "prefix": prefix, "set_root": set_root, "fps": fps, "x_offset": x_offset,
            "frames": list(plate_frames), "format": list(fmt), "ocio_config": _fwd(ocio_config),
            "working_space": "ACEScg", "plate": dict(plate, colorspace=plate_colorspace),
            "camera": {"chan": _fwd(camera_chan), **{k: camera_settings[k] for k in
                                                       ("rot_order", "haperture", "vaperture", "win_translate")}},
            "layers": layers, "set_proxy_obj": _fwd(set_proxy_obj) if set_proxy_obj else None,
            "write": {"file": _fwd(write_pattern), "colorspace": "ACEScg"}, "reads": reads}


BODY = r'''
import os
import nuke

P = DATA["prefix"]
problems = []


def setk(node, knob, value):
    """Set one knob; note (not raise) what this Nuke refuses."""
    try:
        k = node[knob]
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                k.setValue(v, i)
        else:
            k.setValue(value)
    except Exception as e:
        problems.append("%s.%s = %r: %s" % (node.name(), knob, value, e))


def make(classes, name, x, y, **knobs):
    node = None
    for c in classes:
        try:
            node = getattr(nuke.nodes, c)()
            break
        except Exception:
            continue
    if node is None:
        raise RuntimeError("none of %s exist in this Nuke" % (classes,))
    node.setName(P + name)
    node.setXYpos(x + DATA.get("x_offset", 0), y)
    for k, v in knobs.items():
        setk(node, k, v)
    return node


# 1. remove what an earlier run made (only our prefix)
old = [n for n in nuke.allNodes(recurseGroups=False) if n.name().startswith(P)]
for n in old:
    nuke.delete(n)

# 2. every file must be there before anything is built
missing = [f for f in DATA["reads"] if not os.path.exists(f)]
if missing:
    raise RuntimeError("atlas-vfx build: missing files:\n  " + "\n  ".join(missing))

first, last = DATA["frames"]
root = nuke.root()
if DATA["set_root"]:
    setk(root, "colorManagement", "OCIO")
    setk(root, "OCIO_config", "custom")
    setk(root, "customOCIOConfigPath", DATA["ocio_config"])
    setk(root, "workingSpaceLUT", DATA["working_space"])
    setk(root, "first_frame", first)
    setk(root, "last_frame", last)
    setk(root, "fps", DATA["fps"])
    w, h, pa = DATA["format"]
    fname = "AVFX_%dx%d" % (w, h)
    try:
        nuke.addFormat("%d %d 0 0 %d %d %s %s" % (w, h, w, h, pa, fname))
    except Exception as e:
        problems.append("addFormat: %s" % e)
    setk(root, "format", fname)

# 3. plate
pl = DATA["plate"]
plate = make(["Read"], "Plate", 0, 0)
setk(plate, "file", pl["file"])
if pl["start_at"] is None:                      # numbered sequence in Nuke frames
    setk(plate, "first", first)
    setk(plate, "last", last)
    setk(plate, "origfirst", first)
    setk(plate, "origlast", last)
else:                                           # movie: its frame 1 is Nuke frame start_at
    setk(plate, "frame_mode", "start at")
    setk(plate, "frame", str(pl["start_at"]))
setk(plate, "colorspace", pl["colorspace"])
setk(plate, "before", "hold")
setk(plate, "after", "hold")
setk(plate, "label", "%s\n%s" % (pl["kind"], pl["range_note"]))

# 4. layers over the plate, back to front
x, below = 0, plate
for i, L in enumerate(DATA["layers"]):
    rd = make(["Read"], "Layer_" + L["name"], 180 * (i + 1), -200)
    setk(rd, "file", L["pattern"])
    setk(rd, "first", L["first"])
    setk(rd, "last", L["last"])
    setk(rd, "origfirst", L["first"])
    setk(rd, "origlast", L["last"])
    setk(rd, "colorspace", L["colorspace"])
    setk(rd, "premultiplied", L["premultiplied"])
    setk(rd, "before", "black")
    setk(rd, "after", "black")
    mg = make(["Merge2"], "Over_" + L["name"], 0, 120 * (i + 1), operation="over")
    mg.setInput(0, below)                       # B
    mg.setInput(1, rd)                          # A
    below = mg

# 5. write (not rendered)
wr = make(["Write"], "Write_comp", 0, 120 * (len(DATA["layers"]) + 1))
wr.setInput(0, below)
setk(wr, "file", DATA["write"]["file"])
setk(wr, "file_type", "exr")
setk(wr, "colorspace", DATA["write"]["colorspace"])
setk(wr, "channels", "rgba")
setk(wr, "create_directories", True)

# 6. camera from the chan (parsed here, not import_chan, so the rot order and units are ours)
cam = make(["Camera3", "Camera2", "Camera", "Camera4"], "Camera", -300, 0)
c = DATA["camera"]
setk(cam, "rot_order", c["rot_order"])
setk(cam, "haperture", c["haperture"])
setk(cam, "vaperture", c["vaperture"])
setk(cam, "win_translate", c["win_translate"])
rows = [l.split() for l in open(c["chan"]).read().splitlines() if l.strip()]
try:
    for kn in ("translate", "rotate", "focal"):
        cam[kn].setAnimated()
    for r in rows:
        f = int(r[0])
        v = [float(t) for t in r[1:8]]
        for i in range(3):
            cam["translate"].setValueAt(v[i], f, i)
            cam["rotate"].setValueAt(v[3 + i], f, i)
        cam["focal"].setValueAt(v[6], f)
except Exception as e:
    problems.append("camera animation: %s" % e)
setk(cam, "label", "metres, ZXY, %d frames" % len(rows))

# 7. set proxy geometry, for lining up (optional)
if DATA["set_proxy_obj"]:
    try:
        geo = make(["ReadGeo2", "ReadGeo"], "SetProxy", -300, 150)
        setk(geo, "file", DATA["set_proxy_obj"])
    except Exception as e:
        problems.append("set proxy: %s" % e)

msg = "atlas-vfx: built %s (%d layers, frames %d-%d), removed %d old nodes." % (
    DATA["shot"], len(DATA["layers"]), first, last, len(old))
if problems:
    msg += "\nNot set (check by hand):\n  " + "\n  ".join(problems)
print(msg)
'''


def script_text(data: dict) -> str:
    """The script: a fixed header, DATA as one Python literal built by nuke_text.py_literal (no string from the
    data ever touches the source text directly), then BODY. Nothing in `data` can add a statement."""
    head = ("# atlas-vfx Nuke build. Paste into the Script Editor and run. Safe to re-run: it replaces only the\n"
            "# nodes whose names start with DATA['prefix']. Reads only the files in DATA['reads'].\n"
            "# Generated by atlas_vfx/deliver/nuke_builder.py; shot and paths are in DATA below.\n"
            "import json\n"
            f"DATA = {nt.py_literal(data)}\n")
    return head + BODY


def write_script(data: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(script_text(data), encoding="utf-8")
    return p


def node_classes_used(text: str) -> set[str]:
    """Node classes the script can create: names passed to make([...]) and nuke.nodes.X / nuke.createNode('X')."""
    found = set(re.findall(r"nuke\.nodes\.(\w+)", text)) | set(re.findall(r"createNode\(\s*['\"](\w+)", text))
    for group in re.findall(r"make\(\[([^\]]*)\]", text):
        found |= set(re.findall(r"['\"](\w+)['\"]", group))
    return found
