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


LENS_DELIVERY = ("plate", "clean + STMaps")


def find_layer(folder, version: str = "plate") -> dict:
    """A layer folder (Card Project, LTX Return, ...) -> {name, pattern, first, last} from its render.json, for the
    version asked: 'plate' (on the plate as shot) or 'clean' (pinhole, undistorted). A folder from before versions
    existed was made with lens-free cameras, so its one sequence is both."""
    folder = Path(folder)
    rj = sorted(folder.glob("*.render.json"))
    if not rj:
        raise FileNotFoundError(f"no *.render.json in {folder}")
    r = json.loads(rj[0].read_text(encoding="utf-8"))
    versions = r.get("versions") or {"plate": r["pattern"], "clean": r["pattern"]}
    if version not in versions:
        raise ValueError(f"layer {r['card']} ({folder}) has no {version} version, only {sorted(versions)}: "
                         + ("its camera has a lens; give Card Project / LTX Prep the Sensor Lens Maps folder"
                            if version == "plate" else "use lens_delivery plate"))
    size = (r.get("sizes") or {}).get(version)
    return {"name": r["card"], "pattern": _fwd(folder / versions[version]), "first": r["frames"][0],
            "last": r["frames"][1], "colorspace": "ACEScg", "premultiplied": True, "version": version,
            "size": list(size) if size else None}


def plate_source(plate_path, kind: str = "dpx", frame_offset: int = 1001) -> dict:
    """Where the plate conversions live (next to the original, as Mike made them). kind 'virtual': a shot with no
    plate (Shot Setup generate mode); the script starts from a black Constant instead of a Read."""
    if kind == "virtual":
        return {"kind": "virtual", "file": None, "probe": None, "start_at": None,
                "range_note": "no plate: generated shot (black under the layers)"}
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
    raise ValueError(f"plate kind {kind!r}: dpx, prores, original or virtual")


def build_data(*, shot: str, plate: dict, plate_colorspace: str, plate_frames: tuple[int, int], fmt: tuple,
               ocio_config: str, camera_chan: str, camera_settings: dict, layers: list[dict], write_pattern: str,
               set_proxy_obj: str | None = None, prefix: str = "AVFX_", set_root: bool = True, fps: float = 24.0,
               x_offset: int = 0, lens_maps: dict | None = None) -> dict:
    """lens_maps: {"undistort", "redistort"} STMap EXRs (Sensor Lens Maps, G channel) for lens_delivery 'clean +
    STMaps': the clean layers are stacked, redistorted once and merged over the untouched plate."""
    if lens_maps:
        lens_maps = {k: _fwd(lens_maps[k]) for k in ("undistort", "redistort")}
    # names that become the script's file name, Nuke node names and the prefix of the nodes a re-run deletes
    shot = safe_name(shot, "shot")
    if not re.fullmatch(r"\w+", prefix or ""):
        raise ValueError(f"prefix {prefix!r}: letters, digits and _ only, not empty (a re-run deletes every node "
                         f"whose name starts with it)")
    layers = [dict(l, name=safe_name(l["name"], "layer name")) for l in layers]
    reads = ([plate["probe"]] if plate.get("probe") else []) + [_fwd(camera_chan), _fwd(ocio_config)] +         [l["pattern"] % l["first"] for l in layers]
    if lens_maps:
        reads += [lens_maps["undistort"], lens_maps["redistort"]]
    if set_proxy_obj:
        reads.append(_fwd(set_proxy_obj))
    return {"shot": shot, "prefix": prefix, "set_root": set_root, "fps": fps, "x_offset": x_offset,
            "frames": list(plate_frames), "format": list(fmt), "ocio_config": _fwd(ocio_config),
            "working_space": "ACEScg", "plate": dict(plate, colorspace=plate_colorspace),
            "camera": {"chan": _fwd(camera_chan), **{k: camera_settings[k] for k in
                                                       ("rot_order", "haperture", "vaperture", "win_translate")},
                       "units": camera_settings.get("units", "metres")},
            "layers": layers, "set_proxy_obj": _fwd(set_proxy_obj) if set_proxy_obj else None,
            "lens": lens_maps,
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
if pl["kind"] == "virtual":                     # no plate: black under the layers, the script's format
    plate = make(["Constant"], "Plate", 0, 0)
    setk(plate, "channels", "rgba")
    setk(plate, "color", [0, 0, 0, 0])
else:
    plate = make(["Read"], "Plate", 0, 0)
    setk(plate, "file", pl["file"])
    if pl["start_at"] is None:                  # numbered sequence in Nuke frames
        setk(plate, "first", first)
        setk(plate, "last", last)
        setk(plate, "origfirst", first)
        setk(plate, "origlast", last)
    else:                                       # movie: its frame 1 is Nuke frame start_at
        setk(plate, "frame_mode", "start at")
        setk(plate, "frame", str(pl["start_at"]))
    setk(plate, "colorspace", pl["colorspace"])
    setk(plate, "before", "hold")
    setk(plate, "after", "hold")
setk(plate, "label", "%s\n%s" % (pl["kind"], pl["range_note"]))

# 4. layers, back to front. lens 'plate': each (redistorted) layer straight over the plate. lens 'clean + STMaps':
#    the clean (pinhole) layers are stacked, the stack is redistorted ONCE with the lens STMap and goes over the
#    untouched plate (the plate itself is never resampled); an undistorted plate branch is there for lining up.
LENS = DATA.get("lens")


def stmap_read(name, path, x, y):
    rd = make(["Read"], name, x, y)
    setk(rd, "file", path)
    setk(rd, "raw", True)                       # an STMap is data: never colour converted
    setk(rd, "label", "STMap (Raw)")
    return rd


def stmap(name, src, uv, x, y, label):
    st = make(["STMap"], name, x, y)
    st.setInput(0, src)                         # src
    st.setInput(1, uv)                          # stmap
    setk(st, "uv", "rgb")
    setk(st, "label", label)
    return st


x, below, stack = 0, plate, None
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
    setk(rd, "label", L.get("version", "plate"))
    if LENS:
        if stack is None:
            stack = rd
        else:
            mg = make(["Merge2"], "Stack_" + L["name"], 180 * (i + 1), -80, operation="over")
            mg.setInput(0, stack)
            mg.setInput(1, rd)
            stack = mg
    else:
        mg = make(["Merge2"], "Over_" + L["name"], 0, 120 * (i + 1), operation="over")
        mg.setInput(0, below)                   # B
        mg.setInput(1, rd)                      # A
        below = mg
if LENS:
    stmap("Undistort_plate", plate, stmap_read("Read_undistort", LENS["undistort"], -180, 0), -180, 120,
               "plate -> clean (line up the camera here)")
    if stack is not None:
        rdm = stmap("Redistort_layers", stack, stmap_read("Read_redistort", LENS["redistort"], 0, -80), 180, 40,
                    "clean layers -> plate as shot")
        mg = make(["Merge2"], "Over_layers", 0, 120, operation="over")
        mg.setInput(0, plate)
        mg.setInput(1, rdm)
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
setk(cam, "label", "%s, ZXY, %d frames" % (c["units"], len(rows)))

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


# --------------------------------------------------------------------------------------------- .nk text (Indie)
# Nuke Indie lets Python create or touch only 10 nodes per session (live 2026-11-01: the clean + STMaps build needs
# 13). Pasting script text into the Node Graph has no such limit, so the same graph is also written as .nk text:
# open it in a text editor, copy everything, click in the Node Graph, Ctrl+V. Every value goes through
# camera_side/nuke_text (nk_str / nk_name), never pasted raw.
# Stack convention (confirmed empirically in Atlas Camera's exporters/nuke_exporter.py): the LAST push becomes
# input 0. So a Merge2 (B = 0, A = 1) is "push A; push B", an STMap (src = 0, stmap = 1) is "push map; push src".

def _nk_node(cls: str, name: str, knobs: list[tuple[str, str]], x: int, y: int, inputs: int = 0) -> str:
    body = [f" inputs {int(inputs)}"] + [f" {k} {v}" for k, v in knobs]
    body += [f" name {nt.nk_name(name)}", f" xpos {int(x)}", f" ypos {int(y)}"]
    return f"{cls} {{\n" + "\n".join(body) + "\n}"


def _curve(values, first: int) -> str:
    return "{curve x%d %s}" % (first, " ".join(f"{v:.9g}" for v in values))


def _comment_text(value) -> str:
    """A value for a `#` comment line: one line, no control characters. A newline would end the comment and the rest
    would be read as live script on paste (the values come from widgets and camera files, both untrusted)."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in str(value)).split())


def nk_text(data: dict) -> str:
    """The build as Nuke script text, for pasting into Indie's Node Graph (same nodes as the Python build, except
    the Root settings, which a paste can't set: they are listed in the header comment)."""
    from atlas_vfx.camera_side import camera_export as ce

    P = data["prefix"]
    q = nt.nk_str
    first, last = data["frames"]
    w, h, pa = data["format"]
    c1 = _comment_text
    out = ["# atlas-vfx Nuke build for Indie: copy ALL of this, click in the Node Graph, Ctrl+V.",
           f"# Set by hand (a paste can't): Project Settings > frame range {first}-{last}, fps {data['fps']:g}, "
           f"format {w}x{h}, colour management OCIO, config {c1(data['ocio_config'])}, working space "
           f"{c1(data['working_space'])}.",
           f"# shot {c1(data['shot'])}; camera units {c1(data['camera'].get('units', 'metres'))}"]
    sets = []

    def put(text, var):
        out.append(text)
        out.append(f"set N_{var} [stack 0]")
        sets.append(var)

    pl = data["plate"]
    if pl["kind"] == "virtual":                  # no plate: a black Constant in the script's format
        put(_nk_node("Constant", P + "Plate", [("channels", "rgba"), ("color", "{0 0 0 0}"),
                                               ("label", q(f"{pl['kind']}\n{pl['range_note']}"))], 0, 0), "plate")
    else:
        knobs = [("file", q(pl["file"])), ("colorspace", q(pl["colorspace"])), ("before", "hold"),
                 ("after", "hold"), ("label", q(f"{pl['kind']}\n{pl['range_note']}"))]
        if pl["start_at"] is None:
            knobs += [("first", str(first)), ("last", str(last)), ("origfirst", str(first)),
                      ("origlast", str(last))]
        else:
            knobs += [("frame_mode", q("start at")), ("frame", q(str(pl["start_at"])))]
        put(_nk_node("Read", P + "Plate", knobs, 0, 0), "plate")
    for i, L in enumerate(data["layers"]):
        put(_nk_node("Read", P + "Layer_" + L["name"], [
            ("file", q(L["pattern"].replace("%04d", "####"))), ("first", str(L["first"])), ("last", str(L["last"])),
            ("origfirst", str(L["first"])), ("origlast", str(L["last"])), ("colorspace", q(L["colorspace"])),
            ("premultiplied", "true" if L["premultiplied"] else "false"), ("before", "black"), ("after", "black"),
            ("label", q(L.get("version", "plate")))], 180 * (i + 1), -200), f"L{i}")
    lens = data.get("lens")
    top = "plate"
    if lens:
        stack = None
        for i, _L in enumerate(data["layers"]):
            if stack is None:
                stack = f"L{i}"
                continue
            out += [f"push $N_L{i}", f"push $N_{stack}"]          # A = this layer, B = the stack below
            put(_nk_node("Merge2", P + f"Stack_{data['layers'][i]['name']}", [("operation", "over")], 180 * (i + 1),
                         -80, inputs=2), f"S{i}")
            stack = f"S{i}"
        for nm, path, x in (("redistort", lens["redistort"], 0), ("undistort", lens["undistort"], -180)):
            put(_nk_node("Read", P + f"Read_{nm}", [("file", q(path)), ("colorspace", q("Raw")), ("raw", "true"),
                                                     ("label", q("STMap (Raw)"))], x, -80), nm)
        out += ["push $N_undistort", "push $N_plate"]
        put(_nk_node("STMap", P + "Undistort_plate", [("uv", "rgb"),
                                                      ("label", q("plate -> clean (line up the camera here)"))],
                     -180, 120, inputs=2), "ud")
        if stack is not None:
            out += ["push $N_redistort", f"push $N_{stack}"]
            put(_nk_node("STMap", P + "Redistort_layers", [("uv", "rgb"), ("label", q("clean layers -> plate as shot"))],
                         180, 40, inputs=2), "rd")
            out += ["push $N_rd", "push $N_plate"]
            put(_nk_node("Merge2", P + "Over_layers", [("operation", "over")], 0, 120, inputs=2), "over")
            top = "over"
    else:
        for i, L in enumerate(data["layers"]):
            out += [f"push $N_L{i}", f"push $N_{top}"]
            put(_nk_node("Merge2", P + "Over_" + L["name"], [("operation", "over")], 0, 120 * (i + 1), inputs=2),
                f"O{i}")
            top = f"O{i}"
    out.append(f"push $N_{top}")
    wr = data["write"]
    out.append(_nk_node("Write", P + "Write_comp", [("file", q(wr["file"].replace("%04d", "####"))),
                                                    ("file_type", "exr"), ("colorspace", q(wr["colorspace"])),
                                                    ("channels", "rgba"), ("create_directories", "true")],
                        0, 120 * (len(data["layers"]) + 2), inputs=1))
    c = data["camera"]
    rows = ce.read_chan(c["chan"])
    fr = sorted(rows)
    t = [_curve([rows[f][i] for f in fr], fr[0]) for i in range(3)]
    r = [_curve([rows[f][3 + i] for f in fr], fr[0]) for i in range(3)]
    out.append(_nk_node("Camera3", P + "Camera", [
        ("rot_order", "ZXY"), ("translate", "{" + " ".join(t) + "}"), ("rotate", "{" + " ".join(r) + "}"),
        ("focal", "{" + _curve([rows[f][6] for f in fr], fr[0]) + "}"),
        ("haperture", f"{c['haperture']:.9g}"), ("vaperture", f"{c['vaperture']:.9g}"),
        ("win_translate", "{%s %s}" % tuple(f"{v:.9g}" for v in c["win_translate"])),
        ("label", q(f"{c.get('units', 'metres')}, ZXY, {len(fr)} frames"))], -300, 0))
    if data.get("set_proxy_obj"):
        out.append(_nk_node("ReadGeo2", P + "SetProxy", [("file", q(data["set_proxy_obj"]))], -300, 150))
    return "\n".join(out) + "\n"


def write_nk(data: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(nk_text(data), encoding="utf-8")
    return p
