"""Build workflows/DEMO_sensor_pass_s2s3.json: a COPY of DEMO_all_in_one_s2s3.json plus a "Sensor pass" group.

The group shows the chain: the plate tracked in ComfyUI (Plate Read -> Camera Import -> Plate Depth's folder) ->
Sensor Readout Solve / Sensor Lens Maps -> Sensor RS Maps; the plate taken to clean space for LTX (Sensor Apply,
remove camera); an LTX render of the shot -> Sensor Apply (apply camera) -> movie. The demo file is only read.
Node widgets come from the sensor classes' INPUT_TYPES (no running ComfyUI needed); core nodes are copied from the
demo's own instances. Writes the repo copy and the V141 Workflows sidebar copy.

Run: <V141 python> tools\\sensor_workflow.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.sensor.node_apply import AtlasVFXSensorApply  # noqa: E402
from atlas_vfx.sensor.node_lens_maps import AtlasVFXSensorLensMaps  # noqa: E402
from atlas_vfx.sensor.node_readout_solve import AtlasVFXSensorReadoutSolve  # noqa: E402
from atlas_vfx.sensor.node_rs_maps import AtlasVFXSensorRSMaps  # noqa: E402

SRC = ROOT / "workflows" / "DEMO_all_in_one_s2s3.json"
OUT = ROOT / "workflows" / "DEMO_sensor_pass_s2s3.json"
SIDEBAR = Path(r"C:\Users\miike\ComfyUI_V141\ComfyUI\user\default\workflows\atlas-vfx")
WORK = r"C:\Users\miike\atlas-vfx-work\DSCF3000"
WIDGET = {"INT", "FLOAT", "STRING", "BOOLEAN"}
COLOR, BG = "#2a3a2a", "#3f5a3f"


class Builder:
    def __init__(self, wf):
        self.wf = wf
        self.nid = wf["last_node_id"]
        self.lid = wf["last_link_id"]
        self.by_id = {n["id"]: n for n in wf["nodes"]}

    def link(self, src, slot, dst, name):
        dn = self.by_id[dst]
        di = next(i for i, x in enumerate(dn["inputs"]) if x["name"] == name)
        typ = self.by_id[src]["outputs"][slot]["type"]
        self.lid += 1
        dn["inputs"][di]["link"] = self.lid
        out = self.by_id[src]["outputs"][slot]
        out["links"] = (out.get("links") or []) + [self.lid]
        out["slot_index"] = slot
        self.wf["links"].append([self.lid, src, slot, dst, di, typ])

    def add(self, node):
        self.nid += 1
        node["id"] = self.nid
        node.setdefault("flags", {})
        node.setdefault("order", self.nid)
        node.setdefault("mode", 0)
        node.setdefault("properties", {"Node name for S&R": node["type"]})
        node["color"], node["bgcolor"] = COLOR, BG
        self.wf["nodes"].append(node)
        self.by_id[self.nid] = node
        return self.nid

    def ours(self, cls, title, pos, size, values=None):
        """A sensor node from its INPUT_TYPES: widgets in order, sockets for the rest."""
        values = values or {}
        spec = cls.INPUT_TYPES()
        inputs, widgets = [], []
        for group in ("required", "optional"):
            for name, s in spec.get(group, {}).items():
                t, opts = s[0], (s[1] if len(s) > 1 else {})
                if isinstance(t, list) or (t in WIDGET and not opts.get("forceInput")):
                    typ = "COMBO" if isinstance(t, list) else t
                    inputs.append({"name": name, "type": typ, "link": None, "widget": {"name": name}})
                    widgets.append(values.get(name, opts.get("default", t[0] if isinstance(t, list) else "")))
                else:
                    inputs.append({"name": name, "type": t, "link": None})
        outputs = [{"name": n, "type": t, "links": []} for n, t in zip(cls.RETURN_NAMES, cls.RETURN_TYPES)]
        return self.add({"type": cls.__name__, "title": title, "pos": pos, "size": size, "inputs": inputs,
                         "outputs": outputs, "widgets_values": widgets})

    def clone(self, src_id, title, pos, widgets=None, size=None):
        n = copy.deepcopy(self.by_id[src_id])
        for i in n.get("inputs", []):
            i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
        n["title"], n["pos"] = title, pos
        if size:
            n["size"] = size
        if widgets is not None:
            n["widgets_values"] = widgets
        n.pop("id", None)
        n.pop("order", None)
        return self.add(n)


def build():
    wf = json.loads(SRC.read_text(encoding="utf-8"))
    b = Builder(wf)
    plate, camera = 2, 4                                  # Plate Read, Camera Import (.abc) in the demo
    x0, y0 = 0, 2300
    wf["groups"].append({"title": "8  Sensor pass: plate camera off the plate, onto the LTX render",
                         "bounding": [x0, y0 - 40, 3420, 1300], "color": "#3f8f3f", "font_size": 28, "flags": {}})
    note = b.clone(1, "Sensor pass", [x0 - 600, y0 + 40], size=[560, 760], widgets=[
        "# Sensor pass (rolling shutter + lens as STMaps)\n"
        "LTX renders come out global-shutter and rectilinear. The plate is not: the X-H2 reads its sensor top to "
        "bottom over ~30.9 ms (4K HQ), so every row is a slightly different camera. Convention: take the plate's "
        "camera OFF the plate, generate in clean space, put it back ON the render last.\n\n"
        "**Plate side (tracked here in ComfyUI):** Camera Import (group 2) + Plate Depth's folder feed "
        "**Sensor RS Maps**. **Sensor Readout Solve** measures the readout from the solve's tracks and wires in; "
        "readout_source picks measured or the typed 30.9 ms. The .abc camera is a global-shutter solve: its poses "
        "already soak up most of the skew, and the deck reprojection test showed the physical 30.9 ms on top makes "
        "alignment worse. So this graph uses the measured residual (~7 ms). Switch to typed 30.9 ms only with a "
        "camera from a rolling-shutter-aware solve. "
        "**Sensor Lens Maps** takes the lens from the camera (none on this solve) with typed overrides, plus CA "
        "and vignette.\n\n"
        "**Into LTX:** Sensor Apply, *remove camera*, takes the plate to clean space (LTX Prep cannot take an "
        "IMAGE plate yet: see UPSTREAM_TODO).\n\n"
        "**Out of LTX:** Sensor Apply, *apply camera*: vignette (linear) -> CA + distortion -> RS. With ltx_prep "
        "connected it reads the render's frame range and crop windows from LTX Prep.\n\n"
        "Every map is a Raw float ZIP EXR with Nuke Reads (sensor_rs.nk, sensor_lens.nk) next to it."])
    ro = b.ours(AtlasVFXSensorReadoutSolve, "Sensor Readout Solve (from the solve's tracks)", [x0 + 20, y0 + 40],
                [440, 300], {"points_camera": WORK + r"\camera\DSCF3000_f16_GUESS_stairs.shot_camera.json",
                             "frame_step": 3})
    b.link(camera, 0, ro, "shot_camera")
    lens = b.ours(AtlasVFXSensorLensMaps, "Sensor Lens Maps (lens from the camera)", [x0 + 20, y0 + 400],
                  [440, 560], {"map_scale": 0.5})
    b.link(camera, 0, lens, "shot_camera")
    rsm = b.ours(AtlasVFXSensorRSMaps, "Sensor RS Maps: S2-S3 (measured residual, GS-solved camera)",
                 [x0 + 500, y0 + 40], [460, 520], {"depth_folder": WORK + r"\depth\DSCF3000_f16_GUESS_stairs",
                                                   "readout_source": "measured if connected",
                              "map_scale": 0.5})
    b.link(camera, 0, rsm, "shot_camera")
    b.link(ro, 0, rsm, "readout_measured_ms")
    b.link(lens, 0, rsm, "sensor_lens")
    prev = b.clone(3, "RS map (displacement x0.1)", [x0 + 500, y0 + 600], size=[460, 300])
    b.link(rsm, 1, prev, "images")
    rep = b.clone(5, "RS report", [x0 + 500, y0 + 940], size=[460, 280])
    b.link(rsm, 2, rep, "source")
    # plate -> clean space (for LTX Prep)
    cp = b.clone(95, "Comp Preview: plate only (display view)", [x0 + 1000, y0 + 40],
                 widgets=["", 1280, 1, False, ""])
    b.link(plate, 0, cp, "plate_ref")
    rem = b.ours(AtlasVFXSensorApply, "Sensor Apply: REMOVE camera (plate -> clean)", [x0 + 1480, y0 + 40],
                 [440, 420], {"direction": "remove camera"})
    b.link(cp, 0, rem, "images")
    b.link(rsm, 0, rem, "rs_folder")
    b.link(lens, 1, rem, "lens_folder")
    cv1 = b.clone(96, "Clean plate as a movie", [x0 + 1960, y0 + 40])
    b.link(rem, 0, cv1, "images")
    b.link(cp, 1, cv1, "fps")
    sv1 = b.clone(97, "Clean plate (de-RS)", [x0 + 2440, y0 + 40], widgets=["atlas-vfx/SENSOR_plate_clean_S2S3",
                                                                            "auto", "auto"], size=[460, 360])
    b.link(cv1, 0, sv1, "video")
    # LTX render -> plate camera back on
    lv = b.clone(80, "LTX render: night_test (day-to-night, 1130-1226, 1920x1088)", [x0 + 1000, y0 + 560],
                 widgets=["atlas-vfx/LTX_night_test_1130-1226.mp4"])
    gv = b.clone(63, None or "Frames", [x0 + 1000, y0 + 940])
    b.link(lv, 0, gv, "video")
    app = b.ours(AtlasVFXSensorApply, "Sensor Apply: APPLY camera (LTX render -> plate look)",
                 [x0 + 1480, y0 + 560], [440, 420], {"direction": "apply camera", "first_frame": 1130})
    b.link(gv, 0, app, "images")
    b.link(rsm, 0, app, "rs_folder")
    b.link(lens, 1, app, "lens_folder")
    cv2 = b.clone(96, "LTX + sensor as a movie", [x0 + 1960, y0 + 560])
    b.link(app, 0, cv2, "images")
    b.link(gv, 2, cv2, "fps")
    sv2 = b.clone(97, "LTX render with the plate's rolling shutter", [x0 + 2440, y0 + 560],
                  widgets=["atlas-vfx/SENSOR_ltx_night_rs_S2S3", "auto", "auto"], size=[460, 360])
    b.link(cv2, 0, sv2, "video")
    ar = b.clone(5, "What Sensor Apply did", [x0 + 2940, y0 + 560], size=[440, 300])
    b.link(app, 1, ar, "source")
    wf["last_node_id"], wf["last_link_id"] = b.nid, b.lid
    return wf


if __name__ == "__main__":
    wf = build()
    text = json.dumps(wf, indent=1)
    OUT.write_text(text, encoding="utf-8")
    SIDEBAR.mkdir(parents=True, exist_ok=True)
    (SIDEBAR / OUT.name).write_text(text, encoding="utf-8")
    print("wrote", OUT, "and", SIDEBAR / OUT.name, f"({len(wf['nodes'])} nodes, {len(wf['links'])} links)")
