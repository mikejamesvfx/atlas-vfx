"""T8: EXR Write, Camera Export, Nuke Builder (synthetic).

- EXR Write: an sRGB IMAGE + mask becomes premultiplied ACEScg RGBA half with the colourspace in the header;
  check_sequence passes it and fails a missing frame and a wrong colourspace
- Camera Export: the .chan round trips the camera (Nuke ZXY, metres), the .usda transform samples are cam_to_world
  for every frame, apertures and win_translate follow the plate (centred principal point -> 0, shifted -> Nuke units)
- Nuke Builder (plan): the generated script compiles, names every file it reads, and has no NukeX-only node class.
  Then it is run twice against a stand-in `nuke` module: the second run deletes and rebuilds only its prefixed nodes
  (a user node survives), the layers are merged over the plate back to front (B = below, A = layer), the camera gets
  a key on every chan frame with rot order ZXY, and a missing file stops the build before any node is made.

Run: <V141 python> tests\\test_nuke_builder.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("ATLAS_VFX_OUT_ROOTS", tempfile.gettempdir())     # nodes may write under the temp folder

from atlas_vfx.camera_side import camera_export as ce  # noqa: E402
from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.deliver import nuke_builder as nb  # noqa: E402
from test_depth_set_proxy import camera  # noqa: E402

OCIO = str(ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio")


# ------------------------------------------------------------------------------------------ stand-in nuke module

class Knob:
    def __init__(self):
        self.value, self.keys, self.animated = None, {}, False

    def setValue(self, v, idx=None):
        if idx is None:
            self.value = v
        else:
            if not isinstance(self.value, dict):
                self.value = {}
            self.value[idx] = v

    def setAnimated(self):
        self.animated = True

    def setValueAt(self, v, frame, idx=0):
        if not self.animated:
            raise RuntimeError("not animated")
        self.keys[(frame, idx)] = v


class Node:
    def __init__(self, fake, cls):
        self.cls, self.knobs, self.inputs, self._name = cls, {}, {}, f"{cls}{len(fake.all) + 1}"
        fake.all.append(self)

    def __getitem__(self, k):
        return self.knobs.setdefault(k, Knob())

    def name(self):
        return self._name

    def setName(self, n):
        self._name = n

    def setXYpos(self, x, y):
        pass

    def setInput(self, i, n):
        self.inputs[i] = n


def fake_nuke(classes=("Read", "Merge2", "Write", "Camera3", "ReadGeo2")):
    m = types.ModuleType("nuke")
    m.all, m.formats = [], []
    m.nodes = types.SimpleNamespace(**{c: (lambda c=c: Node(m, c)) for c in classes})
    m.allNodes = lambda recurseGroups=False: list(m.all)
    m.delete = lambda n: m.all.remove(n)
    rootnode = Node(m, "Root")
    m.all.remove(rootnode)
    m.root = lambda: rootnode
    m.addFormat = lambda s: m.formats.append(s)
    return m


def run_in_fake(text, fake):
    sys.modules["nuke"] = fake
    try:
        exec(compile(text, "build.py", "exec"), {"__name__": "__nuke__"})
    finally:
        del sys.modules["nuke"]


# ------------------------------------------------------------------------------------------------------- tests

def test_exr_write(tmp: Path):
    import torch

    from atlas_vfx.camera_side.node_exr_write import AtlasVFXEXRWrite

    img = torch.full((3, 16, 24, 3), 0.5)
    mask = torch.zeros((3, 16, 24))
    mask[:, 4:12, 6:18] = 1.0
    res = AtlasVFXEXRWrite().run(img, 1130, "sRGB Encoded Rec.709 (sRGB)", "ACEScg", False, OCIO, "lay", str(tmp / "ew"),
                                 mask=mask)
    pattern = res["result"][0]
    data, attrs = exr_io.read_rgba(pattern % 1131)
    assert abs(data[8, 10, 0] - 0.214) < 0.002 and data[0, 0, 3] == 0 and data[0, 0, 0] == 0, data[8, 10]
    assert attrs["colorInteropID"] == "lin_ap1_scene" and attrs["ocio_colorspace"] == "ACEScg"
    exr_io.check_sequence(pattern, 1130, 1132, 24, 16)
    for bad, want in ((lambda: exr_io.check_sequence(pattern, 1130, 1133), "missing"),
                      (lambda: exr_io.check_sequence(pattern, 1130, 1132, colorspace="ACES2065-1"), "colorspace"),
                      (lambda: exr_io.check_sequence(pattern, 1130, 1132, 48, 32), "size")):
        try:
            bad()
            raise AssertionError(f"check_sequence should fail ({want})")
        except ValueError as e:
            assert want in str(e), e
    print("ok exr write: sRGB 0.5 -> ACEScg 0.214 premultiplied by the mask, header ACEScg / lin_ap1_scene; "
          "check_sequence catches a missing frame, a wrong colourspace and a wrong size")
    return pattern


def test_camera_export(cam: sc.ShotCamera, tmp: Path):
    res = ce.export(cam, tmp / "cam", "syn")
    rows = ce.read_chan(res["chan"])
    assert sorted(rows) == cam.frames
    worst = 0.0
    for n in cam.frames:
        tx, ty, tz, rx, ry, rz, foc = rows[n]
        R = sc.euler_zxy_to_matrix(rx, ry, rz)
        M = cam.cam_to_world(n)
        worst = max(worst, float(np.abs(R - M[:3, :3]).max()), float(np.abs(np.array([tx, ty, tz]) - M[:3, 3]).max()))
        assert abs(foc - cam.intrinsics(n).focal_mm) < 1e-5
    assert worst < 1e-5, worst
    usd = ce.read_usda_transforms(res["usda"])
    assert sorted(usd) == cam.frames
    uworst = max(float(np.abs(usd[n] - cam.cam_to_world(n)).max()) for n in cam.frames)
    assert uworst < 1e-7, uworst
    s = res["settings"]
    assert s["rot_order"] == "ZXY" and s["win_translate"] == [0.0, 0.0] and abs(s["haperture"] - 23.5) < 1e-9
    assert abs(s["vaperture"] - 23.5 * 2160 / 4096) < 1e-9
    # a shifted principal point: +40 px right, +20 px down -> Nuke window units (half width = 1, y up)
    d = copy.deepcopy(cam.data)
    d.setdefault("lens", {})["principal_point_px"] = [2048 + 40, 1080 + 20]
    s2 = ce.nuke_settings(sc.ShotCamera(d))
    assert np.allclose(s2["win_translate"], [40 / 2048, -20 / 2048]), s2["win_translate"]
    text = Path(res["usda"]).read_text(encoding="utf-8")
    assert 'upAxis = "Y"' in text and "metersPerUnit = 1" in text and "timeCodesPerSecond = 24" in text
    print(f"ok camera export: chan round trip within {worst:.1e} (ZXY, metres), usda {len(usd)} transform samples "
          f"within {uworst:.1e}, apertures 23.5 x {s['vaperture']:.3f} mm, principal point -> win_translate")
    return res


def make_shot(tmp: Path, cam_res: dict, layer_pattern: str):
    dpx = tmp / "plate" / "SYN_dpx"
    dpx.mkdir(parents=True)
    (dpx / "SYN.1130.dpx").write_bytes(b"\0")
    obj = tmp / "set.obj"
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    plate = nb.plate_source(tmp / "plate" / "SYN.MOV", "dpx", frame_offset=1130)
    layers = [{"name": "element", "pattern": layer_pattern.replace("\\", "/"), "first": 1130, "last": 1132,
               "colorspace": "ACEScg", "premultiplied": True},
              {"name": "grass", "pattern": layer_pattern.replace("\\", "/"), "first": 1130, "last": 1132,
               "colorspace": "ACEScg", "premultiplied": True}]
    return nb.build_data(shot="SYN_S1", plate=plate, plate_colorspace="F-Log2 F-Gamut", plate_frames=(1130, 1159),
                         fmt=(4096, 2160, 1.0), ocio_config=OCIO, camera_chan=cam_res["chan"],
                         camera_settings=cam_res["settings"], layers=layers, write_pattern=str(tmp / "comp.%04d.exr"),
                         set_proxy_obj=str(obj))


def test_builder(tmp: Path, cam: sc.ShotCamera, cam_res: dict, layer_pattern: str):
    data = make_shot(tmp, cam_res, layer_pattern)
    text = nb.script_text(data)
    compile(text, "build_SYN_S1.py", "exec")
    for f in data["reads"]:
        assert f in text and "\\" not in f, f
    used = nb.node_classes_used(text)
    bad = used & set(nb.NUKEX_ONLY)
    assert not bad and {"Read", "Merge2", "Write"} <= used, (bad, used)
    print(f"ok script: compiles, names all {len(data['reads'])} files it reads, node classes {sorted(used)} "
          f"(none NukeX-only)")

    fake = fake_nuke()
    user = Node(fake, "Grade")
    user.setName("Mike_grade")
    run_in_fake(text, fake)
    first = sorted(n.name() for n in fake.all)
    run_in_fake(text, fake)                                   # idempotent
    second = sorted(n.name() for n in fake.all)
    assert first == second and "Mike_grade" in second, (first, second)
    by = {n.name(): n for n in fake.all}
    o1, o2, wr = by["AVFX_Over_element"], by["AVFX_Over_grass"], by["AVFX_Write_comp"]
    assert o1.inputs[0] is by["AVFX_Plate"] and o1.inputs[1] is by["AVFX_Layer_element"]
    assert o2.inputs[0] is o1 and o2.inputs[1] is by["AVFX_Layer_grass"] and wr.inputs[0] is o2
    assert by["AVFX_Layer_element"]["colorspace"].value == "ACEScg" and by["AVFX_Plate"]["colorspace"].value == "F-Log2 F-Gamut"
    c = by["AVFX_Camera"]
    assert c.cls == "Camera3" and c["rot_order"].value == "ZXY"
    rows = ce.read_chan(cam_res["chan"])
    assert len(c["translate"].keys) == 3 * len(rows) and len(c["focal"].keys) == len(rows)
    n = cam.frames[7]
    assert abs(c["rotate"].keys[(n, 1)] - rows[n][4]) < 1e-9
    root = fake.root()
    assert root["customOCIOConfigPath"].value == OCIO.replace("\\", "/") and root["workingSpaceLUT"].value == "ACEScg"
    assert root["first_frame"].value == 1130 and fake.formats
    print(f"ok build: {len(second) - 1} AVFX nodes, re-run rebuilds the same set and keeps the user's node; plate -> "
          f"element -> grass -> Write; camera {len(rows)} keys, ZXY; root OCIO set")

    # older Nuke without Camera3: falls back to Camera2
    fake2 = fake_nuke(("Read", "Merge2", "Write", "Camera2"))
    run_in_fake(text, fake2)
    assert any(n.cls == "Camera2" for n in fake2.all)
    # a missing file stops it before anything is made
    data["reads"].append((tmp / "nope.exr").as_posix())
    fake3 = fake_nuke()
    try:
        run_in_fake(nb.script_text(data), fake3)
        raise AssertionError("missing file not caught")
    except RuntimeError as e:
        assert "nope.exr" in str(e) and not fake3.all
    print("ok fallbacks: Camera2 when Camera3 is absent; a missing file stops the build before any node is made")


def main() -> int:
    cam = camera()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        layer_pattern = test_exr_write(tmp)
        cam_res = test_camera_export(cam, tmp)
        test_builder(tmp, cam, cam_res, layer_pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
