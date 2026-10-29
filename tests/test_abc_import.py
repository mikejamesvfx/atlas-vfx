"""Alembic camera import (camera_side/abc_io.py + camera_io.from_abc).

The .abc files are written here by a minimal Ogawa writer, so no binary fixture lives in git. Real exports were
checked by hand on ws (docs/dev/NOTES.md 2026-10-20): Nuke 17.1 (matrix op, exact round trip against our builder's
camera), Blender 5.2 (parented empty, euler, scale) and Maya 2026 (rotate order zxy, pivots, scaled group, cm).

- the op stack: translate, rotateY, scale, rotate(axis, angle), matrix; inherits; the chain's scale removed
- rotateY +90 turns the camera's -Z view onto -X (handedness and degrees)
- time: values on a 1/24 sampling starting at 1001/24, constant .ops on the default sampling (as Nuke writes)
  -> frames 1001-1003, not 0-...
- focal kept as horizontal FOV when the file's aperture differs from the sensor; m_per_unit on translation
- identity film back passes, a window translate is refused; two cameras need a name; static camera is held
- HDF5 and non-Alembic files are refused with a message

Run: <V141 python> tests\\test_abc_import.py
"""
from __future__ import annotations

import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import abc_io  # noqa: E402
from atlas_vfx.camera_side import camera_io as cio  # noqa: E402

FAILS = []
DATA = 1 << 63
SENSOR = {"width_mm": 23.5, "height_mm": 12.39}
PLATE = {"path": "", "width": 4096, "height": 2160, "colorspace": "unknown", "range": "full"}


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


# ----------------------------------------------------------------------------------------------- tiny Ogawa writer

class W:
    def __init__(self):
        self.b = bytearray(b"Ogawa\xff\x00\x01" + b"\0" * 8)

    def data(self, payload: bytes) -> int:
        off = len(self.b)
        self.b += struct.pack("<Q", len(payload)) + payload
        return off | DATA

    def group(self, refs) -> int:
        off = len(self.b)
        self.b += struct.pack(f"<Q{len(refs)}Q", len(refs), *refs)
        return off

    def finish(self, root: int, path):
        struct.pack_into("<Q", self.b, 8, root)
        Path(path).write_bytes(bytes(self.b))


POD = {"bool": (0, "?"), "u8": (1, "B"), "f64": (11, "d"), "str": (12, None)}


def scalar(w, name, pod, samples, ts=0):
    """samples: list of tuples (one per sample); stored fully (first 0, last n-1) unless all equal."""
    code, fmt = POD[pod]
    const = all(s == samples[0] for s in samples)
    stored = samples[:1] if const else samples
    refs = []
    for s in stored:
        payload = (b"\0".join(x.encode() for x in s) + b"\0") if pod == "str" else struct.pack(f"<{len(s)}{fmt}", *s)
        refs.append(w.data(b"\0" * 16 + payload))
    extent = len(samples[0])
    first, last = (0, 0) if const else (0, len(samples) - 1)
    hdr = dict(kind=1, pod=code, extent=extent, n=len(samples), first=first, last=last, ts=ts, name=name)
    return w.group(refs), hdr


def compound(w, name, props):
    """props: [(ref, header)]; returns (ref, header) for the compound."""
    hb = bytearray()
    for _, h in props:
        info = h["kind"] | (2 << 2)                         # uint32 size hint
        if h["kind"]:
            info |= (h["pod"] << 4) | 0x0200 | (h["extent"] << 12) | (0x0100 if h["ts"] else 0)
        hb += struct.pack("<I", info)
        if h["kind"]:
            hb += struct.pack("<III", h["n"], h["first"], h["last"])
            if h["ts"]:
                hb += struct.pack("<I", h["ts"])
        nm = h["name"].encode()
        hb += struct.pack("<I", len(nm)) + nm
    ref = w.group([r for r, _ in props] + [w.data(bytes(hb))])
    return ref, dict(kind=0, name=name)


def obj(w, props_ref, children):
    """children: [(name, schema, ref)]"""
    hb = bytearray()
    for name, schema, _ in children:
        nm, md = name.encode(), (f"schema={schema}" if schema else "").encode()
        hb += struct.pack("<I", len(nm)) + nm + b"\xff" + struct.pack("<I", len(md)) + md
    hb += b"\0" * 32
    return w.group([props_ref] + [r for *_, r in children] + [w.data(bytes(hb))])


def xform_obj(w, ops, vals_per_frame, children, inherits=True):
    p_ops = scalar(w, ".ops", "u8", [tuple(ops)] * len(vals_per_frame), ts=0)    # Nuke-like: default sampling
    p_vals = scalar(w, ".vals", "f64", [tuple(v) for v in vals_per_frame], ts=1)
    p_inh = scalar(w, ".inherits", "bool", [(inherits,)] * len(vals_per_frame), ts=1)
    xf = compound(w, ".xform", [p_ops, p_vals, p_inh])
    top, _ = compound(w, "", [xf])
    return obj(w, top, children)


def camera_obj(w, cores, film_back=None):
    props = [scalar(w, ".core", "f64", [tuple(c) for c in cores], ts=1)]
    if film_back:
        ops, ch = film_back
        props.append(scalar(w, ".filmBackOps", "str", [tuple(ops)], ts=0))
        props.append(scalar(w, ".filmBackChannels", "f64", [tuple(ch)], ts=0))
    geom = compound(w, ".geom", props)
    top, _ = compound(w, "", [geom])
    return obj(w, top, [])


def core(focal, hap_cm=2.35, hoff=0.0):
    return [focal, hap_cm, hoff, 1.2393, 0.0, 1.0, 0, 0, 0, 0, 5.6, 5.0, 0.0, 0.5, 0.1, 1e5]


def archive(w, top_children, path, start_frame=1001, fps=24.0):
    top = obj(w, compound(w, "", [])[0], top_children)
    ts = struct.pack("<IdI", 0, 1.0, 1) + struct.pack("<d", 0.0)
    ts += struct.pack("<IdI", 3, 1.0 / fps, 1) + struct.pack("<d", start_frame / fps)
    root = w.group([w.data(struct.pack("<i", 10709)), w.data(struct.pack("<i", 10709)), top,
                    w.data(b"_ai_Application=atlas-vfx test"), w.data(ts), w.data(b"")])
    w.finish(root, path)


# op codes: (type << 4) | hint
T, RY, S, ROT, MAT = 1 << 4, 5 << 4, 0 << 4, 2 << 4, 3 << 4


def rig_vals(i):
    return [1.0 + i, 2.0, -3.0, 30.0 * i, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 10.0 * i]   # T, RY, S, ROT(axis, deg)


def cam_matrix_vals(i):
    M = np.eye(4)
    M[:3, 3] = [0.5, 1.5 + 0.1 * i, 4.0]
    return list(M.T.ravel())                                # Imath row-vector, row-major


def expected(i, inherits=True):
    def rotm(axis, deg):
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        t = math.radians(deg)
        R = np.eye(4)
        R[:3, :3] = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
        return R
    v = rig_vals(i)
    Tm = np.eye(4)
    Tm[:3, 3] = v[0:3]
    rig = Tm @ rotm([0, 1, 0], v[3]) @ np.diag([2.0, 2.0, 2.0, 1.0]) @ rotm(v[7:10], v[10])
    cam = np.asarray(cam_matrix_vals(i)).reshape(4, 4).T
    return (rig @ cam) if inherits else cam


def build(path, inherits=True, film_back=None, two_cams=False, static=False, start_frame=1001):
    w = W()
    n = 1 if static else 3
    cams = [("camShape", "AbcGeom_Camera_v1", camera_obj(w, [core(20.0 + 5 * i) for i in range(n)], film_back))]
    cam_x = xform_obj(w, [MAT], [cam_matrix_vals(i) for i in range(n)], cams, inherits=inherits)
    rig = xform_obj(w, [T, RY, S, ROT], [rig_vals(i) for i in range(n)], [("shotCam", "AbcGeom_Xform_v3", cam_x)])
    kids = [("rig", "AbcGeom_Xform_v3", rig)]
    if two_cams:
        c2 = camera_obj(w, [core(35.0)] * n)
        kids.append(("witness", "AbcGeom_Xform_v3", xform_obj(w, [T], [[0, 0, 0]] * n,
                                                              [("witnessShape", "AbcGeom_Camera_v1", c2)])))
    archive(w, kids, path, start_frame=start_frame)


def test_first_frame_offset():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cam1001.abc"
        build(p)
        base = cio.import_camera(p, plate=PLATE, sensor=SENSOR, fps=24.0)
        c1 = cio.import_camera(p, plate=PLATE, sensor=SENSOR, fps=24.0, first_frame_to=1)
        same = all(np.array_equal(c1.cam_to_world(f - 1000), base.cam_to_world(f)) and
                   c1.intrinsics(f - 1000).focal_mm == base.intrinsics(f).focal_mm for f in base.frames)
        sh = c1.data["extensions"]["atlas_vfx_import"]["frame_shift"]
        check(c1.frames == [1, 2, 3] and same and sh["delta"] == -1000 and sh["source_first"] == 1001,
              "1001 camera renumbered to start at 1, motion and focal untouched", f"{c1.frames}, {sh}")
        q = Path(d) / "cam1.abc"
        build(q, start_frame=1)
        c2 = cio.import_camera(q, plate=PLATE, sensor=SENSOR, fps=24.0, first_frame_to=1001, frames=(1002, 1010))
        check(c2.frames == [1002, 1003] and c2.data["shot"]["frame_start"] == 1002,
              "camera starting at 1 onto 1001, then cropped in the new numbering", str(c2.frames))
        try:
            cio.import_camera(q, plate=PLATE, sensor=SENSOR, fps=24.0, frames=(1130, 1297))
            check(False, "no overlap with the plate is a clear error")
        except cio.CameraImportError as e:
            check("none in" in str(e) or "overlap" in str(e), "no overlap with the plate is a clear error",
                  str(e)[:90])
        st = Path(d) / "static.abc"
        build(st, static=True, start_frame=1)
        c3 = cio.import_camera(st, plate=PLATE, sensor=SENSOR, fps=24.0, first_frame_to=1130, frames=(1130, 1140))
        check(c3.frames == list(range(1130, 1141)), "static camera: offset then held over the plate range")


def test_ops_chain_time():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cam.abc"
        build(p)
        cam_obj, rows, notes = abc_io.camera_samples(p, fps=24.0)
        frames = [f for f, _ in rows]
        check(frames == [1001, 1002, 1003], "frames from the animated sampling, not the .ops default one", frames)
        err = max(np.abs(s.cam_to_world - expected(i)).max() for i, (_, s) in enumerate(rows))
        check(err < 1e-12, "op stack T Ry S Rot(axis) x matrix, parent chain", f"max {err:.1e}")
        check([s.core["focal_mm"] for _, s in rows] == [20.0, 25.0, 30.0], "animated focal")
        p2 = Path(d) / "noinherit.abc"
        build(p2, inherits=False)
        _, rows2, _ = abc_io.camera_samples(p2, fps=24.0)
        err2 = max(np.abs(s.cam_to_world - expected(i, False)).max() for i, (_, s) in enumerate(rows2))
        check(err2 < 1e-12, "inherits off: the camera ignores the rig", f"max {err2:.1e}")

        c = cio.import_camera(p, plate=PLATE, sensor=SENSOR, m_per_unit=0.01, fps=24.0)
        M = c.cam_to_world(1002)
        R = expected(1)[:3, :3] / 2.0
        check(np.allclose(M[:3, :3], R, atol=1e-9) and np.allclose(M[:3, 3], expected(1)[:3, 3] * 0.01),
              "import: rig scale removed from rotation, m_per_unit on translation")
        sens = {"width_mm": 36.0, "height_mm": 36.0 * 2160 / 4096}
        c36 = cio.import_camera(p, plate=PLATE, sensor=sens, fps=24.0)
        check(abs(c36.intrinsics(1001).focal_mm - 20.0 * 36.0 / 23.5) < 1e-9,
              "focal rescaled to keep the file's horizontal FOV on another sensor")


def test_handedness():
    R = abc_io.ops_matrix([RY], [90.0])
    fwd = R[:3, :3] @ np.array([0, 0, -1.0])
    check(np.allclose(fwd, [-1, 0, 0]), "rotateY +90 (degrees) turns -Z onto -X", fwd.round(6).tolist())
    M = abc_io.ops_matrix([MAT], list(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [7, 8, 9, 1.0]]).ravel()))
    check(np.allclose(M[:3, 3], [7, 8, 9]), "matrix op: Imath row-vector translation lands in the last column")


def test_film_back_selection_static_errors():
    with tempfile.TemporaryDirectory() as d:
        ok = Path(d) / "fb_ok.abc"
        build(ok, film_back=(["spostScale"], [1.0, 1.0]))
        c = cio.import_camera(ok, plate=PLATE, sensor=SENSOR, fps=24.0)
        check(len(c.frames) == 3, "identity film back (Nuke's postScale 1 1) passes")
        bad = Path(d) / "fb_bad.abc"
        build(bad, film_back=(["tpreTranslate"], [0.1, 0.0]))
        try:
            cio.import_camera(bad, plate=PLATE, sensor=SENSOR, fps=24.0)
            check(False, "window translate refused")
        except cio.CameraImportError as e:
            check("window translate" in str(e), "window translate refused", str(e)[:80])
        two = Path(d) / "two.abc"
        build(two, two_cams=True)
        try:
            cio.import_camera(two, plate=PLATE, sensor=SENSOR, fps=24.0)
            check(False, "two cameras need a name")
        except cio.CameraImportError as e:
            check("pick one" in str(e) and "witness" in str(e), "two cameras need a name (listed)")
        c2 = cio.import_camera(two, plate=PLATE, sensor=SENSOR, fps=24.0, camera="witness")
        check(abs(c2.intrinsics(1001).focal_mm - 35.0) < 1e-9, "camera= picks the witness camera")
        st = Path(d) / "static.abc"
        build(st, static=True)
        c3 = cio.import_camera(st, plate=PLATE, sensor=SENSOR, fps=24.0, frames=(1001, 1010))
        check(c3.frames == list(range(1001, 1011)), "static camera held over the requested range")
        h5 = Path(d) / "old.abc"
        h5.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
        try:
            cio.import_camera(h5, plate=PLATE, sensor=SENSOR, fps=24.0)
            check(False, "HDF5 Alembic refused")
        except cio.CameraImportError as e:
            check("HDF5" in str(e), "HDF5 Alembic refused with a message")
        junk = Path(d) / "junk.abc"
        junk.write_bytes(b"not alembic at all, just text")
        try:
            cio.import_camera(junk, plate=PLATE, sensor=SENSOR, fps=24.0)
            check(False, "non-Alembic refused")
        except cio.CameraImportError as e:
            check("not an Alembic" in str(e), "non-Alembic refused")
        check(cio.detect_kind(ok) == "abc", "auto detects .abc")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
