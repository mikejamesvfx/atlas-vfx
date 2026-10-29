"""Alembic (.abc) camera reader, pure Python (Ogawa backend). No third-party code or library.

Reads what a camera import needs and nothing more: the object tree, Xform op stacks (translate, rotate, rotateX/Y/Z,
scale, matrix) with inheritance, and the Camera schema's .core sample (focal mm, apertures and film offsets in cm,
squeeze, overscan, fStop, focus distance, shutter, clipping). HDF5 Alembic (pre-2013) is refused with a message.

Ogawa: a 16-byte header ("Ogawa", frozen flag, version, root offset) and a tree of groups (uint64 child count + uint64
child offsets, high bit set = data) and data blocks (uint64 size + bytes). Alembic on Ogawa:
  archive group: 0 version, 1 library version, 2 top object, 3 archive metadata, 4 time samplings, 5 indexed metadata
  object group:  0 property compound, 1..n children, last = object headers (data, 32 hash bytes at the end)
  compound:      0..n properties, last = property headers (data)
  scalar prop:   one data block per stored sample (16-byte digest + POD values)
  array prop:    data at 2i, dims at 2i+1
Matrices returned here are column-vector (x' = M @ x), Y up, camera looking down -Z, like the shot-camera contract.
"""
from __future__ import annotations

import math
import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DATA = 1 << 63
_ACYCLIC_TPC = 1e300           # Alembic stores max(double)/32 for acyclic time samplings
_POD = {0: "?", 1: "B", 2: "b", 3: "H", 4: "h", 5: "I", 6: "i", 7: "Q", 8: "q", 9: "e", 10: "f", 11: "d"}
CORE_FIELDS = ("focal_mm", "h_aperture_cm", "h_film_offset_cm", "v_aperture_cm", "v_film_offset_cm", "squeeze",
               "overscan_left", "overscan_right", "overscan_top", "overscan_bottom", "f_stop", "focus_distance",
               "shutter_open", "shutter_close", "near", "far")


class AbcError(ValueError):
    pass


@dataclass
class Prop:
    name: str
    kind: int                  # 0 compound, 1 scalar, 2 array
    pod: int = -1
    extent: int = 0
    num_samples: int = 0
    first_changed: int = 0
    last_changed: int = 0
    ts_index: int = 0
    metadata: str = ""
    ref: int = 0               # Ogawa child offset of the property group


@dataclass
class Obj:
    name: str
    path: str
    metadata: str
    ref: int
    parent: "Obj | None" = None
    children: list = field(default_factory=list)

    @property
    def schema(self) -> str:
        for kv in self.metadata.split(";"):
            if kv.startswith("schema="):
                return kv[7:]
        return ""


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        try:
            self.buf = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as e:          # empty file
            self._f.close()
            raise AbcError(f"{self.path}: empty file") from e
        head = self.buf[:16]
        if head[:8] == b"\x89HDF\r\n\x1a\n":
            self.close()
            raise AbcError(f"{self.path.name} is an HDF5 Alembic (old format); re-export it as Ogawa")
        if head[:5] != b"Ogawa":
            self.close()
            raise AbcError(f"{self.path.name} is not an Alembic (Ogawa) file")
        root = self._group(struct.unpack_from("<Q", head, 8)[0])
        if len(root) < 5:
            raise AbcError(f"{self.path.name}: archive has {len(root)} children, expected at least 5")
        self.time_samplings = self._time_samplings(self._data(root[4]))
        self.indexed_md = self._indexed_md(self._data(root[5])) if len(root) > 5 else [""]
        self.archive_metadata = self._data(root[3]).decode("utf-8", "replace")
        self.top = Obj("ABC", "/", "", root[2])
        self._load_children(self.top)

    # ------------------------------------------------------------------------------------------- Ogawa
    def close(self):
        try:
            self.buf.close()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _group(self, off) -> list[int]:
        if off & _DATA:
            raise AbcError("expected a group, found data")
        if off == 0:
            return []
        n = struct.unpack_from("<Q", self.buf, off)[0]
        if n > (len(self.buf) - off) // 8:
            raise AbcError(f"corrupt group at {off}")
        return list(struct.unpack_from(f"<{n}Q", self.buf, off + 8))

    def _data(self, ref) -> bytes:
        if not ref & _DATA:
            raise AbcError("expected data, found a group")
        off = ref & ~_DATA
        if off == 0:
            return b""
        size = struct.unpack_from("<Q", self.buf, off)[0]
        return self.buf[off + 8: off + 8 + size]

    # ------------------------------------------------------------------------------------------- archive tables
    @staticmethod
    def _time_samplings(b: bytes) -> list[tuple[float, list[float]]]:
        out, p = [], 0
        while p < len(b):
            _max_sample, tpc, n = struct.unpack_from("<IdI", b, p)
            p += 16
            times = list(struct.unpack_from(f"<{n}d", b, p))
            p += 8 * n
            out.append((tpc, times))
        return out or [(1.0, [0.0])]

    @staticmethod
    def _indexed_md(b: bytes) -> list[str]:
        out, p = [""], 0
        while p < len(b):
            n = b[p]
            out.append(b[p + 1: p + 1 + n].decode("utf-8", "replace"))
            p += 1 + n
        return out

    # ------------------------------------------------------------------------------------------- objects
    def _load_children(self, obj: Obj):
        g = self._group(obj.ref)
        if not g or not g[-1] & _DATA:
            return
        b = self._data(g[-1])
        end, p, k = len(b) - 32, 0, 0
        while p < end:
            (n,) = struct.unpack_from("<I", b, p)
            name = b[p + 4: p + 4 + n].decode("utf-8", "replace")
            p += 4 + n
            mdi = b[p]
            p += 1
            if mdi == 0xFF:
                (m,) = struct.unpack_from("<I", b, p)
                md = b[p + 4: p + 4 + m].decode("utf-8", "replace")
                p += 4 + m
            else:
                md = self.indexed_md[mdi] if mdi < len(self.indexed_md) else ""
            k += 1
            child = Obj(name, (obj.path.rstrip("/") + "/" + name), md, g[k], parent=obj)
            obj.children.append(child)
            self._load_children(child)

    def walk(self, obj: Obj | None = None):
        obj = obj or self.top
        for c in obj.children:
            yield c
            yield from self.walk(c)

    def cameras(self) -> list[Obj]:
        return [o for o in self.walk() if o.schema.startswith("AbcGeom_Camera")]

    # ------------------------------------------------------------------------------------------- properties
    def properties(self, compound_ref: int) -> dict[str, Prop]:
        g = self._group(compound_ref)
        if not g or not g[-1] & _DATA:
            return {}
        b = self._data(g[-1])
        out, p, k = {}, 0, 0
        while p < len(b):
            (info,) = struct.unpack_from("<I", b, p)
            p += 4
            hint = (info & 0x0C) >> 2
            fmt, w = {0: ("<B", 1), 1: ("<H", 2), 2: ("<I", 4)}.get(hint, ("<I", 4))

            def u():
                nonlocal p
                (v,) = struct.unpack_from(fmt, b, p)
                p += w
                return v
            pr = Prop("", info & 0x03, ref=g[k])
            if pr.kind != 0:
                pr.pod = (info & 0xF0) >> 4
                pr.extent = (info & 0xFF000) >> 12
                pr.num_samples = u()
                if info & 0x0200:
                    pr.first_changed, pr.last_changed = u(), u()
                elif info & 0x0800:
                    pr.first_changed = pr.last_changed = 0
                else:
                    pr.first_changed, pr.last_changed = 1, pr.num_samples - 1
                if info & 0x0100:
                    pr.ts_index = u()
            n = u()
            pr.name = b[p: p + n].decode("utf-8", "replace")
            p += n
            mdi = (info & 0xFF00000) >> 20
            if mdi == 0xFF:
                m = u()
                pr.metadata = b[p: p + m].decode("utf-8", "replace")
                p += m
            else:
                pr.metadata = self.indexed_md[mdi] if mdi < len(self.indexed_md) else ""
            out[pr.name] = pr
            k += 1
        return out

    def object_props(self, obj: Obj) -> dict[str, Prop]:
        g = self._group(obj.ref)
        return self.properties(g[0]) if g else {}

    @staticmethod
    def _stored_index(pr: Prop, i: int) -> int:
        i = max(0, min(i, pr.num_samples - 1))
        if pr.first_changed == 0 and pr.last_changed == 0:
            return 0
        if i < pr.first_changed:
            return 0
        return min(i, pr.last_changed)

    def scalar(self, pr: Prop, i: int) -> tuple:
        if pr.kind != 1:
            raise AbcError(f"{pr.name} is not a scalar property")
        if pr.pod not in _POD:
            raise AbcError(f"{pr.name}: POD {pr.pod} not supported")
        g = self._group(pr.ref)
        b = self._data(g[self._stored_index(pr, i)])
        n = max(pr.extent, 1)
        return struct.unpack_from(f"<{n}{_POD[pr.pod]}", b, 16)

    def array(self, pr: Prop, i: int) -> np.ndarray:
        if pr.kind != 2:
            raise AbcError(f"{pr.name} is not an array property")
        g = self._group(pr.ref)
        j = self._stored_index(pr, i)
        b = self._data(g[2 * j])
        if not b:
            return np.zeros(0)
        return np.frombuffer(bytes(b[16:]), dtype="<" + _POD[pr.pod])

    def values(self, pr: Prop, i: int) -> np.ndarray:
        return np.asarray(self.scalar(pr, i) if pr.kind == 1 else self.array(pr, i), float)

    # ------------------------------------------------------------------------------------------- time
    def sample_time(self, pr: Prop, i: int) -> float:
        tpc, times = self.time_samplings[pr.ts_index]
        if tpc > _ACYCLIC_TPC:
            return times[min(i, len(times) - 1)]
        n = len(times)
        return times[i % n] + (i // n) * tpc

    def sample_times(self, pr: Prop) -> list[float]:
        return [self.sample_time(pr, i) for i in range(pr.num_samples)]

    def index_at(self, pr: Prop, t: float) -> int:
        """Sample index nearest time t (held at the ends)."""
        times = self.sample_times(pr)
        return int(np.argmin(np.abs(np.asarray(times) - t))) if times else 0


# ----------------------------------------------------------------------------------------------- Xform ops

def _rot(axis, deg) -> np.ndarray:
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4)
    x, y, z = a / n
    t = math.radians(deg)
    c, s, C = math.cos(t), math.sin(t), 1 - math.cos(t)
    M = np.eye(4)
    M[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return M


def ops_matrix(ops, vals) -> np.ndarray:
    """Alembic XformSample::getMatrix in column-vector form: the first op is outermost (M = O0 @ O1 @ ...)."""
    M, k = np.eye(4), 0
    for op in ops:
        kind = int(op) >> 4
        if kind == 0:                                   # scale
            O = np.diag([vals[k], vals[k + 1], vals[k + 2], 1.0])
            k += 3
        elif kind == 1:                                 # translate
            O = np.eye(4)
            O[:3, 3] = vals[k: k + 3]
            k += 3
        elif kind == 2:                                 # rotate: axis + angle (degrees)
            O = _rot(vals[k: k + 3], vals[k + 3])
            k += 4
        elif kind == 3:                                 # matrix: Imath row-vector, row-major
            O = np.asarray(vals[k: k + 16], float).reshape(4, 4).T
            k += 16
        elif kind in (4, 5, 6):                         # rotateX / Y / Z
            O = _rot(np.eye(3)[kind - 4], vals[k])
            k += 1
        else:
            raise AbcError(f"unknown xform op {op}")
        M = M @ O
    return M


@dataclass
class _Xf:
    obj: Obj
    ops: Prop | None
    vals: Prop | None
    inherits: Prop | None


def _xform(ar: Archive, obj: Obj) -> _Xf | None:
    if not obj.schema.startswith("AbcGeom_Xform"):
        return None
    top = ar.object_props(obj)
    x = top.get(".xform")
    if x is None:
        return _Xf(obj, None, None, None)
    p = ar.properties(x.ref)
    return _Xf(obj, p.get(".ops"), p.get(".vals"), p.get(".inherits"))


def _local(ar: Archive, xf: _Xf, t: float) -> tuple[np.ndarray, bool]:
    inh = True
    if xf.inherits is not None and xf.inherits.num_samples:
        inh = bool(ar.scalar(xf.inherits, ar.index_at(xf.inherits, t))[0])
    if xf.ops is None or xf.ops.num_samples == 0:
        return np.eye(4), inh
    i = ar.index_at(xf.vals, t) if xf.vals is not None and xf.vals.num_samples else 0
    ops = ar.values(xf.ops, ar.index_at(xf.ops, t)).astype(int)
    vals = ar.values(xf.vals, i) if xf.vals is not None and xf.vals.num_samples else np.zeros(0)
    return ops_matrix(ops, vals), inh


# ----------------------------------------------------------------------------------------------- cameras

@dataclass
class CameraSample:
    time: float
    cam_to_world: np.ndarray       # 4x4, column-vector, source units, may carry scale from the hierarchy
    core: dict


def _film_back(ar: Archive, geom: dict[str, Prop]) -> dict:
    """Film back ops (t = translate 2, s = scale 2, m = matrix 9 channels) and whether they amount to identity."""
    ops_p, ch_p = geom.get(".filmBackOps"), geom.get(".filmBackChannels")
    if ops_p is None or not ops_p.num_samples:
        return {"ops": [], "identity": True}
    raw = bytes(ar._data(ar._group(ops_p.ref)[0])[16:])
    ops = [s.decode("utf-8", "replace") for s in raw.split(b"\0") if s]
    if ch_p is None or not ch_p.num_samples:
        return {"ops": ops, "channels": [], "identity": not ops}
    animated = ch_p.last_changed > 0
    ident, k, chans = True, 0, []
    for i in range(ch_p.num_samples if animated else 1):
        ch = ar.values(ch_p, i)
        chans = ch.tolist() if i == 0 else chans
        k = 0
        for op in ops:
            h = op[:1]
            n = {"t": 2, "s": 2, "m": 9}.get(h)
            if n is None:
                return {"ops": ops, "channels": chans, "identity": False, "unknown_op": op}
            v = ch[k: k + n]
            ref = {"t": np.zeros(2), "s": np.ones(2), "m": np.eye(3).ravel()}[h]
            ident &= bool(np.allclose(v, ref, atol=1e-9))
            k += n
    return {"ops": ops, "channels": chans, "animated": animated, "identity": ident}


def find_camera(ar: Archive, which: str | None = None) -> Obj:
    cams = ar.cameras()
    if not cams:
        raise AbcError(f"{ar.path.name} has no camera (objects: "
                       f"{', '.join(o.path for o in list(ar.walk())[:12])})")
    if which:
        hit = [c for c in cams if c.path == which or c.name == which or which in c.path]
        if len(hit) != 1:
            raise AbcError(f"camera {which!r} matches {len(hit)} of: {', '.join(c.path for c in cams)}")
        return hit[0]
    if len(cams) > 1:
        raise AbcError(f"{len(cams)} cameras in {ar.path.name}, pick one: {', '.join(c.path for c in cams)}")
    return cams[0]


def camera_samples(path, which: str | None = None, fps: float = 24.0) -> tuple[Obj, list[tuple[int, CameraSample]],
                                                                                  dict]:
    """[(frame, CameraSample)] for every whole frame between the first and last sample time of the camera and its
    xform chain (frame = round(time * fps)). Also returns notes (time samplings, chain, static or not)."""
    with Archive(path) as ar:
        cam = find_camera(ar, which)
        geom = ar.object_props(cam).get(".geom")
        core = ar.properties(geom.ref).get(".core") if geom else None
        if core is None or core.num_samples == 0:
            raise AbcError(f"{cam.path}: no .core camera sample")
        chain, o = [], cam.parent
        while o is not None and o is not ar.top:
            xf = _xform(ar, o)
            if xf is not None:
                chain.append(xf)
            o = o.parent
        chain.reverse()                                # root first
        # Frame range from the properties that carry the motion. Constant ones (stored once) contribute only their
        # first sample time: Nuke writes .ops on the default time sampling (0, 1, 2 ... seconds), which is not a range.
        times, animated = set(), False
        for pr in [core] + [x.vals for x in chain]:
            if pr is None or not pr.num_samples:
                continue
            if pr.last_changed > 0:
                times.update(ar.sample_times(pr))
                animated = True
            else:
                times.add(ar.sample_time(pr, 0))
        if not times:
            times = {ar.sample_time(core, 0)}
        film_back = _film_back(ar, ar.properties(geom.ref))
        tmin, tmax = min(times), max(times)
        f0, f1 = math.ceil(tmin * fps - 1e-6), math.floor(tmax * fps + 1e-6)
        if f1 < f0:
            f0 = f1 = int(round(tmin * fps))
        out = []
        for f in range(f0, f1 + 1):
            t = f / fps
            W = np.eye(4)
            for xf in chain:
                L, inh = _local(ar, xf, t)
                W = (W @ L) if inh else L
            c = ar.scalar(core, ar.index_at(core, t))
            out.append((f, CameraSample(t, W, dict(zip(CORE_FIELDS, map(float, c))))))
        notes = {"camera": cam.path, "chain": [x.obj.path for x in chain], "film_back_channels": film_back,
                 "time_samplings": [{"tpc": tpc if tpc < _ACYCLIC_TPC else "acyclic", "start": ts[:4]}
                                    for tpc, ts in ar.time_samplings],
                 "sample_time_range_s": [tmin, tmax], "animated": animated,
                 "archive_metadata": ar.archive_metadata[:300]}
        return cam, out, notes
