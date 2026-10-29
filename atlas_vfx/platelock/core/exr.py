"""Minimal uncompressed float32 scanline OpenEXR writer/reader.

No dependencies, so the core installs anywhere. Nuke, Resolve and OIIO read
these. Channels are written in the alphabetical order EXR requires.
"""
from __future__ import annotations

import struct

import numpy as np

_MAGIC = 20000630


def _attr(name: str, typ: str, data: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def _encode(typ: str, value) -> bytes:
    if typ == "string":
        return str(value).encode()
    if typ == "float":
        return struct.pack("<f", float(value))
    if typ == "double":
        return struct.pack("<d", float(value))
    if typ == "int":
        return struct.pack("<i", int(value))
    if typ == "rational":
        n, d = value
        return struct.pack("<iI", int(n), int(d))
    if typ == "v2f":
        return struct.pack("<ff", *map(float, value))
    if typ == "m44f":
        return np.asarray(value, "<f4").reshape(16).tobytes()
    raise ValueError(f"unsupported attribute type {typ}")


def write_exr(path: str, channels: dict, attrs: dict | None = None) -> None:
    """channels: {"R": HxW, "G": HxW, ...} float arrays of equal shape.

    attrs: extra header attributes {name: (type, value)}, type one of
    string, float, double, int, rational, v2f, m44f. Used to carry provenance
    (source frame, retime, solve) into the delivered maps.
    """
    names = sorted(channels)
    arrs = [np.ascontiguousarray(np.asarray(channels[n], dtype="<f4")) for n in names]
    h, w = arrs[0].shape
    for a in arrs:
        if a.shape != (h, w):
            raise ValueError("all channels must share a shape")
    chl = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for n in names) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (struct.pack("<ii", _MAGIC, 2)
              + _attr("channels", "chlist", chl)
              + _attr("compression", "compression", b"\0")
              + _attr("dataWindow", "box2i", box)
              + _attr("displayWindow", "box2i", box)
              + _attr("lineOrder", "lineOrder", b"\0")
              + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
              + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + b"".join(_attr(k, t, _encode(t, v)) for k, (t, v) in sorted((attrs or {}).items()))
              + b"\0")
    line_bytes = 4 * w * len(names)
    start = len(header) + 8 * h
    offsets = struct.pack(f"<{h}Q", *[start + y * (8 + line_bytes) for y in range(h)])
    with open(path, "wb") as f:
        f.write(header)
        f.write(offsets)
        for y in range(h):
            f.write(struct.pack("<ii", y, line_bytes))
            for a in arrs:
                f.write(a[y].tobytes())


def read_exr(path: str) -> dict:
    """Reads files written by write_exr (uncompressed float32 scanline only)."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, ver = struct.unpack_from("<ii", buf, 0)
    if magic != _MAGIC:
        raise ValueError("not an EXR")
    p = 8
    attrs = {}
    while buf[p] != 0:
        e = buf.index(b"\0", p); name = buf[p:e].decode(); p = e + 1
        e = buf.index(b"\0", p); typ = buf[p:e].decode(); p = e + 1
        (size,) = struct.unpack_from("<i", buf, p); p += 4
        attrs[name] = (typ, buf[p:p + size]); p += size
    p += 1
    if attrs["compression"][1] != b"\0":
        raise ValueError("only uncompressed EXR supported")
    chl = attrs["channels"][1]
    names, q = [], 0
    while chl[q] != 0:
        e = chl.index(b"\0", q); names.append(chl[q:e].decode()); q = e + 1 + 16
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    offsets = struct.unpack_from(f"<{h}Q", buf, p)
    out = {n: np.empty((h, w), np.float32) for n in names}
    for off in offsets:
        y, _ = struct.unpack_from("<ii", buf, off)
        o = off + 8
        for n in names:
            out[n][y - y0] = np.frombuffer(buf, "<f4", w, o); o += 4 * w
    return out


def write_stmap(path: str, stmap: np.ndarray, valid: np.ndarray | None = None, attrs: dict | None = None) -> None:
    """STMap EXR for Nuke's STMap node: s in R, t in G, B zero, A = valid."""
    h, w = stmap.shape[:2]
    a = np.ones((h, w), np.float32) if valid is None else np.asarray(valid, np.float32)
    write_exr(path, {"R": stmap[..., 0], "G": stmap[..., 1], "B": np.zeros((h, w), np.float32), "A": a}, attrs)


def write_rgba(path: str, rgb: np.ndarray, alpha: np.ndarray | None = None, attrs: dict | None = None) -> None:
    ch = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
    if alpha is not None:
        ch["A"] = alpha
    write_exr(path, ch, attrs)
