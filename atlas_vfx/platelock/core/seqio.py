"""EXR sequence input with header metadata.

Plates arrive as EXR sequences (debayered, conformed), not camera raw. The
header parser below reads attributes from ANY OpenEXR file, whatever its
compression, because headers are never compressed; pixels go through
whichever EXR library is installed (OpenImageIO, then OpenEXR, then imageio),
falling back to the built-in reader for uncompressed files.

camera_metadata() pulls out what the lock can use (fps, shutter, focal
length, sensor size, timecode, lens/camera names) and lists anything that
looks like gyro or readout data, so you can see what your conform carried
across. Nothing is assumed about which keys a given tool writes; known
standard names are tried first, then keys are matched loosely.
"""
from __future__ import annotations

import glob
import os
import re
import struct

import numpy as np

_MAGIC = 20000630
_COMPRESSION = ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k"]


def _cstr(buf, p):
    e = buf.index(b"\0", p)
    return buf[p:e].decode("utf-8", "replace"), e + 1


def _decode(typ: str, data: bytes):
    try:
        if typ == "string":
            return data.decode("utf-8", "replace")
        if typ == "int":
            return struct.unpack("<i", data)[0]
        if typ == "float":
            return struct.unpack("<f", data)[0]
        if typ == "double":
            return struct.unpack("<d", data)[0]
        if typ == "v2i":
            return list(struct.unpack("<ii", data))
        if typ == "v2f":
            return list(struct.unpack("<ff", data))
        if typ == "v3f":
            return list(struct.unpack("<fff", data))
        if typ == "v2d":
            return list(struct.unpack("<dd", data))
        if typ == "v3d":
            return list(struct.unpack("<ddd", data))
        if typ == "box2i":
            return list(struct.unpack("<iiii", data))
        if typ == "box2f":
            return list(struct.unpack("<ffff", data))
        if typ == "m33f":
            return np.frombuffer(data, "<f4").reshape(3, 3).tolist()
        if typ == "m44f":
            return np.frombuffer(data, "<f4").reshape(4, 4).tolist()
        if typ == "m33d":
            return np.frombuffer(data, "<f8").reshape(3, 3).tolist()
        if typ == "m44d":
            return np.frombuffer(data, "<f8").reshape(4, 4).tolist()
        if typ == "rational":
            n, d = struct.unpack("<iI", data)
            return n / d if d else float("nan")
        if typ == "compression":
            c = data[0]
            return _COMPRESSION[c] if c < len(_COMPRESSION) else c
        if typ == "lineOrder":
            return ["increasing_y", "decreasing_y", "random_y"][data[0]] if data[0] < 3 else data[0]
        if typ == "chlist":
            out, q = [], 0
            while data[q] != 0:
                name, q = _cstr(data, q)
                ptype, = struct.unpack_from("<i", data, q)
                out.append({"name": name, "type": ["uint", "half", "float"][ptype] if ptype < 3 else ptype})
                q += 16
            return out
        if typ == "timecode":
            tc, ub = struct.unpack("<II", data)

            def bcd(v):
                return (v & 0xF) + 10 * ((v >> 4) & 0xF)
            fr = bcd(tc & 0x3F)
            sec = bcd((tc >> 8) & 0x7F)
            mn = bcd((tc >> 16) & 0x7F)
            hr = bcd((tc >> 24) & 0x3F)
            return f"{hr:02d}:{mn:02d}:{sec:02d}:{fr:02d}"
        if typ == "stringvector":
            out, q = [], 0
            while q + 4 <= len(data):
                n, = struct.unpack_from("<i", data, q)
                q += 4
                out.append(data[q:q + n].decode("utf-8", "replace"))
                q += n
            return out
        if typ == "floatvector":
            return np.frombuffer(data, "<f4").tolist()
        if typ == "chromaticities":
            return list(struct.unpack("<8f", data))
    except Exception:  # noqa: BLE001 - unknown or malformed attribute: keep raw
        pass
    return {"type": typ, "bytes": len(data)}


def read_exr_header(path: str) -> dict:
    """All header attributes of an EXR (first part if multi-part)."""
    with open(path, "rb") as f:
        buf = f.read(1 << 20)  # headers are small; 1 MB is generous
    magic, ver = struct.unpack_from("<ii", buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"{path}: not an OpenEXR file")
    p = 8
    attrs = {}
    while buf[p] != 0:
        name, p = _cstr(buf, p)
        typ, p = _cstr(buf, p)
        size, = struct.unpack_from("<i", buf, p)
        p += 4
        attrs[name] = _decode(typ, buf[p:p + size])
        p += size
    attrs["_version_flags"] = {"tiled": bool(ver & 0x200), "long_names": bool(ver & 0x400),
                               "deep": bool(ver & 0x800), "multipart": bool(ver & 0x1000)}
    return attrs


_ALIASES = {
    "fps": ["framesPerSecond", "fps", "frameRate", "FramesPerSecond", "nuke/input/frame_rate"],
    "shutter_angle": ["shutterAngle", "ShutterAngle", "shutter_angle"],
    "exposure_time": ["expTime", "exposureTime", "ExposureTime", "shutterSpeed"],
    "focal_mm": ["focalLength", "FocalLength", "lensFocalLength", "focal_length"],
    "sensor_width_mm": ["sensorPhysicalWidth", "sensorAcrossDimension", "sensorWidth", "SensorWidth"],
    "sensor_height_mm": ["sensorPhysicalHeight", "sensorHeight", "SensorHeight"],
    "aperture": ["aperture", "fNumber", "FNumber", "tStop"],
    "focus_distance": ["focus", "focusDistance", "FocusDistance"],
    "iso": ["isoSpeed", "ISO", "iso"],
    "timecode": ["timeCode", "timecode", "TimeCode"],
    "camera": ["cameraModel", "camera", "CameraModel", "cameraMake"],
    "lens": ["lensModel", "lens", "LensModel", "lensMake"],
    "capture_date": ["capDate", "captureDate"],
}
_LOOSE = {
    "gyro": re.compile(r"gyro|imu|angular|quaternion|orientation", re.I),
    "readout": re.compile(r"readout|rolling|shutter_?time|line_?time", re.I),
    "lens_extra": re.compile(r"distortion|anamorph|squeeze", re.I),
}


def camera_metadata(header: dict) -> dict:
    """Normalise the bits of an EXR header the lock can use."""
    out = {"source_keys": {}}
    for field, keys in _ALIASES.items():
        for k in keys:
            if k in header:
                out[field] = header[k]
                out["source_keys"][field] = k
                break
        else:
            low = {kk.lower(): kk for kk in header}
            for k in keys:
                if k.lower() in low:
                    out[field] = header[low[k.lower()]]
                    out["source_keys"][field] = low[k.lower()]
                    break
    for name, rx in _LOOSE.items():
        hits = {k: header[k] for k in header if rx.search(k)}
        if hits:
            out[name] = hits
    dw = header.get("dataWindow")
    if dw:
        out["width"], out["height"] = dw[2] - dw[0] + 1, dw[3] - dw[1] + 1
    if "shutter_angle" not in out and "exposure_time" in out and "fps" in out:
        try:
            out["shutter_angle"] = float(out["exposure_time"]) * float(out["fps"]) * 360.0
            out["source_keys"]["shutter_angle"] = "derived from exposure_time * fps"
        except (TypeError, ValueError):
            pass
    out["compression"] = header.get("compression")
    out["channels"] = [c["name"] for c in header.get("channels", [])] if isinstance(header.get("channels"), list) else None
    return out


def focal_px(focal_mm: float, sensor_width_mm: float, width_px: int) -> float:
    """Focal length in pixels for the delivered image width (use the active, cropped sensor width)."""
    return float(focal_mm) / float(sensor_width_mm) * float(width_px)


# ------------------------------------------------------------------ pixels

def _read_pixels(path: str, channels=("R", "G", "B")) -> np.ndarray:
    errs = []
    try:
        import OpenImageIO as oiio  # type: ignore
        inp = oiio.ImageInput.open(path)
        if inp is None:
            raise RuntimeError(oiio.geterror())
        spec = inp.spec()
        names = list(spec.channelnames)
        px = inp.read_image(0, 0, 0, spec.nchannels, "float")
        inp.close()
        px = np.asarray(px).reshape(spec.height, spec.width, spec.nchannels)
        idx = [names.index(c) for c in channels if c in names]
        return px[..., idx].astype(np.float32)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        errs.append(f"OpenImageIO: {e}")
    try:
        import OpenEXR  # type: ignore
        with OpenEXR.File(path) as f:  # OpenEXR >= 3.3 python API
            ch = f.channels()
            if "RGB" in ch and set(channels) == {"R", "G", "B"}:
                return np.asarray(ch["RGB"].pixels, np.float32)
            return np.stack([np.asarray(ch[c].pixels, np.float32) for c in channels if c in ch], -1)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        errs.append(f"OpenEXR: {e}")
    try:
        import imageio.v3 as iio  # type: ignore
        px = np.asarray(iio.imread(path), np.float32)
        return px[..., :len(channels)] if px.ndim == 3 else px[..., None]
    except Exception as e:  # noqa: BLE001
        errs.append(f"imageio: {e}")
    from .exr import read_exr
    try:
        d = read_exr(path)
        return np.stack([d[c] for c in channels if c in d], -1)
    except Exception as e:  # noqa: BLE001
        errs.append(f"built-in (uncompressed only): {e}")
    raise RuntimeError(f"could not read {path}. Install OpenImageIO or OpenEXR. Tried: " + "; ".join(errs))


def list_sequence(pattern: str) -> list[str]:
    """Accepts a folder, a glob, 'shot.####.exr' or 'shot.%04d.exr'."""
    if os.path.isdir(pattern):
        files = glob.glob(os.path.join(pattern, "*.exr"))
    else:
        g = re.sub(r"#+", "*", pattern)
        g = re.sub(r"%0?\d*d", "*", g)
        files = glob.glob(g)

    def frame_no(f):
        m = re.findall(r"(\d+)(?=\.[^.]+$)", os.path.basename(f))
        return int(m[-1]) if m else 0
    return sorted(files, key=frame_no)


def read_sequence(pattern: str, first: int | None = None, last: int | None = None,
                  channels=("R", "G", "B"), every: int = 1):
    """Load an EXR sequence. Returns (frames (N,H,W,C) float32, headers, files)."""
    files = list_sequence(pattern)
    if not files:
        raise FileNotFoundError(f"no EXR files match {pattern}")
    files = files[slice(first, None if last is None else last + 1, every)]
    headers = [read_exr_header(f) for f in files]
    frames = np.stack([_read_pixels(f, channels) for f in files])
    return frames, headers, files


def sequence_report(headers: list[dict]) -> dict:
    """Metadata summary over a sequence, flagging anything that changes per frame."""
    metas = [camera_metadata(h) for h in headers]
    first = metas[0]
    varying = sorted({k for m in metas[1:] for k in m
                      if k not in ("source_keys", "timecode") and m.get(k) != first.get(k)})
    return {"frames": len(headers), "first": first, "varying_keys": varying,
            "timecode_first": first.get("timecode"), "timecode_last": metas[-1].get("timecode")}
