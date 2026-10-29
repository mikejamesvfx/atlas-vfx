"""Plate reading: probe, decode with the range from the stream, OCIO to scene linear / display, tracking proxies.

Ported from prototypes/vfx_probe/plate_proxy.py and track_proxy.py, with the range bug fixed: those forced
in_range=tv, but DSCF3000 is full range (pc), so its codes were stretched. Here the range comes from ffprobe
(override only for broken files), and every colour step is a named OCIO transform.

Decoded images are normalised code values in [0, 1] (10-bit code / 1023 for a full-range decode), which is what an
OCIO log input space such as 'F-Log2 F-Gamut' expects. The plate is read in place, never copied or written.

No ComfyUI imports here (the nodes live in node_plate_read.py / node_tracking_proxy.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("atlas_vfx")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCIO = REPO_ROOT / "config" / "atlas_aces2.0_studio_flog2.ocio"
# where every node writes by default: ~/atlas-vfx-work, or ATLAS_VFX_WORK (env vars are trusted, widgets are not)
WORK_ROOT = Path(os.environ.get("ATLAS_VFX_WORK") or Path.home() / "atlas-vfx-work")
NUKE_FRAME_OFFSET = 1001
_SAFE_NAME = re.compile(r"\w[\w .\-]*")
DEFAULT_DISPLAY = "sRGB - Display"
DEFAULT_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
WORKING_SPACE = "ACEScg"

# camera body -> fallback input space when the file does not say (cameras tag log files as bt709). Fuji files do say:
# see fuji_recording_mode, which wins over this.
BODY_DEFAULT_SPACE = {
    "FUJIFILM X-H2": "F-Log2 F-Gamut",
}

# FujiFilm maker note 0x3803 VideoRecordingMode (values as in exiftool's FujiFilm.pm) -> OCIO input space
FUJI_RECORDING_MODE = {0x00: "Normal", 0x10: "F-Log", 0x20: "HLG", 0x30: "F-Log2"}
FUJI_MODE_SPACE = {"F-Log": "F-Log F-Gamut", "F-Log2": "F-Log2 F-Gamut"}
_FUJI_3803 = re.compile(rb"\x03\x38\x04\x00\x01\x00\x00\x00(....)", re.S)    # tag 0x3803, LONG, count 1, LE


def fuji_recording_mode(path: str | os.PathLike, max_moov: int = 256 << 20) -> str | None:
    """'F-Log' / 'F-Log2' / 'HLG' / 'Normal' from a Fuji MOV's MVTG maker notes (in moov), or None if absent.
    Reads only the top-level atom headers and the moov atom; never the media data."""
    import struct
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            off = 0
            while off + 8 <= size:
                fh.seek(off)
                n, typ = struct.unpack(">I4s", fh.read(8))
                hdr = 8
                if n == 1:
                    n, hdr = struct.unpack(">Q", fh.read(8))[0], 16
                elif n == 0:
                    n = size - off
                if n < hdr:
                    return None
                if typ == b"moov":
                    if n > max_moov:
                        return None
                    data = fh.read(n - hdr)
                    i = data.find(b"MVTG")
                    if i < 0:
                        return None
                    m = _FUJI_3803.search(data, i)
                    if not m:
                        return None
                    v = struct.unpack("<I", m.group(1))[0]
                    return FUJI_RECORDING_MODE.get(v, f"unknown 0x{v:x}")
                off += n
    except OSError:
        return None
    return None

_RANGE = {"pc": "full", "jpeg": "full", "tv": "legal", "mpeg": "legal"}
_MATRIX = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte170m": "bt601",
           "bt470bg": "bt601", "fcc": "fcc", "smpte240m": "smpte240m"}


class PlateError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- paths from widgets

def clean_path(value) -> str:
    """A path from a widget: whitespace and quotes stripped, `~` and %VARS% / $VARS expanded, so a saved workflow can
    say `~/atlas-vfx-work/...` and run from any home folder. Empty stays empty."""
    s = str(value or "").strip().strip('"').strip("'")
    return os.path.expandvars(os.path.expanduser(s)) if s else ""


def ocio_path(value) -> str:
    """The OCIO config to use: the widget's path, or the shipped config when blank."""
    return clean_path(value) or str(DEFAULT_OCIO)


def safe_name(name, what: str = "name") -> str:
    """One file or folder name component from a widget: letters, digits, space, _ . - only; not empty, not . or ..,
    no leading dot, no trailing dot. Refused with a ValueError, never repaired, so `..\\x` or `a'''b` fails loudly
    instead of landing in a path or a generated script."""
    s = str(name or "").strip()
    if not s or s in (".", "..") or s.endswith(".") or not _SAFE_NAME.fullmatch(s):
        raise ValueError(f"{what} {s!r}: letters, digits, space, _ . - only, one component (it becomes a file name)")
    return s


def _norm(p) -> Path:
    return Path(os.path.normcase(str(Path(p).resolve())))


def _inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def out_roots() -> list[Path]:
    """Folders nodes may write under: WORK_ROOT plus those in ATLAS_VFX_OUT_ROOTS (os.pathsep separated), resolved
    and case-folded. Read at call time so a test or a session can add one."""
    roots = [WORK_ROOT]
    roots += [Path(s.strip()) for s in os.environ.get("ATLAS_VFX_OUT_ROOTS", "").split(os.pathsep) if s.strip()]
    return [_norm(r) for r in roots]


def resolve_out_dir(user_value, default, *, plate_path=None, what: str = "out_dir") -> Path:
    """The folder (or file) a node writes to. An empty widget value means `default`. The result, with `..` resolved,
    must sit under one of out_roots(); it may never be inside this repo (production data stays out of git) or in
    the plate's folder (plates are read-only inputs). Anything else is a ValueError before anything is written."""
    s = clean_path(user_value)
    p = Path(s) if s else Path(default)
    r = _norm(p)
    if _inside(r, _norm(REPO_ROOT)):
        raise ValueError(f"{what} {p}: inside the atlas-vfx repo; production data stays out of git")
    if plate_path is not None and _inside(r, _norm(Path(plate_path).resolve().parent)):
        raise ValueError(f"{what} {p}: in the plate's folder; plates are read-only inputs")
    roots = out_roots()
    if not any(_inside(r, root) for root in roots):
        raise ValueError(f"{what} {p}: outside the work root(s) {[str(x) for x in roots]}; set ATLAS_VFX_WORK or "
                         f"add the folder to ATLAS_VFX_OUT_ROOTS")
    return p.resolve()


# --------------------------------------------------------------------------------------------- probe

def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise PlateError(f"{cmd[0]} failed ({p.returncode}): {p.stderr.strip()[-800:]}")
    return p.stdout


def ffprobe(path: str | os.PathLike) -> dict:
    return json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


def _is_rgb(pix_fmt: str) -> bool:
    return pix_fmt.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "rgba", "bgra", "gray", "ya"))


def camera_body(info: dict) -> str | None:
    tags = {k.lower(): v for k, v in (info.get("format", {}).get("tags") or {}).items()}
    comment = tags.get("comment", "")
    m = re.match(r"\s*FUJIFILM DIGITAL CAMERA\s+(\S+)", comment)
    if m:
        return f"FUJIFILM {m.group(1)}"
    make, model = tags.get("make") or tags.get("com.apple.quicktime.make"), \
        tags.get("model") or tags.get("com.apple.quicktime.model")
    if make or model:
        return " ".join(x for x in (make, model) if x)
    return None


def quick_sha256(path: str | os.PathLike, chunk: int = 8 << 20) -> str:
    """sha256 of (size, first 8 MB, last 8 MB): cheap identity for multi-GB plates. Not a full-file hash."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(chunk, size - chunk))
            h.update(fh.read(chunk))
    return h.hexdigest()


@lru_cache(maxsize=8)
def ocio_config(path: str | os.PathLike | None):
    """The OCIO config at path, or the shipped one when path is blank."""
    import PyOpenColorIO as ocio

    return ocio.Config.CreateFromFile(ocio_path(path))


def plate_ref(path: str | os.PathLike, first: int = 0, last: int | None = None,
              ocio_config_path: str | os.PathLike | None = None, input_space: str = "auto",
              range_override: str = "auto", frame_offset: int = NUKE_FRAME_OFFSET) -> dict:
    """Probe a movie plate and describe it. JSON-safe dict, passed between nodes as ATLAS_PLATE_SEQ.

    first/last are plate frame indices (0-based, inclusive). Nuke frame = index + frame_offset.
    input_space 'auto' reads the Fuji recording-mode tag (F-Log / F-Log2) from the file, else the camera body default
    (F-Log2 F-Gamut for an X-H2, with a warning), and fails if there is neither.
    range_override 'auto' trusts the stream; 'full'/'legal' only for files whose flag is known to be wrong.
    """
    p = Path(clean_path(path))
    if not p.is_file():
        raise PlateError(f"plate not found: {p}")
    if "%" in p.name or "#" in p.name:
        raise PlateError("image sequences are not supported by Plate Read yet; give it the movie file")
    info = ffprobe(p)
    if not info.get("streams"):
        raise PlateError(f"no video stream in {p}")
    s = info["streams"][0]
    pix_fmt = s.get("pix_fmt", "")
    width, height = int(s["width"]), int(s["height"])
    fps = Fraction(s.get("r_frame_rate", "0/1"))
    if fps <= 0:
        raise PlateError(f"no frame rate in {p}")

    n = s.get("nb_frames")
    if n and n != "N/A":
        frame_count = int(n)
    else:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                    "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(p)])
        frame_count = int(out.strip().split(",")[0])

    stream_range = s.get("color_range", "unknown")
    if range_override in ("full", "legal"):
        rng, rng_src = range_override, f"override (stream says {stream_range})"
        if _RANGE.get(stream_range, rng) != rng:
            log.warning("atlas-vfx: %s range overridden to %s, stream says %s", p.name, rng, stream_range)
    elif range_override != "auto":
        raise PlateError(f"range_override must be auto, full or legal, not {range_override!r}")
    elif _is_rgb(pix_fmt):
        rng, rng_src = "full", f"rgb pixel format {pix_fmt}"
    elif stream_range in _RANGE:
        rng, rng_src = _RANGE[stream_range], f"stream color_range={stream_range}"
    else:
        rng, rng_src = "legal", f"assumed (stream color_range={stream_range})"
        log.warning("atlas-vfx: %s has no range flag, assuming legal; set range_override if wrong", p.name)

    cs = s.get("color_space", "unknown")
    matrix, matrix_src = (_MATRIX[cs], "stream") if cs in _MATRIX else ("bt709", f"assumed (stream {cs})")

    body = camera_body(info)
    cfg_path = ocio_path(ocio_config_path)
    if input_space == "auto":
        mode = fuji_recording_mode(p) if (body or "").startswith("FUJIFILM") else None
        if mode in FUJI_MODE_SPACE:
            space, space_src = FUJI_MODE_SPACE[mode], f"file: FujiFilm VideoRecordingMode {mode} (tag 0x3803)"
        elif mode is not None:
            raise PlateError(f"{p.name} was recorded as {mode!r}, not F-Log/F-Log2; set input_space")
        else:
            space = BODY_DEFAULT_SPACE.get(body or "")
            if space is None:
                raise PlateError(f"no default input space for camera body {body!r}; set input_space")
            space_src = f"default for {body} (file has no recording-mode tag)"
            log.warning("atlas-vfx: %s has no Fuji recording-mode tag; assuming %s", p.name, space)
    else:
        space, space_src = input_space, "user"
    cfg = ocio_config(cfg_path)
    if cfg.getColorSpace(space) is None:
        raise PlateError(f"colorspace {space!r} not in {cfg_path}")

    last = frame_count - 1 if last is None or last < 0 else int(last)
    first = int(first)
    if not (0 <= first <= last < frame_count):
        raise PlateError(f"frames {first}-{last} outside plate 0-{frame_count - 1}")

    tc = (s.get("tags") or {}).get("timecode") or (info.get("format", {}).get("tags") or {}).get("timecode")
    return {
        "kind": "atlas_vfx.plate_ref", "version": 1,
        "path": str(p.resolve()), "quick_sha256": quick_sha256(p), "size_bytes": p.stat().st_size,
        "codec": s.get("codec_name"), "profile": s.get("profile"), "pix_fmt": pix_fmt,
        "width": width, "height": height, "fps": [fps.numerator, fps.denominator], "frame_count": frame_count,
        "range": rng, "range_source": rng_src, "color_matrix": matrix, "color_matrix_source": matrix_src,
        "chroma_location": s.get("chroma_location") or None,
        "stream_color_tags": {k: s.get(k) for k in ("color_range", "color_space", "color_transfer",
                                                     "color_primaries")},
        "colorspace": space, "colorspace_source": space_src, "ocio_config": cfg_path,
        "working_space": WORKING_SPACE,
        "camera_body": body, "timecode": tc,
        "first": first, "last": last, "frame_offset": int(frame_offset),
        "nuke_first": first + int(frame_offset), "nuke_last": last + int(frame_offset),
    }


# --------------------------------------------------------------------------------------------- decode

def proxy_size(ref: dict, width: int | None) -> tuple[int, int]:
    if not width or width >= ref["width"]:
        return ref["width"], ref["height"]
    h = int(round(width * ref["height"] / ref["width"] / 2)) * 2
    return int(width), max(h, 2)


# ffmpeg only decodes and hands over the native planes; the Y'CbCr -> R'G'B' maths is done here, because swscale's
# 10-bit full-range conversion is off by about a code and tints neutrals (measured 2026-10-19: Y=300 neutral came
# back as R 299.1, G 301.4, B 298.9).
_YUV_FMT = re.compile(r"^yuvj?(420|422|444)p(\d+)?(le)?$")
_KR_KB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593),
          "smpte240m": (0.212, 0.087), "fcc": (0.30, 0.11)}


def _yuv_layout(pix_fmt: str):
    m = _YUV_FMT.match(pix_fmt)
    if not m:
        return None
    sub, bits = m.group(1), int(m.group(2) or 8)
    if bits > 8 and not m.group(3):
        return None   # big-endian high bit depth: not seen in practice
    sx, sy = {"420": (2, 2), "422": (2, 1), "444": (1, 1)}[sub]
    return sx, sy, bits


def _upsample(c: np.ndarray, factor: int, axis: int, offset: float) -> np.ndarray:
    """Linear chroma upsampling. Luma sample k sits at chroma coordinate (k - offset) / factor: offset 0 for
    co-sited chroma, 0.5 for chroma centred between luma samples."""
    if factor == 1:
        return c
    n = c.shape[axis] * factor
    pos = np.clip((np.arange(n) - offset) / factor, 0, c.shape[axis] - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, c.shape[axis] - 1)
    w = (pos - i0).astype(np.float32)
    shape = [1, 1]
    shape[axis] = n
    w = w.reshape(shape)
    return np.take(c, i0, axis=axis) * (1 - w) + np.take(c, i1, axis=axis) * w


def _raw_frames(ref: dict, first: int, last: int, luma_only: bool, step: int = 1):
    """Yields (index, planes) straight from ffmpeg: (Y, Cb, Cr) native integer planes for Y'CbCr, or an
    HxWx3 uint16 array (rgb48le) for RGB formats."""
    W, H = ref["width"], ref["height"]
    lay = _yuv_layout(ref["pix_fmt"])
    if lay is None and not _is_rgb(ref["pix_fmt"]):
        raise PlateError(f"pixel format {ref['pix_fmt']} not supported by Plate Read")
    if lay:
        sx, sy, bits = lay
        dt = np.dtype("<u2") if bits > 8 else np.dtype("u1")
        ny, nc = W * H, (W // sx) * (H // sy)
        out_fmt, nbytes = ref["pix_fmt"], (ny + 2 * nc) * dt.itemsize
    else:
        out_fmt, nbytes = "rgb48le", W * H * 6
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", ref["path"], "-map", "0:v:0",
           "-vf", f"select=between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{step}))",
           "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", out_fmt, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    i = first
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            if lay:
                a = np.frombuffer(buf, dt)
                y = a[:ny].reshape(H, W)
                planes = (y, None, None) if luma_only else (
                    y, a[ny:ny + nc].reshape(H // sy, W // sx), a[ny + nc:].reshape(H // sy, W // sx))
            else:
                planes = np.frombuffer(buf, "<u2").reshape(H, W, 3)
            yield i, planes
            i += step
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise PlateError(f"ffmpeg decode failed ({proc.returncode}): {err.strip()[-800:]}")
    want_end = first + ((last - first) // step + 1) * step
    if i != want_end:
        raise PlateError(f"ffmpeg stopped at frame {i - step}, expected {first}-{last} step {step}: "
                         f"{err.strip()[-400:]}")


def _norm_luma(y: np.ndarray, bits: int, full: bool) -> np.ndarray:
    """H.273: full range Y/(2^n - 1); legal range (Y - 16*2^(n-8)) / (219*2^(n-8)). Not clipped."""
    y = y.astype(np.float32)
    if full:
        return y / float((1 << bits) - 1)
    s = float(1 << (bits - 8))
    return (y - 16 * s) / (219 * s)


def _norm_chroma(c: np.ndarray, bits: int, full: bool) -> np.ndarray:
    c = c.astype(np.float32) - float(1 << (bits - 1))
    return c / float((1 << bits) - 1) if full else c / (224 * float(1 << (bits - 8)))


def _siting(ref: dict) -> tuple[float, float]:
    """(horizontal, vertical) chroma offsets from the stream's chroma_location (H.273 / ffmpeg names)."""
    loc = ref.get("chroma_location") or "left"
    return {"left": (0.0, 0.5), "center": (0.5, 0.5), "topleft": (0.0, 0.0), "top": (0.5, 0.0),
            "bottomleft": (0.0, 1.0), "bottom": (0.5, 1.0)}.get(loc, (0.0, 0.5))


def _resize(img: np.ndarray, width: int | None, ref: dict) -> np.ndarray:
    W, H = proxy_size(ref, width)
    if (W, H) == (ref["width"], ref["height"]):
        return img
    import cv2

    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def decode_rgb(ref: dict, first: int | None = None, last: int | None = None, width: int | None = None,
               step: int = 1):
    """Yields (plate frame index, float32 HxWx3 normalised code values in the plate's input space).
    Y'CbCr -> R'G'B' with the plate's range and matrix (H.273), chroma upsampled linearly at its siting."""
    f0 = ref["first"] if first is None else first
    f1 = ref["last"] if last is None else last
    lay = _yuv_layout(ref["pix_fmt"])
    full = ref["range"] == "full"
    for i, planes in _raw_frames(ref, f0, f1, luma_only=False, step=max(1, int(step))):
        if lay is None:
            rgb = planes.astype(np.float32) / 65535.0
        else:
            sx, sy, bits = lay
            ox, oy = _siting(ref)
            y = _norm_luma(planes[0], bits, full)
            cb = _upsample(_upsample(_norm_chroma(planes[1], bits, full), sx, 1, ox), sy, 0, oy)
            cr = _upsample(_upsample(_norm_chroma(planes[2], bits, full), sx, 1, ox), sy, 0, oy)
            kr, kb = _KR_KB[ref["color_matrix"]]
            kg = 1.0 - kr - kb
            r = y + 2 * (1 - kr) * cr
            b = y + 2 * (1 - kb) * cb
            g = (y - kr * r - kb * b) / kg
            rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        yield i, _resize(rgb, width, ref)


def decode_luma16(ref: dict, first: int | None = None, last: int | None = None, width: int | None = None,
                  step: int = 1):
    """Yields (plate frame index, uint16 HxW luma, range-expanded to 0-65535). For tracking proxies and charts."""
    f0 = ref["first"] if first is None else first
    f1 = ref["last"] if last is None else last
    lay = _yuv_layout(ref["pix_fmt"])
    full = ref["range"] == "full"
    for i, planes in _raw_frames(ref, f0, f1, luma_only=True, step=max(1, int(step))):
        if lay is None:
            rgb = planes.astype(np.float32) / 65535.0
            y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        else:
            y = _norm_luma(planes[0], lay[2], full)
        y = _resize(np.clip(y, 0, 1), width, ref)
        yield i, (y * 65535.0 + 0.5).astype(np.uint16)


# --------------------------------------------------------------------------------------------- colour

@lru_cache(maxsize=32)
def _cpu(config_path: str, src: str, dst: str | None, display: str | None, view: str | None):
    import PyOpenColorIO as ocio

    cfg = ocio_config(config_path)
    if dst is not None:
        proc = cfg.getProcessor(src, dst)
    else:
        proc = cfg.getProcessor(ocio.DisplayViewTransform(src=src, display=display, view=view))
    return proc.getDefaultCPUProcessor()


def to_working(img: np.ndarray, ref: dict, dst: str | None = None) -> np.ndarray:
    """Input space -> scene linear (ACEScg unless dst given). Named OCIO transform; returns a new float32 array."""
    out = np.ascontiguousarray(img, dtype=np.float32).copy()
    _cpu(ref["ocio_config"], ref["colorspace"], dst or ref.get("working_space", WORKING_SPACE), None, None) \
        .applyRGB(out)
    return out


def to_display(img: np.ndarray, ref: dict, display: str = DEFAULT_DISPLAY, view: str = DEFAULT_VIEW) -> np.ndarray:
    """Input space -> display-referred through the named OCIO display/view, clipped to [0, 1]."""
    out = np.ascontiguousarray(img, dtype=np.float32).copy()
    _cpu(ref["ocio_config"], ref["colorspace"], None, display, view).applyRGB(out)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------------------------- tracking proxy

def tracking_proxy_frame(y16: np.ndarray, clahe) -> np.ndarray:
    """Log luma, per-frame 0.5-99.5 % stretch, CLAHE on 16 bit, 8 bit out (the prototype's recipe)."""
    lo, hi = np.percentile(y16, 0.5), np.percentile(y16, 99.5)
    y = np.clip((y16.astype(np.float32) - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return (clahe.apply((y * 65535).astype(np.uint16)) >> 8).astype(np.uint8)


def default_track_dir(ref: dict, width: int) -> Path:
    return WORK_ROOT / Path(ref["path"]).stem / f"track_{ref['first']:04d}-{ref['last']:04d}_w{width}"


def write_tracking_proxies(ref: dict, out_dir: str | os.PathLike | None = None, width: int = 1024,
                           clip_limit: float = 3.0, tiles: int = 8, keep_every: int = 0):
    """Writes plate_<index>.png per frame + track_proxy.json. Returns (out_dir, kept previews as uint8 HxW)."""
    import cv2

    out = resolve_out_dir(out_dir, default_track_dir(ref, width), plate_path=ref["path"])
    out.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    kept, written = [], []
    for i, y16 in decode_luma16(ref, width=width):
        y8 = tracking_proxy_frame(y16, clahe)
        name = f"plate_{i:04d}.png"
        if not cv2.imwrite(str(out / name), y8):
            raise PlateError(f"could not write {out / name}")
        written.append(name)
        if keep_every and (i - ref["first"]) % keep_every == 0:
            kept.append(y8)
    W, H = proxy_size(ref, width)
    write_sidecar(out / "track_proxy.json", "AtlasVFXTrackingProxy",
                  inputs={"plate_ref": ref, "width": width, "clip_limit": clip_limit, "tiles": tiles},
                  outputs={"size": [W, H], "frames": [ref["first"], ref["last"]], "files": len(written),
                           "pattern": "plate_%04d.png (plate frame index)",
                           "recipe": "luma code values (range from stream), per-frame 0.5-99.5% stretch, "
                                     "CLAHE 16-bit, >>8"})
    return out, kept


# --------------------------------------------------------------------------------------------- sidecar

@lru_cache(maxsize=1)
def tool_versions() -> dict:
    v = {}
    try:
        v["ffmpeg"] = _run(["ffmpeg", "-version"]).splitlines()[0]
    except Exception as e:  # noqa: BLE001 - sidecar must not fail the node
        v["ffmpeg"] = f"unavailable: {e}"
    try:
        import PyOpenColorIO as ocio
        v["opencolorio"] = ocio.__version__
    except ImportError:
        v["opencolorio"] = None
    try:
        import cv2
        v["opencv"] = cv2.__version__
    except ImportError:
        v["opencv"] = None
    v["numpy"] = np.__version__
    return v


def write_sidecar(path: str | os.PathLike, node: str, inputs: dict, outputs: dict) -> Path:
    import datetime
    import platform

    try:
        from atlas_vfx import __version__ as pack_version  # absent after the split; fine
    except ImportError:
        pack_version = None
    doc = {
        "node": node, "pack_version": pack_version,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(), "python": platform.python_version(), "tools": tool_versions(),
        "inputs": inputs, "outputs": outputs,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p
