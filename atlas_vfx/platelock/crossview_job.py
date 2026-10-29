"""Cross-view job: plate EXR sequence + AI EXR sequence + ID mattes -> locked AI, frame for frame.

For every frame N (plate N <-> AI N):
  - both frames to ACEScct through OCIO, for matching only
  - per ID: cross-view warp solved at proc_width (2K default), maps scaled to full plate resolution
  - the original linear AI frame is warped per ID (bilinear, in its own colourspace)
  - comp: plate ID mattes are the alphas; AI colour is edge extended where the plate matte reaches
    past the warped AI matte, so plate edges never pick up the AI's invented background

Writes under out_dir (outside the repo):
  <id>/<id>.####.exr         warped AI, RGBA half ZIP, A = plate ID matte, colourspace tagged
  <id>/<id>_stmap.####.exr   STMap into the AI frame of the same number, float ZIP, A = confidence
  comp/comp.####.exr         all IDs over the AI frame (or the plate, see base)
  crossview.json             config, inputs, per-frame and per-ID stats, versions (sidecar)
  platelock_crossview.nk     Reads for every sequence, ready to wire into STMap nodes
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..camera_side import nuke_text as nt
from . import colour
from .core import crossview as cv
from .core import seqio, warp


def _frame_files(pattern: str) -> dict[int, str]:
    files = seqio.list_sequence(pattern)
    out = {}
    for f in files:
        m = re.findall(r"(\d+)(?=\.[^.]+$)", os.path.basename(f))
        if m:
            out[int(m[-1])] = f
    if not out:
        raise FileNotFoundError(f"no frames match {pattern}")
    return out


def _read(path: str, channels=("R", "G", "B")) -> np.ndarray:
    return seqio._read_pixels(path, channels)


def _read_matte(path: str, channel: str) -> np.ndarray:
    chans = (channel,) if channel else ("A",)
    try:
        m = _read(path, chans)
    except Exception:  # noqa: BLE001
        m = None
    if m is None or m.size == 0 or m.shape[-1] == 0:
        m = _read(path, ("R",))
    return np.clip(m[..., 0], 0, 1).astype(np.float32)


def _write(path: Path, rgba: np.ndarray, colorspace: str, attrs: dict, half: bool):
    from ..camera_side import exr_io
    exr_io.write_rgba(path, rgba, colorspace=colorspace, attrs=attrs, half=half, premultiplied=False)


REPO = Path(__file__).resolve().parents[2]


def inside_repo(path, repo: Path = REPO) -> bool:
    """True if path is the repo or under it. Path containment, not string prefix: atlas-vfx-work is outside."""
    p = Path(os.path.normcase(str(Path(path).resolve())))
    r = Path(os.path.normcase(str(Path(repo).resolve())))
    return p == r or r in p.parents


def _core_hash() -> str:
    h = hashlib.sha1()
    core = Path(__file__).parent / "core"
    for f in sorted(core.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def run(plate: str, ai: str, plate_ids: dict, out_dir: str, *, ai_ids: dict | None = None,
        plate_space: str = "ACEScg", ai_space: str = "ACEScg", ocio_config: str | None = None,
        first: int | None = None, last: int | None = None, matte_channel: str = "A",
        cfg: cv.CVConfig | None = None, base: str = "ai", write_comp: bool = True,
        progress=None) -> dict:
    """plate_ids / ai_ids: {name: matte sequence pattern}. base: 'ai' or 'plate' under the IDs in comp."""
    cfg = cfg or cv.CVConfig()
    od = Path(out_dir)
    if inside_repo(od):
        raise ValueError("platelock: out_dir is inside the atlas-vfx repo; production data stays out of git")
    od.mkdir(parents=True, exist_ok=True)

    P = _frame_files(plate)
    A = _frame_files(ai)
    PM = {k: _frame_files(v) for k, v in plate_ids.items()}
    AM = {k: _frame_files(v) for k, v in (ai_ids or {}).items()}
    frames = sorted(set(P) & set(A))
    if first is not None:
        frames = [f for f in frames if f >= first]
    if last is not None:
        frames = [f for f in frames if f <= last]
    if not frames:
        raise ValueError("plate and AI sequences share no frame numbers in range")
    for k, fm in PM.items():
        miss = [f for f in frames if f not in fm]
        if miss:
            raise ValueError(f"plate matte {k} is missing frames {miss[:5]}{'...' if len(miss) > 5 else ''}")

    head = seqio.read_exr_header(P[frames[0]])
    per_frame = {}
    t0 = time.time()
    for n, f in enumerate(frames):
        plate_lin = _read(P[f])
        ai_lin = _read(A[f])
        H, W = plate_lin.shape[:2]
        ah, aw = ai_lin.shape[:2]
        pm = {k: _read_matte(PM[k][f], matte_channel) for k in PM}
        am = {k: _read_matte(AM[k][f], matte_channel) for k in AM if f in AM[k]} or None
        if am is not None:
            am = {k: cv.resize(v, W, H) for k, v in am.items()}

        plate_log = colour.to_match(plate_lin, plate_space, ocio_config)
        ai_log = colour.to_match(ai_lin, ai_space, ocio_config)
        ai_log = cv.resize(ai_log, W, H)          # solve in plate pixels; maps point into the AI at its size
        res = cv.crossview_frame(plate_log, ai_log, pm, am, cfg)

        warped, walpha, stats = {}, {}, {}
        for k, r in res.ids.items():
            maps = cv.scale_maps(r.maps, res.proc_wh, (W, H), (aw, ah))
            conf = cv.resize(r.confidence, W, H, area=False)
            rgb = warp.remap(ai_lin, *maps, border="replicate")
            warped[k] = rgb
            if am is not None and k in am:
                # AI matte at its own size for sampling through the full-res map
                amat = _read_matte(AM[k][f], matte_channel)
                walpha[k] = warp.remap(amat, *maps)
            d = od / k
            attrs = {"source_frame": f, "plate_colorspace": plate_space, "id": k}
            _write(d / f"{k}.{f:04d}.exr", np.dstack([rgb, pm[k]]), ai_space, attrs, half=True)
            st = warp.to_stmap(maps, aw, ah)
            _write(d / f"{k}_stmap.{f:04d}.exr", np.dstack([st, np.zeros((H, W), np.float32), conf]),
                   "Raw", {**attrs, "stmap": "nuke s,t into AI frame; A = confidence"}, half=False)
            stats[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                        for kk, vv in r.stats.items() if not kk.startswith("_")}

        if write_comp:
            base_img = ai_lin if base == "ai" else colour.convert(plate_lin, plate_space, ai_space, ocio_config)
            if base_img.shape[:2] != (H, W):
                base_img = cv.resize(base_img, W, H)
            comp, cover = cv.comp_by_plate_mattes(base_img, warped, pm, base_img, walpha or None)
            _write(od / "comp" / f"comp.{f:04d}.exr", np.dstack([comp, cover]), ai_space,
                   {"source_frame": f, "base": base}, half=True)
        per_frame[f] = stats
        if progress:
            progress(n + 1, len(frames))

    summary = _summarise(per_frame)
    from atlas_vfx import __version__
    side = {
        "schema": "platelock.crossview/1", "atlas_vfx": __version__, "core_hash": _core_hash(),
        "python": platform.python_version(), "seconds": round(time.time() - t0, 1),
        "inputs": {"plate": plate, "ai": ai, "plate_ids": plate_ids, "ai_ids": ai_ids or {},
                   "plate_space": plate_space, "ai_space": ai_space,
                   "ocio_config": str(ocio_config or colour.DEFAULT_OCIO), "match_space": colour.MATCH_SPACE,
                   "matte_channel": matte_channel, "base": base},
        "plate_header": {k: v for k, v in seqio.camera_metadata(head).items() if k != "source_keys"},
        "frames": [frames[0], frames[-1]], "config": asdict(cfg), "summary": summary,
        "per_frame": {str(k): v for k, v in per_frame.items()},
    }
    (od / "crossview.json").write_text(json.dumps(side, indent=1, default=str))
    (od / "platelock_crossview.nk").write_text(_nuke(od, list(PM), frames, ai))
    return side


def _summarise(per_frame: dict) -> dict:
    out = {}
    ids = sorted({k for v in per_frame.values() for k in v})
    for k in ids:
        rows = [v[k] for v in per_frame.values() if k in v]

        def col(name):
            vals = [r[name] for r in rows if name in r and r[name] is not None and np.isfinite(r[name])]
            return vals
        s = {}
        for name in ("residual_px_median", "confident_fraction", "edge_px_raw", "iou_raw"):
            v = col(name)
            if v:
                s[name] = {"mean": float(np.mean(v)), "worst": float(np.max(v) if "px" in name else np.min(v))}
        out[k] = s
    return out


def _nuke(od: Path, ids: list, frames: list, ai: str) -> str:
    f0, f1 = frames[0], frames[-1]
    o = str(od).replace("\\", "/")
    lines = ["# platelock cross-view: generated Reads. Wire Read_AI into STMap.src and each _stmap Read into",
             "# STMap.stmap (UV channels rgb); the baked layers carry the plate ID matte in alpha.",
             f"Root {{\n inputs 0\n first_frame {f0}\n last_frame {f1}\n}}"]

    def read(path, name, x):
        return (f"Read {{\n inputs 0\n file {nt.nk_str(path)}\n first {int(f0)}\n last {int(f1)}\n on_error black\n"
                f" name {nt.nk_name(name)}\n xpos {x}\n ypos 0\n}}")
    lines.append(read(ai.replace("\\", "/").replace("%04d", "####"), "Read_AI", 0))
    lines.append(read(f"{o}/comp/comp.####.exr", "Read_comp", 150))
    for n, k in enumerate(ids):
        lines.append(read(f"{o}/{k}/{k}_stmap.####.exr", f"Read_{k}_stmap", 300 + 300 * n))
        lines.append(read(f"{o}/{k}/{k}.####.exr", f"Read_{k}", 450 + 300 * n))
    return "\n".join(lines) + "\n"
