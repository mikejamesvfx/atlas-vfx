"""Lens + readout from a filmed chart take: the shot-level half of camera_side/grid_rs.py. One code path for
tools/grid_rs_shoot.py and the Chart Calibrate node (catalogue 49).

  lens     static holds of the chart (take A). Every Nth frame is detected together with the next frame; a view is
           kept only if the chart barely moved between them (< still_px), so rolling shutter can't bend it.
           K at plate size, k1 k2, focal in mm on the sensor preset, principal point, rms.
  readout  fast pans / tilts with the whole chart in view (takes B-E). Every frame is detected, labelled, and fitted
           with per-frame poses plus one readout, against the lens from take A. A take with no motion (locked off)
           is reported as not measurable, never as a number.

Frames come through the plate reader (plate.py: range from the stream, Y'CbCr planes straight from ffmpeg), luma
only, stretched per frame to 8 bit (the chart is black and white; the log curve doesn't matter for corners). The
take is read in place, never written next to. No ComfyUI imports; no imports from layers/ or deliver/.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from . import grid_rs, sensors
from . import plate as pl

log = logging.getLogger("atlas_vfx")

DEFAULT_PRESET = "fujifilm_xh2_dci4k_24p_dis_off"
MIN_STILL_VIEWS = 4
# A readout is reported only when the fit can see it: sigma under this many ms and under this fraction of the value.
# A pan at 2 deg/frame measures to a few hundredths of a ms; a locked-off take comes back with sigma in the tens of
# ms or infinite (tests/test_grid_rs.py).
MAX_SIGMA_MS = 2.0
MAX_SIGMA_FRAC = 0.1
DIST_CONVENTION = "OpenCV / Sensor Lens Maps brown [k1, k2, p1, p2, k3], pixels at plate size"


def default_out_dir() -> Path:
    return pl.WORK_ROOT / "grid"


def chart_ref(path, first: int = 0, last: int | None = None) -> dict:
    """ATLAS_PLATE_SEQ for a chart take. The colour space is 'Raw' on purpose: only luma code values are used, so the take
    needn't carry a recording-mode tag or a known body. Range still comes from the stream."""
    return pl.plate_ref(path, first, last, input_space="Raw")


def stretch8(y16: np.ndarray) -> np.ndarray:
    """uint16 luma -> uint8, 0.5-99.5 % stretch per frame (sampled every 8th pixel)."""
    y = y16.astype(np.float32)
    lo, hi = np.percentile(y[::8, ::8], 0.5), np.percentile(y[::8, ::8], 99.5)
    return (np.clip((y - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8)


def detections(ref: dict, first: int, last: int, step: int = 1, progress=None) -> dict:
    """{plate frame index: grid_rs.detect() or None} for first..last (inclusive) every step frames."""
    out = {}
    if first > last:
        return out
    for f, y16 in pl.decode_luma16(ref, first, last, step=step):
        out[f] = grid_rs.detect(stretch8(y16))
        if progress:
            progress(f, out[f])
    return out


def _preset(name: str) -> dict:
    return sensors.get_preset(name or DEFAULT_PRESET)


def _aspect_warning(ref: dict, preset: dict, name: str) -> str | None:
    pw, ph = preset["resolution"]
    if abs(ref["width"] / ref["height"] - pw / ph) > 0.01 * pw / ph:
        return (f"take is {ref['width']}x{ref['height']}, preset {name} is {pw}x{ph}: a different aspect means a "
                f"different sensor area, so focal_mm is only as good as the preset's width_mm")
    return None


def plate_info(ref: dict) -> dict:
    fps = ref["fps"][0] / ref["fps"][1]
    return {"width": ref["width"], "height": ref["height"], "fps": fps, "frames": ref["frame_count"],
            "range": ref["range"], "range_source": ref["range_source"]}


def calibrate_lens(ref: dict, *, preset: str = DEFAULT_PRESET, first: int | None = None, last: int | None = None,
                   every: int = 12, still_px: float = 1.0, focal_guess_mm: float = 16.47, progress=None) -> dict:
    """Take A: still views -> K, k1 k2, focal (mm on the preset sensor), principal point. ValueError if fewer than
    MIN_STILL_VIEWS still views of the chart are found."""
    f0 = ref["first"] if first is None else int(first)
    f1 = ref["last"] if last is None else int(last)
    every = max(1, int(every))
    p = _preset(preset)
    # two decode passes at `every`: the sampled frames, then the frame after each
    dets = detections(ref, f0, f1, every, progress)
    dets.update(detections(ref, f0 + 1, f1, every, progress))
    starts = list(range(f0, f1, every))                 # s + 1 must exist
    views, kept = [], []
    for s in starts:
        d0, d1 = dets.get(s), dets.get(s + 1)
        if d0 is None or d1 is None or d0["shape"] != d1["shape"]:
            continue
        move = float(np.median(np.linalg.norm(d0["xy"] - d1["xy"], axis=1)))
        if move < still_px:
            views.append(d0)
            kept.append((s, round(move, 2), len(d0["xy"])))
    log.info("atlas-vfx chart lens: %d still views of %d sampled", len(views), len(starts))
    if len(views) < MIN_STILL_VIEWS:
        raise ValueError(f"chart lens: {len(views)} still views of the chart in frames {f0}-{f1} (every {every}); "
                         f"need >= {MIN_STILL_VIEWS} (hold the chart still for 2+ frames, or lower every)")
    W, H = ref["width"], ref["height"]
    cal = grid_rs.calibrate(views, (W, H), focal_guess_px=focal_guess_mm / p["width_mm"] * W)
    K = cal["K"]
    warnings = [w for w in [_aspect_warning(ref, p, preset)] if w]
    return {"K": K, "dist_k1_k2_p1_p2_k3": cal["dist"], "rms_px": cal["rms_px"], "views": kept,
            "n_views": len(views), "n_sampled": len(starts),
            "focal_px": float(K[0, 0]), "focal_mm": float(K[0, 0] * p["width_mm"] / W),
            "k1": float(cal["dist"][0]), "k2": float(cal["dist"][1]),
            "principal_point_px": [float(K[0, 2]), float(K[1, 2])],
            "principal_point_offset_px": [float(K[0, 2] - W / 2), float(K[1, 2] - H / 2)],
            "plate": plate_info(ref), "sensor_preset": preset, "sensor_width_mm": p["width_mm"],
            "frames": [f0, f1], "every": every, "still_px": still_px, "focal_guess_mm": focal_guess_mm,
            "distortion_convention": DIST_CONVENTION, "warnings": warnings}


def read_lens(path) -> dict:
    """A lens JSON from calibrate_lens, read as data (json only) and checked: K 3x3 and 5 distortion numbers, finite."""
    s = pl.clean_path(path)
    if not s:
        raise ValueError("chart readout needs the lens JSON from the lens step (take A)")
    p = Path(s)
    if not p.is_file():
        raise ValueError(f"lens JSON not found: {p}")
    try:
        lens = json.loads(p.read_text(encoding="utf-8"))
        K = np.array(lens["K"], float)
        dist = np.array(lens["dist_k1_k2_p1_p2_k3"], float)
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"lens JSON {p.name}: not a chart lens calibration ({e})") from None
    if K.shape != (3, 3) or dist.shape != (5,) or not (np.all(np.isfinite(K)) and np.all(np.isfinite(dist))):
        raise ValueError(f"lens JSON {p.name}: K must be 3x3 and the distortion 5 finite numbers")
    if not K[0, 0] > 0 or not K[1, 1] > 0:
        raise ValueError(f"lens JSON {p.name}: focal must be > 0")
    lens["K"], lens["dist_k1_k2_p1_p2_k3"], lens["_path"] = K, dist, str(p)
    return lens


def measure_readout(ref: dict, lens: dict, *, first: int | None = None, last: int | None = None,
                    progress=None) -> dict:
    """Takes B-E: per-frame poses + one readout against the take-A lens. 'measurable' False (and readout_ms None)
    when the fit can't see the readout, e.g. a locked-off take. ValueError if the chart isn't found in 3 consecutive
    frames, or the lens was calibrated at another plate size."""
    f0 = ref["first"] if first is None else int(first)
    f1 = ref["last"] if last is None else int(last)
    K, dist = np.asarray(lens["K"], float), np.asarray(lens["dist_k1_k2_p1_p2_k3"], float)
    lp = lens.get("plate") or {}
    if lp.get("width") and (lp.get("width"), lp.get("height")) != (ref["width"], ref["height"]):
        raise ValueError(f"lens JSON is for {lp['width']}x{lp['height']}, this take is {ref['width']}x"
                         f"{ref['height']}: calibrate the lens on a take in the same recording mode")
    info = plate_info(ref)
    dets = detections(ref, f0, f1, 1, progress)
    found = [f for f, d in dets.items() if d is not None]
    full = [f for f in found if dets[f]["shape"] == (grid_rs.CHART["cols"] - 1, grid_rs.CHART["rows"] - 1)]
    log.info("atlas-vfx chart readout: chart in %d of %d frames (%d whole)", len(found), len(dets), len(full))
    obs = grid_rs.index_sequence(dets, K, dist)
    rs = grid_rs.fit_readout(obs, K, ref["height"], fps=info["fps"])
    gs = grid_rs.fit_readout(obs, K, ref["height"], solve_readout=False, readout_start=0.0, fps=info["fps"])
    ms, sig = float(rs["readout_ms"]), float(rs["sigma_ms"])
    measurable = math.isfinite(sig) and sig <= MAX_SIGMA_MS and sig <= MAX_SIGMA_FRAC * max(ms, 1e-9)
    why = None
    if not measurable:
        why = (f"not measurable: sigma {sig:.2f} ms on a fit of {ms:.2f} ms (needs sigma <= {MAX_SIGMA_MS} ms and "
               f"<= {MAX_SIGMA_FRAC:.0%} of the value); the camera turned {rs['rot_deg_per_frame_median']:.3f} "
               f"deg/frame. A locked-off or slow take can't show the readout: pan or tilt across the chart.")
    return {"measurable": measurable, "readout_ms": ms if measurable else None, "sigma_ms": sig,
            "readout_ms_fit": ms, "not_measurable_reason": why,
            "rms_px_rs": rs["rms_px"], "rms_px_gs": gs["rms_px"], "median_px_rs": rs["median_px"],
            "rot_deg_per_frame_median": rs["rot_deg_per_frame_median"], "n_frames": rs["n_frames"],
            "n_corners": rs["n_obs"], "segments": rs["segments"], "frames_detected": len(found),
            "frames_full_chart": len(full), "frames": [f0, f1], "lens": lens.get("_path"),
            "lens_focal_mm": lens.get("focal_mm"), "lens_k1_k2": [float(dist[0]), float(dist[1])],
            "fit": rs["message"], "plate": info}


def json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def finite(o):
    """inf / nan -> None so the JSON stays standard (json.dumps writes Infinity otherwise)."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return finite(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def write_result(out_dir: Path, name: str, step: str, ref: dict, args: dict, result: dict, *, node: str) -> Path:
    """<out_dir>/<name>_<step>.json (the result plus an _inputs block, as the tool always wrote) and
    <name>_<step>.sidecar.json (pl.write_sidecar). name must already be pl.safe_name'd."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = finite(dict(result))
    doc["_inputs"] = {"plate": ref["path"], "plate_quick_sha256": ref["quick_sha256"], "args": args,
                      "recording_mode": pl.fuji_recording_mode(ref["path"]), "range": ref["range"],
                      "range_source": ref["range_source"], "by": node, "chart": grid_rs.CHART}
    path = out_dir / f"{name}_{step}.json"
    path.write_text(json.dumps(doc, indent=1, default=json_default, allow_nan=False), encoding="utf-8")
    keys = ("focal_mm", "k1", "k2", "principal_point_offset_px", "rms_px", "n_views") if step == "lens" else \
        ("measurable", "readout_ms", "sigma_ms", "rms_px_rs", "rms_px_gs", "rot_deg_per_frame_median", "n_frames")
    pl.write_sidecar(out_dir / f"{name}_{step}.sidecar.json", node,
                     inputs={"plate_ref": ref, "step": step, **finite(args)},
                     outputs={"path": str(path), **{k: doc.get(k) for k in keys}})
    return path
