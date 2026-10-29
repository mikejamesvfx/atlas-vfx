"""Sensor pass jobs: what the nodes (and the demo script) run. Plate side writes maps; render side applies them.

Plate side (tracked in ComfyUI): Plate Read -> Camera Import / solve -> Plate Depth give the ATLAS_TRACKED_CAMERA and the depth
folder; the readout comes from the sensor preset by the plan's rule (decide_readout: chart > RS-aware solve > spec;
Sensor Readout Solve is a deprecated diagnostic); Sensor Lens Maps takes the lens from the camera (typed values
override). Every value's origin goes in the sidecar.
Render side: Sensor Apply puts the maps on an LTX render (frames and crop windows from ATLAS_VFX_LTX_PREP), or takes them off
the plate on the way into LTX Prep.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import plate as pl
from atlas_vfx.camera_side import plate_depth as pdp
from atlas_vfx.camera_side import sensors
from atlas_vfx.contracts import shot_camera as sc

from . import apply as ap
from . import lens as ln
from . import maps_io as mio
from . import rs

DISPLAY_VIEW = "display view (ACES 2.0 SDR, as LTX Prep)"
WARP_ONLY = "warp only (no colour change, no vignette)"


# ------------------------------------------------------------------------------- plate side

def depth_frames(folder) -> dict[int, Path]:
    d = mio.frames_in(folder, "depth")
    if not d:
        raise FileNotFoundError(f"no depth.<frame>.exr in {folder}")
    return d


PRESET_RULE = "preset rule"
READOUT_SOURCES = [PRESET_RULE, "measured if connected", "typed"]
GS_CAMERA_WARNING = ("camera has no RS-aware solve (extensions.atlas_vfx_rs_solve): the full readout through a "
                     "GS-solved camera lost 4 % on the deck test (2026-10-20); prefer an RS Solve camera")


def resolve_readout(typed_ms: float, measured_ms: float | None, source: str,
                    chart_ms: float | None = None) -> tuple[float, str]:
    """The two original modes. 'measured if connected': a chart measurement if connected, else Sensor Readout
    Solve's value if connected, else the typed one."""
    if source != "typed" and chart_ms and chart_ms > 0:
        return float(chart_ms), "measured (chart)"
    if source == "typed" or measured_ms is None or measured_ms <= 0:
        why = "typed" if source == "typed" else "typed (no measured readout connected)"
        return float(typed_ms), why
    return float(measured_ms), "measured (Sensor Readout Solve)"


def decide_readout(cam_data: dict, typed_ms: float, source: str, *, measured_ms: float | None = None,
                   chart_ms: float | None = None) -> dict:
    """Which readout Sensor RS Maps uses, and the record of why. 'preset rule' (plan, Phase 1b): chart measurement
    > this camera's RS-aware solve > the sensor preset (generic spec), never a Readout Solve residual from a
    global-shutter camera; the typed value only when none of those exist. The other modes are resolve_readout."""
    if source != PRESET_RULE:
        ms, why = resolve_readout(typed_ms, measured_ms, source, chart_ms)
        return {"mode": source, "readout_ms": ms, "readout_from": why}
    name = sensors.preset_for_camera(cam_data)
    cands = sensors.readout_candidates(name, chart_ms=chart_ms, camera=cam_data, gs_residual_ms=measured_ms)
    won = sensors.pick_readout(cands)
    d = {"mode": source, "preset": name, "candidates": cands}
    if won:
        d.update(readout_ms=won["value"], readout_from=f"{won['status']} ({won['kind']}): {won['source']}",
                 winner=won, passed_over=won["passed_over"], rejected=won["rejected"])
    else:
        d.update(readout_ms=float(typed_ms), readout_from="typed (no preset or evidence for this camera)",
                 rejected=[{**c, "why": sensors.NEVER[c["kind"]]} for c in cands if c["kind"] in sensors.NEVER])
    if "atlas_vfx_rs_solve" not in (cam_data.get("extensions") or {}):
        d["warning"] = GS_CAMERA_WARNING
    return d


def make_rs_maps(cam: sc.ShotCamera, depth_folder, out_dir, readout_ms: float, *, readout_from: str = "typed",
                 frames=None, scale: float = 1.0, shutter_angle: float = 0.0, blur_samples: int = 5,
                 direction: str = "both", iters: int = 3, lens: ln.BrownConrady | None = None,
                 lens_from: str = "camera", far_m: float | None = None, roundtrip_every: int = 12,
                 readout_decision: dict | None = None, progress=None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    depth = depth_frames(depth_folder)
    frames = [f for f in (frames or cam.frames) if f in depth]
    if not frames:
        raise ValueError(f"no frames of {cam.name} have depth in {depth_folder}")
    W = int(round(cam.data["plate"]["width"] * scale))
    H = int(round(cam.data["plate"]["height"] * scale))
    fps = float(cam.data["shot"]["fps"])
    lens = lens or ln.BrownConrady.from_camera(cam)
    want_rs = direction in ("both", "apply RS")
    want_de = direction in ("both", "remove RS")
    blur = blur_samples if shutter_angle > 0 and want_rs else 0
    exp_fr = shutter_angle / 360.0
    per = []
    for k, f in enumerate(frames):
        z, valid, _ = pdp.read_exr(depth[f])
        m_rs, m_de = rs.rs_maps(cam, f, z, readout_ms / 1000.0, fps, iters=iters, width=W, height=H, lens=lens,
                                valid=valid, far_m=far_m, want_ders=want_de)
        attrs = {"frame": f, "camera": cam.name, "readout_ms": readout_ms, "fps": fps}
        if want_rs:
            mio.write_stmap(out / f"rs.{f:04d}.exr", m_rs, {**attrs, "map": "rs (apply rolling shutter)"})
        if want_de:
            mio.write_stmap(out / f"ders.{f:04d}.exr", m_de, {**attrs, "map": "ders (remove rolling shutter)"})
        for b in range(blur):
            off = ((b + 0.5) / blur - 0.5) * exp_fr
            mb, _ = rs.rs_maps(cam, f, z, readout_ms / 1000.0, fps, width=W, height=H, lens=lens, valid=valid,
                               far_m=far_m, time_offset=off, want_ders=False)
            mio.write_stmap(out / f"rs_blur{b}.{f:04d}.exr", mb, {**attrs, "map": f"rs blur sample {b}",
                                                                   "time_offset_frames": off})
        row = {"frame": f, **rs.skew_px_per_row(m_rs)}
        for key in ("dx_per_row", "dy_per_row", "dx_mean_px", "dy_mean_px"):
            row[key] /= 1.0 if key.endswith("row") else scale             # px at plate size
        if want_de and (k % max(1, roundtrip_every) == 0 or k == len(frames) - 1):
            row["roundtrip"] = {kk: (v / scale if kk.endswith("px") and v is not None else v)
                                for kk, v in rs.roundtrip(m_rs, m_de).items()}
        per.append(row)
        if progress:
            progress(k + 1, len(frames))
    rts = [r["roundtrip"]["median_px"] for r in per if r.get("roundtrip") and r["roundtrip"]["median_px"] is not None]
    report = {
        "schema": "atlas_vfx.sensor_rs/1", "camera": cam.name, "frames": [frames[0], frames[-1]], "count": len(frames),
        "map_size": [W, H], "plate_size": [cam.data["plate"]["width"], cam.data["plate"]["height"]],
        "readout_ms": readout_ms, "readout_from": readout_from, "readout_frames": readout_ms / 1000.0 * fps,
        "readout_decision": readout_decision,
        "fps": fps, "row_time": "t(y) = t_frame + (y/H - 0.5) * T_readout, y = sensor (distorted) row",
        "lens": lens.to_dict(), "lens_from": lens_from, "depth_folder": str(depth_folder), "far_m": far_m,
        "iters": iters, "direction": direction, "shutter_angle": shutter_angle, "blur_samples": blur,
        "maps": {"rs": "rs.%04d.exr" if want_rs else None, "ders": "ders.%04d.exr" if want_de else None,
                 "blur": [f"rs_blur{b}.%04d.exr" for b in range(blur)]},
        "colorspace": "Raw (STMap, Nuke s/t)",
        "skew_px_per_row_median": float(np.median([r["dx_per_row"] for r in per])),
        "skew_px_per_row_max_abs": float(np.max(np.abs([r["dx_per_row"] for r in per]))),
        "roundtrip_median_px": float(np.median(rts)) if rts else None,
        "roundtrip_worst_median_px": float(np.max(rts)) if rts else None,
        "seconds": round(time.time() - t0, 1), "per_frame": per,
    }
    (out / "sensor_rs.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "sensor_rs.nk").write_text(mio.nuke_rs(out, frames[0], frames[-1], blur), encoding="utf-8")
    pl.write_sidecar(out / "sensor_rs.sidecar.json", "AtlasVFXSensorRSMaps",
                     inputs={"camera": cam.name, "camera_source": cam.data.get("source"),
                             "depth_folder": str(depth_folder), "readout_ms": readout_ms,
                             "readout_from": readout_from,
                             "readout_mode": (readout_decision or {}).get("mode"),
                             "readout_winner": (readout_decision or {}).get("winner"),
                             "lens_from": lens_from, "scale": scale},
                     outputs={k: v for k, v in report.items() if k != "per_frame"})
    return report


def lens_from_inputs(cam: sc.ShotCamera | None, source: str, typed: dict, plate_size=None) -> tuple[ln.BrownConrady, str]:
    """source 'camera' (camera lens block; typed values ignored), 'typed' (typed k's on the camera's intrinsics),
    'camera + typed' (typed values replace the camera's where non-zero)."""
    names = ("k1", "k2", "k3", "p1", "p2")
    if cam is not None:
        base = ln.BrownConrady.from_camera(cam)
    else:
        if not plate_size:
            raise ValueError("Sensor Lens: no camera, so plate size and focal are needed")
        W, H = plate_size
        f = typed.get("focal_px") or W
        base = ln.BrownConrady(f, f, W / 2.0, H / 2.0)
    if source == "camera" and cam is not None:
        model = cam.data.get("lens", {}).get("distortion_model", "none")
        return base, f"camera ({cam.name}, {model})"
    if source == "typed":
        return ln.BrownConrady(base.fx, base.fy, base.cx, base.cy, **{n: float(typed.get(n, 0)) for n in names}), "typed"
    over = {n: float(typed[n]) for n in names if float(typed.get(n, 0)) != 0}
    from dataclasses import replace
    return replace(base, **over), f"camera + typed {sorted(over)}"


def make_lens_maps(lens: ln.BrownConrady, plate_size, out_dir, *, ca_r=0.0, ca_b=0.0, v1=0.0, v2=0.0, v3=0.0,
                   scale: float = 1.0, lens_from: str = "camera", ca_from: str = "typed",
                   vignette_from: str = "typed") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, H = [int(round(v * scale)) for v in plate_size]
    L = lens.scaled(scale) if scale != 1 else lens
    maps = ln.lens_maps(L, W, H, ca_r, ca_b)
    for kind in ("undistort", "redistort"):
        for c in ln.CHANNELS:
            mio.write_stmap(out / f"{kind}_{c.lower()}.exr", maps[kind][c], {"map": f"{kind} {c}"})
    g = ln.vignette_gain(W, H, v1, v2, v3, L.cx, L.cy)
    mio.write_exr(out / "vignette.exr", {"R": g, "G": g, "B": g},
                  {"map": "vignette gain: multiply linear ACEScg to apply, divide to remove"})
    # the inverse check: undistort(redistort(p)) per channel
    X, Y = ln.pixel_grid(W, H)
    err = {}
    for c in ln.CHANNELS:
        px, py = ln.from_st(maps["redistort"][c], W, H)
        st = rs.sample(maps["undistort"][c].astype(np.float64), px, py, W, H)
        qx, qy = ln.from_st(st, W, H)
        inside = (px > 2) & (px < W - 2) & (py > 2) & (py < H - 2)
        e = np.hypot(qx - X, qy - Y)[inside]
        err[c] = float(np.median(e)) / scale if e.size else None
    report = {"schema": "atlas_vfx.sensor_lens/1", "lens": lens.to_dict(), "lens_from": lens_from,
              "ca": {"R": ca_r, "B": ca_b, "G": 0.0}, "ca_from": ca_from,
              "vignette": {"v1": v1, "v2": v2, "v3": v3, "r": "1 at the corners"}, "vignette_from": vignette_from,
              "plate_size": list(plate_size), "map_size": [W, H], "colorspace": "Raw (STMap, Nuke s/t; gain)",
              "roundtrip_median_px": err,
              "files": sorted(p.name for p in out.glob("*.exr"))}
    (out / "sensor_lens.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "sensor_lens.nk").write_text(mio.nuke_lens(out), encoding="utf-8")
    pl.write_sidecar(out / "sensor_lens.sidecar.json", "AtlasVFXSensorLensMaps",
                     inputs={"lens_from": lens_from, "scale": scale}, outputs=report)
    return report


# ------------------------------------------------------------------------------- render side

class Maps:
    """Lazy access to an RS folder and/or a lens folder, cut to an image's plate window."""

    def __init__(self, rs_folder: str | None = None, lens_folder: str | None = None):
        self.rs_folder = Path(rs_folder) if rs_folder else None
        self.lens_folder = Path(lens_folder) if lens_folder else None
        self.rs_report = json.loads((self.rs_folder / "sensor_rs.json").read_text()) if self.rs_folder else None
        self.lens_report = json.loads((self.lens_folder / "sensor_lens.json").read_text()) if self.lens_folder else None
        rep = self.rs_report or self.lens_report
        if rep is None:
            raise ValueError("Sensor Apply: give an RS maps folder, a lens maps folder, or both")
        self.plate_size = tuple(rep["plate_size"])
        self._lens = {}

    def _map_window(self, st, win):
        """Map at map size -> plate size if needed -> window."""
        W, H = self.plate_size
        if win is None:
            return st
        st = ap.resize_map(st, W, H)
        return ap.window_map(st, win, (W, H))

    def rs(self, frame: int, kind: str, win=None) -> list:
        if not self.rs_folder:
            return []
        if kind == "rs" and self.rs_report["maps"]["blur"]:
            names = [f"rs_blur{b}" for b in range(len(self.rs_report["maps"]["blur"]))]
        else:
            names = [kind]
        out = []
        for n in names:
            p = self.rs_folder / f"{n}.{frame:04d}.exr"
            if not p.is_file():
                raise FileNotFoundError(f"Sensor Apply: no {p.name} in {self.rs_folder} (maps cover "
                                        f"{self.rs_report['frames']})")
            out.append(self._map_window(mio.read_stmap(p), win))
        return out

    def lens(self, kind: str, win=None) -> dict:
        if not self.lens_folder:
            return {}
        if kind not in self._lens:
            self._lens[kind] = {c: mio.read_stmap(self.lens_folder / f"{kind}_{c.lower()}.exr") for c in ln.CHANNELS}
        return {c: self._map_window(m, win) for c, m in self._lens[kind].items()}

    def gain(self, win=None):
        if not self.lens_folder:
            return None
        v = self.lens_report["vignette"]
        if not any(v[k] for k in ("v1", "v2", "v3")):
            return None
        g = mio.read_exr(self.lens_folder / "vignette.exr")[0][..., 0]
        if win is None:
            return g
        import cv2
        W, H = self.plate_size
        return ap.window_gain(cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR), win)


def parse_window(text: str):
    s = (text or "").strip()
    if not s:
        return None
    v = [int(round(float(x))) for x in s.replace(";", ",").split(",")]
    if len(v) != 4:
        raise ValueError(f"window must be x0,y0,w,h in plate pixels, got {text!r}")
    return v


def colour_procs(colorspace: str, ocio_config: str):
    """(to ACEScg, from ACEScg) CPU processors, or (None, None) for ACEScg / warp only."""
    if colorspace in ("ACEScg", WARP_ONLY):
        return None, None
    import PyOpenColorIO as ocio
    cfg = pl.ocio_config(ocio_config)
    if colorspace == DISPLAY_VIEW:
        fwd = cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW))
        inv = cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScg", display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW,
                                                         direction=ocio.TRANSFORM_DIR_INVERSE))
        return inv.getDefaultCPUProcessor(), fwd.getDefaultCPUProcessor()
    return (cfg.getProcessor(colorspace, "ACEScg").getDefaultCPUProcessor(),
            cfg.getProcessor("ACEScg", colorspace).getDefaultCPUProcessor())


def _apply_proc(proc, img):
    if proc is None:
        return img
    out = np.ascontiguousarray(img[..., :3], np.float32).copy()
    proc.applyRGB(out)
    if img.shape[2] > 3:
        out = np.concatenate([out, img[..., 3:]], -1)
    return out


def apply_batch(images: np.ndarray, maps: Maps, *, direction: str = "apply camera", first_frame: int,
                windows=None, colorspace: str = WARP_ONLY, ocio_config: str | None = None,
                use_vignette: bool = True, use_lens: bool = True, use_rs: bool = True, edge: str = "edge",
                progress=None) -> tuple[np.ndarray, dict]:
    """images (N, h, w, C) -> same shape. Image i is Nuke frame first_frame + i showing plate window windows[i]
    ([x0, y0, w, h] plate px, or None = the whole plate), resized to the image size."""
    to_lin, from_lin = colour_procs(colorspace, ocio_config or str(pl.DEFAULT_OCIO))
    if colorspace == WARP_ONLY:
        use_vignette = False
    out = np.empty_like(images, dtype=np.float32)
    applied = []
    for i in range(len(images)):
        n = first_frame + i
        win = windows[i] if windows else None
        img = _apply_proc(to_lin, images[i].astype(np.float32))
        gain = maps.gain(win) if use_vignette else None
        if direction == "apply camera":
            r = ap.apply(img, gain=gain, lens=maps.lens("redistort", win) if use_lens else None,
                         rs=maps.rs(n, "rs", win) if use_rs else None, edge=edge)
        else:
            de = maps.rs(n, "ders", win) if use_rs else []
            r = ap.remove(img, gain=gain, lens=maps.lens("undistort", win) if use_lens else None,
                          ders=de[0] if de else None, edge=edge)
        r = _apply_proc(from_lin, r)
        out[i] = np.clip(r, 0.0, 1.0) if colorspace == DISPLAY_VIEW else r
        applied.append(n)
        if progress:
            progress(i + 1, len(images))
    report = {"direction": direction, "frames": [applied[0], applied[-1]] if applied else None,
              "count": len(applied), "windows": "per frame (ATLAS_VFX_LTX_PREP)" if windows else "whole plate",
              "colorspace": colorspace, "vignette": bool(use_vignette and maps.lens_folder is not None),
              "lens": bool(use_lens and maps.lens_folder is not None),
              "rs": bool(use_rs and maps.rs_folder is not None), "edge": edge,
              "rs_folder": str(maps.rs_folder) if maps.rs_folder else None,
              "lens_folder": str(maps.lens_folder) if maps.lens_folder else None,
              "order": ("vignette (linear) -> CA + distortion -> RS" if direction == "apply camera"
                        else "de-RS -> undistort + CA -> devignette (linear)")}
    return out, report


def frames_windows_from_prep(ltx_prep: dict, n_images: int):
    """(first Nuke frame, per-frame windows or None) from an ATLAS_VFX_LTX_PREP (LTX Prep's output)."""
    f0, f1 = ltx_prep["frames"]
    if n_images < f1 - f0 + 1:
        raise ValueError(f"LTX gave {n_images} frames for {f0}-{f1}")
    wins = ltx_prep.get("windows")
    if wins:
        return f0, [list(w) for w in wins[:n_images]]
    return f0, None
