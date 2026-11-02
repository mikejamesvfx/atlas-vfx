"""Clean space: the plate undistorted, possibly bigger than the plate (overscan, so barrel corners are kept).

A shot camera with a lens has a clean twin: the same poses and focal in pixels, no lens, on the clean frame (plate +
overscan each side, principal point moved by the overscan). Card Project renders pinhole layers through it, LTX
works at its size, and the Nuke Builder's clean branch uses its wider aperture. Sensor Lens Maps (sensor_lens.json:
plate_size, clean_size, overscan_px) says how big the clean frame is.

Pure python; reads only the lens folder's JSON. No ComfyUI, no sensor imports (camera_side moves to Atlas Camera).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from atlas_vfx.contracts import shot_camera as sc


def lens_report(lens_folder) -> dict:
    p = Path(lens_folder) / "sensor_lens.json"
    if not p.is_file():
        raise FileNotFoundError(f"no sensor_lens.json in {lens_folder} (a Sensor Lens Maps folder)")
    return json.loads(p.read_text(encoding="utf-8"))


def clean_size(lens_folder) -> tuple[int, int]:
    rep = lens_report(lens_folder)
    return tuple(rep.get("clean_size") or rep["plate_size"])


def clean_camera(cam: sc.ShotCamera, lens_folder) -> sc.ShotCamera:
    """The pinhole camera of the clean frame: plate size -> clean size, sensor widened with it (focal in pixels
    unchanged), principal point shifted by the overscan, lens none. Frames, poses and focal_mm are the same."""
    rep = lens_report(lens_folder)
    W, H = (int(v) for v in rep["plate_size"])
    Wc, Hc = (int(v) for v in rep.get("clean_size") or rep["plate_size"])
    if (cam.data["plate"]["width"], cam.data["plate"]["height"]) != (W, H):
        raise ValueError(f"lens maps are for a {W}x{H} plate, the camera's plate is "
                         f"{cam.data['plate']['width']}x{cam.data['plate']['height']}")
    k = cam.intrinsics(cam.frames[0])
    ox, oy = (Wc - W) / 2.0, (Hc - H) / 2.0
    d = copy.deepcopy(cam.data)
    d["plate"]["width"], d["plate"]["height"] = Wc, Hc
    d["sensor"]["width_mm"] = cam.data["sensor"]["width_mm"] * Wc / W
    d["sensor"]["height_mm"] = cam.data["sensor"]["height_mm"] * Hc / H
    d["sensor"].pop("preset", None)                    # a widened sensor is no longer the preset's
    d["lens"] = {"distortion_model": "none", "distortion_params": []}
    if abs(k.cx - W / 2) > 1e-9 or abs(k.cy - H / 2) > 1e-9 or ox or oy:
        d["lens"]["principal_point_px"] = [k.cx + ox, k.cy + oy]
    d.setdefault("extensions", {})["atlas_vfx_clean_space"] = {
        "of": cam.name, "plate_size": [W, H], "clean_size": [Wc, Hc], "overscan_px": [ox, oy],
        "lens": cam.data.get("lens"), "lens_folder": str(lens_folder)}
    sc.validate(d)
    return sc.ShotCamera(d)
