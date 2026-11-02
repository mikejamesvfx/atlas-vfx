"""Checks shared by every node that takes a shot camera: the camera must be in the units, frame numbering and lens
space the node assumes, or the node says so. A solved camera (Solve Plate) can be unscaled, nodal or carry a lens;
before the tracker every camera was an import, and nodes took metres, a matching frame offset and a pinhole for
granted.

Each check returns the message (or None) and logs it; strict=True raises instead. Pure python, no ComfyUI.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("atlas_vfx")


class CameraCheckError(ValueError):
    pass


def _data(cam) -> dict:
    return cam.data if hasattr(cam, "data") else cam


def _emit(msg: str | None, who: str, strict: bool) -> str | None:
    if msg is None:
        return None
    msg = f"{who}: {msg}"
    if strict:
        raise CameraCheckError(msg)
    log.warning("atlas-vfx %s", msg)
    return msg


def is_nodal(cam) -> bool:
    d = _data(cam)
    ext = (d.get("extensions") or {}).get("atlas_vfx_solve_plate") or {}
    if (ext.get("degeneracy") or {}).get("nodal"):
        return True
    C = np.array([np.asarray(f["cam_to_world"], float)[:3, 3] for f in d["frames"]])
    return len(C) > 1 and float(np.ptp(C, axis=0).max()) < 1e-9


def has_lens(cam) -> bool:
    lens = _data(cam).get("lens") or {}
    return lens.get("distortion_model", "none") != "none" and any(abs(float(v)) > 0
                                                                   for v in lens.get("distortion_params") or [])


def require_metric(cam, who: str, what: str = "metre widgets", strict: bool = False) -> str | None:
    """The camera's world is in metres (Measure & Scale, an import with a known scale), not a solve's own units."""
    d = _data(cam)
    method = (d.get("scale") or {}).get("method", "unscaled")
    if is_nodal(cam):
        return _emit(f"{d['shot']['name']} is a nodal solve (camera turning about one point, no depth), so its "
                     f"{what} mean nothing; place things in its rotation only, or solve a shot with parallax", who,
                     strict)
    if method == "unscaled":
        return _emit(f"{d['shot']['name']} is unscaled (scale.method unscaled: solve units, not metres), so its "
                     f"{what} are in solve units; run Measure & Scale first", who, strict)
    return None


def check_frame_offset(cam, plate_ref: dict, who: str, strict: bool = True) -> str | None:
    """The camera numbers frames the way the plate does (shot frame = plate index + frame_offset)."""
    d = _data(cam)
    co = d["shot"].get("plate_frame_offset")
    po = plate_ref.get("frame_offset")
    if co is None or po is None or int(co) == int(po):
        return None
    return _emit(f"camera {d['shot']['name']} numbers plate frames with offset {co}, the plate with {po}: every "
                 f"frame would be {int(po) - int(co):+d} off. Re-import the camera with the plate's offset", who,
                 strict)


def lens_needs_clean(cam, has_lens_folder: bool, who: str, allow_raw: bool = False) -> str | None:
    """A node that comps pinhole layers onto the plate needs the plate in clean (undistorted) space when the camera
    has a lens: Sensor Lens Maps of that camera, or allow_raw to accept the misregistration knowingly."""
    if not has_lens(cam) or has_lens_folder:
        return None
    lens = _data(cam)["lens"]
    msg = (f"camera {_data(cam)['shot']['name']} has a {lens['distortion_model']} lens "
           f"{[round(float(v), 5) for v in lens['distortion_params']]} but no lens folder is connected, so pinhole "
           f"layers would sit on the distorted plate. Connect Sensor Lens Maps of this camera (lens_source camera)")
    return _emit(msg + (" (allowed: allow_raw)" if allow_raw else ""), who, strict=not allow_raw)
