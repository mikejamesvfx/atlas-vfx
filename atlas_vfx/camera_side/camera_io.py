"""Camera import: Nuke Camera JSON (from the export snippet), Nuke .chan, COLMAP/pycolmap models -> shot-camera@0.1.

Axes: the contract is Nuke/USD (right-handed, Y up, camera looks down -Z). COLMAP cameras are x right, y down,
z forward, so R_c2w_nuke = R_c2w_colmap @ diag(1, -1, -1); the centre is unchanged. World axes are whatever the
source had (a raw solve is not gravity aligned; Measure & Scale does that in T4).

Sensor size always comes from a preset or the caller, never from a Nuke camera: Nuke's vaperture is usually its
default 15.7. If the Nuke haperture differs from the real sensor width, focal is rescaled so the horizontal field of
view Nuke solved is kept.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

NUKE_TO_FROM_COLMAP = np.diag([1.0, -1.0, -1.0])
HAPERTURE_TOL = 0.005   # relative


class CameraImportError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- Euler, any order

def _axis(a: str, deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    if a == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if a == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_matrix(order: str, rx: float, ry: float, rz: float) -> np.ndarray:
    """Nuke rot_order semantics: the first letter is applied first, so 'ZXY' -> R = Ry @ Rx @ Rz."""
    order = order.upper()
    if sorted(order) != ["X", "Y", "Z"]:
        raise CameraImportError(f"bad rot_order {order!r}")
    ang = {"X": rx, "Y": ry, "Z": rz}
    R = np.eye(3)
    for a in order:
        R = _axis(a, ang[a]) @ R
    return R


def _c2w(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# --------------------------------------------------------------------------------------------- plate block

def plate_block(plate_ref: dict | None, width: int | None = None, height: int | None = None) -> dict:
    """shot-camera 'plate' block from a ATLAS_PLATE_SEQ (atlas_vfx.plate_ref), or a bare one from a resolution."""
    if plate_ref:
        b = {"path": plate_ref["path"], "width": plate_ref["width"], "height": plate_ref["height"],
             "colorspace": plate_ref["colorspace"], "ocio_config": plate_ref["ocio_config"],
             "range": plate_ref["range"]}
        for k in ("timecode", "camera_body"):
            if plate_ref.get(k):
                b[k] = plate_ref[k]
        return b
    if not (width and height):
        raise CameraImportError("need a plate_ref or width/height")
    return {"path": "", "width": int(width), "height": int(height), "colorspace": "unknown", "range": "full"}


def _frame_filter(frames, keep):
    if keep is None:
        return frames
    lo, hi = keep
    return [f for f in frames if lo <= f[0] <= hi]


def _finish(rows, *, name, plate, sensor, source, scale, lens=None, frame_offset=None, extensions=None,
            fps=24.0) -> sc.ShotCamera:
    """rows: [(shot frame, cam_to_world 4x4, focal_mm)], any order. Checks contiguity with a clear message."""
    if not rows:
        raise CameraImportError("no frames to import (outside the plate range? offset_first_frame renumbers the "
                                "camera, e.g. 'to plate start', or switch crop_to_plate off)")
    rows = sorted(rows, key=lambda r: r[0])
    nums = [r[0] for r in rows]
    gaps = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    if gaps:
        show = ", ".join(map(str, gaps[:12])) + (" ..." if len(gaps) > 12 else "")
        raise CameraImportError(f"{len(gaps)} frames missing between {nums[0]} and {nums[-1]}: {show} "
                                f"(unregistered in the solve? import a range without gaps)")
    return sc.build(name=name, frame_start=nums[0], fps=fps, plate=plate, sensor=sensor,
                    cam_to_world=[r[1] for r in rows], focal_mm=[r[2] for r in rows], source=source,
                    scale=scale, lens=lens, plate_frame_offset=frame_offset, extensions=extensions)


def _scale_block(m_per_unit: float, method: str | None) -> dict:
    if method is None:
        method = "unscaled" if m_per_unit == 1.0 else "known_lengths"
    return {"method": method, "m_per_source_unit": float(m_per_unit)}


# --------------------------------------------------------------------------------------------- Nuke

def from_nuke_json(path, *, plate: dict, sensor: dict, m_per_unit: float = 1.0, scale_method: str | None = None,
                   name: str | None = None, frames: tuple[int, int] | None = None, frame_offset: int = 1001,
                   fps: float = 24.0) -> sc.ShotCamera:
    """Camera JSON written by the Nuke export snippet (howling-yowies nuke_export_for_claude.py)."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    rot_order = d.get("rot_order", "ZXY")
    xo = d.get("xform_order", "SRT")
    if xo not in ("SRT", "RST"):
        raise CameraImportError(f"xform_order {xo} not supported (translation must be applied last)")
    rows, notes = [], {"rot_order": rot_order, "xform_order": xo}
    hap_seen = set()
    for fr in _frame_filter([(r["frame"], r) for r in d["frames"]], frames):
        n, r = fr
        win = r.get("win_translate", [0, 0])
        if any(abs(v) > 1e-9 for v in win):
            raise CameraImportError(f"frame {n}: win_translate {win} (lens shift) not supported yet")
        hap = float(r.get("haperture", sensor["width_mm"]))
        hap_seen.add(round(hap, 6))
        focal = float(r["focal"])
        if abs(hap / sensor["width_mm"] - 1) > HAPERTURE_TOL:
            focal *= sensor["width_mm"] / hap      # keep Nuke's horizontal field of view
        R = euler_to_matrix(rot_order, *r["rotate"])
        rows.append((int(n), _c2w(R, np.asarray(r["translate"], float) * m_per_unit), focal))
    notes["nuke_haperture_mm"] = sorted(hap_seen)
    notes["nuke_vaperture_mm_ignored"] = sorted({round(float(r.get("vaperture", 0)), 6) for r in d["frames"]})
    if any(abs(h / sensor["width_mm"] - 1) > HAPERTURE_TOL for h in hap_seen):
        notes["focal_rescaled"] = f"focal * {sensor['width_mm']} / haperture (horizontal FOV kept)"
    return _finish(rows, name=name or d.get("name", p.stem), plate=plate, sensor=sensor, fps=fps,
                   source={"kind": "import", "solver": f"Nuke {d.get('class', 'Camera')} {d.get('name', '')}".strip(),
                           "path": str(p.resolve())},
                   scale=_scale_block(m_per_unit, scale_method), frame_offset=frame_offset,
                   extensions={"atlas_vfx_import": notes})


def read_chan(path) -> list[tuple]:
    rows = []
    for k, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split()
        if len(parts) not in (7, 8):
            raise CameraImportError(f"{path}:{k}: expected 7 or 8 columns (frame tx ty tz rx ry rz [focal]), "
                                    f"got {len(parts)}")
        rows.append((int(round(float(parts[0]))), *map(float, parts[1:])))
    return rows


def from_nuke_chan(path, *, plate: dict, sensor: dict, rot_order: str = "ZXY", focal_mm: float | None = None,
                   m_per_unit: float = 1.0, scale_method: str | None = None, name: str | None = None,
                   frames: tuple[int, int] | None = None, frame_offset: int = 1001,
                   fps: float = 24.0) -> sc.ShotCamera:
    """Nuke .chan: frame tx ty tz rx ry rz [focal]. Without a focal column, focal_mm is required. The chan carries
    no film back: focal is taken as relative to the sensor given."""
    p = Path(path)
    rows = []
    for r in _frame_filter(read_chan(p), frames):
        f = r[7] if len(r) == 8 else focal_mm
        if not f:
            raise CameraImportError(f"{p.name} has no focal column; pass focal_mm")
        rows.append((r[0], _c2w(euler_to_matrix(rot_order, *r[4:7]), np.array(r[1:4]) * m_per_unit), float(f)))
    return _finish(rows, name=name or p.stem, plate=plate, sensor=sensor, fps=fps,
                   source={"kind": "import", "solver": "Nuke .chan", "path": str(p.resolve())},
                   scale=_scale_block(m_per_unit, scale_method), frame_offset=frame_offset,
                   extensions={"atlas_vfx_import": {"rot_order": rot_order}})


# --------------------------------------------------------------------------------------------- Alembic

def from_abc(path, *, plate: dict, sensor: dict, camera: str | None = None, m_per_unit: float = 1.0,
             scale_method: str | None = None, name: str | None = None, frames: tuple[int, int] | None = None,
             frame_offset: int = 1001, fps: float = 24.0) -> sc.ShotCamera:
    """Alembic camera (Ogawa; Nuke, Maya, Houdini, Blender, 3DE exports). Frame = sample time x fps, so a camera
    written at frames 1001-1598 at 24 fps lands on 1001-1598. The world matrix is the camera's whole Xform chain
    (inherits honoured); scale in the chain is removed from the rotation and reported. Apertures are cm in Alembic:
    as for Nuke JSON, the sensor comes from the preset and focal is rescaled if the file's horizontal aperture
    differs, so the horizontal field of view is kept. Film offsets (lens shift) and squeeze are refused for now."""
    from . import abc_io

    p = Path(path)
    try:
        cam_obj, samples, notes = abc_io.camera_samples(p, camera, fps=fps)
    except abc_io.AbcError as e:
        raise CameraImportError(str(e)) from e
    fb = notes.get("film_back_channels") or {}
    if not fb.get("identity", True):
        raise CameraImportError(f"{p.name}: film back ops {fb.get('ops')} = {fb.get('channels')} "
                                f"(window translate/scale) not supported yet")
    if frames is not None and not notes["animated"] and samples:
        samples = [(f, samples[0][1]) for f in range(frames[0], frames[1] + 1)]     # static camera: hold
    rows, hap_seen, scale_dev = [], set(), 0.0
    for f, s in _frame_filter(samples, frames):
        c = s.core
        if abs(c["h_film_offset_cm"]) > 1e-9 or abs(c["v_film_offset_cm"]) > 1e-9:
            raise CameraImportError(f"frame {f}: film offset {c['h_film_offset_cm']}, {c['v_film_offset_cm']} cm "
                                    f"(lens shift) not supported yet")
        if abs(c["squeeze"] - 1.0) > 1e-6:
            raise CameraImportError(f"frame {f}: lens squeeze {c['squeeze']} (anamorphic) not supported yet")
        hap = c["h_aperture_cm"] * 10.0
        hap_seen.add(round(hap, 6))
        focal = c["focal_mm"]
        if abs(hap / sensor["width_mm"] - 1) > HAPERTURE_TOL:
            focal *= sensor["width_mm"] / hap
        M = np.asarray(s.cam_to_world, float)
        R = M[:3, :3]
        sx = np.linalg.norm(R, axis=0)
        scale_dev = max(scale_dev, float(np.abs(sx - 1).max()))
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            raise CameraImportError(f"frame {f}: the camera's world matrix is mirrored (negative scale in the chain)")
        rows.append((int(f), _c2w(R, M[:3, 3] * m_per_unit), float(focal)))
    if not rows:
        lo, hi = samples[0][0], samples[-1][0]
        raise CameraImportError(f"{p.name}: camera samples cover frames {lo}-{hi}, none in {frames} "
                                f"(offset_first_frame renumbers it, e.g. 'to plate start')")
    notes = {**notes, "abc_haperture_mm": sorted(hap_seen), "chain_scale_removed": round(scale_dev, 6)}
    if any(abs(h / sensor["width_mm"] - 1) > HAPERTURE_TOL for h in hap_seen):
        notes["focal_rescaled"] = f"focal * {sensor['width_mm']} / haperture (horizontal FOV kept)"
    return _finish(rows, name=name or p.stem, plate=plate, sensor=sensor, fps=fps,
                   source={"kind": "import", "solver": f"Alembic {cam_obj.path}", "path": str(p.resolve())},
                   scale=_scale_block(m_per_unit, scale_method), frame_offset=frame_offset,
                   extensions={"atlas_vfx_import": notes})


# --------------------------------------------------------------------------------------------- COLMAP

_SINGLE_F = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


def _largest_reconstruction(folder: Path):
    import pycolmap

    cands = []
    if (folder / "images.bin").is_file() or (folder / "images.txt").is_file():
        cands = [folder]
    else:
        for sub in (folder / "sparse", folder):
            if sub.is_dir():
                cands += [d for d in sorted(sub.iterdir())
                          if d.is_dir() and ((d / "images.bin").is_file() or (d / "images.txt").is_file())]
            if cands:
                break
    if not cands:
        raise CameraImportError(f"no COLMAP model under {folder}")
    recs = [(pycolmap.Reconstruction(str(c)), c) for c in cands]
    return max(recs, key=lambda rc: rc[0].num_reg_images())


def from_colmap(folder, *, plate: dict, sensor: dict, frame_offset: int = 1001, m_per_unit: float = 1.0,
                scale_method: str | None = None, name: str | None = None, frames: tuple[int, int] | None = None,
                frame_regex: str = r"(\d+)(?!.*\d)", fps: float = 24.0, solver: str | None = None) -> sc.ShotCamera:
    """pycolmap model (the folder, its sparse/, or sparse/N; the model with most registered images wins).
    Image names give the plate frame index (last number in the name); shot frame = index + frame_offset.
    The model may be at proxy resolution: intrinsics are scaled to the plate by width."""
    import pycolmap

    folder = Path(folder)
    rec, model_dir = _largest_reconstruction(folder)
    cams = {}
    for cid, cam in rec.cameras.items():
        model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
        s = plate["width"] / cam.width
        if abs(cam.height * s - plate["height"]) > 1.0:
            raise CameraImportError(f"COLMAP camera {cid} is {cam.width}x{cam.height}, not the plate's aspect "
                                    f"({plate['width']}x{plate['height']})")
        prm = list(cam.params)
        if model in _SINGLE_F:
            fx = fy = prm[0]
            cx, cy = prm[1], prm[2]
            extra = prm[3:]
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = prm[:4]
            extra = prm[4:]
        else:
            raise CameraImportError(f"COLMAP camera model {model} not supported")
        if abs(fx / fy - 1) > 1e-4:
            raise CameraImportError(f"COLMAP camera {cid}: fx {fx} != fy {fy} (non-square pixels not supported)")
        if model == "SIMPLE_RADIAL":
            dist = ("simple_radial", extra) if any(abs(k) > 0 for k in extra) else ("none", [])
        elif any(abs(k) > 0 for k in extra):
            raise CameraImportError(f"COLMAP {model} distortion {extra} not mapped to the contract yet")
        else:
            dist = ("none", [])
        cams[cid] = {"model": model, "focal_mm": fx * sensor["width_mm"] / cam.width,
                     "pp": [cx * s, cy * s], "dist": dist, "size": [cam.width, cam.height], "f_px": fx}
    if len({(round(c["pp"][0], 3), round(c["pp"][1], 3), c["dist"][0], tuple(c["dist"][1])) for c in cams.values()}) > 1:
        raise CameraImportError("several COLMAP cameras with different principal points/distortion: not supported")

    rx = re.compile(frame_regex)
    rows = []
    for img in rec.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        m = rx.search(img.name)
        if not m:
            raise CameraImportError(f"no frame number in image name {img.name!r}")
        n = int(m.group(1)) + frame_offset
        pose = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(pose.rotation.matrix(), float)
        t = np.asarray(pose.translation, float)
        R_c2w = R.T @ NUKE_TO_FROM_COLMAP
        centre = -R.T @ t
        rows.append((n, _c2w(R_c2w, centre * m_per_unit), cams[img.camera_id]["focal_mm"]))
    rows = [r for r in rows if frames is None or frames[0] <= r[0] <= frames[1]]

    c0 = next(iter(cams.values()))
    lens = {"distortion_model": c0["dist"][0], "distortion_params": [float(k) for k in c0["dist"][1]]}
    centre_pp = [plate["width"] / 2.0, plate["height"] / 2.0]
    if max(abs(a - b) for a, b in zip(c0["pp"], centre_pp)) > 0.01:
        lens["principal_point_px"] = [float(v) for v in c0["pp"]]
    return _finish(rows, name=name or folder.name, plate=plate, sensor=sensor, fps=fps, lens=lens,
                   source={"kind": "solve", "solver": solver or f"pycolmap {pycolmap.__version__}",
                           "path": str(model_dir.resolve()),
                           "rmse_px": float(rec.compute_mean_reprojection_error()),
                           "rmse_px_at_width": int(c0["size"][0])},
                   scale=_scale_block(m_per_unit, scale_method), frame_offset=frame_offset,
                   extensions={"atlas_vfx_import": {"colmap_model": c0["model"], "colmap_size": c0["size"],
                                                    "m_per_unit": float(m_per_unit),
                                                    "colmap_f_px": c0["f_px"],
                                                    "axis_conversion": "R_c2w_nuke = R_c2w_colmap @ diag(1,-1,-1)"}})


def colmap_points(folder, *, plate_width: int, frame_offset: int = 1001, m_per_unit: float = 1.0,
                  frame_regex: str = r"(\d+)(?!.*\d)") -> dict:
    """The solve's 3D points in the same world as from_colmap's camera (centres are not moved, so points are not
    either), plus which points each frame saw and where (plate pixels). For plane fits in Measure & Scale.
    Returns {"xyz": (N,3), "ids": (N,), "observations": {shot frame: (point indices (K,), xy plate px (K,2))}}."""
    rec, _ = _largest_reconstruction(Path(folder))
    ids = np.array(sorted(rec.points3D), dtype=np.int64)
    index = {int(i): k for k, i in enumerate(ids)}
    xyz = np.array([rec.points3D[int(i)].xyz for i in ids], float) * m_per_unit
    rx = re.compile(frame_regex)
    obs = {}
    for img in rec.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        s = plate_width / cam.width
        k, xy = [], []
        for p in img.points2D:
            if p.has_point3D():
                k.append(index[int(p.point3D_id)])
                xy.append(np.asarray(p.xy, float) * s)
        n = int(rx.search(img.name).group(1)) + frame_offset
        obs[n] = (np.array(k, dtype=np.int64), np.array(xy, float).reshape(-1, 2))
    return {"xyz": xyz, "ids": ids, "observations": obs}


# --------------------------------------------------------------------------------------------- dispatch

def detect_kind(path) -> str:
    p = Path(path)
    if p.is_dir():
        return "colmap"
    if p.suffix.lower() == ".chan":
        return "nuke_chan"
    if p.suffix.lower() == ".abc":
        return "abc"
    if p.suffix.lower() == ".json":
        head = json.loads(p.read_text(encoding="utf-8"))
        if head.get("contract") == sc.CONTRACT:
            return "shot_camera"
        if "frames" in head and head.get("class", "").startswith("Camera"):
            return "nuke_json"
    raise CameraImportError(f"cannot tell what kind of camera {p} is; set kind")


def shift_frames(cam: sc.ShotCamera, delta: int, reason: str = "") -> sc.ShotCamera:
    """The same camera renumbered by delta frames (shot range, every frame, and a note in extensions)."""
    if not delta:
        return cam
    d = copy.deepcopy(cam.data)
    for fr in d["frames"]:
        fr["frame"] += delta
    d["shot"]["frame_start"] += delta
    d["shot"]["frame_end"] += delta
    note = d.setdefault("extensions", {}).setdefault("atlas_vfx_import", {})
    note["frame_shift"] = {"delta": int(delta), "source_first": cam.frames[0], "reason": reason}
    return sc.ShotCamera(d)


def _from_shot_camera(path, *, frames=None, name=None) -> sc.ShotCamera:
    """A shot camera JSON as it is (world, source, extensions kept), cropped to `frames` and renamed like the other
    importers do. Before 2026-10-26 both were ignored, so 'crop to plate' handed on every frame."""
    cam = sc.load(path)
    if frames is not None:
        cam = crop_frames(cam, frames)
    if name and name != cam.name:
        d = copy.deepcopy(cam.data)
        d.setdefault("extensions", {}).setdefault("atlas_vfx_import", {})["renamed_from"] = cam.name
        d["shot"]["name"] = name
        cam = sc.ShotCamera(d)
    return cam


def crop_frames(cam: sc.ShotCamera, frames: tuple[int, int]) -> sc.ShotCamera:
    """Keep frames inside [lo, hi]. A single-frame (static) camera is held over the whole range."""
    lo, hi = frames
    d = copy.deepcopy(cam.data)
    if len(d["frames"]) == 1:
        one = d["frames"][0]
        d["frames"] = [dict(one, frame=f) for f in range(lo, hi + 1)]
    else:
        d["frames"] = [fr for fr in d["frames"] if lo <= fr["frame"] <= hi]
    if not d["frames"]:
        raise CameraImportError(f"camera frames {cam.frames[0]}-{cam.frames[-1]} do not overlap {lo}-{hi} "
                                f"(offset the first frame, or switch crop off)")
    d["shot"]["frame_start"] = d["frames"][0]["frame"]
    d["shot"]["frame_end"] = d["frames"][-1]["frame"]
    return sc.ShotCamera(d)


def import_camera(path, kind: str = "auto", first_frame_to: int | None = None, **kw) -> sc.ShotCamera:
    """first_frame_to: renumber so the camera's first frame becomes this (e.g. 1 or 1001), before any crop to
    `frames` (which is in the renumbered frames, like the plate's)."""
    if first_frame_to is not None:
        crop = kw.pop("frames", None)
        cam = import_camera(path, kind, **kw)
        cam = shift_frames(cam, int(first_frame_to) - cam.frames[0], f"first frame to {first_frame_to}")
        return crop_frames(cam, crop) if crop is not None else cam
    kind = detect_kind(path) if kind == "auto" else kind
    abc_camera = kw.pop("camera", None)        # Alembic only: which camera, when the file has several
    if kind == "shot_camera":
        return _from_shot_camera(path, frames=kw.get("frames"), name=kw.get("name"))
    if kind == "nuke_json":
        kw.pop("rot_order", None)
        kw.pop("focal_mm", None)
        return from_nuke_json(path, **kw)
    if kind == "nuke_chan":
        return from_nuke_chan(path, **kw)
    if kind == "abc":
        kw.pop("rot_order", None)
        kw.pop("focal_mm", None)
        return from_abc(path, camera=abc_camera, **kw)
    if kind == "colmap":
        kw.pop("rot_order", None)
        kw.pop("focal_mm", None)
        return from_colmap(path, **kw)
    raise CameraImportError(f"unknown kind {kind!r}")
