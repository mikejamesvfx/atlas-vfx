"""Measure & Scale: known lengths picked in the plate -> metres, up axis, ground, gravity-aligned shot camera.

Ported and generalised from prototypes/vfx_probe/door_scale3.py and ground_check.py. Picks format and how to make
them in Nuke: docs/PICKING.md. In short:

  measurements: each has two ends a and b, each end picked in one or more frames. An end picked in 2+ frames is
                triangulated through the shot camera; an end picked in one frame needs a "plane" (frame + region of
                the solve's points) and is the ray/plane hit. For "vertical": true, a is the bottom and b the top.
  plane_distance measurements ('kind': 'plane_distance'): a and b are plane specs ({frames, region}); the length
                is the perpendicular distance between them (deck to step pad = 4 risers). Robust where picks are
                not: hundreds of solve points per plane, and with 'vertical' the plane normal gives up.
  points:       labelled single points (deck spot for the character, card anchors...), same rules.
  ground:       {"frame" or "frames", "region"} of solve points (plane fit; pool frames so the points cover an
                area, not a strip) and/or "points": [labels] picked on the ground.

Scale = median of length_m / measured length, refused if the spread is over max_spread_pct unless forced.
Up = mean of the vertical measurements (or the ground normal if there are none). A ground plane more than
max_ground_tilt_deg off the verticals is refused unless forced. The output world is metres, Y up,
ground at y = 0, the first frame's camera at x = z = 0 looking towards -Z.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side.plate import clean_path
from atlas_vfx.contracts import shot_camera as sc


class MeasureError(ValueError):
    pass


LINE_RATIO = 0.05   # plane points whose 2nd spread is under 5 % of the 1st are a line, not a plane


# --------------------------------------------------------------------------------------------- picks

def load_picks(src, plate_height: int) -> dict:
    """Read a picks JSON (path or dict) and normalise it: int frames, top-left pixel coordinates."""
    d = json.loads(Path(src).read_text(encoding="utf-8")) if not isinstance(src, dict) else copy.deepcopy(src)
    origin = d.get("origin", "nuke_bottom_left")
    if origin not in ("nuke_bottom_left", "top_left"):
        raise MeasureError(f"picks origin must be nuke_bottom_left or top_left, not {origin!r}")

    def conv(frames: dict) -> dict:
        out = {}
        for f, xy in frames.items():
            x, y = float(xy[0]), float(xy[1])
            out[int(f)] = (x, plate_height - y) if origin == "nuke_bottom_left" else (x, y)
        return out

    def conv_region(r):
        if r is None:
            return None
        x0, y0, x1, y1 = map(float, r)
        if origin == "nuke_bottom_left":
            y0, y1 = plate_height - y0, plate_height - y1
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def conv_plane(pl):
        """{"frame": F} or {"frames": [F, ...]} + region; points from every listed frame are pooled."""
        if pl is None:
            return None
        frames = pl.get("frames") or [pl["frame"]]
        return {"frames": tuple(int(f) for f in frames), "region": conv_region(pl["region"])}

    meas = []
    for m in d.get("measurements", []):
        kind = m.get("kind", "points")
        if kind == "plane_distance":     # a and b are plane specs; length = perpendicular distance between them
            a, b = conv_plane(m["a"]), conv_plane(m["b"])
        elif kind == "points":
            a, b = conv(m["a"]), conv(m["b"])
        else:
            raise MeasureError(f"{m['label']}: kind must be points or plane_distance, not {kind!r}")
        meas.append({"label": m["label"], "length_m": float(m["length_m"]), "vertical": bool(m.get("vertical")),
                     "kind": kind, "a": a, "b": b, "plane": conv_plane(m.get("plane")),
                     "use_for_scale": bool(m.get("use_for_scale", True))})
    pts = [{"label": p["label"], "frames": conv(p["frames"]), "plane": conv_plane(p.get("plane"))}
           for p in d.get("points", [])]
    g = d.get("ground") or {}
    ground = {"plane": conv_plane(g) if "region" in g else None, "points": list(g.get("points", []))}
    if not meas:
        raise MeasureError("picks have no measurements")
    return {"measurements": meas, "points": pts, "ground": ground, "origin": origin}


# --------------------------------------------------------------------------------------------- geometry

def ray(cam: sc.ShotCamera, frame: int, uv) -> tuple[np.ndarray, np.ndarray]:
    """World ray (origin, unit direction) through plate pixel uv (top-left origin)."""
    k = cam.intrinsics(frame)
    M = cam.cam_to_world(frame)
    d_cam = np.array([(uv[0] - k.cx) / k.fx, -(uv[1] - k.cy) / k.fy, -1.0])
    d = M[:3, :3] @ d_cam
    return M[:3, 3].copy(), d / np.linalg.norm(d)


def triangulate(cam: sc.ShotCamera, obs: dict) -> tuple[np.ndarray, float]:
    """Least-squares point closest to the rays of 2+ observations {frame: uv}. Returns (X, reprojection rms px)."""
    if len(obs) < 2:
        raise MeasureError("triangulation needs the point in 2+ frames")
    A, b = np.zeros((3, 3)), np.zeros(3)
    for f, uv in obs.items():
        o, d = ray(cam, f, uv)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e10:
        raise MeasureError("rays are parallel: pick the point in frames further apart")
    X = np.linalg.solve(A, b)
    err = [np.linalg.norm(cam.project(f, X)[0][0] - np.asarray(uv)) for f, uv in obs.items()]
    return X, float(np.sqrt(np.mean(np.square(err))))


def fit_plane(pts: np.ndarray, thresh: float | None = None, iters: int = 400, seed: int = 0):
    """RANSAC plane + least-squares refine. Returns (centre, unit normal, inlier mask, rms)."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        raise MeasureError(f"plane fit needs 3+ points, got {len(pts)}")
    rng = np.random.default_rng(seed)
    if thresh is None:
        ext = np.linalg.norm(pts.max(0) - pts.min(0))
        thresh = 0.01 * ext
    best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        c = pts[best].mean(0)
        n = np.linalg.svd(pts[best] - c)[2][-1]
        best = np.abs((pts - c) @ n) < thresh
    c = pts[best].mean(0)
    n = np.linalg.svd(pts[best] - c)[2][-1]
    rms = float(np.sqrt(np.mean(((pts[best] - c) @ n) ** 2)))
    sv = np.linalg.svd(pts[best] - c, compute_uv=False)
    if sv[1] < LINE_RATIO * sv[0]:
        raise MeasureError(f"plane points are nearly a line (spread ratio {sv[1] / sv[0]:.3f}): any plane through "
                           f"them fits. Widen the region or pool more frames")
    return c, n, best, rms


def ray_plane(o, d, c, n) -> np.ndarray:
    den = d @ n
    if abs(den) < 1e-9:
        raise MeasureError("ray is parallel to the plane")
    return o + d * ((c - o) @ n) / den


def points_in_region(cam: sc.ShotCamera, frame: int, points: dict, region) -> np.ndarray:
    """Indices of solve points seen in `frame` inside region (x0, y0, x1, y1), top-left plate pixels. Uses the
    solve's own observations when present (no occluded points), else projection with depth > 0."""
    x0, y0, x1, y1 = region
    obs = points.get("observations", {}).get(frame)
    if obs is not None:
        k, xy = obs
        m = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        return k[m]
    uv, depth = cam.project(frame, points["xyz"])
    m = (depth > 0) & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    return np.nonzero(m)[0]


def _plane_from_region(cam, points, plane):
    if points is None:
        raise MeasureError("a plane pick needs the solve's 3D points (COLMAP model); this camera has none")
    idx = np.unique(np.concatenate([points_in_region(cam, f, points, plane["region"]) for f in plane["frames"]]))
    if len(idx) < 15:
        raise MeasureError(f"only {len(idx)} solve points in region {plane['region']} at frames {plane['frames']}")
    c, n, inl, rms = fit_plane(points["xyz"][idx])
    return {"centre": c, "normal": n, "rms": rms, "points": int(inl.sum()), "of": int(len(idx))}


def _resolve(cam, obs: dict, plane, points, cache) -> dict:
    """One picked point -> {"X", "method", "rms_px", "frames"}."""
    if len(obs) >= 2:
        X, rms = triangulate(cam, obs)
        return {"X": X, "method": "triangulated", "rms_px": rms, "frames": sorted(obs)}
    if len(obs) == 1 and plane is not None:
        key = (plane["frames"], plane["region"])
        if key not in cache:
            cache[key] = _plane_from_region(cam, points, plane)
        pl = cache[key]
        (f, uv), = obs.items()
        o, d = ray(cam, f, uv)
        return {"X": ray_plane(o, d, pl["centre"], pl["normal"]), "method": "ray_plane", "rms_px": None,
                "frames": [f], "plane_rms": pl["rms"], "plane_points": pl["points"]}
    raise MeasureError("a point picked in one frame needs a plane; otherwise pick it in 2+ frames")


def _angle(u, v) -> float:
    return math.degrees(math.acos(float(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


# --------------------------------------------------------------------------------------------- measure

def measure(cam: sc.ShotCamera, picks: dict, points: dict | None = None, max_spread_pct: float = 5.0,
            force: bool = False, max_ground_tilt_deg: float = 5.0) -> dict:
    """Everything Measure & Scale works out, in the INPUT camera's units, plus the similarity to apply."""
    cache: dict = {}
    meas = []
    for m in picks["measurements"]:
        if m.get("kind") == "plane_distance":
            pa = _plane_from_region(cam, points, m["a"])
            pb = _plane_from_region(cam, points, m["b"])
            # along the mean of both normals: two fitted planes are never exactly parallel, and a centre-to-plane
            # distance along one normal picks up (angle x horizontal offset). Keep the regions close together too.
            na = pa["normal"] * np.sign(pa["normal"] @ (pb["centre"] - pa["centre"]))
            nb = pb["normal"] * np.sign(pb["normal"] @ na)
            n = (na + nb) / np.linalg.norm(na + nb)
            units = float((pb["centre"] - pa["centre"]) @ n)
            par = _angle(pa["normal"] * np.sign(pa["normal"] @ pb["normal"]), pb["normal"])
            a = {"X": pa["centre"], "method": "plane", "rms_px": None, "frames": list(m["a"]["frames"]),
                 "plane_rms": pa["rms"], "plane_points": pa["points"]}
            b = {"X": pa["centre"] + n * units, "method": "plane", "rms_px": None, "frames": list(m["b"]["frames"]),
                 "plane_rms": pb["rms"], "plane_points": pb["points"], "planes_parallel_within_deg": par}
            meas.append({**m, "A": a, "B": b, "measured_units": units, "m_per_unit": m["length_m"] / units})
            continue
        a = _resolve(cam, m["a"], m["plane"], points, cache)
        b = _resolve(cam, m["b"], m["plane"], points, cache)
        units = float(np.linalg.norm(b["X"] - a["X"]))
        if units <= 0:
            raise MeasureError(f"{m['label']}: both ends at the same place")
        meas.append({**m, "A": a, "B": b, "measured_units": units, "m_per_unit": m["length_m"] / units})
    use = [x for x in meas if x["use_for_scale"]]
    if not use:
        raise MeasureError("no measurement is marked for scale")
    mpu = np.array([x["m_per_unit"] for x in use])
    s = float(np.median(mpu))
    spread = float((mpu.max() - mpu.min()) / s * 100) if len(use) > 1 else 0.0
    if spread > max_spread_pct and not force:
        detail = ", ".join(f"{x['label']} {x['m_per_unit']:.4f}" for x in use)
        raise MeasureError(f"scale spread {spread:.1f} % > {max_spread_pct} % ({detail} m/unit); check the picks or "
                           f"lengths, or force")

    labelled = {p["label"]: _resolve(cam, p["frames"], p["plane"], points, cache) for p in picks["points"]}

    # ground
    g = picks["ground"]
    ground = {"plane": None, "points": []}
    if g.get("plane"):
        ground["plane"] = _plane_from_region(cam, points, g["plane"])
    for lab in g.get("points", []):
        if lab not in labelled:
            raise MeasureError(f"ground point {lab!r} is not in points")
        ground["points"].append(labelled[lab]["X"])

    # up
    verts = [x for x in meas if x["vertical"]]
    C0 = cam.position(cam.frames[0])
    if verts:
        ups = [(x["B"]["X"] - x["A"]["X"]) / x["measured_units"] for x in verts]
        up = np.sum(ups, axis=0)
        up /= np.linalg.norm(up)
        up_spread = max(_angle(u, up) for u in ups)
        up_method = f"mean of {len(verts)} vertical measurement(s): " + ", ".join(x["label"] for x in verts)
    elif ground["plane"] is not None:
        up = ground["plane"]["normal"].copy()
        if (C0 - ground["plane"]["centre"]) @ up < 0:
            up = -up
        up_spread = None
        up_method = "ground plane normal"
    else:
        raise MeasureError("no vertical measurement and no ground plane: cannot find up")

    # ground height along up
    if ground["plane"] is not None:
        gh = float(ground["plane"]["centre"] @ up)
        fr = g["plane"]["frames"]
        ground_method = (f"plane of {ground['plane']['points']} solve points (frames "
                         f"{fr[0]}{'-' + str(fr[-1]) if len(fr) > 1 else ''}, {len(fr)} pooled)")
        normal_vs_up = _angle(ground["plane"]["normal"] * np.sign(ground["plane"]["normal"] @ up), up)
        if verts and normal_vs_up > max_ground_tilt_deg and not force:
            raise MeasureError(f"ground plane normal is {normal_vs_up:.1f} deg off the verticals (limit "
                               f"{max_ground_tilt_deg}): wrong region, sloping ground, or bad vertical picks; "
                               f"fix, or force")
        if ground["points"]:
            ground_method += f" (+{len(ground['points'])} picked points, reported only)"
    elif ground["points"]:
        gh = float(np.median([p @ up for p in ground["points"]]))
        ground_method = f"median of {len(ground['points'])} picked ground point(s)"
        normal_vs_up = None
    else:
        raise MeasureError("no ground: give ground.region (solve points) or ground.points (picked)")

    # similarity: X' = s R (X - origin); up -> +Y, first camera's forward (flattened) -> -Z, origin under camera 0
    fwd = -cam.cam_to_world(cam.frames[0])[:3, 2]
    flat = fwd - (fwd @ up) * up
    if np.linalg.norm(flat) < 1e-6:
        flat = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    z = -flat / np.linalg.norm(flat)
    x = np.cross(up, z)
    R = np.stack([x, up, z])
    origin = C0 - ((C0 @ up) - gh) * up

    def to_new(X):
        return s * R @ (np.asarray(X) - origin)

    return {
        "m_per_unit": s, "spread_pct": spread, "max_spread_pct": max_spread_pct, "forced": bool(force and
                                                                                               spread > max_spread_pct),
        "measurements": [{"label": x["label"], "length_m": x["length_m"], "measured_units": x["measured_units"],
                          "m_per_unit": x["m_per_unit"], "vertical": x["vertical"], "use_for_scale": x["use_for_scale"],
                          "a": {k: v for k, v in x["A"].items() if k != "X"},
                          "b": {k: v for k, v in x["B"].items() if k != "X"},
                          "length_after_m": float(np.linalg.norm(x["B"]["X"] - x["A"]["X"]) * s)} for x in meas],
        "up_method": up_method, "up_spread_deg": up_spread,
        "ground_method": ground_method, "ground_normal_vs_up_deg": normal_vs_up,
        "ground_rms_m": ground["plane"]["rms"] * s if ground["plane"] else None,
        "points_m": {k: to_new(v["X"]).tolist() for k, v in labelled.items()},
        "points_detail": {k: {kk: vv for kk, vv in v.items() if kk != "X"} for k, v in labelled.items()},
        "measurement_ends_m": {x["label"]: [to_new(x["A"]["X"]).tolist(), to_new(x["B"]["X"]).tolist()]
                               for x in meas},
        "camera_height_m": {str(f): float(to_new(cam.position(f))[1]) for f in cam.frames[:: max(1, len(cam.frames)
                                                                                                 // 8)]},
        "transform": {"scale": s, "rotation": R.tolist(), "origin_src_units": origin.tolist(),
                      "rule": "X_new = scale * rotation @ (X_src - origin)"},
    }


def apply(cam: sc.ShotCamera, result: dict, name: str | None = None) -> sc.ShotCamera:
    """The scaled, gravity-aligned shot camera (Y up, ground y = 0)."""
    tr = result["transform"]
    s, R, o = tr["scale"], np.array(tr["rotation"]), np.array(tr["origin_src_units"])
    data = copy.deepcopy(cam.data)
    for fr in data["frames"]:
        M = np.array(fr["cam_to_world"])
        N = np.eye(4)
        N[:3, :3] = R @ M[:3, :3]
        N[:3, 3] = s * R @ (M[:3, 3] - o)
        fr["cam_to_world"] = N.tolist()
    prev = data["scale"]["m_per_source_unit"] if data["scale"]["method"] != "unscaled" else 1.0
    data["scale"] = {
        "method": "known_lengths", "m_per_source_unit": float(prev * s),
        "measurements": [{"label": m["label"], "length_m": m["length_m"], "measured_units": m["measured_units"],
                          "frames": sorted(set(m["a"]["frames"]) | set(m["b"]["frames"]))}
                         for m in result["measurements"] if m["use_for_scale"]],
        "spread_pct": result["spread_pct"], "up_axis_method": result["up_method"], "ground_y_m": 0.0,
    }
    if name:
        data["shot"]["name"] = name
    ext = data.setdefault("extensions", {})
    ext["atlas_vfx_measure"] = {k: result[k] for k in ("transform", "up_spread_deg", "ground_method",
                                                        "ground_normal_vs_up_deg", "ground_rms_m", "points_m",
                                                        "forced")}
    return sc.ShotCamera(data)


def points_for_camera(cam: sc.ShotCamera, folder: str = "", max_reproj_px: float = 3.0):
    """The COLMAP solve's 3D points in this camera's world: the given folder or the camera's own solve, scaled by
    the import's m_per_unit and moved by Measure & Scale's transform if the camera went through it. Checked by
    reprojecting what the solve observed (median must be under max_reproj_px at plate size).
    Returns (points dict as camera_io.colmap_points, note) or (None, note) when there is no solve."""
    from . import camera_io as cio

    src = cam.data["source"]
    folder = clean_path(folder) or (src.get("path") if src.get("kind") == "solve" else "")
    if not folder:
        return None, "none (no solve folder)"
    ext = cam.data.get("extensions", {})
    mpu = float(ext.get("atlas_vfx_import", {}).get("m_per_unit", 1.0))
    pts = cio.colmap_points(folder, plate_width=cam.data["plate"]["width"],
                            frame_offset=cam.data["shot"].get("plate_frame_offset", 1001), m_per_unit=mpu)
    if "atlas_vfx_measure" in ext:
        pts["xyz"] = transform_points(ext["atlas_vfx_measure"], pts["xyz"])
    errs = []
    for f in cam.frames[:: max(1, len(cam.frames) // 10)]:
        if f in pts["observations"] and len(pts["observations"][f][0]):
            k, xy = pts["observations"][f]
            uv, _ = cam.project(f, pts["xyz"][k], ignore_distortion=True)
            errs.append(np.median(np.linalg.norm(uv - xy, axis=1)))
    if not errs:
        raise ValueError(f"solve points in {folder} share no frames with {cam.name}")
    med = float(np.median(errs))
    if med > max_reproj_px:
        raise ValueError(f"solve points in {folder} are not in {cam.name}'s world (reprojection {med:.1f} px); "
                         f"they need the camera imported from that solve")
    return pts, f"{folder} ({len(pts['xyz'])} points, reprojection {med:.2f} px)"


def points_in_camera(cam: sc.ShotCamera, solve_folder: str = "", points_camera: str = "", *, who: str = "RS Solve",
                     max_reproj_px: float = 3.0):
    """The solve's points + tracks in `cam`'s world. points_camera: a shot camera JSON imported from that solve (and
    maybe through Measure & Scale) whose world they are moved into; it must be the same world as cam (checked).
    Shared by RS Solve and the deprecated Sensor Readout Solve."""
    pcam = sc.load(clean_path(points_camera)) if points_camera.strip() else cam
    folder = clean_path(solve_folder) or pcam.data["source"].get("path", "")
    pts, note = points_for_camera(pcam, folder, max_reproj_px=max_reproj_px)
    if pts is None:
        raise ValueError(f"{who}: no solve points ({note}); give solve_folder")
    if pcam is not cam:
        errs = []
        for f in cam.frames[:: max(1, len(cam.frames) // 10)]:
            if f in pts["observations"] and len(pts["observations"][f][0]):
                k, xy = pts["observations"][f]
                uv, _ = cam.project(f, pts["xyz"][k], ignore_distortion=True)
                errs.append(float(np.median(np.linalg.norm(uv - xy, axis=1))))
        if not errs or np.median(errs) > max_reproj_px:
            raise ValueError(f"{who}: {pcam.name}'s points do not reproject through {cam.name} "
                             f"({np.median(errs) if errs else 'no shared frames'} px)")
        note += f"; moved into {pcam.name}'s world, reprojects through {cam.name} at {np.median(errs):.2f} px"
    return pts, note, pcam


def transform_points(result: dict, xyz) -> np.ndarray:
    tr = result["transform"]
    return (tr["scale"] * (np.asarray(xyz, float) - tr["origin_src_units"]) @ np.array(tr["rotation"]).T)
