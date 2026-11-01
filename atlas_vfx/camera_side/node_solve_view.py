"""Atlas VFX Solve View (catalogue 69): a solved camera and its points in a three.js panel (web/atlas_vfx_solve_view.js).

The node builds the payload (camera per frame, points coloured by reprojection error, per-frame error, a few
tracking proxies with the points drawn on them) and passes the seeds widget through. The panel is display only: it
draws the payload and, when you click on a plate frame, appends a seed to the seeds widget, which feeds Track 2D.
Without the frontend the node runs the same (seeds pass through unchanged).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import measure as ms
from . import plate as pl
from . import tracks2d

log = logging.getLogger("atlas_vfx")
UI_KEY = "atlas_solve_view"


def _point_errors(cam: sc.ShotCamera, pts: dict) -> tuple[np.ndarray, dict]:
    """Worst reprojection per point and median per frame (plate px, through the camera's own lens)."""
    worst = np.zeros(len(pts["xyz"]))
    per_frame = {}
    have = set(cam.frames)
    for f, (k, xy) in pts["observations"].items():
        if f not in have or not len(k):
            continue
        uv, _ = cam.project(f, pts["xyz"][k])
        e = np.linalg.norm(uv - xy, axis=1)
        np.maximum.at(worst, k, e)
        per_frame[f] = float(np.median(e))
    return worst, per_frame


def display_frame(cam: sc.ShotCamera, xyz) -> dict | None:
    """How the panel levels a solve that Measure & Scale has not oriented (a raw solve keeps COLMAP's world, often
    upside down in a Y-up viewer): up = the cameras' mean up, the first camera's forward (flattened) -> -Z, origin
    under the first camera at the points' low ground. Display only: the camera data is not changed. None when the
    camera is already gravity aligned."""
    if "atlas_vfx_measure" in (cam.data.get("extensions") or {}):
        return None
    ups = np.array([cam.cam_to_world(f)[:3, 1] for f in cam.frames])
    up = ups.mean(0)
    if np.linalg.norm(up) < 1e-6:
        return None
    up /= np.linalg.norm(up)
    fwd = -cam.cam_to_world(cam.frames[0])[:3, 2]
    flat = fwd - (fwd @ up) * up
    if np.linalg.norm(flat) < 1e-6:
        flat = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    z = -flat / np.linalg.norm(flat)
    R = np.stack([np.cross(up, z), up, z])
    C0 = cam.position(cam.frames[0])
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    ground = float(np.percentile((xyz - C0) @ up, 5)) if len(xyz) else 0.0
    return {"R": R, "origin": C0 + min(ground, 0.0) * up}


HAND_HEIGHT_M = 1.6          # an unscaled solve's metres guessed from a handheld camera this far above the ground


def check_cube(cam: sc.ShotCamera, disp: dict | None, xyz) -> dict | None:
    """A 1 m check cube standing on a solve point where it stays in view longest, in the panel's (display) world, with its
    8 corners in the camera's world for projecting onto the plate. Size from the camera's scale; an unscaled solve
    guesses it from a handheld camera height (HAND_HEIGHT_M above the ground), and says so."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    if not len(xyz):
        return None
    if disp is None:
        if disp is None and cam.data.get("scale", {}).get("method") not in (None, "unscaled"):
            R, o = np.eye(3), np.zeros(3)                       # Measure & Scale: metres, Y up, ground y = 0
        else:
            return None
    else:
        R, o = disp["R"], disp["origin"]
    d = (xyz - o) @ R.T                                         # display world: Y up, ground near y = 0
    scale = cam.data.get("scale", {})
    if scale.get("method") not in (None, "unscaled"):
        side, how = 1.0 / float(scale.get("m_per_source_unit", 1.0) or 1.0), "1 m (scaled camera)"
        side = 1.0 if disp is None else side
    else:
        h = float(np.median([(R @ (cam.position(f) - o))[1] for f in cam.frames]))
        if h <= 0:
            return None
        side, how = h / HAND_HEIGHT_M, f"1 m if the camera was {HAND_HEIGHT_M} m above the ground (unscaled solve)"
    unit = np.array([[x, y, z] for x in (-.5, .5) for y in (0, 1) for z in (-.5, .5)]) * side
    W, H = cam.data["plate"]["width"], cam.data["plate"]["height"]

    def seen(c):                       # frames that show the whole cube, standing at c (display world)
        world = (unit + c) @ R + o
        n = 0
        for f in cam.frames[:: max(1, len(cam.frames) // 24)]:
            M = cam.cam_to_world(f)
            if not (((world - M[:3, 3]) @ M[:3, :3])[:, 2] < 0).all():
                continue
            uv, _ = cam.project(f, world)
            n += bool(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all())
        return n

    # standing on a solve point (the ground, a step, the deck), wherever it stays whole in view longest
    cands = d[np.random.default_rng(0).choice(len(d), min(300, len(d)), replace=False)]
    c = max(cands, key=seen)
    corners = unit + c
    world = corners @ R + o                                     # back to the camera's world
    return {"centre": c.tolist(), "side": float(side), "note": how, "corners_disp": corners.tolist(),
            "corners_world": world}


CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _proxy_images(cam: sc.ShotCamera, pts: dict | None, count: int, cube: dict | None = None) -> list[dict]:
    """Up to `count` tracking proxies written into ComfyUI's temp folder as JPEG, with the plate-px points each saw.
    Many frames (playback in the panel) carry fewer points each, so the payload stays small."""
    import cv2

    ext = (cam.data.get("extensions") or {}).get("atlas_vfx_solve_plate") or {}
    if not ext.get("proxy_dir") or count <= 0:
        return []
    # the camera is graph data: only read proxies from a folder outputs may be written to
    proxy = pl.resolve_out_dir(ext["proxy_dir"], pl.WORK_ROOT, what="proxy_dir")
    try:
        import folder_paths

        tmp = Path(folder_paths.get_temp_directory())
    except ImportError:
        return []
    off = cam.data["shot"].get("plate_frame_offset", 1001)
    frames = cam.frames
    pick = sorted({frames[round(i * (len(frames) - 1) / max(count - 1, 1))] for i in range(min(count, len(frames)))})
    out = []
    sub = "atlas_solve_view"
    (tmp / sub).mkdir(parents=True, exist_ok=True)
    for f in pick:
        src = proxy / f"plate_{f - off:04d}.png"
        a = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        name = pl.safe_name(f"{cam.name}_{f}.jpg")
        cv2.imwrite(str(tmp / sub / name), a, [cv2.IMWRITE_JPEG_QUALITY, 85])
        obs = []
        if pts is not None and f in pts["observations"]:
            k, xy = pts["observations"][f]
            uv, _ = cam.project(f, pts["xyz"][k])
            e = np.linalg.norm(uv - xy, axis=1)
            sel = np.arange(len(k))[:: max(1, len(k) // (1500 if len(pick) <= 48 else 400))]
            obs = [[round(float(uv[i, 0]), 1), round(float(uv[i, 1]), 1), round(float(e[i]), 2)] for i in sel]
        box = None
        if cube is not None:
            M = cam.cam_to_world(f)
            zc = (cube["corners_world"] - M[:3, 3]) @ M[:3, :3]
            if (zc[:, 2] < 0).all():                            # every corner in front of the camera
                uv, _ = cam.project(f, cube["corners_world"])
                box = [[round(float(a), 1), round(float(b), 1)] for a, b in uv]
        out.append({"frame": f, "filename": name, "subfolder": sub, "type": "temp", "points": obs, "cube": box})
    return out


def payload(cam: sc.ShotCamera, max_points: int = 20000, images: int = 6, seed: int = 0) -> dict:
    k = cam.intrinsics(cam.frames[0])
    W, H = cam.data["plate"]["width"], cam.data["plate"]["height"]
    pts, note = ms.points_for_camera(cam, max_reproj_px=1e9) if cam.data["source"].get("kind") == "solve" \
        else (None, "no solve points")
    body = {
        "name": cam.name, "plate": [W, H], "frame_offset": int(cam.data["shot"].get("plate_frame_offset", 1001)), "fov_x_deg": float(np.degrees(2 * np.arctan(W / 2 / k.fx))),
        "frames": [],
        "scale": cam.data.get("scale", {}), "points_note": note, "points": [], "err": [], "frame_err": {},
        "report": ((cam.data.get("extensions") or {}).get("atlas_vfx_solve_plate") or {}),
    }
    if pts is not None and len(pts["xyz"]):
        worst, per_frame = _point_errors(cam, pts)
        idx = np.arange(len(pts["xyz"]))
        if len(idx) > max_points:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
        body["points"] = [round(float(v), 4) for v in pts["xyz"][idx].ravel()]
        body["err"] = [round(float(v), 2) for v in worst[idx]]
        body["frame_err"] = {str(f): round(v, 3) for f, v in per_frame.items()}
    disp = display_frame(cam, pts["xyz"] if pts is not None else [])
    mats = {f: cam.cam_to_world(f) for f in cam.frames}
    if disp is not None:
        R, o = disp["R"], disp["origin"]
        for f, M in mats.items():
            N = np.eye(4)
            N[:3, :3] = R @ M[:3, :3]
            N[:3, 3] = R @ (M[:3, 3] - o)
            mats[f] = N
        if body["points"]:
            xyz = (np.array(body["points"]).reshape(-1, 3) - o) @ R.T
            body["points"] = [round(float(v), 4) for v in xyz.ravel()]
        body["display"] = "levelled for display by the cameras' up (Measure & Scale orients the camera itself)"
    body["frames"] = [{"f": f, "m": [round(v, 6) for v in M.ravel().tolist()]} for f, M in mats.items()]
    cube = check_cube(cam, disp, pts["xyz"] if pts is not None else [])
    if cube is not None:
        body["cube"] = {k: cube[k] for k in ("centre", "side", "note", "corners_disp")}
        body["cube_edges"] = CUBE_EDGES
    body["images"] = _proxy_images(cam, pts, images, cube)
    return body


class AtlasVFXSolveView:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("seeds",)
    DESCRIPTION = ("Looks at a solve in 3D: the points (green under 0.5 px, yellow to 2 px, red beyond, at plate "
                   "size), the camera path and its frustum at the scrubbed frame, the reprojection error per frame, "
                   "and a few plate frames with the solve's points drawn on them. Click a plate frame to add a seed "
                   "for Track 2D (the seeds widget, passed out unchanged when nothing is clicked).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "max_points": ("INT", {"default": 20000, "min": 100, "max": 200000}),
                "plate_frames": ("INT", {"default": 6, "min": 0, "max": 2000,
                                         "tooltip": "tracking proxies sent to the Plate tab; the shot's frame count "
                                                    "plays the whole shot (Play button)"}),
                "seeds": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    def run(self, shot_camera, max_points, plate_frames, seeds):
        cam = sc.ShotCamera(shot_camera)
        tracks2d.parse_seeds(seeds)                  # untrusted widget: refuse bad JSON here, not in Track 2D
        body = payload(cam, max_points, plate_frames)
        log.info("atlas-vfx Solve View: %s, %d frames, %d points", cam.name, len(body["frames"]),
                 len(body["err"]))
        return {"ui": {UI_KEY: [body]}, "result": (seeds,)}


NODE_CLASS_MAPPINGS = {"AtlasVFXSolveView": AtlasVFXSolveView}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSolveView": "Atlas VFX Solve View"}

if __name__ == "__main__":              # dump a payload for the panel harness: python -m ... <shot_camera.json>
    import sys

    print(json.dumps(payload(sc.load(sys.argv[1]), 2000, 0))[:2000])
