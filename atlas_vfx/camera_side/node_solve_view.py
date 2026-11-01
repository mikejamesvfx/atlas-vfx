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


def _proxy_images(cam: sc.ShotCamera, pts: dict | None, count: int) -> list[dict]:
    """Up to `count` tracking proxies written into ComfyUI's temp folder as JPEG, with the plate-px points each saw."""
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
            sel = np.arange(len(k))[:: max(1, len(k) // 1500)]
            obs = [[round(float(uv[i, 0]), 1), round(float(uv[i, 1]), 1), round(float(e[i]), 2)] for i in sel]
        out.append({"frame": f, "filename": name, "subfolder": sub, "type": "temp", "points": obs})
    return out


def payload(cam: sc.ShotCamera, max_points: int = 20000, images: int = 6, seed: int = 0) -> dict:
    k = cam.intrinsics(cam.frames[0])
    W, H = cam.data["plate"]["width"], cam.data["plate"]["height"]
    pts, note = ms.points_for_camera(cam, max_reproj_px=1e9) if cam.data["source"].get("kind") == "solve" \
        else (None, "no solve points")
    body = {
        "name": cam.name, "plate": [W, H], "frame_offset": int(cam.data["shot"].get("plate_frame_offset", 1001)), "fov_x_deg": float(np.degrees(2 * np.arctan(W / 2 / k.fx))),
        "frames": [{"f": f, "m": [round(v, 6) for v in cam.cam_to_world(f).ravel().tolist()]} for f in cam.frames],
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
    body["images"] = _proxy_images(cam, pts, images)
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
                "plate_frames": ("INT", {"default": 6, "min": 0, "max": 48}),
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
