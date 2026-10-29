"""Atlas VFX Set Proxy (catalogue 10): deck + wall planes from depth and solve points -> OBJ + USDA + JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import measure as ms
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.camera_side import plate_depth as pd
from atlas_vfx.contracts import shot_camera as sc

from . import set_proxy as sp


class AtlasVFXSetProxy:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("obj", "usda", "report")
    DESCRIPTION = ("Fits the deck (horizontal, near deck_y) and the wall (vertical, facing the camera) to unprojected "
                   "Plate Depth plus the solve's points, trims each to what was seen, writes OBJ + USDA (metres, "
                   "Y up). Needs a camera that went through Measure & Scale.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "depth_folder": ("STRING", {"default": ""}),
                "solve_points": ("STRING", {"default": ""}),
                "deck_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "deck_band": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 10.0, "step": 0.01}),
                "max_tilt_deg": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 45.0, "step": 0.5,
                                           "tooltip": "planes must be this close to level / plumb (camera is gravity aligned)"}),
                "inlier_m": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 1.0, "step": 0.001}),
                "frame_every": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "pixel_step": ("INT", {"default": 8, "min": 1, "max": 64}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, shot_camera, depth_folder, solve_points, deck_y, deck_band, max_tilt_deg, inlier_m, frame_every,
            pixel_step, name, out_dir):
        cam = sc.ShotCamera(shot_camera)
        if "atlas_vfx_measure" not in cam.data.get("extensions", {}):
            raise ValueError("Set Proxy needs a gravity-aligned camera in metres: run Measure & Scale first")
        folder = Path(pl.clean_path(depth_folder))
        files = sorted(folder.glob("depth.*.exr"))
        if not files:
            raise ValueError(f"no depth.*.exr in {folder}")
        clouds, used = [], []
        for k, f in enumerate(files):
            if k % frame_every:
                continue
            Z, A, attrs = pd.read_exr(f)
            if attrs.get("camera") != cam.name:
                raise ValueError(f"{f.name} was made for camera {attrs.get('camera')!r}, not {cam.name!r}")
            n = int(attrs["frame"])
            clouds.append(pd.unproject(cam, n, Z, step=pixel_step, mask=A))
            used.append(n)
        # only the depth's frame range: solve points seen there, camera positions there
        lo, hi = min(used), max(used)
        pts, note = ms.points_for_camera(cam, solve_points)
        if pts is not None:
            seen = np.unique(np.concatenate([pts["observations"][f][0] for f in range(lo, hi + 1)
                                             if f in pts["observations"]]))
            clouds.append(pts["xyz"][seen])
        cloud = np.vstack(clouds)
        res = sp.fit_set(cloud, [cam.position(f) for f in cam.frames if lo <= f <= hi], deck_y=deck_y,
                         deck_band=deck_band, max_tilt_deg=max_tilt_deg, thresh=inlier_m)
        name = pl.safe_name(name.strip() or f"{cam.name}_set")
        out = pl.resolve_out_dir(out_dir, folder.parent.parent / "set_proxy")
        planes = {"deck": res["deck"], "wall": res["wall"]}
        obj = sp.write_obj(out / f"{name}.obj", planes)
        usda = sp.write_usda(out / f"{name}.usda", planes)
        report = sp.to_json(res)
        report.update({"camera": cam.name, "depth_frames": used, "solve_points": note})
        (out / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXSetProxy",
                         inputs={"camera": cam.name, "depth_folder": str(folder), "solve_points": note,
                                 "deck_y": deck_y, "deck_band": deck_band, "max_tilt_deg": max_tilt_deg,
                                 "inlier_m": inlier_m, "frame_every": frame_every, "pixel_step": pixel_step},
                         outputs={"obj": str(obj), "usda": str(usda), "json": str(out / f"{name}.json")})
        d, w = res["deck"], res["wall"]
        ext_d, ext_w = np.ptp(d["outline"], 0), np.ptp(w["outline"], 0)
        line = (f"deck: y {d['height_at_centre_m']:+.3f} m at its centre, tilt {d['tilt_deg']:.2f} deg, rms "
                f"{d['rms'] * 100:.1f} cm, {ext_d[0]:.1f} x {ext_d[2]:.1f} m | wall: normal "
                f"{np.round(w['normal'], 3).tolist()}, tilt {w['tilt_deg']:.2f} deg, rms {w['rms'] * 100:.1f} cm, "
                f"{max(ext_w[0], ext_w[2]):.1f} x {ext_w[1]:.1f} m | {res['points_used']} points")
        return {"ui": {"text": [line]}, "result": (str(obj), str(usda), json.dumps(report, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXSetProxy": AtlasVFXSetProxy}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSetProxy": "Atlas VFX Set Proxy"}
