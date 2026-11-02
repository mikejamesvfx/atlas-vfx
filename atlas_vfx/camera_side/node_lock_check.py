"""Atlas VFX Lock Check (catalogue 8): grid card + measured points through every frame on the display proxy -> mp4."""
from __future__ import annotations

import json

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import camera_checks as cc
from . import lock_check as lc
from . import measure as ms
from . import plate as pl


def levelled_card(cam: sc.ShotCamera, width_m: float, height_m: float, divisions: int):
    """For a raw solve (no Measure & Scale): levelled by the cameras' up as Solve View shows it, a ground card under
    Solve View's 1 m check cube (on a solve point that stays in view) plus the cube's edges, in the camera's world.
    None when there are no solve points."""
    from .node_solve_view import CUBE_EDGES, check_cube, display_frame

    pts, _ = ms.points_for_camera(cam, max_reproj_px=1e9)
    if pts is None or not len(pts["xyz"]):
        return None
    disp = display_frame(cam, pts["xyz"])
    cube = check_cube(cam, disp, pts["xyz"]) if disp is not None else None
    if cube is None:
        return None
    R, o, s = disp["R"], disp["origin"], cube["side"]
    c = np.array(cube["centre"])
    f0 = R @ -cam.cam_to_world(cam.frames[0])[:3, 2]
    v = np.array([f0[0], 0.0, f0[2]])
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0, 0, -1.0])
    u = np.cross(v, [0, 1.0, 0])
    U, V = u * width_m * s, v * height_m * s
    q = c - U / 2 - V / 2
    lines = []
    for k in range(divisions + 1):
        a = k / divisions
        lines.append((q + U * a, q + U * a + V))
        lines.append((q + V * a, q + V * a + U))
    corners = np.array(cube["corners_disp"])
    lines += [(corners[a], corners[b]) for a, b in CUBE_EDGES]
    world = [(a @ R + o, b @ R + o) for a, b in lines]        # display world -> the camera's world
    return world, c @ R + o, f"auto: under the 1 m check cube, levelled by the cameras' up ({cube['note']})"


def _centre(cam: sc.ShotCamera, spec: str, points: dict):
    spec = spec.strip()
    if spec in ("", "auto"):
        if cc.is_nodal(cam):
            raise ValueError("Lock Check: a nodal camera has no height above any ground, so the auto card (ground, "
                             "4 m ahead) means nothing; give a centre in solve units or a measured point")
        f0 = -cam.cam_to_world(cam.frames[0])[:3, 2]
        flat = np.array([f0[0], 0.0, f0[2]])
        flat /= np.linalg.norm(flat)
        c = cam.position(cam.frames[0]) + 4.0 * flat
        return np.array([c[0], 0.0, c[2]]), "auto: ground, 4 m ahead of the first camera"
    if spec in points:
        return np.array(points[spec], float), f"measured point {spec}"
    try:
        xyz = np.array([float(v) for v in spec.split(",")])
        assert xyz.shape == (3,)
        return xyz, "explicit"
    except (ValueError, AssertionError):
        raise ValueError(f"card centre {spec!r}: use auto, a measured point label ({sorted(points)}) or x,y,z")


class AtlasVFXLockCheck:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("mp4", "preview")
    DESCRIPTION = ("Projects a gridded card (green) and the measured points (red) through every frame onto the "
                   "display proxy. If the card stays glued to its surface, the camera is right.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "centre": ("STRING", {"default": "auto", "tooltip": "auto | measured point label | x,y,z metres"}),
                "orientation": (["ground", "vertical"], {"default": "ground"}),
                "width_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "height_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "divisions": ("INT", {"default": 6, "min": 1, "max": 50}),
                "proxy_width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 2}),
                "preview_every": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "out_path": ("STRING", {"default": ""}),
            },
            "optional": {"measure_report": ("STRING", {"forceInput": True})},
        }

    def run(self, plate_ref, shot_camera, centre, orientation, width_m, height_m, divisions, proxy_width,
            preview_every, out_path, measure_report=None):
        import torch

        cam = sc.ShotCamera(shot_camera)
        cc.check_frame_offset(cam, plate_ref, "Lock Check")
        cc.require_metric(cam, "Lock Check", "card width and height (metres)")
        points = {}
        if measure_report and measure_report.strip() not in ("", "None", "null"):
            points = json.loads(measure_report).get("points_m", {})
        elif "atlas_vfx_measure" in cam.data.get("extensions", {}):
            points = cam.data["extensions"]["atlas_vfx_measure"].get("points_m", {})
        lev = None
        if centre.strip() in ("", "auto") and orientation == "ground" and not points and not cc.is_nodal(cam)                 and cam.data["source"].get("kind") == "solve":
            lev = levelled_card(cam, width_m, height_m, divisions)
        if lev is not None:
            lines, c, how = lev
        else:
            c, how = _centre(cam, centre, points)
            lines = lc.card_lines(cam, c, orientation, width_m, height_m, divisions)
        out = pl.resolve_out_dir(out_path, pl.WORK_ROOT / pl.plate_stem(plate_ref) / "lock_check" /
                                 f"{cam.name}_{orientation}.mp4", plate_path=plate_ref["path"], what="out_path")
        path, kept = lc.render(plate_ref, cam, out, lines=lines, marks=points, width=proxy_width,
                               keep_every=preview_every)
        pl.write_sidecar(path.with_suffix(".sidecar.json"), "AtlasVFXLockCheck",
                         inputs={"plate_ref": plate_ref, "camera": cam.name, "centre": c.tolist(), "centre_from": how,
                                 "orientation": orientation, "size_m": [width_m, height_m], "divisions": divisions,
                                 "marks": points},
                         outputs={"mp4": str(path),
                                  "frames": [max(plate_ref["nuke_first"], cam.frames[0]),
                                             min(plate_ref["nuke_last"], cam.frames[-1])]})
        return (str(path), torch.from_numpy(np.stack(kept)))


NODE_CLASS_MAPPINGS = {"AtlasVFXLockCheck": AtlasVFXLockCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXLockCheck": "Atlas VFX Lock Check"}
