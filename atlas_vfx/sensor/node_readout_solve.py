"""Atlas VFX Sensor Readout Solve (deprecated 2026-10-26, a diagnostic): the readout residual left in a GS solve's
tracks. It reads low on a GS camera, so Sensor RS Maps' preset rule never uses it (plan, Phase 1b S2)."""
from __future__ import annotations

import json
import logging

import numpy as np

from atlas_vfx.camera_side import measure
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import readout_solve
from ._node_util import CATEGORY, work_dir

log = logging.getLogger("atlas_vfx")

def points_in_camera(cam: sc.ShotCamera, solve_folder: str = "", points_camera: str = ""):
    """camera_side.measure.points_in_camera (moved there for RS Solve, catalogue 47)."""
    return measure.points_in_camera(cam, solve_folder, points_camera, who="Sensor Readout Solve")


class AtlasVFXSensorReadoutSolve:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("readout_ms", "report")
    DESCRIPTION = ("DEPRECATED (2026-10-26), kept as a diagnostic. On a global-shutter solve it reads low (23.5 vs "
                   "26.6 ms RS-aware on DSCF3000): the GS solve has already bent its poses and points to soak up some "
                   "of the rolling shutter, so the residual is not the sensor. Sensor RS Maps' 'preset rule' never "
                   "uses this value; the readout comes from a chart measurement, an RS-aware solve, or the preset "
                   "spec. What it does: re-projects the solve's 2D tracks and 3D points row by row (platelock's RS "
                   "camera model) for a sweep of readouts, with a small per-frame pose correction. Check the curve "
                   "in the report: a flat minimum means the move is too gentle to measure.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "solve_folder": ("STRING", {"default": "", "tooltip": "pycolmap model folder; blank = the camera's "
                                            "own solve (source.path)"}),
                "points_camera": ("STRING", {"default": "", "tooltip": "shot camera JSON imported from that solve "
                                             "(e.g. Measure & Scale's output) whose world the points go into; blank = "
                                             "this camera. Use it when this camera is an export (.abc) of a solve."}),
                "frame_step": ("INT", {"default": 2, "min": 1, "max": 24}),
                "refine_pose": ("BOOLEAN", {"default": True}),
                "max_readout_ms": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 100.0, "step": 0.5}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, shot_camera, solve_folder, points_camera, frame_step, refine_pose, max_readout_ms, out_dir):
        log.warning("atlas-vfx Sensor Readout Solve is deprecated (diagnostic only): a GS solve's residual is not "
                    "the sensor readout; Sensor RS Maps' preset rule ignores it (chart > RS-aware solve > spec)")
        cam = sc.ShotCamera(shot_camera)
        pts, note, pcam = points_in_camera(cam, solve_folder, points_camera)
        fps = float(cam.data["shot"]["fps"])
        grid = np.linspace(0.0, max_readout_ms / 1000.0 * fps, 41)
        frames = cam.frames[1:-1:frame_step]
        est = readout_solve.estimate_readout(cam, pts["xyz"], pts["observations"], frames=frames, grid=grid,
                                             refine_pose=refine_pose)
        est.update({"camera": cam.name, "points": note, "points_camera": pcam.name})
        out = work_dir(cam.data, "readout", out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{cam.name}.readout.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{cam.name}.readout.sidecar.json", "AtlasVFXSensorReadoutSolve",
                         inputs={"camera": cam.name, "solve_folder": solve_folder, "points_camera": pcam.name,
                                 "frame_step": frame_step, "refine_pose": refine_pose},
                         outputs={k: v for k, v in est.items() if k != "curve"})
        return (float(est["readout_ms"]), json.dumps(est, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXSensorReadoutSolve": AtlasVFXSensorReadoutSolve}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSensorReadoutSolve": "Atlas VFX Sensor Readout Solve"}
