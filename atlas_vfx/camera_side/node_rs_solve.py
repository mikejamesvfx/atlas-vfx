"""Atlas VFX RS Solve (catalogue 47): refit a global-shutter shot camera so its poses and points follow a rolling
shutter, with the readout free or held. Wraps camera_side/rs_solve.solve_shot (the same code as tools/rs_clip_solve.py).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from atlas_vfx.contracts import shot_camera as sc

from . import camera_io as cio
from . import measure as ms
from . import plate as pl
from . import rs_solve, sensors

log = logging.getLogger("atlas_vfx")

FREE_START_FRAMES = 0.3          # free readout start when there is no preset (12.5 ms at 24p; NOTES: start-insensitive)
MAX_PROFILE = 20
# Same-world check on the solve's points, at plate size. A GS solve of an RS plate carries the skew in its residuals,
# so this is looser than Measure & Scale's 3 px: the clip tool's 6 px (1.5 px on a 1024 px tracking proxy).
POINTS_MAX_REPROJ_PX = 6.0


def parse_profile(text: str) -> list[float]:
    """'0, 10, 20, 30.9' -> readouts in ms. A widget is untrusted: numbers only, 0-100 ms, at most MAX_PROFILE."""
    parts = [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]
    if len(parts) > MAX_PROFILE:
        raise ValueError(f"RS Solve: at most {MAX_PROFILE} profile readouts, got {len(parts)}")
    out = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"RS Solve: profile_ms entry {p!r} is not a number") from None
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"RS Solve: profile readout {v} ms outside 0-100 ms")
        out.append(v)
    return out


def start_readout(cam_data: dict, typed_ms: float, held: bool) -> tuple[float, str]:
    """readout_ms > 0 is used as given. 0 = the sensor preset's readout (generic spec); with no preset a free solve
    starts at 0.3 frames and a held one is refused."""
    if typed_ms > 0:
        return float(typed_ms), "typed"
    name = sensors.preset_for_camera(cam_data)
    won = sensors.pick_readout(sensors.get_preset(name)["readout_evidence"]) if name else None
    if won:
        return won["value"], f"preset {name} ({won['status']}: {won['source']})"
    if held:
        raise ValueError("RS Solve: readout held needs readout_ms > 0 or a camera with a sensor preset")
    fps = float(cam_data["shot"]["fps"])
    return FREE_START_FRAMES / fps * 1000.0, f"{FREE_START_FRAMES} frames (no sensor preset)"


class AtlasVFXRSSolve:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_TRACKED_CAMERA", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("shot_camera", "readout_ms", "path", "report")
    DESCRIPTION = ("Rolling-shutter-aware refit of a global-shutter solve. A GS solve of a rolling-shutter plate is "
                   "bent: poses and points soak up the row-to-row skew, so putting the readout on top counts it "
                   "twice. This re-solves poses and points with the readout in the projection (row time from the "
                   "observed row, pose slerped between frames) and moves the result back onto the input world, so "
                   "layers placed there stay put. readout 'free' solves it (starting at readout_ms), 'held' keeps "
                   "readout_ms; readout_ms 0 = the sensor preset's spec. The output camera carries the readout in "
                   "extensions.atlas_vfx_rs_solve, which Sensor RS Maps' preset rule uses. profile_ms re-solves at "
                   "fixed readouts to show how deep the valley is (a flat one = the move is too gentle to measure). "
                   "Points come from the camera's own COLMAP solve, or solve_folder (+ points_camera when this "
                   "camera is an export of that solve).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "solve_folder": ("STRING", {"default": "", "tooltip": "pycolmap model folder; blank = the camera's "
                                            "own solve (source.path)"}),
                "points_camera": ("STRING", {"default": "", "tooltip": "shot camera JSON imported from that solve "
                                             "(e.g. Measure & Scale's output) whose world the points go into; blank = "
                                             "this camera"}),
                "readout": (["free", "held"], {"default": "free"}),
                "readout_ms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                         "tooltip": "start value (free) or the value (held); 0 = the sensor "
                                                    "preset's spec"}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "0 = camera's first"}),
                "last_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "0 = camera's last; the "
                                       "frames must be contiguous"}),
                "max_points": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "0 = all"}),
                "max_iter": ("INT", {"default": 100, "min": 5, "max": 500}),
                "profile_ms": ("STRING", {"default": "", "tooltip": "readouts to re-solve at, e.g. "
                                          "0,10,20,30.9,40 (ms); blank = no profile"}),
                "profile_points": ("INT", {"default": 8000, "min": 500, "max": 1000000}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, shot_camera, solve_folder, points_camera, readout, readout_ms, first_frame, last_frame, max_points,
            max_iter, profile_ms, profile_points, name, out_dir):
        cam = sc.ShotCamera(shot_camera)
        held = readout == "held"
        profile = parse_profile(profile_ms)
        name = pl.safe_name(name.strip() or f"{cam.name}_rs")
        plate_path = Path(cam.data["plate"]["path"]) if cam.data["plate"]["path"] else Path("plate")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / plate_path.stem / "camera")
        start_ms, start_from = start_readout(cam.data, readout_ms, held)
        warnings = []
        dist = (cam.data.get("lens") or {}).get("distortion_model", "none")
        if dist not in ("none", None):
            warnings.append(f"camera lens distortion '{dist}' is ignored: the RS solve is pinhole")
        if "atlas_vfx_rs_solve" in (cam.data.get("extensions") or {}):
            warnings.append("input camera is already an RS solve; solving again from its poses")
        for w in warnings:
            log.warning("atlas-vfx RS Solve (%s): %s", cam.name, w)
        pts, note, pcam = ms.points_in_camera(cam, solve_folder, points_camera, max_reproj_px=POINTS_MAX_REPROJ_PX)
        frames = (first_frame or cam.frames[0], last_frame or cam.frames[-1])
        res = rs_solve.solve_shot(cam, pts, readout_ms=start_ms, solve_readout=not held, frames=frames,
                                  max_points=max_points or None, max_iter=max_iter, profile_ms=profile,
                                  profile_points=profile_points, name=name)
        rep, rows = res["report"], res["profile"]
        for w in rs_solve.world_warnings(rep):
            warnings.append(w)
            log.warning("atlas-vfx RS Solve (%s): %s", cam.name, w)
        solved = (rep["frames"][0], rep["frames"][1])
        rs_cam = res["camera"]
        if [rs_cam.frames[0], rs_cam.frames[-1]] != list(solved):
            # frames outside the solve would keep their GS poses under a camera that says "apply this readout":
            # RS Maps would put the readout on GS poses there (the skew counted twice), so they are dropped
            rs_cam = cio.crop_frames(rs_cam, solved)
            warnings.append(f"output cropped to the solved frames {solved[0]}-{solved[1]}")
            log.warning("atlas-vfx RS Solve (%s): %s", cam.name, warnings[-1])
        res["camera"] = rs_cam
        path = sc.save(rs_cam, out / f"{name}.shot_camera.json")
        full = {"camera": cam.name, "points": note, "points_camera": pcam.name, "readout": readout,
                "readout_start_ms": start_ms, "readout_start_from": start_from, "warnings": warnings,
                "solve": rep, "profile": rows}
        (out / f"{name}.rs_solve.json").write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.shot_camera.sidecar.json", "AtlasVFXRSSolve",
                         inputs={"camera": cam.name, "camera_source": cam.data.get("source"), "points": note,
                                 "points_camera": pcam.name, "readout": readout, "readout_ms": readout_ms,
                                 "readout_start_ms": start_ms, "readout_start_from": start_from,
                                 "frames": list(frames), "max_points": max_points, "max_iter": max_iter,
                                 "profile_ms": profile, "profile_points": profile_points},
                         outputs={"path": str(path), "report": str(out / f"{name}.rs_solve.json"),
                                  "readout_ms": rep["readout_ms"], "readout_solved": rep["readout_solved"],
                                  "reproj_input_gs": rep["reproj_input_gs"], "reproj_solved": rep["reproj_solved"],
                                  "message": rep["message"], "warnings": warnings})
        head = (f"{name}: readout {rep['readout_ms']:.2f} ms ({'solved' if not held else 'held'}), median "
                f"{rep['reproj_input_gs']['median_px']:.2f} -> {rep['reproj_solved']['median_px']:.2f} px, "
                f"{rep['message']}")
        return {"ui": {"text": [head]},
                "result": (res["camera"].data, float(rep["readout_ms"]), str(path), json.dumps(full, indent=1,
                                                                                               default=str))}


NODE_CLASS_MAPPINGS = {"AtlasVFXRSSolve": AtlasVFXRSSolve}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXRSSolve": "Atlas VFX RS Solve"}
