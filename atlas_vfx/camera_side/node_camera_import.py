"""Atlas VFX Camera Import (catalogue 5): Nuke JSON / .chan / Alembic .abc / COLMAP model / shot camera -> ATLAS_TRACKED_CAMERA."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import camera_io as cio
from . import plate as pl
from .sensors import PRESETS, preset_for_body, sensor_block

ROT_ORDERS = ["ZXY", "XYZ", "XZY", "YXZ", "YZX", "ZYX"]
FIRST_FRAME = {"off": None, "to 1": 1, "to 1001": 1001, "to plate start": "plate"}


def _source_hash(p: Path) -> dict:
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in (".bin", ".txt"))
    out = {}
    for f in files[:20]:
        out[str(f.relative_to(p) if p.is_dir() else f.name)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


class AtlasVFXCameraImport:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_TRACKED_CAMERA", "STRING", "STRING")
    RETURN_NAMES = ("shot_camera", "path", "summary")
    DESCRIPTION = ("Imports a camera into shot-camera@0.1 (metres, Y up, -Z forward). Sensor comes from the preset, "
                   "never from Nuke's vaperture. Alembic: any app's animated camera (Nuke, Maya, Houdini, Blender), "
                   "frame = sample time x plate fps, whole parent chain, focal kept as horizontal FOV. COLMAP models: image numbers are plate indices, axes converted, "
                   "intrinsics scaled from proxy to plate. Saves <name>.shot_camera.json + sidecar.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "", "tooltip": "Nuke camera JSON / .chan / .abc / COLMAP "
                                                                     "folder / shot camera JSON; ~ expands"}),
                "kind": (["auto", "nuke_json", "nuke_chan", "abc", "colmap", "shot_camera"], {"default": "auto"}),
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "sensor_preset": (["auto", *PRESETS, "custom"], {"default": "auto"}),
                "sensor_width_mm": ("FLOAT", {"default": 23.5, "min": 1.0, "max": 100.0, "step": 0.01}),
                "sensor_height_mm": ("FLOAT", {"default": 12.39, "min": 1.0, "max": 100.0, "step": 0.01}),
                "m_per_unit": ("FLOAT", {"default": 1.0, "min": 1e-6, "max": 1e6, "step": 0.0001,
                                         "tooltip": "metres per source unit (1 = leave unscaled)"}),
                "rot_order": (ROT_ORDERS, {"default": "ZXY", "tooltip": ".chan only; Nuke JSON carries its own"}),
                "chan_focal_mm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5000.0, "step": 0.001,
                                            "tooltip": ".chan without a focal column only"}),
                "crop_to_plate": ("BOOLEAN", {"default": True,
                                              "tooltip": "keep only the plate_ref's frames (nuke_first-nuke_last)"}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "offset_first_frame": (list(FIRST_FRAME), {
                    "default": "off",
                    "tooltip": "renumber the camera so its first frame lands here (e.g. a Maya/Blender camera at 1 "
                               "onto a 1001 plate, or a 1001 camera onto 1). 'plate start' is the plate file's first "
                               "frame (frame_offset), not the cut. The camera must start where the plate starts. "
                               "Applied before crop_to_plate."}),
                "abc_camera": ("STRING", {"default": "", "tooltip": ".abc only: camera name or path when the file "
                                                                     "has several (e.g. AVFX_Camera)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, source_path, **kw):
        """Re-run when the source changes on disk (file, or any file in a COLMAP folder)."""
        p = Path(pl.clean_path(source_path))
        try:
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            return ";".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files[:50])
        except OSError:
            return float("nan")

    def run(self, source_path, kind, plate_ref, sensor_preset, sensor_width_mm, sensor_height_mm, m_per_unit,
            rot_order, chan_focal_mm, crop_to_plate, name, out_dir, offset_first_frame="off", abc_camera=""):
        src = Path(pl.clean_path(source_path))
        if sensor_preset == "auto":
            sensor_preset = preset_for_body(plate_ref.get("camera_body"), plate_ref["width"], plate_ref["height"])
            if sensor_preset is None:
                raise ValueError(f"no sensor preset for {plate_ref.get('camera_body')} at "
                                 f"{plate_ref['width']}x{plate_ref['height']}; pick one or use custom")
        sensor = sensor_block(sensor_preset, sensor_width_mm, sensor_height_mm)
        rng = (plate_ref["nuke_first"], plate_ref["nuke_last"]) if crop_to_plate else None
        default = f"{Path(plate_ref['path']).stem}_{src.stem}" + (f"_{rng[0]}-{rng[1]}" if rng else "")
        name = pl.safe_name(name.strip() or default)
        fps = plate_ref["fps"][0] / plate_ref["fps"][1]
        first_to = FIRST_FRAME.get(offset_first_frame)
        if first_to == "plate":
            first_to = plate_ref["frame_offset"]     # first frame of the plate file, not of the cut (nuke_first)
        cam = cio.import_camera(src, kind, plate=cio.plate_block(plate_ref), sensor=sensor, m_per_unit=m_per_unit,
                                rot_order=rot_order, focal_mm=chan_focal_mm or None, name=name, frames=rng,
                                frame_offset=plate_ref["frame_offset"], fps=fps, camera=abc_camera.strip() or None,
                                first_frame_to=first_to)
        shift = cam.data.get("extensions", {}).get("atlas_vfx_import", {}).get("frame_shift")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / Path(plate_ref["path"]).stem / "camera")
        path = cio.sc.save(cam, out / f"{name}.shot_camera.json")
        pl.write_sidecar(out / f"{name}.shot_camera.sidecar.json", "AtlasVFXCameraImport",
                         inputs={"source_path": str(src), "kind": kind, "source_sha256": _source_hash(src),
                                 "plate_ref": plate_ref, "sensor": sensor, "m_per_unit": m_per_unit,
                                 "rot_order": rot_order, "chan_focal_mm": chan_focal_mm, "crop": rng,
                                 "abc_camera": abc_camera, "offset_first_frame": offset_first_frame},
                         outputs={"path": str(path), "frames": [cam.frames[0], cam.frames[-1]]})
        k = cam.intrinsics(cam.frames[0])
        summary = (f"{name}: {len(cam.frames)} frames {cam.frames[0]}-{cam.frames[-1]}, focal {k.focal_mm:.3f} mm, "
                   f"sensor {sensor['width_mm']} x {sensor['height_mm']} mm, scale {cam.data['scale']}, "
                   f"source {cam.data['source'].get('solver')}"
                   + (f", renumbered {shift['delta']:+d} (source started at {shift['source_first']})" if shift else ""))
        return (cam.data, str(path), summary)


NODE_CLASS_MAPPINGS = {"AtlasVFXCameraImport": AtlasVFXCameraImport}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXCameraImport": "Atlas VFX Camera Import"}
