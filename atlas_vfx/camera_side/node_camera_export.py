"""Atlas VFX Camera Export (catalogue 25): shot camera -> Nuke .chan (metres, ZXY) + Nuke camera settings JSON +
time-sampled USD camera (.usda)."""
from __future__ import annotations

import json
from pathlib import Path

from atlas_vfx.contracts import shot_camera as sc

from . import camera_export as ce
from . import plate as pl


class AtlasVFXCameraExport:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("chan", "usda", "nuke_camera_json")
    DESCRIPTION = ("Writes <name>.chan (frame tx ty tz rx ry rz focal; metres x units_per_m; import on a Camera with "
                   "rot_order ZXY), <name>.nuke_camera.json (apertures, win_translate, rot order, frames) and "
                   "<name>.usda (Y up, metres, one transform sample per frame).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "units_per_m": ("FLOAT", {"default": 1.0, "min": 1e-4, "max": 1e4, "step": 0.01}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, shot_camera, units_per_m, name, out_dir):
        cam = sc.ShotCamera(shot_camera)
        plate_path = Path(cam.data["plate"]["path"]) if cam.data["plate"]["path"] else Path("plate")
        name = pl.safe_name(name.strip() or cam.name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / plate_path.stem / "camera_export")
        res = ce.export(cam, out, name, units_per_m)
        pl.write_sidecar(out / f"{name}.export.sidecar.json", "AtlasVFXCameraExport",
                         inputs={"camera": cam.name, "units_per_m": units_per_m}, outputs=res)
        return {"ui": {"text": [json.dumps(res["settings"])]},
                "result": (res["chan"], res["usda"], res["nuke_camera"])}


NODE_CLASS_MAPPINGS = {"AtlasVFXCameraExport": AtlasVFXCameraExport}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXCameraExport": "Atlas VFX Camera Export"}
