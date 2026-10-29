"""Atlas VFX Sensor Lens Maps: undistort / redistort STMaps per channel (lateral CA) and a vignette gain."""
from __future__ import annotations

import json

from atlas_vfx.contracts import shot_camera as sc

from . import jobs
from ._node_util import CATEGORY, grid_image, work_dir


class AtlasVFXSensorLensMaps:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_VFX_SENSOR_LENS", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("sensor_lens", "lens_folder", "preview", "report")
    DESCRIPTION = ("Brown-Conrady lens (k1 k2 k3 p1 p2, OpenCV convention) from the shot camera's lens block, typed "
                   "values as override, plus lateral CA (R and B radial scale vs G) and vignette (linear ACEScg gain, "
                   "r = 1 in the corners). Writes undistort_{r,g,b} / redistort_{r,g,b} STMaps, vignette.exr and "
                   "Nuke Reads. Feed sensor_lens into Sensor RS Maps so row times use sensor (distorted) rows, and "
                   "lens_folder into Sensor Apply.")

    @classmethod
    def INPUT_TYPES(cls):
        f = {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.0001}
        return {
            "required": {
                "lens_source": (["camera", "camera + typed", "typed"], {"default": "camera"}),
                "k1": ("FLOAT", f), "k2": ("FLOAT", f), "k3": ("FLOAT", f), "p1": ("FLOAT", f), "p2": ("FLOAT", f),
                "ca_r": ("FLOAT", {"default": 0.0, "min": -0.01, "max": 0.01, "step": 0.00005,
                                   "tooltip": "R radial scale - 1 relative to G (0.0004 = 0.04 %)"}),
                "ca_b": ("FLOAT", {"default": 0.0, "min": -0.01, "max": 0.01, "step": 0.00005}),
                "vig_1": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "vig_2": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "vig_3": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "map_scale": ("FLOAT", {"default": 1.0, "min": 0.125, "max": 1.0, "step": 0.125}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "plate_ref": ("ATLAS_PLATE_SEQ",),
            },
        }

    def run(self, lens_source, k1, k2, k3, p1, p2, ca_r, ca_b, vig_1, vig_2, vig_3, map_scale, out_dir,
            shot_camera=None, plate_ref=None):
        import torch

        from . import apply as ap
        cam = sc.ShotCamera(shot_camera) if shot_camera else None
        if cam is not None:
            size = (cam.data["plate"]["width"], cam.data["plate"]["height"])
        elif plate_ref:
            size = (plate_ref["width"], plate_ref["height"])
        else:
            raise ValueError("Sensor Lens Maps: connect a shot_camera (or a plate_ref for typed lenses)")
        lens, lens_from = jobs.lens_from_inputs(cam, lens_source, {"k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2},
                                                size)
        out = work_dir(cam.data if cam else {"plate": {"path": plate_ref["path"]}},
                       f"{cam.name if cam else 'typed'}_lens", out_dir)
        rep = jobs.make_lens_maps(lens, size, out, ca_r=ca_r, ca_b=ca_b, v1=vig_1, v2=vig_2, v3=vig_3,
                                  scale=map_scale, lens_from=lens_from)
        maps = jobs.Maps(lens_folder=str(out))
        g = grid_image(1024, int(round(1024 * size[1] / size[0])))
        prev = ap.apply(g, lens=maps.lens("redistort"), gain=maps.gain())
        sl = {"lens": lens.to_dict(), "plate_size": list(size), "folder": str(out), "lens_from": lens_from}
        return (sl, str(out), torch.from_numpy(prev[None]), json.dumps(rep, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXSensorLensMaps": AtlasVFXSensorLensMaps}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSensorLensMaps": "Atlas VFX Sensor Lens Maps"}
