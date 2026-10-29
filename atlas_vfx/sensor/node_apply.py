"""Atlas VFX Sensor Apply: plate camera onto an LTX render (apply), or off the plate on the way into LTX (remove)."""
from __future__ import annotations

import json

from atlas_vfx.camera_side import plate as pl

from . import jobs
from ._node_util import CATEGORY, progress_bar


def _spaces():
    base = [jobs.DISPLAY_VIEW, jobs.WARP_ONLY, "ACEScg"]
    try:
        cfg = pl.ocio_config(str(pl.DEFAULT_OCIO))
        return base + [c.getName() for c in cfg.getColorSpaces() if c.getName() not in base]
    except Exception:  # noqa: BLE001 - no OCIO here: the explicit choices still work
        return base


class AtlasVFXSensorApply:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    DESCRIPTION = ("apply camera: an LTX render (or any clean footage) gets the plate's sensor, in order vignette "
                   "(linear ACEScg) -> lateral CA + distortion -> rolling shutter. remove camera: the plate goes to "
                   "clean space, de-RS -> undistort -> devignette, for LTX Prep. Connect ltx_prep and the render's "
                   "frames and crop windows come from it (a cropped or work-size render gets the matching part of "
                   "the plate-resolution maps); otherwise first_frame + window. Colour is explicit: the display view "
                   "LTX Prep used (inverted to ACEScg for the vignette), ACEScg, any OCIO colourspace, or warp only.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["apply camera", "remove camera"], {"default": "apply camera"}),
                "rs_folder": ("STRING", {"default": "", "tooltip": "Sensor RS Maps folder (blank = no RS)"}),
                "lens_folder": ("STRING", {"default": "", "tooltip": "Sensor Lens Maps folder (blank = no lens)"}),
                "first_frame": ("INT", {"default": 1130, "min": 0, "max": 999999,
                                        "tooltip": "Nuke frame of image 0 (ignored when ltx_prep is connected)"}),
                "window": ("STRING", {"default": "", "tooltip": "x0,y0,w,h plate px the images show; blank = whole "
                                      "plate (ignored when ltx_prep is connected)"}),
                "colorspace": (_spaces(), {"default": jobs.DISPLAY_VIEW}),
                "use_vignette": ("BOOLEAN", {"default": True}),
                "use_lens": ("BOOLEAN", {"default": True}),
                "use_rs": ("BOOLEAN", {"default": True}),
                "edge": (["edge", "black"], {"default": "edge"}),
            },
            "optional": {
                "ltx_prep": ("ATLAS_VFX_LTX_PREP",),
            },
        }

    def run(self, images, direction, rs_folder, lens_folder, first_frame, window, colorspace, use_vignette,
            use_lens, use_rs, edge, ltx_prep=None):
        import numpy as np
        import torch
        maps = jobs.Maps(pl.clean_path(rs_folder) or None, pl.clean_path(lens_folder) or None)
        imgs = images.cpu().numpy().astype(np.float32)
        if ltx_prep:
            f0, wins = jobs.frames_windows_from_prep(ltx_prep, len(imgs))
            imgs = imgs[: ltx_prep["frames"][1] - ltx_prep["frames"][0] + 1]
            ocio_cfg = ltx_prep["plate_ref"]["ocio_config"]
        else:
            f0, w = first_frame, jobs.parse_window(window)
            wins = [w] * len(imgs) if w else None
            ocio_cfg = str(pl.DEFAULT_OCIO)
        out, rep = jobs.apply_batch(imgs, maps, direction=direction, first_frame=f0, windows=wins,
                                    colorspace=colorspace, ocio_config=ocio_cfg, use_vignette=use_vignette,
                                    use_lens=use_lens, use_rs=use_rs, edge=edge, progress=progress_bar())
        rep["frames_from"] = "ltx_prep" if ltx_prep else "first_frame/window"
        return (torch.from_numpy(out), json.dumps(rep, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXSensorApply": AtlasVFXSensorApply}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSensorApply": "Atlas VFX Sensor Apply"}
