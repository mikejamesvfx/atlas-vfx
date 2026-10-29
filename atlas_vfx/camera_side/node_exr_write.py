"""Atlas VFX EXR Write (catalogue 23): an IMAGE batch (+ optional alpha) -> RGBA half EXR sequence in ACEScg with the
colourspace in the header, Nuke frame numbers, and a sidecar. Any input colourspace is converted by a named OCIO
transform; nothing is guessed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import exr_io
from . import plate as pl


class AtlasVFXEXRWrite:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pattern",)
    DESCRIPTION = ("Writes <name>.<frame>.exr: RGBA half, zip, in `colorspace` (default ACEScg), converted from "
                   "`input_colorspace` with the OCIO config. Alpha from the mask (1 = opaque) or 1; premultiplies "
                   "unless the input already is. colorInteropID + atlas_vfx:ocio_colorspace in every header.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "first_frame": ("INT", {"default": 1130, "min": 0, "max": 1_000_000}),
                "input_colorspace": ("STRING", {"default": "ACEScg",
                                                "tooltip": "OCIO name of the IMAGE's colourspace, e.g. ACEScg or "
                                                           "'sRGB Encoded Rec.709 (sRGB)' for generator output"}),
                "colorspace": ("STRING", {"default": "ACEScg"}),
                "input_premultiplied": ("BOOLEAN", {"default": False}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "name": ("STRING", {"default": "layer"}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {"mask": ("MASK",)},
        }

    def run(self, images, first_frame, input_colorspace, colorspace, input_premultiplied, ocio_config, name, out_dir,
            mask=None):
        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / "exr_write" / name)
        proc = None
        if input_colorspace != colorspace:
            proc = pl.ocio_config(ocio_config).getProcessor(input_colorspace, colorspace).getDefaultCPUProcessor()
        imgs = images.cpu().numpy()
        masks = mask.cpu().numpy() if mask is not None else None
        for i, img in enumerate(imgs):
            rgb = np.ascontiguousarray(img[..., :3].astype(np.float32))
            if masks is not None:
                a = masks[min(i, len(masks) - 1)].astype(np.float32)
            elif img.shape[-1] == 4:
                a = img[..., 3].astype(np.float32)
            else:
                a = np.ones(rgb.shape[:2], np.float32)
            if proc is not None:
                if input_premultiplied:                       # colour conversions act on straight colour
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rgb = np.ascontiguousarray(np.where(a[..., None] > 0, rgb / a[..., None], 0).astype(np.float32))
                proc.applyRGB(rgb)
                rgb = rgb * a[..., None]
            elif not input_premultiplied:
                rgb = rgb * a[..., None]
            exr_io.write_rgba(out / f"{name}.{first_frame + i:04d}.exr", np.dstack([rgb, a]), colorspace=colorspace,
                              attrs={"layer": name, "frame": first_frame + i, "source_colorspace": input_colorspace})
        pattern = str(out / f"{name}.%04d.exr")
        last = first_frame + len(imgs) - 1
        check = exr_io.check_sequence(pattern, first_frame, last, colorspace=colorspace)
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXEXRWrite",
                         inputs={"frames": [first_frame, last], "input_colorspace": input_colorspace,
                                 "input_premultiplied": input_premultiplied, "has_mask": mask is not None,
                                 "ocio_config": ocio_config},
                         outputs=check)
        return {"ui": {"text": [json.dumps(check)]}, "result": (pattern,)}


NODE_CLASS_MAPPINGS = {"AtlasVFXEXRWrite": AtlasVFXEXRWrite}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXEXRWrite": "Atlas VFX EXR Write"}
