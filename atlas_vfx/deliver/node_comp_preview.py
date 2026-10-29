"""Atlas VFX Comp Preview: plate + layer EXR folders (back to front) -> a movie in ComfyUI (IMAGE frames + fps)."""
from __future__ import annotations

import json

from . import comp_preview as cpv


class AtlasVFXCompPreview:
    CATEGORY = "Atlas VFX/Deliver"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "fps", "report")
    DESCRIPTION = ("Comps the plate with every layer EXR sequence over it, back to front, in ACEScg, then through the "
                   "OCIO display view (ACES 2.0 SDR) with an optional frame burn-in. Feed frames + fps into Create "
                   "Video and Save Video to watch the shot in ComfyUI. A preview, not the delivery: Nuke builds that.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "layer_folders": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "one EXR folder per line, BACK to FRONT"}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 16}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 50}),
                "burn_in": ("BOOLEAN", {"default": True}),
                "label": ("STRING", {"default": ""}),
            }
        }

    def run(self, plate_ref, layer_folders, width, frame_step, burn_in, label):
        import torch
        try:
            import comfy.utils  # type: ignore
            bar = comfy.utils.ProgressBar(100)

            def progress(n, total):
                bar.update_absolute(int(100 * n / max(total, 1)))
        except ImportError:
            progress = None
        frames, report = cpv.render(plate_ref, cpv.parse_folders(layer_folders), width=width, step=frame_step,
                                    burn_in=burn_in, label=label.strip(), progress=progress)
        fps = report["fps"] / frame_step
        return (torch.from_numpy(frames), float(fps), json.dumps(report, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXCompPreview": AtlasVFXCompPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXCompPreview": "Atlas VFX Comp Preview"}
