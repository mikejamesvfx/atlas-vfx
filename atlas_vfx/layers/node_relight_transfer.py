"""Atlas VFX Relight Transfer: the light of a low-res relit take (LTX relight IC-LoRA) moved onto the full-res element,
keeping the element's own pixels. See light_transfer.py."""
from __future__ import annotations

import json

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import light_transfer as lt
from . import relight as rl


class AtlasVFXRelightTransfer:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "gain", "info")
    DESCRIPTION = ("Takes the light from a relit take and keeps the pixels of the element. relit is the same framing at "
                   "any size: an LTX relight IC-LoRA run (2.3 Relight at 1280x704, 2.5 Day-to-Night / Golden-Hour at "
                   "FullHD), as LTX Return gives it or raw. The change of light, log2(relit / element), is measured at "
                   "analysis_width, blurred inside the matte only and brought to full res by a guided filter on the "
                   "element (plus MoGe depth / normals when wired), so the element keeps every pixel. mode sh also "
                   "fits the light as spherical harmonics on the normals and re-shades at the normals' sharpness "
                   "(wire Atlas VFX Normals from MoGe-3 run at element res; a shot camera keeps the fit in world "
                   "space). gain is the log2 gain at full res for QC. Outside the matte the element passes through.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "element": ("IMAGE", {"tooltip": "full-res element, same framing as relit"}),
                "matte": ("MASK",),
                "relit": ("IMAGE", {"tooltip": "the relit take, any size, same framing"}),
                "encoding": (["linear", "srgb"], {"default": "linear",
                                                  "tooltip": "how element and relit are encoded; output matches"}),
                "mode": (list(lt.MODES), {"default": "guided"}),
                "analysis_width": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 16,
                                           "tooltip": "width the gain is measured at; the LoRA's native width. "
                                                      "0 = the relit take's width"}),
                "blur_px": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                      "tooltip": "gain blur at analysis size, inside the matte only"}),
                "max_stops": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 10.0, "step": 0.25}),
                "temporal": ("INT", {"default": 1, "min": 0, "max": 12,
                                     "tooltip": "average the gain (and SH fit) over +-N frames"}),
                "eps": ("FLOAT", {"default": 0.01, "min": 1e-5, "max": 1.0, "step": 0.001,
                                  "tooltip": "guided filter smoothness (stops squared)"}),
                "geo_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05,
                                         "tooltip": "weight of depth / normals in the guide"}),
                "min_r2": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "sh: a frame whose fit explains less falls back to guided"}),
            },
            "optional": {
                "normals": ("IMAGE", {"tooltip": "Atlas VFX Normals output (OpenGL, encoded 0-1); needed for sh"}),
                "moge_geometry": ("MOGE_GEOMETRY", {"tooltip": "depth as an extra guide"}),
                "shot_camera": ("ATLAS_TRACKED_CAMERA", {"tooltip": "sh fit in world space"}),
                "first_frame": ("INT", {"default": 1001, "min": 0, "max": 1_000_000,
                                        "tooltip": "shot frame of the first image (for shot_camera)"}),
            },
        }

    def run(self, element, matte, relit, encoding, mode, analysis_width, blur_px, max_stops, temporal, eps,
            geo_weight, min_r2, normals=None, moge_geometry=None, shot_camera=None, first_frame=1001):
        import torch

        el = element.float().cpu().numpy()[..., :3]
        rel = relit.float().cpu().numpy()[..., :3]
        if encoding == "srgb":
            el, rel = rl.srgb_to_lin(el), rl.srgb_to_lin(rel)
        m = matte.float().cpu().numpy()
        if m.ndim == 2:
            m = m[None]
        n = lt.decode_normals(normals.float().cpu().numpy()) if normals is not None else None
        depth = None
        if moge_geometry is not None and "depth" in moge_geometry:
            depth = moge_geometry["depth"].float().cpu().numpy()
        rots = None
        if shot_camera is not None:
            cam = sc.ShotCamera(shot_camera)
            try:
                rots = np.stack([cam.cam_to_world(first_frame + t)[:3, :3] for t in range(len(el))]).astype(np.float32)
            except KeyError as e:
                raise ValueError(f"shot_camera: {e.args[0]} (first_frame {first_frame}, {len(el)} images)") from None
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(len(el))
        except ImportError:  # outside ComfyUI (tests)
            pass
        out, gain, rep = lt.transfer(el, rel, m, mode=mode, analysis_width=analysis_width, blur_px=blur_px,
                                     max_stops=max_stops, temporal=temporal, eps=eps, normals=n, depth=depth,
                                     cam_rotations=rots, geo_weight=geo_weight, min_r2=min_r2,
                                     progress=(lambda t: pbar.update(1)) if pbar else None)
        if encoding == "srgb":
            out = rl.lin_to_srgb(out)
        if rots is not None:
            rep["shot_frames"] = [int(first_frame), int(first_frame + len(el) - 1)]
        return (torch.from_numpy(np.ascontiguousarray(out)), torch.from_numpy(np.ascontiguousarray(gain)),
                json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXRelightTransfer": AtlasVFXRelightTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXRelightTransfer": "Atlas VFX Relight Transfer"}
