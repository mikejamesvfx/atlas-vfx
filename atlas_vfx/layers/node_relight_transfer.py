"""Atlas VFX Relight Transfer: the light of a low-res relit take (LTX relight IC-LoRA) moved onto the full-res element,
keeping the element's own pixels. See light_transfer.py."""
from __future__ import annotations

import json

import numpy as np

from atlas_vfx.contracts import shot_camera as sc

from . import element_colour as ec
from . import light_transfer as lt
from . import ltx_integration as li
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
                "encoding": (["linear", "srgb", ec.VIEW_INVERSE], {"default": "linear", "tooltip": (
                    "how element and relit are encoded; output matches. Display pictures from LTX: the inverse "
                    "display view (the ACES 2.0 SDR view LTX Prep used). srgb = the plain sRGB curve")}),
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
                "subsample": ("INT", {"default": 2, "min": 1, "max": 16,
                                      "tooltip": "fast guided filter: solve at 1/N res, apply at full res. 2 is "
                                                 "within 0.01 stop of exact at 4K; 4 is faster, ~0.2 stop at edges; "
                                                 "1 = exact"}),
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
                "relit_mirrored": ("BOOLEAN", {"default": False,
                                               "tooltip": "the relit take was rendered on a left-right flipped "
                                                          "reference (the sun-on-the-left trick): flip it back here. "
                                                          "Leave off for LTX Return output (Return already flips)"}),
                "holdout": ("MASK", {"tooltip": "roto over what the relight invented rather than lit (an object "
                                                "that is not in the plate, a border LTX drew): its light is "
                                                "ignored and filled from around it. Any size, same framing"}),
                "lift_limit": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.25,
                                         "tooltip": "where the relit take lifts the element more than this (stops), "
                                                    "the plate is too dark to carry it (noise x 2^lift): fade to the "
                                                    "relit take's own pixels there (soft but clean). 3 keeps the "
                                                    "plate's texture on a dusk plate lifted 2-3 stops. 0 = off"}),
                "holdout_grow": ("INT", {"default": 2, "min": 0, "max": 64,
                                         "tooltip": "grow the holdout by this many analysis px"}),
            },
        }

    def run(self, element, matte, relit, encoding, mode, analysis_width, blur_px, max_stops, temporal, eps,
            subsample, geo_weight, min_r2, normals=None, moge_geometry=None, shot_camera=None, first_frame=1001,
            relit_mirrored=False, holdout=None, lift_limit=3.0, holdout_grow=2):
        import torch

        el = element.float().cpu().numpy()[..., :3]
        rel = relit.float().cpu().numpy()[..., :3]
        if relit_mirrored:
            rel = li.mirror(rel)
        if encoding == "srgb":
            el, rel = rl.srgb_to_lin(el), rl.srgb_to_lin(rel)
        elif encoding == ec.VIEW_INVERSE:
            el, rel = ec.to_acescg(el, ""), ec.to_acescg(rel, "")
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
                                     max_stops=max_stops, temporal=temporal, eps=eps, subsample=subsample, normals=n, depth=depth,
                                     cam_rotations=rots, geo_weight=geo_weight, min_r2=min_r2,
                                     holdout=None if holdout is None else holdout.float().cpu().numpy(),
                                     holdout_grow=holdout_grow, lift_limit=lift_limit,
                                     progress=(lambda t: pbar.update(1)) if pbar else None)
        if encoding == "srgb":
            out = rl.lin_to_srgb(out)
        elif encoding == ec.VIEW_INVERSE:
            out = np.ascontiguousarray(out, np.float32).copy()
            ec.from_acescg_processor("", ec.VIEW_INVERSE).applyRGB(out)
            out = np.clip(out, 0.0, 1.0)
        rep["relit_mirrored"] = bool(relit_mirrored)
        if rots is not None:
            rep["shot_frames"] = [int(first_frame), int(first_frame + len(el) - 1)]
        return (torch.from_numpy(np.ascontiguousarray(out)), torch.from_numpy(np.ascontiguousarray(gain)),
                json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AtlasVFXRelightTransfer": AtlasVFXRelightTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXRelightTransfer": "Atlas VFX Relight Transfer"}
