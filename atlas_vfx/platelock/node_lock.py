"""Atlas VFX Plate Lock: lock the best AI take to its plate (retime, re-shoot background, warp people, restack)."""
from __future__ import annotations

import json
import logging

import numpy as np

from . import _comfy
from .core import pipeline
from .node_qc import summary

log = logging.getLogger("atlas_vfx")


class AtlasVFXPlateLock:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "ATLAS_VFX_PLATELOCK_RESULT", "STRING")
    RETURN_NAMES = ("comp", "background", "background_valid", "result", "report")
    DESCRIPTION = ("Takes the AI take apart and puts it back on the plate's clock and camera. Each performer is "
                   "retimed, warped (correspondences, then an edge snap to the plate matte) and restacked by depth "
                   "over the AI background re-photographed through the plate camera with rolling shutter. "
                   "AI frames, masks and depth are resized to the plate if they differ (same aspect assumed). "
                   "Depth must be metric or inverse-metric, not a normalised preview; 'flat' treats the AI "
                   "background as a card, fine for a locked-off shot.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE",),
                "ai": ("IMAGE",),
                "plate_camera": ("ATLAS_VFX_PLATELOCK_TRACK",),
                "ai_camera": ("ATLAS_VFX_PLATELOCK_TRACK",),
                "people": ("ATLAS_VFX_PLATELOCK_PEOPLE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 1e-6, "max": 1e6, "step": 0.001,
                                    "tooltip": "AI solve units -> plate units (e.g. performer height ratio)"}),
                "depth_mode": (["flat", "metric", "inverse"], {"default": "flat"}),
                "flat_distance": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1e5, "step": 0.1}),
                "shutter_deg": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "motion_blur_samples": ("INT", {"default": 1, "min": 1, "max": 32}),
                "neighbours": ("INT", {"default": 2, "min": 0, "max": 24,
                                       "tooltip": "AI frames either side tried first to fill the background"}),
                "edge_snap": ("BOOLEAN", {"default": True}),
                "snap_max_px": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 64.0, "step": 0.5}),
            },
            "optional": {
                "ai_depth": ("IMAGE", {"tooltip": "per-frame depth of the AI take, channel 0"}),
                "plate_matte": ("MASK", {"tooltip": "screen key of the plate (Screen Key node or a Nuke pull)"}),
            },
        }

    def run(self, plate, ai, plate_camera, ai_camera, people, scale, depth_mode, flat_distance, shutter_deg,
            motion_blur_samples, neighbours, edge_snap, snap_max_px, ai_depth=None, plate_matte=None):
        P = _comfy.image_np(plate)
        A = _comfy.image_np(ai)
        N, H, W = P.shape[:3]
        M = A.shape[0]
        if (A.shape[1], A.shape[2]) != (H, W):
            ar_p, ar_a = W / H, A.shape[2] / A.shape[1]
            if abs(ar_p - ar_a) / ar_p > 0.01:
                log.warning("platelock: AI aspect %.4f differs from plate %.4f; resizing anyway (crop first)",
                            ar_a, ar_p)
            A = _comfy.resize_stack(A, W, H)
        ai_cam = ai_camera if (ai_camera.intr.width, ai_camera.intr.height) == (W, H) else ai_camera.scaled(W, H)
        plate_cam = plate_camera if (plate_camera.intr.width, plate_camera.intr.height) == (W, H) \
            else plate_camera.scaled(W, H)

        if depth_mode == "flat" or ai_depth is None:
            if depth_mode != "flat":
                log.warning("platelock: no ai_depth connected, using flat depth %.2f", flat_distance)
            D = np.full((M, H, W), flat_distance, np.float32)
        else:
            D = _comfy.mask_np(ai_depth)
            D = _comfy.resize_stack(D, W, H)
            if depth_mode == "inverse":
                D = 1.0 / np.maximum(D, 1e-6)
        if D.shape[0] < M:
            raise ValueError(f"ai_depth has {D.shape[0]} frames, the AI take has {M}")

        ppl = []
        for p in people:
            ppl.append(pipeline.PersonInput(
                p.name, _comfy.resize_stack(p.plate_masks, W, H), _comfy.resize_stack(p.ai_masks, W, H),
                p.depth, p.correspondences))
            if ppl[-1].plate_masks.shape[0] != N or ppl[-1].ai_masks.shape[0] != M:
                raise ValueError(f"{p.name}: mask frame counts must match plate ({N}) and AI ({M})")

        key = None
        if plate_matte is not None:
            key = _comfy.resize_stack(_comfy.mask_np(plate_matte), W, H)

        cfg = pipeline.LockConfig(shutter=shutter_deg / 360.0, mb_samples=motion_blur_samples,
                                  neighbours=neighbours, snap=edge_snap, snap_max_shift=snap_max_px)
        try:
            import comfy.utils  # type: ignore
            bar = comfy.utils.ProgressBar(100)

            def progress(stage, frac):
                bar.update_absolute(int(frac * 100))
        except ImportError:
            progress = None
        res = pipeline.lock_shot(P, A, D, plate_cam, ai_cam, ppl, scale, cfg, plate_key=key, progress=progress)
        text = "before\n" + summary(res.report_before) + "\n\nafter\n" + summary(res.report_after)
        res.inputs = {"plate_size": [W, H], "ai_frames": M, "scale": scale, "depth_mode": depth_mode,
                      "config": cfg.__dict__}
        return (_comfy.to_image(res.comp), _comfy.to_image(res.bg), _comfy.to_mask(res.bg_valid), res, text)


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLock": AtlasVFXPlateLock}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLock": "Atlas VFX Plate Lock"}
