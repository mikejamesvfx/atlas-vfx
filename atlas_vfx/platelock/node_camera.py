"""Atlas VFX Plate Lock Camera: a camera track for the plate or the AI take, with rolling shutter."""
from __future__ import annotations

import json

from . import _comfy, adapters
from .core.camera import CameraTrack, Intrinsics, readout_frames
from .core.seqio import focal_px


class AtlasVFXPlateLockCamera:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("ATLAS_VFX_PLATELOCK_TRACK", "STRING")
    RETURN_NAMES = ("camera", "info")
    DESCRIPTION = ("Builds the camera the lock works in. 'shot_camera' reads an atlas-vfx shot-camera@0.1 JSON "
                   "(Nuke axes, converted to OpenCV internally). 'platelock_json' reads a platelock.track/1 file "
                   "(any solver adapter can write one). 'locked_off' is a static camera from focal and sensor, the "
                   "common bluescreen case. Readout is the sensor's top-to-bottom time; 0 = global shutter "
                   "(use 0 for AI takes). Measure it off the plate rather than trusting a spec sheet.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "the sequence this camera belongs to (for size and frame count)"}),
                "mode": (["locked_off", "shot_camera", "platelock_json"], {"default": "locked_off"}),
                "path": ("STRING", {"default": "", "tooltip": "JSON path for shot_camera / platelock_json"}),
                "focal_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 2000.0, "step": 0.01}),
                "sensor_width_mm": ("FLOAT", {"default": 23.1, "min": 1.0, "max": 100.0, "step": 0.01,
                                              "tooltip": "ACTIVE sensor width for the delivered image (crop mode)"}),
                "readout_ms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "rs_anchor": (["centre", "top"], {"default": "centre",
                                                  "tooltip": "which row a solved pose represents"}),
            }
        }

    def run(self, frames, mode, path, focal_mm, sensor_width_mm, readout_ms, fps, rs_anchor):
        n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
        ro = readout_frames(readout_ms, fps)
        p = _comfy.clean_path(path)
        if mode == "shot_camera":
            tr = adapters.track_from_shot_camera(p, ro, rs_anchor)
        elif mode == "platelock_json":
            tr = CameraTrack.load(p).with_readout(ro, rs_anchor) if ro > 0 else CameraTrack.load(p)
        else:
            f = focal_px(focal_mm, sensor_width_mm, w)
            tr = adapters.static_track(Intrinsics.from_fov(w, h, f), n, ro).with_readout(ro, rs_anchor)
        if (tr.intr.width, tr.intr.height) != (w, h):
            tr = tr.scaled(w, h)
        if tr.n < n:
            raise ValueError(f"camera has {tr.n} frames but the sequence has {n}")
        info = {"mode": mode, "frames": tr.n, "width": w, "height": h, "fx": tr.intr.fx,
                "readout_frames": tr.readout, "rs_anchor": tr.rs_anchor}
        return (tr, json.dumps(info, indent=1))


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockCamera": AtlasVFXPlateLockCamera}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockCamera": "Atlas VFX Plate Lock Camera"}
