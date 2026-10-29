"""Atlas VFX Sensor RS Maps: per-frame rolling-shutter STMaps (apply and remove) from the shot camera + plate depth."""
from __future__ import annotations

import json
import logging

from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import jobs
from . import lens as ln
from . import maps_io as mio
from ._node_util import CATEGORY, map_preview, progress_bar, work_dir

log = logging.getLogger("atlas_vfx")

class AtlasVFXSensorRSMaps:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("rs_folder", "preview", "report")
    DESCRIPTION = ("Rolling shutter as per-frame STMaps from the tracked plate: each sensor row t(y) = t_frame + "
                   "(y/H - 0.5) * readout. Every plate pixel is lifted with Plate Depth's z (metres) through the pose "
                   "at its row's time and re-photographed at the frame time. Writes rs.####.exr (apply RS to clean / "
                   "LTX footage) and ders.####.exr (take it off the plate before LTX Prep), Raw float ZIP, plus Nuke "
                   "Reads and a sidecar recording where the readout and lens came from and which source won. "
                   "readout_source 'preset rule' (default): chart measurement (readout_chart_ms) > this camera's "
                   "RS-aware solve > the sensor preset's spec; a Sensor Readout Solve value "
                   "(readout_measured_ms, a GS residual) is never used and is logged as rejected. readout_ms is "
                   "used only when the camera has no preset, or with 'typed'. 'measured if connected' is the old "
                   "mode (chart, else Readout Solve, else typed). Spec values: X-H2 8K 30.9 ms (DCI 4K HQ about "
                   "29); 4K sub-sampled 15.3; 4K/60 crop 13.5; BMPCC 6K 19.8.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "depth_folder": ("STRING", {"default": "", "tooltip": "Plate Depth output (depth.<frame>.exr)"}),
                "readout_ms": ("FLOAT", {"default": 30.9, "min": 0.0, "max": 100.0, "step": 0.1,
                                         "tooltip": "typed readout: used with readout_source = typed, when the "
                                                    "camera has no sensor preset, or (old mode) when nothing "
                                                    "measured is connected"}),
                "readout_source": (jobs.READOUT_SOURCES, {"default": jobs.PRESET_RULE,
                                   "tooltip": "preset rule: chart > RS-aware solve > preset spec, never a GS "
                                              "Readout Solve residual"}),
                "direction": (["both", "apply RS", "remove RS"], {"default": "both"}),
                "shutter_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                                            "tooltip": "0 = no per-row motion blur (default). > 0 writes blur "
                                                       "sub-sample maps that Sensor Apply averages"}),
                "blur_samples": ("INT", {"default": 5, "min": 2, "max": 16}),
                "map_scale": ("FLOAT", {"default": 1.0, "min": 0.125, "max": 1.0, "step": 0.125}),
                "iters": ("INT", {"default": 3, "min": 1, "max": 10}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "0 = camera's first"}),
                "last_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "0 = camera's last"}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "readout_measured_ms": ("FLOAT", {"forceInput": True,
                                                  "tooltip": "Sensor Readout Solve (deprecated): used only by "
                                                             "'measured if connected'"}),
                "sensor_lens": ("ATLAS_VFX_SENSOR_LENS",),
                "readout_chart_ms": ("FLOAT", {"forceInput": True,
                                               "tooltip": "readout measured on the chart (S1); wins every mode "
                                                          "but typed"}),
            },
        }

    def run(self, shot_camera, depth_folder, readout_ms, readout_source, direction, shutter_angle, blur_samples,
            map_scale, iters, first_frame, last_frame, out_dir, readout_measured_ms=None, sensor_lens=None,
            readout_chart_ms=None):
        import torch
        cam = sc.ShotCamera(shot_camera)
        dec = jobs.decide_readout(cam.data, readout_ms, readout_source, measured_ms=readout_measured_ms,
                                  chart_ms=readout_chart_ms)
        ro, ro_from = dec["readout_ms"], dec["readout_from"]
        log.info("atlas-vfx Sensor RS Maps (%s): %.2f ms from %s", cam.name, ro, ro_from)
        for r in dec.get("rejected", []):
            log.warning("atlas-vfx Sensor RS Maps: not using %.2f ms (%s)", r["value"], r["why"])
        if dec.get("warning"):
            log.warning("atlas-vfx Sensor RS Maps (%s): %s", cam.name, dec["warning"])
        if sensor_lens:
            lens = ln.BrownConrady(**sensor_lens["lens"])
            lens_from = f"Sensor Lens Maps ({sensor_lens['lens_from']})"
        else:
            lens, lens_from = ln.BrownConrady.from_camera(cam), "camera"
        f0 = first_frame or cam.frames[0]
        f1 = last_frame or cam.frames[-1]
        frames = [f for f in cam.frames if f0 <= f <= f1]
        out = work_dir(cam.data, f"{cam.name}_rs", out_dir)
        rep = jobs.make_rs_maps(cam, pl.clean_path(depth_folder), out, ro, readout_from=ro_from, frames=frames,
                                scale=map_scale, shutter_angle=shutter_angle, blur_samples=blur_samples,
                                direction=direction, iters=iters, lens=lens, lens_from=lens_from,
                                readout_decision=dec, progress=progress_bar())
        mid = rep["per_frame"][len(rep["per_frame"]) // 2]["frame"]
        kind = "ders" if direction == "remove RS" else "rs"
        prev = map_preview(mio.read_stmap(out / f"{kind}.{mid:04d}.exr"), gain=0.1 / map_scale)
        short = {k: v for k, v in rep.items() if k != "per_frame"}
        head = {k: short[k] for k in ("readout_ms", "readout_from", "skew_px_per_row_median", "roundtrip_median_px")}
        return {"ui": {"text": [json.dumps(head)]},
                "result": (str(out), torch.from_numpy(prev[None]), json.dumps(short, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXSensorRSMaps": AtlasVFXSensorRSMaps}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSensorRSMaps": "Atlas VFX Sensor RS Maps"}
