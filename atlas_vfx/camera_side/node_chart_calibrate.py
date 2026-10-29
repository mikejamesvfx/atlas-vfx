"""Atlas VFX Chart Calibrate (catalogue 49): lens and rolling-shutter readout from a filmed chart take. Wraps
camera_side/chart_calibrate.py, the same code as tools/grid_rs_shoot.py."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from . import chart_calibrate as cc
from . import plate as pl
from . import sensors

log = logging.getLogger("atlas_vfx")


class AtlasVFXChartCalibrate:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
                    "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("lens_path", "focal_mm", "k1", "k2", "pp_offset_x_px", "pp_offset_y_px",
                    "readout_path", "readout_ms", "sigma_ms", "rms_rs_px", "rms_gs_px", "measurable", "report")
    DESCRIPTION = ("Lens and rolling-shutter readout from a filmed checker chart (the 96 x 48 in grid, 31 x 15 inner "
                   "corners). mode 'lens' (take A, static holds): every Nth frame is paired with the next and kept "
                   "only if the chart didn't move, then OpenCV fits focal, k1 k2 and the principal point at plate size; "
                   "writes <name>_lens.json. mode 'readout' (takes B-E, fast pans with the whole chart in view): every "
                   "frame's corners are fitted with per-frame poses plus one readout, against take A's lens JSON; "
                   "writes <name>_readout.json. readout_ms wires into Sensor RS Maps' readout_chart_ms, where the "
                   "preset rule ranks it first. A take with no motion is reported as not measurable: measurable is "
                   "False and readout_ms is 0, which RS Maps reads as no chart value. The take is only read (luma, "
                   "range from the stream); outputs go to atlas-vfx-work/grid by default. In readout mode the lens "
                   "outputs echo the lens JSON used.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "chart take (MOV); read only"}),
                "mode": (["lens", "readout"], {"default": "lens",
                                               "tooltip": "lens = take A (static holds); readout = takes B-E "
                                                          "(pans), needs lens_json"}),
                "sensor_preset": (sorted(sensors.PRESETS), {"default": cc.DEFAULT_PRESET,
                                                             "tooltip": "sensor area, for focal in mm"}),
                "lens_json": ("STRING", {"default": "", "tooltip": "readout mode: the lens step's <name>_lens.json "
                                         "(wire lens_path from a lens-mode node)"}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 999999,
                                        "tooltip": "plate frame index, 0-based"}),
                "last_frame": ("INT", {"default": -1, "min": -1, "max": 999999,
                                       "tooltip": "inclusive; -1 = the take's last"}),
                "every": ("INT", {"default": 12, "min": 1, "max": 1000,
                                  "tooltip": "lens mode: sample every N frames (each paired with the next)"}),
                "still_px": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0, "step": 0.05,
                                       "tooltip": "lens mode: max chart motion to the next frame for a still view"}),
                "focal_guess_mm": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 1000.0, "step": 0.01,
                                             "tooltip": "lens mode: starting focal (the zoom mark)"}),
                "name": ("STRING", {"default": "", "tooltip": "file name stem; blank = the take's name"}),
                "out_dir": ("STRING", {"default": "", "tooltip": "blank = atlas-vfx-work/grid"}),
            }
        }

    def run(self, video_path, mode, sensor_preset, lens_json, first_frame, last_frame, every, still_px,
            focal_guess_mm, name, out_dir):
        if mode not in ("lens", "readout"):
            raise ValueError(f"Chart Calibrate: mode must be lens or readout, not {mode!r}")
        if sensor_preset not in sensors.PRESETS:
            raise ValueError(f"Chart Calibrate: unknown sensor preset {sensor_preset!r}")
        ref = cc.chart_ref(video_path, int(first_frame), None if last_frame < 0 else int(last_frame))
        name = pl.safe_name(name.strip() or Path(ref["path"]).stem)
        out = pl.resolve_out_dir(out_dir, cc.default_out_dir(), plate_path=ref["path"])
        args = {"mode": mode, "sensor_preset": sensor_preset, "first_frame": ref["first"], "last_frame": ref["last"],
                "name": name}
        if mode == "lens":
            args.update(every=int(every), still_px=float(still_px), focal_guess_mm=float(focal_guess_mm))
            res = cc.calibrate_lens(ref, preset=sensor_preset, every=every, still_px=still_px,
                                    focal_guess_mm=focal_guess_mm)
            for w in res["warnings"]:
                log.warning("atlas-vfx Chart Calibrate (%s): %s", name, w)
            lens_path = cc.write_result(out, name, "lens", ref, args, res, node="AtlasVFXChartCalibrate")
            lens, readout_path = res, ""
            ro = {"readout_ms": 0.0, "sigma_ms": 0.0, "rms_px_rs": 0.0, "rms_px_gs": 0.0, "measurable": False}
            head = (f"{name} lens: focal {res['focal_mm']:.3f} mm ({res['focal_px']:.1f} px), k1 {res['k1']:+.5f} "
                    f"k2 {res['k2']:+.5f}, pp offset {res['principal_point_offset_px'][0]:+.1f}, "
                    f"{res['principal_point_offset_px'][1]:+.1f} px, rms {res['rms_px']:.3f} px, "
                    f"{res['n_views']} still views")
        else:
            lens = cc.read_lens(lens_json)
            args.update(lens_json=lens["_path"])
            res = cc.measure_readout(ref, lens)
            readout_path = str(cc.write_result(out, name, "readout", ref, args, res, node="AtlasVFXChartCalibrate"))
            lens_path = lens["_path"]
            ro = dict(res)
            if not res["measurable"]:
                log.warning("atlas-vfx Chart Calibrate (%s): %s", name, res["not_measurable_reason"])
                ro["readout_ms"] = 0.0
            head = (f"{name} readout: " + (f"{res['readout_ms']:.2f} +- {res['sigma_ms']:.2f} ms" if res["measurable"]
                                            else "NOT MEASURABLE") +
                    f" | rms RS {res['rms_px_rs']:.3f} vs GS {res['rms_px_gs']:.3f} px | turn "
                    f"{res['rot_deg_per_frame_median']:.2f} deg/frame | {res['n_frames']} frames")
        # lens outputs: this run's calibration, or (readout mode) the lens JSON it used, same plate size
        K, dist = np.asarray(lens["K"], float), np.asarray(lens["dist_k1_k2_p1_p2_k3"], float)
        W, H = ref["width"], ref["height"]
        focal_mm = float(lens.get("focal_mm") or K[0, 0] * sensors.get_preset(sensor_preset)["width_mm"] / W)
        sig = float(ro["sigma_ms"])
        report = {"mode": mode, "out_dir": str(out), **{k: v for k, v in res.items() if k != "K"}}
        log.info("atlas-vfx Chart Calibrate: %s", head)
        return {"ui": {"text": [head]},
                "result": (str(lens_path), focal_mm, float(dist[0]), float(dist[1]), float(K[0, 2] - W / 2),
                           float(K[1, 2] - H / 2), readout_path, float(ro["readout_ms"]),
                           sig if math.isfinite(sig) else -1.0, float(ro["rms_px_rs"]), float(ro["rms_px_gs"]),
                           bool(ro["measurable"]), json.dumps(cc.finite(report), indent=1, default=cc.json_default))}


NODE_CLASS_MAPPINGS = {"AtlasVFXChartCalibrate": AtlasVFXChartCalibrate}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXChartCalibrate": "Atlas VFX Chart Calibrate"}
