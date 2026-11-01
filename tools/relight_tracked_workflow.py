"""Save (and optionally queue) the tracked relight graph: relight a COMP of DSCF3000 S2-S3 with LTX, the camera,
lens and sun coming from the plate (Mike, 2026-11-01: "relight a comp", sun from Physical Sky + the tracked camera).

  Plate Read 0-296 -> Solve Mask -> Solve Plate -> Measure & Scale (S2-S3 picks, metres)
      -> Sensor Lens Maps (the solved lens, auto overscan): LTX works in clean space
      -> Physical Sky (Burning Palms, the plate's capture time, the tracked camera) -> light rig
      -> Sun Direction (from the rig: where the real sun is relative to the camera, in the LoRA's words, mirror)
  Shot Range 1130-1226 (97 frames) -> Plate Read -> LTX Prep (reference = comp: plate + the element, mask_mode
      none: the whole comp is relit, clean space) -> Light Ball -> Latent Pre-render (the reference encoded once,
      cached by frame hash) -> Guide From Latent -> LTX-2.5 distilled + LTX-2.3 Relight IC-LoRA -> LTX Return (the
      relit comp, plate size) -> Relight Transfer Files (the element at 4K, normals in the tracked camera)

Built from tools/ltx_relight_workflow.graph_4k(), so its tuned LTX settings stay. heading_deg is the compass bearing
the camera faces on its first frame (the tracked camera only knows its turns, not north): 180 until measured.

Run: <V141 python> tools\\relight_tracked_workflow.py [--heading 180] [--queue]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ltx_relight_workflow as lrw  # noqa: E402
import ui_workflow  # noqa: E402
from atlas_vfx import registry  # noqa: E402

FOOTAGE = "~/Videos/atlas-vfx_footage"
PICKS = "~/atlas-vfx-work/DSCF3000/picks/DSCF3000_S2S3_picks_GUESS.json"
CAPTURE_UTC = "2025-12-25T07:58:19Z"           # DSCF3000 creation_time (ffprobe), 6:58 pm AEDT
BURNING_PALMS = (-34.1906, 151.0425, 5.0)


def graph(heading_deg: float = 180.0) -> dict:
    g = lrw.graph_4k()
    # --- camera: the tracker
    g["200"] = {"class_type": "AtlasVFXPlateRead", "inputs": {
        "path": f"{FOOTAGE}/DSCF3000.MOV", "first": 0, "last": 296, "ocio_config": "", "input_space": "auto",
        "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 96}}
    g["201"] = {"class_type": "AtlasVFXSolveMask", "inputs": {
        "plate_ref": ["200", 0], "classes": "person. car. bicycle. dog. tree. bush. flag.", "window": 16,
        "box_threshold": 0.35, "text_threshold": 0.25, "dilate_px": 6, "proxy_width": 1024}}
    g["202"] = {"class_type": "AtlasVFXSolvePlate", "inputs": {
        "plate_ref": ["200", 0], "sensor_preset": "auto", "sensor_width_mm": 23.5, "sensor_height_mm": 12.39,
        "focal_guess_mm": 16.0, "solve_focal": True, "solve_distortion": True, "lens_model": "k1 k2 (RADIAL)",
        "proxy_width": 1024, "max_features": 8192, "overlap": 15, "reject_px": 2.0, "name": "DSCF3000_relight_solve",
        "out_dir": "", "masks": ["201", 0], "mask_dilate_px": 4, "seed": 0, "camera_move": "auto"}}
    g["203"] = {"class_type": "AtlasVFXMeasureScale", "inputs": {
        "shot_camera": ["202", 0], "picks_path": PICKS, "solve_points": "", "max_spread_pct": 5.0, "force": False,
        "name": "DSCF3000_relight_metric", "out_dir": ""}}
    g["204"] = {"class_type": "AtlasVFXSensorLensMaps", "inputs": {
        "lens_source": "camera", "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "ca_r": 0.0, "ca_b": 0.0,
        "vig_1": 0.0, "vig_2": 0.0, "vig_3": 0.0, "map_scale": 1.0, "out_dir": "", "shot_camera": ["203", 0],
        "overscan": "auto"}}
    # --- the sun: Physical Sky at the plate's time and place, seen from the tracked camera
    lat, lon, alt = BURNING_PALMS
    g["210"] = {"class_type": "AtlasVFXPhysicalSky", "inputs": {
        "latitude": lat, "longitude": lon, "altitude_m": alt, "capture_utc": CAPTURE_UTC, "time_offset_h": 0.0,
        "heading_deg": float(heading_deg), "fov_x": 69.5, "tilt_deg": 0.0, "visibility_km": 40.0,
        "ground_albedo": 0.25, "exposure_ev": 0.0, "sun_mode": "metadata", "sun_azimuth_offset": 0.0,
        "sun_elevation_offset": 0.0, "sun_azimuth": 250.0, "sun_elevation": 15.0, "width": 1024,
        "latlong_centre": "camera heading", "name": "DSCF3000_relight_sky", "out_dir": "", "shot_camera": ["203", 0],
        "dataset": "", "ocio_config": "", "display": "sRGB - Display", "view": "ACES 2.0 - SDR 100 nits (Rec.709)",
        "sky_model": "auto"}}
    g["72"]["inputs"].update({"light_rig": ["210", 1], "rig_frame": "middle"})
    # --- LTX: the whole comp, clean space
    g["2"]["inputs"].update({"mask_mode": "none", "sensor_lens_folder": ["204", 1], "reference": "comp"})
    # --- the reference encoded once (Latent Pre-render), the guide added from the latent
    g["60"] = {"class_type": "AtlasVFXLatentPrerender", "inputs": {
        "images": ["5", 0], "vae": ["15", 0], "name": "DSCF3000_relight_ref", "out_dir": ""}}
    g["43"] = {"class_type": "AtlasVFXGuideFromLatent", "inputs": {
        "positive": ["41", 0], "negative": ["41", 1], "vae": ["15", 0], "latent": ["42", 0], "guide": ["60", 0],
        "strength": 1.0, "latent_downscale_factor": ["17", 1]}}
    g["54"]["inputs"]["name"] = "comp_relit"
    g["57"]["inputs"].update({"shot_camera": ["203", 0], "name": "element_relit_4k_tracked"})
    g["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["202", 2]}}
    return g


def seed(nodes):
    classes, _, _ = registry.load_nodes()
    for n in nodes:
        c = classes.get(n["class_type"])
        if c is not None:
            ui_workflow._INFO[n["class_type"]] = {"input": c.INPUT_TYPES(), "output": list(c.RETURN_TYPES),
                                                  "output_name": list(c.RETURN_NAMES)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heading", type=float, default=180.0, help="compass bearing of the first frame's view")
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--queue", action="store_true", help="also queue it on --api")
    a = ap.parse_args()
    ui_workflow.API = a.api
    g = graph(a.heading)
    seed(g.values())
    print("saved", ui_workflow.save(g, "relight_tracked_dscf3000_s2s3",
                                    "Relight a comp: tracker -> lens (clean space) -> Physical Sky sun -> LTX relight"))
    if a.queue:
        body = json.dumps({"prompt": g, "client_id": "atlas-vfx-relight-" + uuid.uuid4().hex[:6]}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{a.api}/prompt", data=body, headers={
            "Content-Type": "application/json"}), timeout=60).read())
        print("queued", r.get("prompt_id"), r.get("node_errors"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
