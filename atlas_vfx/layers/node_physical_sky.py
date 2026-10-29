"""Atlas VFX Physical Sky (catalogue 48): GPS + capture time -> clear sky HDRI (ACEScg lat-long) and the sun light
rig Relight uses, rendered by Atlas Scene's Prague sky (reached through atlas_vfx/_atlas_scene.py)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .. import _atlas_scene
from ..camera_side import exr_io
from ..camera_side import plate as pl
from . import physical_sky as ps

log = logging.getLogger("atlas_vfx")


def _f(d, lo, hi, step=0.01, tip=""):
    return ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step, "tooltip": tip})


def dataset_path(widget: str) -> Path:
    """The Prague dataset: the widget, else ATLAS_PRAGUE_DATASET (Atlas Scene's default). An existing .dat file."""
    s = pl.clean_path(widget) or pl.clean_path(os.environ.get("ATLAS_PRAGUE_DATASET", ""))
    if not s:
        raise ValueError("no Prague dataset: fill the dataset widget or set ATLAS_PRAGUE_DATASET to the official "
                         "Prague Sky Model .dat (ground level, 103 MB, is enough)")
    p = Path(s)
    if p.suffix.lower() != ".dat" or not p.is_file():
        raise ValueError(f"Prague dataset {p}: not an existing .dat file")
    return p


def preview_data_url(prev, max_width: int = 768, quality: int = 80) -> str:
    """The display-referred preview as a small JPEG data URL, for the Sun Globe's sky dome (display only)."""
    import base64

    import cv2

    img = np.clip(np.asarray(prev, np.float32), 0.0, 1.0)
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray((img[..., ::-1] * 255.0 + 0.5).astype(np.uint8)),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def globe_ui(sun, cam, fov, aspect, centre, latitude, longitude, capture_utc, time_offset_h, sun_mode, heading_deg,
             sky_rendered, prev) -> dict:
    """What the Sun Globe panel (web/atlas_vfx_sun_globe.js) cannot work out itself, sent as ui.atlas_sun_globe[0]:
    the sun actually used, the NOAA sun before any artist offset (`derived`), the capture day's sun path with its
    events, the camera used (a shot camera's fov / first-frame tilt), and the preview for the dome. `inputs` echoes
    the widgets the path depends on so the panel can tell when the path is stale. Display only: nothing here feeds
    back into the graph."""
    derived = sun.get("derived")
    if cam.get("camera") == "shot_camera":
        camera = {"source": "shot_camera", "heading_deg": float(heading_deg), "fov_x": float(fov),
                  "aspect": float(aspect), "tilt_deg": float(cam.get("first_frame_tilt_deg", 0.0)),
                  "frames": int(cam.get("frames", 1))}
    else:
        camera = {"source": "typed", "heading_deg": float(heading_deg), "fov_x": float(fov), "aspect": float(aspect),
                  "tilt_deg": float(cam.get("tilt_deg", 0.0)), "frames": 1}
    return json.loads(json.dumps({
        "version": 1,
        "sun": {"azimuth_deg": float(sun["azimuth_deg"]), "elevation_deg": float(sun["elevation_deg"]),
                "mode": sun["mode"], "renderable": bool(sun["renderable"]),
                "derived": ({"azimuth_deg": float(derived["azimuth_deg"]), "elevation_deg": float(derived["elevation_deg"])}
                            if derived else None),
                "capture_utc": sun.get("capture_utc"), "sun_utc": sun.get("sun_utc"),
                "time_offset_h": float(time_offset_h)},
        "inputs": {"latitude": float(latitude), "longitude": float(longitude), "capture_utc": str(capture_utc or ""),
                   "time_offset_h": float(time_offset_h), "sun_mode": str(sun_mode), "heading_deg": float(heading_deg)},
        "path": ps.sun_path(latitude, longitude, capture_utc),
        "clock": ps.capture_clock(capture_utc),
        "camera": camera,
        "latlong_centre_deg": float(centre),
        "sky_rendered": bool(sky_rendered),
        "min_elevation_deg": ps.PRAGUE_MIN_ELEVATION,
        "preview": preview_data_url(prev),
        "convention": "sky frame right-handed, Y up, north -Z, east +X; azimuth clockwise from north, elevation up",
    }))


class AtlasVFXPhysicalSky:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "ATLAS_VFX_LIGHT_RIG", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("hdri_acescg", "light_rig", "preview", "info", "exr_path")
    DESCRIPTION = (
        "Turns the footage's GPS and capture time into a physically based clear sky and sun. Sun position: NOAA, "
        "geometric, from latitude / longitude and the UTC capture time (+ time_offset_h to relight to another "
        "hour); 'offset' adds an artist nudge, 'manual' takes azimuth / elevation as typed. Sky: the official "
        "Prague Sky Model as Atlas Scene integrates it (Atlas Scene must be reachable; see ATLAS_SCENE_ROOT), "
        "dataset from the widget or ATLAS_PRAGUE_DATASET. "
        "hdri_acescg: scene-linear, unclipped, float32 lat-long, width x width/2, converted from Linear Rec.709 to "
        "ACEScg by the named OCIO transform, also written as <name>.exr (float, ACEScg in the header) with a "
        "sidecar. Lat-long convention (Nuke's): seen from inside, the centre column looks down -Z and is compass "
        "north (or the camera heading with latlong_centre = camera heading), +X (east) a quarter to the right, "
        "+Y at the top; azimuth grows to the right, so a sun at azimuth A, elevation e sits at column "
        "((A - centre + 180) mod 360) / 360 * width, row (90 - e) / 180 * height. "
        "light_rig: sun direction in camera space (heading = compass direction the camera looks; a shot camera "
        "gives the field of view and every frame's rotation, its first frame looks at heading), sun colour (max 1) "
        "and sun_strength against a sky of 1; plug it into Relight's sun_rig to replace the plate's sun estimate. "
        "Sun under -4.2 deg (Prague's limit): the sky is NOT rendered, the HDRI is black, a warning is logged and "
        "info says sky_rendered false; the rig keeps the exact direction with zero strength. "
        "preview: the HDRI through the named OCIO display / view, auto-exposed, sun ringed, camera FOV marked.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latitude": _f(-34.1906, -90.0, 90.0, 0.0001, "GPS latitude, degrees north (typed when the file has none)"),
                "longitude": _f(151.0425, -180.0, 180.0, 0.0001, "GPS longitude, degrees east"),
                "altitude_m": _f(5.0, 0.0, 15000.0, 1.0, "used only with the full dataset (ground level ignores it)"),
                "capture_utc": ("STRING", {"default": "2025-12-25T07:39:25Z",
                                           "tooltip": "ISO 8601 with its offset: ...Z or ...+11:00"}),
                "time_offset_h": _f(0.0, -48.0, 48.0, 0.05, "relight to another hour"),
                "heading_deg": _f(180.0, 0.0, 360.0, 0.1, "compass direction the camera looks (0 N, 90 E)"),
                "fov_x": _f(50.0, 1.0, 170.0, 0.1, "horizontal field of view (the shot camera's wins)"),
                "tilt_deg": _f(0.0, -89.0, 89.0, 0.1, "camera tilt up (typed camera only)"),
                "visibility_km": _f(40.0, 20.0, 131.8, 0.1, "Prague's range: 20 (hazy) to 131.8 (very clear)"),
                "ground_albedo": _f(0.25, 0.0, 1.0, 0.01),
                "exposure_ev": _f(0.0, -30.0, 30.0, 0.1, "applied to the HDRI (the rig is normalised)"),
                "sun_mode": (["metadata", "offset", "manual"], {"default": "metadata"}),
                "sun_azimuth_offset": _f(0.0, -180.0, 180.0, 0.1),
                "sun_elevation_offset": _f(0.0, -90.0, 90.0, 0.1),
                "sun_azimuth": _f(250.0, 0.0, 360.0, 0.1, "manual mode"),
                "sun_elevation": _f(15.0, -90.0, 90.0, 0.1, "manual mode"),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 4,
                                  "tooltip": "lat-long width (height = width / 2), a multiple of 4"}),
                "latlong_centre": (["north", "camera heading"], {"default": "north"}),
                "name": ("STRING", {"default": "physical_sky"}),
                "out_dir": ("STRING", {"default": "", "tooltip": "blank = ~/atlas-vfx-work/physical_sky/<name>"}),
            },
            "optional": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "dataset": ("STRING", {"default": "", "tooltip": "Prague .dat; blank = ATLAS_PRAGUE_DATASET"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "display": ("STRING", {"default": pl.DEFAULT_DISPLAY}),
                "view": ("STRING", {"default": pl.DEFAULT_VIEW}),
            },
        }


    def run(self, latitude, longitude, altitude_m, capture_utc, time_offset_h, heading_deg, fov_x, tilt_deg,
            visibility_km, ground_albedo, exposure_ev, sun_mode, sun_azimuth_offset, sun_elevation_offset, sun_azimuth,
            sun_elevation, width, latlong_centre, name, out_dir, shot_camera=None, dataset="", ocio_config="",
            display=pl.DEFAULT_DISPLAY, view=pl.DEFAULT_VIEW):
        import torch

        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / "physical_sky" / name)
        width = int(width)
        if width < 64 or width % 4:
            raise ValueError(f"width {width}: at least 64 and a multiple of 4")
        sun = ps.effective_sun(sun_mode, latitude, longitude, capture_utc, time_offset_h, sun_azimuth_offset,
                               sun_elevation_offset, sun_azimuth, sun_elevation)
        az, el = sun["azimuth_deg"], sun["elevation_deg"]
        dirs, cam = ps.sun_in_camera(az, el, heading_deg, tilt_deg, shot_camera)
        fov = float(cam.get("fov_x", fov_x))
        aspect = float(cam.get("aspect", 16 / 9))
        centre = 0.0 if latlong_centre == "north" else float(heading_deg)
        warnings, notes = [], []
        render = None
        if sun["renderable"]:
            ds = dataset_path(dataset)
            render = ps.render_sky(az, el, width=width, centre_deg=centre, visibility_km=visibility_km,
                                   ground_albedo=ground_albedo, exposure_ev=exposure_ev, altitude_m=altitude_m,
                                   dataset=str(ds), source=f"atlas-vfx Physical Sky, sun mode {sun_mode}")
            rgb709, e_sun, e_sky = render["lin_rec709"], render["e_sun"], render["e_sky_up"]
            notes += render["notes"]
        else:
            ds = None
            msg = (f"sun at {el:.2f} deg is below Prague's {ps.PRAGUE_MIN_ELEVATION} deg: sky NOT rendered, HDRI is "
                   "black; the rig keeps the sun direction with zero strength")
            log.warning("atlas-vfx Physical Sky: %s", msg)
            warnings.append(msg)
            rgb709 = np.zeros((width // 2, width, 3), np.float32)
            e_sun = e_sky = np.zeros(3)
        if cam["camera"] == "shot_camera" and tilt_deg:
            notes.append("tilt_deg ignored: the shot camera's rotation is used")
        acescg = ps.to_working(rgb709, ocio_config)
        rig = ps.light_rig(dirs, e_sun, e_sky, ground_albedo, el, fov, aspect)
        rig.update(sun_azimuth_deg=round(az, 6), sun_elevation_deg=round(el, 6))
        prev, pev = ps.preview(acescg, az, el, centre, float(heading_deg), fov, ocio_config, display, view)

        exr = exr_io.write_rgba(out / f"{name}.exr", np.dstack([acescg, np.ones(acescg.shape[:2], np.float32)]),
                                colorspace=ps.WORKING, half=False, premultiplied=True,
                                attrs={"layer": "physical_sky", "source_colorspace": ps.REC709,
                                       "latlong": ps.LATLONG_CONVENTION, "latlong_centre_deg": centre,
                                       "sun_azimuth_deg": az, "sun_elevation_deg": el,
                                       "sky_rendered": int(render is not None)})
        info = self._info(sun, cam, centre, latitude, longitude, altitude_m, visibility_km, ground_albedo,
                          exposure_ev, width, fov, aspect, ds, render, rig, pev, display, view, ocio_config,
                          warnings, notes, exr)
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXPhysicalSky",
                         inputs={k: info["inputs"][k] for k in info["inputs"]}, outputs=info)
        text = json.dumps(info, indent=1, default=str)
        summary = (f"sun az {az:.2f} el {el:.2f} ({sun_mode}); sky {'rendered' if render else 'NOT rendered'}; "
                   f"sun_strength {rig['sun_strength']}; {exr}")
        globe = globe_ui(sun, cam, fov, aspect, centre, latitude, longitude, capture_utc, time_offset_h, sun_mode,
                         heading_deg, render is not None, prev)
        return {"ui": {"text": [summary], "atlas_sun_globe": [globe]},
                "result": (torch.from_numpy(np.ascontiguousarray(acescg))[None], rig,
                           torch.from_numpy(np.ascontiguousarray(prev))[None], text, str(exr))}


    @staticmethod
    def _info(sun, cam, centre, latitude, longitude, altitude_m, visibility_km, ground_albedo, exposure_ev, width,
              fov, aspect, ds, render, rig, pev, display, view, ocio_config, warnings, notes, exr) -> dict:
        rel = ((sun["azimuth_deg"] - cam["heading_deg"] + 180.0) % 360.0) - 180.0
        info = {
            "sun": {**sun, "relative_to_heading_deg": round(rel, 4)},
            "sky_rendered": render is not None,
            "warnings": warnings, "notes": notes,
            "inputs": {"latitude": latitude, "longitude": longitude, "altitude_m": altitude_m,
                       "visibility_km": visibility_km, "ground_albedo": ground_albedo, "exposure_ev": exposure_ev,
                       "width": width, "latlong_centre_deg": centre, "camera": cam, "fov_x": fov, "aspect": aspect,
                       "ocio_config": ocio_config, "display": display, "view": view},
            "latlong": {"convention": ps.LATLONG_CONVENTION, "width": width, "height": width // 2,
                        "sun_pixel": [round(v, 3) for v in ps.sun_pixel(sun["azimuth_deg"], sun["elevation_deg"],
                                                                        width, width // 2, centre)]},
            "colour": {"rendered_in": ps.REC709, "hdri": ps.WORKING, "transform": f"OCIO {ps.REC709} -> {ps.WORKING}",
                       "preview": f"OCIO {ps.WORKING} -> {display} / {view}", "preview_exposure_ev": round(pev, 4),
                       "dtype": "float32, unclipped"},
            "light_rig": {k: rig[k] for k in ("frames", "sun_colour", "sun_strength", "sun_visible", "colourspace",
                                              "normalised")}
                         | {"sun_dir_first": rig["sun_dir"][0], "sun_uv_first": rig["sun_uv"][0],
                            "sun_in_front_first": rig["sun_in_front"][0], "sky": rig["sky"][0], "ground": rig["ground"][0]},
            "files": {"exr": str(exr)},
        }
        if render is not None:
            info["provenance"] = {
                "sky_model": "Prague Sky Model (Charles University, Apache-2.0) via Atlas Scene",
                "atlas_scene": _atlas_scene.report(),
                "evaluator": render["evaluator"],
                "dataset": {"path": str(ds), "bytes": ds.stat().st_size, "quick_sha256": pl.quick_sha256(ds),
                            "capabilities": render["capabilities"]},
                "altitude_used_m": render["altitude_used_m"],
                "atlas_scene_metadata": render["metadata"],
                "hdri_sun_energy": render["hdri_sun_energy"],
                "e_sun_rec709": np.asarray(render["e_sun"]).round(6).tolist(),
                "e_sky_up_rec709": np.asarray(render["e_sky_up"]).round(6).tolist(),
            }
            if sun.get("sun_utc") and sun["mode"] in ("metadata", "offset"):
                chk = ps.atlas_scene_cross_check(latitude, longitude, sun["sun_utc"])
                if chk.get("ok"):
                    chk["diff_deg"] = [round(((chk["azimuth_deg"] - sun["derived"]["azimuth_deg"] + 180) % 360) - 180, 6),
                                       round(chk["elevation_deg"] - sun["derived"]["elevation_deg"], 6)]
                info["provenance"]["atlas_scene_derive_sun"] = chk
        else:
            info["provenance"] = {"sky_model": "not run (sun below Prague's domain)",
                                  "atlas_scene": _atlas_scene.report()}
        return json.loads(json.dumps(info, default=str))


NODE_CLASS_MAPPINGS = {"AtlasVFXPhysicalSky": AtlasVFXPhysicalSky}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPhysicalSky": "Atlas VFX Physical Sky"}
