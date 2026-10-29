"""Physical Sky core: GPS + capture time -> clear sky HDRI and sun light rig.

Packaging, not new science. The sun position is NOAA (layers/solar.py). Two sky models:
  prague         Atlas Scene's (`atlas_scene.environment.render_environment` with the `atlas_scene_prague`
                 evaluator, the official Prague Sky Model), reached only through atlas_vfx/_atlas_scene.py. Atlas
                 Scene owns it; nothing of it is copied here.
  hosek-wilkie   the analytic fallback that ships with atlas-vfx (layers/sky_hosek.py, the authors' BSD-3 model
                 ported to numpy), rendered by `render_latlong` below. Sea level, sun 0-90 deg, approximate.

Frames and the lat-long convention (the part that bit on 2026-10-28):
  sky frame      right-handed, Y up, north = -Z, east = +X. Compass azimuth A (clockwise from north), elevation e:
                 d = (sin A cos e, sin e, -cos A cos e).
  lat-long       seen from inside, Nuke's layout and Atlas Scene's `latlong_directions`: column u in [0, 1) is
                 longitude 2 pi u - pi, row v is latitude pi/2 - pi v; the centre column looks down -Z, a quarter
                 to the right is +X, the top row is +Y. With `centre` c the centre column is compass c, so a sun at
                 (A, e) lands at x = ((A - c + 180) mod 360) / 360 * W - 0.5, y = (90 - e) / 180 * H - 0.5 (pixel
                 centres). Azimuth grows to the RIGHT (north, east, south, west), as you turn right on the spot.
  Atlas Scene    its Prague boundary maps Atlas (x, y, z) to Prague (x, -z, y) and passes the compass azimuth
                 straight in, so in its own lat-long the sun sits at x = (270 - A) / 360 * W: azimuth grows to the
                 LEFT (compass mirrored). `_Oriented` below hands Atlas Scene's evaluator the mirrored directions,
                 so the image we get back is right-handed. tests/test_physical_sky.py pins the sun pixel.
  camera         OpenGL camera space (x right, y up, z toward the viewer, looking down -Z), as Relight expects:
                 right = cross(forward, up) in the right-handed sky frame.

Colour: Atlas Scene returns unclipped scene-linear Rec.709 primaries ("Linear Rec.709 (sRGB)"); the HDRI goes to
ACEScg by the named OCIO transform with the shipped config; float32 throughout (half overflows on the sun disk).
Below Prague's domain (sun under -4.2 deg) the sky is not rendered: the HDRI is black, a warning is logged and the
info says so; the sun direction is still exact and the rig carries it with zero strength.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import _atlas_scene
from ..camera_side import plate as pl
from ..contracts import shot_camera as sc
from . import sky_hosek, solar

log = logging.getLogger("atlas_vfx")

REC709 = "Linear Rec.709 (sRGB)"
WORKING = "ACEScg"
PRAGUE_MIN_ELEVATION = -4.2           # Prague's (and Atlas Scene's) solar elevation domain starts here
PRAGUE_VISIBILITY_KM = (20.0, 131.8)
HOSEK_MIN_ELEVATION = 0.0             # Hosek-Wilkie's domain starts at the horizon (no twilight)
SKY_MODELS = ("prague", "hosek-wilkie")
MIN_ELEVATION = {"prague": PRAGUE_MIN_ELEVATION, "hosek-wilkie": HOSEK_MIN_ELEVATION}
SUN_RADIUS_DEG = 0.2667               # the Prague model's sun disk radius (PragueSkyModel.cpp SUN_RADIUS)
LUM709 = np.array([0.2126, 0.7152, 0.0722], np.float64)
LATLONG_CONVENTION = ("lat-long seen from inside: centre column = compass `centre` looking down -Z, +X a quarter "
                      "right, +Y top; sky frame north -Z, east +X, Y up; azimuth grows to the right")


# ------------------------------------------------------------------------------------------------ directions
def azel_to_dir(az_deg, el_deg):
    """Sky-frame unit direction(s) for compass azimuth / elevation in degrees."""
    a, e = np.broadcast_arrays(np.radians(np.asarray(az_deg, np.float64)), np.radians(np.asarray(el_deg, np.float64)))
    return np.stack([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)], -1)


def dir_to_azel(d):
    d = np.asarray(d, np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(d[..., 0], -d[..., 2])) % 360.0, np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))


def sun_pixel(az_deg, el_deg, width, height, centre_deg=0.0):
    """(x, y) pixel coordinates (pixel centres are integers) of a direction in our lat-long."""
    x = ((float(az_deg) - float(centre_deg) + 180.0) % 360.0) / 360.0 * width - 0.5
    y = (90.0 - float(el_deg)) / 180.0 * height - 0.5
    return x, y


def latlong_to_atlas(centre_deg):
    """3x3 matrix taking a direction of our lat-long (centre compass c) to the Atlas Scene world direction that its
    Prague boundary reads as the same compass azimuth and elevation. det = -1: the mirror described above."""
    c = math.radians(centre_deg)
    return np.array([[-math.sin(c), 0.0, -math.cos(c)],
                     [0.0, 1.0, 0.0],
                     [-math.cos(c), 0.0, math.sin(c)]], np.float64)


# ------------------------------------------------------------------------------------------------ cameras
def camera_to_sky(heading_deg, tilt_deg=0.0):
    """3x3 camera-to-sky rotation (columns: camera x right, y up, z back) of a camera looking at compass heading,
    tilted up by tilt_deg, no roll."""
    f = azel_to_dir(heading_deg, tilt_deg)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        raise ValueError("camera looks straight up or down; heading is undefined")
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def compass_of(v) -> float:
    """Compass-style angle of a Y-up world direction: 0 at -Z, 90 at +X (the sky frame's convention)."""
    v = np.asarray(v, np.float64)
    if math.hypot(v[0], v[2]) < 1e-9:
        raise ValueError("camera looks straight up or down; heading is undefined")
    return math.degrees(math.atan2(v[0], -v[2])) % 360.0


def sun_in_camera(az_deg, el_deg, heading_deg, tilt_deg=0.0, shot_camera: dict | None = None):
    """Per-frame sun direction in OpenGL camera space, plus what the camera is.

    Typed camera: one frame, a level-roll camera at heading/tilt. Shot camera (shot-camera@0.1, Y up): heading is
    the compass heading of its FIRST frame's view; the solved world is turned about Y to match, then every frame's
    own rotation applies (pans and tilts follow the solve). Returns (dirs (F, 3), meta)."""
    if shot_camera is None:
        R = camera_to_sky(heading_deg, tilt_deg)
        d = R.T @ azel_to_dir(az_deg, el_deg)
        return d[None], {"camera": "typed", "frames": 1, "heading_deg": float(heading_deg), "tilt_deg": float(tilt_deg)}
    cam = sc.ShotCamera(shot_camera)
    frames = cam.frames
    R0 = cam.cam_to_world(frames[0])[:3, :3]
    c0 = compass_of(-R0[:, 2])
    delta = c0 - float(heading_deg)
    d_world = azel_to_dir(float(az_deg) + delta, el_deg)
    dirs = np.stack([cam.cam_to_world(f)[:3, :3].T @ d_world for f in frames])
    k = cam.intrinsics(frames[0])
    tilt0 = math.degrees(math.asin(max(-1.0, min(1.0, float(-R0[1, 2])))))
    return dirs, {"camera": "shot_camera", "name": cam.name, "frames": len(frames), "frame_range": [frames[0], frames[-1]],
                  "heading_deg": float(heading_deg), "world_yaw_deg": round(delta, 6),
                  "first_frame_tilt_deg": round(tilt0, 6),
                  "fov_x": math.degrees(2 * math.atan(k.width / (2 * k.fx))), "aspect": k.width / k.height,
                  "tilt_deg": "from the shot camera"}


def project_uv(d_cam, fov_x_deg, aspect):
    """Normalised image position (0..1, top-left origin) of camera-space directions; the inverse of
    relight.sun_direction with behind = depth. Points behind the camera get front=False."""
    d = np.asarray(d_cam, np.float64).reshape(-1, 3)
    tx = math.tan(math.radians(fov_x_deg) / 2)
    z = np.where(np.abs(d[:, 2]) < 1e-9, -1e-9, d[:, 2])
    u = 0.5 + (d[:, 0] / -z) / (2 * tx)
    v = 0.5 - (d[:, 1] / -z) * aspect / (2 * tx)
    return np.stack([u, v], -1), d[:, 2] < 0


# ------------------------------------------------------------------------------------------------ the sun
def effective_sun(mode: str, latitude, longitude, capture_utc: str, time_offset_h=0.0, az_offset=0.0, el_offset=0.0,
                  manual_az=0.0, manual_el=0.0, min_elevation=PRAGUE_MIN_ELEVATION) -> dict:
    """The sun the sky is rendered for. metadata: NOAA from GPS + capture time (+ offset hours); offset: that plus an
    artist offset in azimuth / elevation; manual: the typed azimuth / elevation (capture time only recorded)."""
    out = {"mode": mode, "time_offset_h": float(time_offset_h)}
    if mode in ("metadata", "offset") or str(capture_utc or "").strip():
        t0 = solar.parse_utc(capture_utc)
        t = solar.shifted(t0, time_offset_h)
        az, el = solar.sun_at(latitude, longitude, t)
        out.update(capture_utc=t0.isoformat().replace("+00:00", "Z"), sun_utc=t.isoformat().replace("+00:00", "Z"),
                   derived={"azimuth_deg": az, "elevation_deg": el, "method": "NOAA (layers/solar.py), geometric"})
    if mode == "metadata":
        az, el = out["derived"]["azimuth_deg"], out["derived"]["elevation_deg"]
    elif mode == "offset":
        az = out["derived"]["azimuth_deg"] + float(az_offset)
        el = out["derived"]["elevation_deg"] + float(el_offset)
        out["offset"] = {"azimuth_deg": float(az_offset), "elevation_deg": float(el_offset)}
    elif mode == "manual":
        az, el = float(manual_az), float(manual_el)
    else:
        raise ValueError(f"sun mode {mode!r}: metadata, offset or manual")
    if not -90.0 <= el <= 90.0:
        raise ValueError(f"sun elevation {el:.3f} deg is outside -90..90 (check the elevation offset)")
    out.update(azimuth_deg=az % 360.0, elevation_deg=el, renderable=el >= float(min_elevation))
    return out


GOLDEN_ELEVATION = 4.0                # the Sun Globe's GOLDEN button puts the sun here (inside golden hour, -4..6 deg)


def capture_clock(capture_utc: str) -> dict:
    """The clock the capture time was written in, for showing times: its UTC offset in minutes when the string
    names one other than Z (e.g. '+11:00'), else None (times show as UTC)."""
    s = str(capture_utc or "").strip()
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        t = None
    if t is None or t.utcoffset() is None or s[-1:] in ("Z", "z"):
        return {"utc_offset_min": None, "label": "UTC"}
    m = int(round(t.utcoffset().total_seconds() / 60.0))
    return {"utc_offset_min": m, "label": f"{'+' if m >= 0 else '-'}{abs(m) // 60:02d}:{abs(m) % 60:02d}"}


def _sun_el(latitude, longitude, t0, offset_h):
    return solar.sun_at(latitude, longitude, solar.shifted(t0, offset_h))


def _crossing(latitude, longitude, t0, a, b, level, it=50):
    """Bisect the time offset (hours) in [a, b] where the elevation crosses `level` (it does, by the caller)."""
    fa = _sun_el(latitude, longitude, t0, a)[1] - level
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = _sun_el(latitude, longitude, t0, m)[1] - level
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _event(latitude, longitude, t0, offset_h) -> dict:
    az, el = _sun_el(latitude, longitude, t0, offset_h)
    t = solar.shifted(t0, offset_h)
    return {"offset_h": round(offset_h, 6), "utc": t.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "azimuth_deg": round(az, 4), "elevation_deg": round(el, 4)}


def sun_path(latitude, longitude, capture_utc: str, step_min: float = 10.0) -> dict | None:
    """The capture day's sun, for the Sun Globe. Samples every `step_min` minutes over the local solar day that holds
    the capture (solar midnight to solar midnight, local mean solar time = UTC + longitude / 15 h), each sample
    keyed by the time_offset_h that gives it (hours from the capture). Events, found on solar.sun_at itself (not
    the samples): sunrise / sunset (geometric elevation crossing 0, bisected), solar noon (highest sun, ternary
    search) and golden hour (crossing GOLDEN_ELEVATION, morning and evening). A polar day or night has no
    sunrise / sunset (None, and `polar` says which). None when there is no capture time."""
    if not str(capture_utc or "").strip():
        return None
    t0 = solar.parse_utc(capture_utc)
    solar_h = (t0.hour + t0.minute / 60.0 + t0.second / 3600.0 + float(longitude) / 15.0) % 24.0
    start = -solar_h
    n = int(round(24 * 60 / float(step_min)))
    offs = [start + i * float(step_min) / 60.0 for i in range(n + 1)]
    pts = [_sun_el(latitude, longitude, t0, o) for o in offs]
    az = [p[0] for p in pts]
    el = [p[1] for p in pts]
    i_max = max(range(len(el)), key=el.__getitem__)
    a, b = offs[max(i_max - 1, 0)], offs[min(i_max + 1, n)]
    for _ in range(60):
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if _sun_el(latitude, longitude, t0, m1)[1] < _sun_el(latitude, longitude, t0, m2)[1]:
            a = m1
        else:
            b = m2
    noon = _event(latitude, longitude, t0, 0.5 * (a + b))

    def crossings(level):
        up = down = None
        for i in range(n):
            lo, hi = el[i] - level, el[i + 1] - level
            if lo < 0 <= hi and up is None:
                up = _event(latitude, longitude, t0, _crossing(latitude, longitude, t0, offs[i], offs[i + 1], level))
            elif lo >= 0 > hi and down is None:
                down = _event(latitude, longitude, t0, _crossing(latitude, longitude, t0, offs[i], offs[i + 1], level))
        return up, down

    rise, sset = crossings(0.0)
    g_am, g_pm = crossings(GOLDEN_ELEVATION)
    polar = None
    if rise is None and sset is None:
        polar = "day" if min(el) > 0 else ("night" if max(el) < 0 else None)
    return {"step_min": float(step_min), "start_offset_h": round(start, 6), "end_offset_h": round(offs[-1], 6),
            "offset_h": [round(o, 6) for o in offs], "azimuth_deg": [round(v, 4) for v in az],
            "elevation_deg": [round(v, 4) for v in el], "sunrise": rise, "sunset": sset, "noon": noon,
            "golden_morning": g_am, "golden_evening": g_pm, "golden_elevation_deg": GOLDEN_ELEVATION, "polar": polar,
            "method": "NOAA (layers/solar.py), geometric (no refraction); events bisected on sun_at"}


def atlas_scene_cross_check(latitude, longitude, sun_utc: str) -> dict:
    """Atlas Scene's own derive_sun for the same instant, where it accepts it (elevation >= -4.2)."""
    try:
        sun_mod = _atlas_scene.import_module("environment.sun")
        cam_mod = _atlas_scene.import_module("scene.camera")
        s = sun_mod.derive_sun(cam_mod.CaptureLocation(float(latitude), float(longitude)), cam_mod.CaptureTime(utc=sun_utc))
    except _atlas_scene.AtlasSceneMissing:
        raise
    except Exception as e:  # noqa: BLE001 - derive_sun refuses below -4.2; record why
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}
    return {"ok": True, "azimuth_deg": s.azimuth_deg, "elevation_deg": s.elevation_deg}


# ------------------------------------------------------------------------------------------------ Atlas Scene render
class _Oriented:
    """Atlas Scene evaluator seen through our lat-long frame: directions are mapped by latlong_to_atlas before the
    Prague evaluator sees them. Only a change of axes; the radiance is Atlas Scene's."""

    def __init__(self, inner, centre_deg: float):
        self.inner = inner
        self.M = latlong_to_atlas(centre_deg)
        self.implementation_version = (f"{getattr(inner, 'implementation_version', type(inner).__name__)}"
                                       f"+atlas_vfx_latlong(centre={float(centre_deg):.6f})")

    def _map(self, directions):
        return np.asarray(directions, np.float64).reshape(-1, 3) @ self.M.T

    def capabilities(self):
        return self.inner.capabilities()

    def sky_radiance(self, directions):
        return self.inner.sky_radiance(self._map(directions))

    def sun_radiance(self, directions):
        return self.inner.sun_radiance(self._map(directions))

    def transmittance(self, directions, distance_m):
        return self.inner.transmittance(self._map(directions), distance_m)

    def in_scatter(self, directions, distance_m):
        return self.inner.in_scatter(self._map(directions), distance_m)


def _sun_irradiance(ev, d_sun, n=81):
    """Normal irradiance (Rec.709 linear, model units) of the sun disk: the evaluator's sun radiance integrated over a
    small square around the sun direction (lat-long frame), n x n samples, 1.3 x the disk radius."""
    s = np.asarray(d_sun, np.float64)
    a = np.array([0.0, 1.0, 0.0]) if abs(s[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(s, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(t1, s)
    R = math.radians(SUN_RADIUS_DEG) * 1.3
    g = (np.arange(n) + 0.5) / n * 2 * R - R
    A, B = np.meshgrid(g, g)
    d = s[None] + A.reshape(-1, 1) * t1[None] + B.reshape(-1, 1) * t2[None]
    w = 1.0 / np.linalg.norm(d, axis=1) ** 3              # solid angle of a tangent-plane cell
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    L = np.asarray(ev.sun_radiance(d), np.float64).reshape(-1, 3)
    return (L * w[:, None]).sum(0) * (2 * R / n) ** 2


def _sky_irradiance_up(ev, nlon=256, nlat=64):
    """Irradiance on an up-facing surface from the sky alone (no sun), Rec.709 linear, model units."""
    lat = (np.arange(nlat) + 0.5) / nlat * (math.pi / 2)
    lon = (np.arange(nlon) + 0.5) / nlon * 2 * math.pi - math.pi
    LON, LAT = np.meshgrid(lon, lat)
    d = np.stack([np.sin(LON) * np.cos(LAT), np.sin(LAT), -np.cos(LON) * np.cos(LAT)], -1).reshape(-1, 3)
    L = np.asarray(ev.sky_radiance(d), np.float64).reshape(-1, 3)
    w = (np.sin(LAT) * np.cos(LAT)).reshape(-1) * (math.pi / 2 / nlat) * (2 * math.pi / nlon)
    return (L * w[:, None]).sum(0)


def hdri_sun_energy(img, az, el, centre, exposure_ev, e_sun):
    """How much of the sun's irradiance the point-sampled lat-long actually holds (1.0 = all of it). The disk is
    about 3 px wide at 2K, so this varies with where the sun falls between pixel centres."""
    H, W = img.shape[:2]
    r0 = max(0, int((90.0 - el - 2.0) / 180.0 * H))            # only the rows within 2 deg of the sun
    r1 = min(H, int((90.0 - el + 2.0) / 180.0 * H) + 2)
    img = img[r0:r1]
    yy, xx = np.mgrid[r0:r1, 0:W]
    lon = 2 * math.pi * (xx + 0.5) / W - math.pi
    lat = math.pi / 2 - math.pi * (yy + 0.5) / H
    d = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], -1)
    s = azel_to_dir(az - centre, el)
    ang = np.degrees(np.arccos(np.clip(d @ s, -1, 1)))
    near, ring = ang < 1.0, (ang >= 1.0) & (ang < 1.5)
    if not near.any() or not ring.any():
        return None
    bg = np.median(img[ring], 0)
    dom = (np.cos(lat) * (math.pi / H) * (2 * math.pi / W))[near]
    e_img = ((img[near] - bg) * dom[:, None]).sum(0)
    want = float(LUM709 @ (np.asarray(e_sun) * 2.0 ** exposure_ev))
    return float(LUM709 @ e_img) / want if want > 0 else None


def render_sky(az, el, *, width, centre_deg=0.0, visibility_km=40.0, ground_albedo=0.25, exposure_ev=0.0,
               altitude_m=0.0, dataset: str, source: str = "atlas-vfx Physical Sky") -> dict:
    """Render the lat-long (width x width/2, float32, Linear Rec.709, unclipped) with Atlas Scene's Prague sky for a
    sun at (az, el), plus the sun / sky irradiance the light rig needs. Raises AtlasSceneMissing with the fix when
    Atlas Scene or its Prague extension cannot be reached; ValueError for values outside Prague's domain."""
    width = int(width)
    if width < 64 or width % 4:
        raise ValueError(f"width {width}: at least 64 and a multiple of 4")
    if not PRAGUE_VISIBILITY_KM[0] <= visibility_km <= PRAGUE_VISIBILITY_KM[1]:
        raise ValueError(f"visibility {visibility_km} km outside Prague's {PRAGUE_VISIBILITY_KM[0]}-"
                         f"{PRAGUE_VISIBILITY_KM[1]} km")
    if not 0.0 <= ground_albedo <= 1.0:
        raise ValueError(f"ground albedo {ground_albedo} outside 0-1")
    if el < PRAGUE_MIN_ELEVATION:
        raise ValueError(f"sun elevation {el:.3f} below Prague's {PRAGUE_MIN_ELEVATION} deg; nothing to render")
    _atlas_scene.prague()                                   # the compiled extension, or a clear AtlasSceneMissing
    model = _atlas_scene.import_module("environment.model")
    render = _atlas_scene.import_module("environment.render")
    cache = _atlas_scene.import_module("environment.cache")
    evaluator = _atlas_scene.import_module("environment.evaluator")

    env = model.Environment()
    env.set_physical_sky(model="prague", mode="simple")
    a = env.atmosphere
    a.dataset.path = str(dataset)
    a.visibility_km.value = float(visibility_km)
    a.ground_albedo.value = float(ground_albedo)
    a.exposure_ev.value = float(exposure_ev)
    env.sun = model.SunState.with_derived(float(az), float(el), source=source)
    inner = evaluator.create_evaluator(env)                 # loads the dataset; holds env.sun / env.atmosphere
    caps = inner.capabilities()
    notes = []
    alt_used = 0.0
    if altitude_m and altitude_m > 0:
        if caps.altitude:
            a.dataset.kind = model.DatasetKind("full")
            a.observer_altitude_m.value = float(altitude_m)
            alt_used = float(altitude_m)
        else:
            notes.append(f"altitude {altitude_m} m not used: the dataset is ground level only (zero altitude)")
    ev = _Oriented(inner, centre_deg)
    img = render.render_environment(env, width, width // 2, evaluator=ev, cache=cache.EnvironmentCache())
    px = np.nan_to_num(np.asarray(img.pixels, np.float32), nan=0.0, posinf=3.0e38, neginf=0.0)
    d_sun = azel_to_dir(float(az) - float(centre_deg), float(el))
    e_sun = _sun_irradiance(ev, d_sun)
    e_sky = _sky_irradiance_up(ev)
    return {"lin_rec709": px, "e_sun": e_sun, "e_sky_up": e_sky, "capabilities": caps.to_dict(),
            "altitude_used_m": alt_used, "notes": notes, "metadata": dict(img.metadata),
            "evaluator": ev.implementation_version, "hdri_sun_energy": hdri_sun_energy(
                px, az, el, centre_deg, exposure_ev, e_sun)}


# ------------------------------------------------------------------------------------------------ Hosek-Wilkie render
def render_latlong(ev, width, ground_albedo, exposure_ev, e_sun, e_sky_up, el):
    """Our own lat-long render for an evaluator already in the lat-long frame (sky + sun per pixel centre, the
    lower hemisphere filled with a Lambertian ground albedo / pi * (E_sky_up + E_sun sin el), as light_rig and Atlas
    Scene fill it), x 2^exposure_ev. float32 (H, W, 3) Linear Rec.709. Used for Hosek-Wilkie; the Prague path keeps
    Atlas Scene's renderer."""
    W, H = int(width), int(width) // 2
    lon = 2 * math.pi * (np.arange(W) + 0.5) / W - math.pi
    ground = float(ground_albedo) / math.pi * (np.asarray(e_sky_up, np.float64) +
                                               np.asarray(e_sun, np.float64) * max(math.sin(math.radians(el)), 0.0))
    out = np.empty((H, W, 3), np.float32)
    rows = max(1, (1 << 18) // W)
    for r0 in range(0, H, rows):
        lat = math.pi / 2 - math.pi * (np.arange(r0, min(H, r0 + rows)) + 0.5) / H
        LON, LAT = np.meshgrid(lon, lat)
        d = np.stack([np.sin(LON) * np.cos(LAT), np.sin(LAT), -np.cos(LON) * np.cos(LAT)], -1).reshape(-1, 3)
        L = np.asarray(ev.sky_radiance(d), np.float64) + np.asarray(ev.sun_radiance(d), np.float64)
        L[d[:, 1] <= 0.0] = ground
        out[r0:r0 + LAT.shape[0]] = (L * 2.0 ** float(exposure_ev)).reshape(LAT.shape[0], W, 3)
    return out


def turbidity_for_visibility(visibility_km: float) -> tuple[float, dict]:
    """Hosek-Wilkie turbidity for a meteorological visibility, from data/hosek_visibility_turbidity.json
    (piecewise linear in log visibility, clamped to the table). Approximate; the table records how it was made."""
    t = json.loads((Path(__file__).resolve().parent / "data" / "hosek_visibility_turbidity.json").read_text(
        encoding="utf-8"))
    v = np.array([p[0] for p in t["points"]], np.float64)
    T = np.array([p[1] for p in t["points"]], np.float64)
    x = float(np.clip(visibility_km, v[0], v[-1]))
    return float(np.interp(math.log(x), np.log(v), T)), {"method": t["method"], "provenance": t["provenance"]}


def render_sky_hosek(az, el, *, width, centre_deg=0.0, visibility_km=40.0, ground_albedo=0.25, exposure_ev=0.0,
                     altitude_m=0.0) -> dict:
    """render_sky's result with the Hosek-Wilkie fallback (atlas_vfx/layers/sky_hosek.py) instead of Prague."""
    width = int(width)
    if width < 64 or width % 4:
        raise ValueError(f"width {width}: at least 64 and a multiple of 4")
    if not PRAGUE_VISIBILITY_KM[0] <= visibility_km <= PRAGUE_VISIBILITY_KM[1]:
        raise ValueError(f"visibility {visibility_km} km outside {PRAGUE_VISIBILITY_KM[0]}-{PRAGUE_VISIBILITY_KM[1]} km")
    if el < HOSEK_MIN_ELEVATION:
        raise ValueError(f"sun elevation {el:.3f} below Hosek-Wilkie's {HOSEK_MIN_ELEVATION} deg; nothing to render")
    notes = []
    if altitude_m and altitude_m > 0:
        notes.append(f"altitude {altitude_m} m not used: Hosek-Wilkie is sea level")
    turb, mapping = turbidity_for_visibility(visibility_km)
    inner = sky_hosek.HosekWilkieEvaluator.from_compass(float(az), float(el), turb, float(ground_albedo))
    ev = _Oriented(inner, centre_deg)
    d_sun = azel_to_dir(float(az) - float(centre_deg), float(el))
    e_sun = _sun_irradiance(ev, d_sun)
    e_sky = _sky_irradiance_up(ev)
    px = render_latlong(ev, width, ground_albedo, exposure_ev, e_sun, e_sky, float(el))
    return {"lin_rec709": px, "e_sun": e_sun, "e_sky_up": e_sky, "capabilities": inner.capabilities().to_dict(),
            "altitude_used_m": 0.0, "notes": notes, "metadata": {"renderer": "atlas_vfx physical_sky.render_latlong"},
            "evaluator": ev.implementation_version, "turbidity": turb, "turbidity_mapping": mapping,
            "model": "hosek-wilkie", "hdri_sun_energy": hdri_sun_energy(px, az, el, centre_deg, exposure_ev, e_sun)}


# ------------------------------------------------------------------------------------------------ light rig
def light_rig(sun_dirs_cam, e_sun, e_sky_up, ground_albedo, sun_el, fov_x_deg, aspect,
              source="physical sky (Prague via Atlas Scene)") -> dict:
    """ATLAS_VFX_LIGHT_RIG (the dict Light From Plate makes and Relight reads) from the physical sky.

    Normalised so the sky ambient seen by an up-facing surface (E_sky_up / pi) has luminance 1:
      sky          E_sky_up / pi                                  (per frame, Linear Rec.709)
      ground       albedo / pi * (E_sky_up + E_sun * sin(el))     (Lambertian ground, as Atlas Scene fills it)
      sun_colour   E_sun / max channel                            (max channel 1, like Light From Plate)
      sun_strength max channel of E_sun / pi, in the same units: sun_colour * sun_strength is the physical key a
                   surface facing the sun gets, against a sky of 1. Relight's 'physical sky' mode uses it.
    A sun under the horizon (or a black sky) gives zero colour and strength; the direction stays exact."""
    dirs = np.asarray(sun_dirs_cam, np.float64).reshape(-1, 3)
    F = len(dirs)
    e_sun = np.maximum(np.asarray(e_sun, np.float64), 0.0)
    e_sky = np.maximum(np.asarray(e_sky_up, np.float64), 0.0)
    sky = e_sky / math.pi
    ground = float(ground_albedo) / math.pi * (e_sky + e_sun * max(math.sin(math.radians(sun_el)), 0.0))
    ref = float(LUM709 @ sky)
    k = 1.0 / ref if ref > 0 else 0.0
    peak = float(e_sun.max())
    colour = e_sun / peak if peak > 0 else np.zeros(3)
    uv, front = project_uv(dirs, fov_x_deg, aspect)
    return {"frames": F, "fov_x": float(fov_x_deg), "aspect": float(aspect),
            "sun_uv": uv.round(4).tolist(), "sun_in_front": front.tolist(),
            "sun_dir": dirs.round(6).tolist(), "sun_colour": colour.round(6).tolist(),
            "sun_strength": round(peak / math.pi * k, 6),
            "sky": [(sky * k).round(6).tolist()] * F, "ground": [(ground * k).round(6).tolist()] * F,
            "colourspace": REC709, "normalised": "sky (E_sky_up / pi) luminance = 1",
            "source": source, "sun_visible": bool(peak > 0)}


# ------------------------------------------------------------------------------------------------ colour + preview
def to_working(rgb709, ocio_config_path=""):
    """Linear Rec.709 -> ACEScg by the named OCIO transform (shipped config when blank). float32, unclipped."""
    out = np.ascontiguousarray(rgb709, np.float32).copy()
    pl.ocio_config(ocio_config_path).getProcessor(REC709, WORKING).getDefaultCPUProcessor().applyRGB(out)
    return out


def preview(acescg, sun_az, sun_el, centre, heading, fov_x, ocio_config_path="", display=pl.DEFAULT_DISPLAY,
            view=pl.DEFAULT_VIEW, max_width=1024):
    """Display-referred look at the HDRI through the named OCIO display / view, auto-exposed so the median sky
    sits at 0.18; magenta ring on the sun, yellow lines at the camera's horizontal field of view. Returns
    (image float32 0-1, exposure EV applied)."""
    import cv2
    import PyOpenColorIO as ocio

    img = np.asarray(acescg, np.float32)
    H, W = img.shape[:2]
    if W > max_width:
        img = cv2.resize(img, (max_width, max_width // 2), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
    up = img[: H // 2].reshape(-1, 3) @ np.array([0.2722, 0.6741, 0.0537], np.float32)   # AP1 luminance
    med = float(np.median(up)) if up.size else 0.0
    pev = math.log2(0.18 / med) if med > 0 else 0.0
    out = np.ascontiguousarray(img * np.float32(2.0 ** pev))
    cfg = pl.ocio_config(ocio_config_path)
    cfg.getProcessor(ocio.DisplayViewTransform(src=WORKING, display=display, view=view)).getDefaultCPUProcessor() \
        .applyRGB(out)
    out = np.clip(out, 0.0, 1.0)
    x, y = sun_pixel(sun_az, sun_el, W, H, centre)          # ringed even under the horizon: it is still the sun
    cv2.circle(out, (int(round(x)), int(round(y))), max(6, W // 120), (1.0, 0.0, 1.0), 2, cv2.LINE_AA)
    for edge in (-fov_x / 2, fov_x / 2):
        x, _ = sun_pixel(heading + edge, 0.0, W, H, centre)
        xi = int(round(x)) % W
        out[H // 4: 3 * H // 4, xi] = (1.0, 1.0, 0.0)
    xh, _ = sun_pixel(heading, 0.0, W, H, centre)
    out[H // 2 - 4: H // 2 + 5, int(round(xh)) % W] = (1.0, 1.0, 0.0)
    return out.astype(np.float32), pev
