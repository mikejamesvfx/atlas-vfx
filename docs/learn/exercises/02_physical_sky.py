"""Learning exercise 02: Physical Sky, from a place and a time to a sky, a sun and a light rig.

Guide: docs/learn/02-physical-sky.md. Runs with no plate, no Atlas Scene, no GPU and no network: the sky is the
Hosek-Wilkie fallback that ships in atlas_vfx/layers/sky_hosek.py. Everything is synthetic (a place, a time, a
typed camera). Nothing is written to disk.

Run from the repo root with the ComfyUI venv's python (it needs numpy and PyOpenColorIO, both in V141):
    python docs/learn/exercises/02_physical_sky.py

Exits non-zero if any check fails. Takes a few seconds.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TMP = Path(tempfile.mkdtemp(prefix="atlas_vfx_learn_sky_"))
os.environ["ATLAS_VFX_WORK"] = str(TMP / "work")     # set before atlas_vfx is imported; nothing is written anyway
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from atlas_vfx.layers import physical_sky as ps  # noqa: E402
from atlas_vfx.layers import solar  # noqa: E402
from atlas_vfx.layers.node_sun_direction import from_light_rig  # noqa: E402

FAILS: list[str] = []


def check(cond, label, detail=""):
    print(("   ok   " if cond else "   FAIL ") + label + (f"  ({detail})" if str(detail) != "" else ""))
    if not cond:
        FAILS.append(label)


def step(n, title):
    print(f"\n{n}. {title}")


# The places and times we use. Burning Palms is the repo's check shot (DSCF2977, 6:39 pm AEDT, Christmas Day 2025).
SYDNEY = (-33.8688, 151.2093)
BURNING_PALMS = (-34.1906, 151.0425)
DSCF2977_UTC = "2025-12-25T07:39:25Z"
OBLIQUITY = 23.44            # the tilt of the earth's axis, degrees: the solstice sun sits this far off the equator

t_start = time.time()

# ------------------------------------------------------------------------------------------------ 1
step(1, "A capture time must say which clock it is in")
t_utc = solar.parse_utc(DSCF2977_UTC)
t_local = solar.parse_utc("2025-12-25T18:39:25+11:00")       # the same instant written in AEDT
check(t_utc == t_local, "'...07:39:25Z' and '...18:39:25+11:00' are the same instant", t_utc.isoformat())
try:
    solar.parse_utc("2025-12-25T18:39:25")                   # no offset: local or UTC? an hour is 15 deg of sun
    check(False, "a bare time is refused")
except ValueError as e:
    check(True, "a bare time is refused", str(e)[:60] + "...")

# ------------------------------------------------------------------------------------------------ 2
step(2, "Where is the sun? NOAA's algorithm (layers/solar.py), checked against simple astronomy")
# At solar noon on a solstice the sun's elevation is 90 - |latitude - declination|, declination = -23.44 in
# December (southern summer) and +23.44 in June. Find noon with sun_path (it searches for the highest sun).
for label, day, decl in (("summer solstice", "2025-12-21T02:00:00Z", -OBLIQUITY),
                         ("winter solstice", "2025-06-21T02:00:00Z", +OBLIQUITY)):
    path = ps.sun_path(*SYDNEY, day)
    noon = path["noon"]
    expect = 90.0 - abs(SYDNEY[0] - decl)
    print(f"   Sydney {label}: solar noon {noon['utc']} UTC, sun az {noon['azimuth_deg']:.2f}, "
          f"el {noon['elevation_deg']:.2f} (simple astronomy says {expect:.2f})")
    check(abs(noon["elevation_deg"] - expect) < 0.3, f"{label} noon elevation matches 90 - |lat - decl|")
    az = noon["azimuth_deg"]
    check(min(az, 360 - az) < 1.0, f"{label} noon sun is due north (Sydney is south of the tropic)", round(az, 3))
    print(f"   sunrise {path['sunrise']['utc']}  sunset {path['sunset']['utc']}  "
          f"evening golden hour (sun at {path['golden_elevation_deg']} deg) {path['golden_evening']['utc']}")
# Try changing: SYDNEY to a place north of the tropic (e.g. (51.5, -0.13), London). The noon sun swings to the south
# (azimuth ~180). Or a date in the arctic summer: sun_path reports polar = "day" and no sunrise or sunset.

az, el = solar.sun_at(*BURNING_PALMS, t_utc)
print(f"   DSCF2977 (Burning Palms, 6:39 pm AEDT): az {az:.3f}, el {el:.3f}")
check(abs(az - 252.355) < 0.01 and abs(el - 16.121) < 0.01, "matches the repo's pinned DSCF2977 sun (252.355 / 16.121)")

# ------------------------------------------------------------------------------------------------ 3
step(3, "The sky frame and the lat-long layout (the part that bit on 2026-10-28)")
dirs = ps.azel_to_dir([0, 90, 180, 270], 0)
for name, d in zip("NESW", dirs):
    print(f"   {name} on the horizon -> {np.round(d, 3).tolist()}")
check(np.allclose(dirs, [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0]], atol=1e-12),
      "right-handed, Y up: north -Z, east +X, south +Z, west -X")
W, H = 2048, 1024
x_n, _ = ps.sun_pixel(0, 0, W, H)
x_e, _ = ps.sun_pixel(90, 0, W, H)
check(x_n == 1023.5 and x_e == 1535.5, "north is the centre column, east a quarter to the RIGHT (azimuth grows right)",
      f"north x {x_n}, east x {x_e}")
x_c, _ = ps.sun_pixel(200, 0, W, H, centre_deg=200)
check(x_c == 1023.5, "latlong_centre 'camera heading' puts the heading in the middle column")
check(abs(np.linalg.det(ps.latlong_to_atlas(0.0)) + 1) < 1e-12,
      "our frame -> Atlas Scene's Prague frame is a mirror (det -1), handled by _Oriented")

# ------------------------------------------------------------------------------------------------ 4
step(4, "The sun as the camera sees it (typed camera: heading + fov + tilt)")
d_cam, meta = ps.sun_in_camera(az, el, heading_deg=180.0, tilt_deg=0.0)
x, y, z = d_cam[0]
print(f"   camera faces south (heading 180); sun in OpenGL camera space x {x:+.3f} y {y:+.3f} z {z:+.3f}")
check(x > 0 and y > 0 and z < 0, "a WSW sun is to the right, above, and in front (camera looks down -Z)")
uv, front = ps.project_uv(d_cam, fov_x_deg=69.5, aspect=16 / 9)
print(f"   projected: u {uv[0][0]:.2f}, v {uv[0][1]:.2f}; in front {bool(front[0])}  (u > 1 means off frame right)")
check(uv[0][0] > 1.0, "72 deg off the lens with a 69.5 deg lens: the sun is off frame right")
# Try changing: heading_deg to 250. The sun moves to u ~ 0.5 (dead ahead, a backlit shot).

# A tracked (shot) camera: frame 0 looks at heading_deg; later frames follow the solve. Two frames, panned 10 deg left.
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402


def yaw(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    return M


cam = sc.build(name="learn_pan", frame_start=1001, fps=24.0,
               plate={"path": "synthetic", "width": 4096, "height": 2160, "colorspace": "ACEScg", "range": "full"},
               sensor={"width_mm": 23.5, "height_mm": 12.39}, cam_to_world=[yaw(0.0), yaw(10.0)], focal_mm=16.0,
               source={"kind": "synthetic", "solver": "docs/learn/exercises/02_physical_sky.py"})
d_shot, meta = ps.sun_in_camera(az, el, 180.0, shot_camera=cam.data)
u0, u1 = ps.project_uv(d_shot, meta["fov_x"], meta["aspect"])[0][:, 0]
print(f"   shot camera: fov_x {meta['fov_x']:.1f} deg from 16 mm on 23.5 mm; world turned {meta['world_yaw_deg']:+.2f} "
      f"deg so frame 1001 faces south; sun u {u0:.3f} -> {u1:.3f} as the camera pans left")
check(u1 > u0, "panning the camera left pushes the sun further to frame right")

# ------------------------------------------------------------------------------------------------ 5
step(5, "Render a Hosek-Wilkie sky (no Atlas Scene needed) and look at the pixels")
t0 = time.time()
r = ps.render_sky_hosek(az, el, width=1024, centre_deg=0.0, visibility_km=40.0, ground_albedo=0.25, exposure_ev=0.0)
img = r["lin_rec709"]                                         # (512, 1024, 3) float32, Linear Rec.709, unclipped
print(f"   {img.shape[1]}x{img.shape[0]} in {time.time() - t0:.1f} s; visibility 40 km -> turbidity {r['turbidity']:.3f}")
lum = img @ ps.LUM709
iy, ix = np.unravel_index(int(np.argmax(lum)), lum.shape)
px, py = ps.sun_pixel(az, el, 1024, 512)
print(f"   brightest pixel ({ix}, {iy}) lum {lum[iy, ix]:.1f}; predicted sun pixel ({px:.1f}, {py:.1f}); "
      f"max over the sky {img.max():.1f} (way over 1: scene-linear, not display)")
check(abs(ix - px) <= 3 and abs(iy - py) <= 3, "the brightest pixel is where the azimuth / elevation say")
lower = img[img.shape[0] // 2 + 1:]
check(float(np.ptp(lower.reshape(-1, 3), axis=0).max()) < 1e-6, "below the horizon: one flat Lambertian ground colour",
      [round(float(v), 3) for v in lower[0, 0]])


def sky_at(a, e):
    """Luminance of the sky alone (no sun disc) in one direction, straight from the evaluator."""
    from atlas_vfx.layers import sky_hosek
    ev = sky_hosek.HosekWilkieEvaluator.from_compass(az, el, r["turbidity"], 0.25)
    return float(ev.sky_radiance(sky_hosek.compass_to_dir(a, e)[None])[0] @ ps.LUM709)


near, far, zen, hor = sky_at(az + 5, el), sky_at(az + 90, el), sky_at(0, 89.9), sky_at(az + 180, 2)
print(f"   sky luminance: 5 deg from the sun {near:.3f}, 90 deg round {far:.3f}, zenith {zen:.3f}, "
      f"low horizon opposite the sun {hor:.3f}")
check(near > 3 * far, "the circumsolar glow: the sky is much brighter near the sun")
print(f"   sun irradiance E_sun {np.round(r['e_sun'], 2).tolist()}, sky on an up-facing card E_sky_up "
      f"{np.round(r['e_sky_up'], 2).tolist()}  (Rec.709, radiometric CIE-weighted units)")
print(f"   HDRI holds {r['hdri_sun_energy']:.2f} x the sun's true energy (point-sampled disc; the rig integrates it)")
# Try changing: width to 256. The disc is under 1.5 px there and can fall between pixel centres, so the HDRI's share
# of the sun (hdri_sun_energy) wanders; the rig below doesn't care.

# ------------------------------------------------------------------------------------------------ 6
step(6, "Haze: visibility_km -> Hosek turbidity, and what it does to the light")
rows = []
for vis in (20.0, 40.0, 100.0):
    rv = ps.render_sky_hosek(az, el, width=256, visibility_km=vis)
    rig_v = ps.light_rig(d_cam, rv["e_sun"], rv["e_sky_up"], 0.25, el, 69.5, 16 / 9)
    z = rv["lin_rec709"][2, 128]                              # a pixel near the zenith
    rows.append((vis, rv["turbidity"], rig_v["sun_strength"], float(z[2] / max(z[0], 1e-9))))
    print(f"   visibility {vis:5.1f} km -> turbidity {rv['turbidity']:.2f}: sun_strength {rig_v['sun_strength']:.2f}, "
          f"zenith blue/red {rows[-1][3]:.2f}")
check(rows[0][1] > rows[1][1] > rows[2][1], "hazier air = higher turbidity")
check(rows[2][2] > rows[0][2], "clearer air = a harder sun against its sky (bigger sun_strength)")
check(rows[2][3] > rows[0][3], "clearer air = a bluer zenith")

# ------------------------------------------------------------------------------------------------ 7
step(7, "The light rig Relight reads (ATLAS_VFX_LIGHT_RIG), normalised to a sky of 1")
rig = ps.light_rig(d_cam, r["e_sun"], r["e_sky_up"], 0.25, el, 69.5, 16 / 9)
sky_lum = float(np.asarray(rig["sky"][0]) @ ps.LUM709)
print(f"   sky {rig['sky'][0]} (luminance {sky_lum:.4f}), ground {rig['ground'][0]}")
print(f"   sun_colour {rig['sun_colour']} (max channel 1), sun_strength {rig['sun_strength']:.3f}, "
      f"sun_dir {rig['sun_dir'][0]}")
check(abs(sky_lum - 1.0) < 1e-3, "the sky ambient is normalised to luminance 1")
check(abs(max(rig["sun_colour"]) - 1.0) < 1e-9 and rig["sun_colour"][0] >= rig["sun_colour"][2],
      "sun colour peaks at 1, and a 16 deg evening sun is warmer than blue")
check(4.0 < rig["sun_strength"] < 12.0, "sun_strength: the key is several times the sky (NOTES: Prague gave 7.07)",
      round(rig["sun_strength"], 3))
# In Relight with sun_strength_from 'physical sky': sun_gain = trim x sun_strength x (the plate's own sky luminance).

# ------------------------------------------------------------------------------------------------ 8
step(8, "Colour: the sky is rendered in Linear Rec.709 and lands in ACEScg by a NAMED OCIO transform")
red = ps.to_working(np.array([[[1.0, 0.0, 0.0]]], np.float32))[0, 0]
white = ps.to_working(np.array([[[1.0, 1.0, 1.0]]], np.float32))[0, 0]
print(f"   OCIO '{ps.REC709}' -> '{ps.WORKING}': red {[round(float(v), 4) for v in red]}, "
      f"white {[round(float(v), 4) for v in white]}")
check(np.allclose(red, [0.6131, 0.0702, 0.0206], atol=0.005), "Rec.709 red lands inside AP1, as expected")
check(np.allclose(white, [1.0, 1.0, 1.0], atol=0.005), "white stays white (no hidden gain)")
hot = ps.to_working(img[int(py) - 2:int(py) + 3, int(px) - 2:int(px) + 3])
check(float(hot.max()) > 1000.0 and np.isfinite(hot).all(), "unclipped float32: the sun survives the hop",
      f"peak {float(hot.max()):.0f}")

# ------------------------------------------------------------------------------------------------ 9
step(9, "Sun Direction: the same rig, in the Relight LoRA's words")
words = from_light_rig(rig)
print(f"   {words['azimuth_in_camera_deg']:+.0f} deg from the camera's back, {words['elevation_deg']:.0f} deg up -> "
      f"'{words['direction']}', '{words['height']}'")
check(words["direction"] == "the right" and words["height"] == "low", "a WSW sun from a south-facing camera: the right, low")

# ------------------------------------------------------------------------------------------------ 10
step(10, "Off the edge of the model: night and twilight are refused, never faked")
night = ps.effective_sun("metadata", *BURNING_PALMS, "2025-12-25T15:00:00Z", min_elevation=ps.HOSEK_MIN_ELEVATION)
print(f"   2 am AEDT: sun el {night['elevation_deg']:.2f}, renderable {night['renderable']}")
check(not night["renderable"], "a sun under the horizon is not renderable for Hosek-Wilkie")
try:
    ps.render_sky_hosek(250.0, -2.0, width=256)
    check(False, "Hosek-Wilkie refuses a twilight sun (-2 deg)")
except ValueError as e:
    check(True, "Hosek-Wilkie refuses a twilight sun (-2 deg); Prague (Atlas Scene) goes down to -4.2", str(e)[:50])
later = ps.effective_sun("metadata", *BURNING_PALMS, "2025-12-25T15:00:00Z", time_offset_h=-7.0)
check(later["renderable"] and later["sun_utc"] == "2025-12-25T08:00:00Z", "time_offset_h relights to another hour",
      f"7 pm AEDT: el {later['elevation_deg']:.2f}")

print(f"\n{'PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  ({time.time() - t_start:.1f} s)")
sys.exit(1 if FAILS else 0)
