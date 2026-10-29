"""Physical Sky (catalogue 48): NOAA sun vs a published reference and vs Atlas Scene's derive_sun, the lat-long
convention (the sun's brightest pixel lands where its azimuth / elevation say), the colour hop to ACEScg, the light
rig into Relight, shot-camera frames, below-horizon behaviour, confined outputs, and the adapter failing clearly
when Atlas Scene is missing while the node still registers.

The sky render needs Atlas Scene + its Prague extension + a Prague dataset on this machine (ws has them). If any is
missing this test FAILS (it never skips): set ATLAS_SCENE_ROOT / ATLAS_SCENE_PRAGUE_NATIVE / ATLAS_PRAGUE_DATASET.

Run: <V141 python> tests\\test_physical_sky.py   (plain script, exits non-zero on failure; run from a neutral cwd)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="atlas_vfx_sky_"))
os.environ["ATLAS_VFX_WORK"] = str(TMP / "work")          # before atlas_vfx is imported: plate.WORK_ROOT reads it
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)
sys.path.insert(0, str(ROOT))

from atlas_vfx import _atlas_scene  # noqa: E402
from atlas_vfx.camera_side import exr_io  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import physical_sky as ps  # noqa: E402
from atlas_vfx.layers import relight as rl  # noqa: E402
from atlas_vfx.layers import solar  # noqa: E402

FAILS: list[str] = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if str(detail) != "" else ""), flush=True)
    if not cond:
        FAILS.append(label)


def dataset() -> str:
    """ATLAS_PRAGUE_DATASET from this process, else (Windows) the user's persisted environment, as Atlas Scene's
    docs tell you to set it. Missing = a loud failure, not a skip."""
    p = os.environ.get("ATLAS_PRAGUE_DATASET", "")
    if not p and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                p = winreg.QueryValueEx(k, "ATLAS_PRAGUE_DATASET")[0]
        except OSError:
            p = ""
    if not p or not Path(p).is_file():
        raise SystemExit(f"FAIL: no Prague dataset (ATLAS_PRAGUE_DATASET={p!r}); the sky render cannot be tested")
    os.environ["ATLAS_PRAGUE_DATASET"] = p
    return p


def node_args(**over):
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    req = AtlasVFXPhysicalSky.INPUT_TYPES()["required"]
    d = {k: (v[1]["default"] if len(v) > 1 and "default" in v[1] else v[0][0]) for k, v in req.items()}
    d["out_dir"] = ""
    d.update(over)
    return d


def run_node(**over):
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    r = AtlasVFXPhysicalSky().run(**node_args(**over))["result"]
    return r[0][0].numpy(), r[1], r[2][0].numpy(), json.loads(r[3]), r[4]


# ------------------------------------------------------------------------------------------------ NOAA sun
def test_noaa_vs_published():
    """NREL SPA (Reda & Andreas 2004, spa_tester.c): 2003-10-17 12:30:30 at UTC-7, lat 39.742476, lon -105.1786,
    1830.14 m, 820 mbar, 11 C -> topocentric zenith 50.111622 deg (refraction applied), azimuth 194.340241 deg.
    NOAA is geometric and geocentric, so the published zenith gets SPA's own refraction term added back and the
    solar parallax (8.794 arcsec x sin zenith) taken off before comparing."""
    az, el = solar.sun_at(39.742476, -105.1786, datetime(2003, 10, 17, 19, 30, 30, tzinfo=timezone.utc))
    e_app = 90.0 - 50.111622
    e0 = e_app
    for _ in range(5):                                       # SPA eq. 42 refraction at the geometric elevation
        refr = (820 / 1010) * (283 / (273 + 11)) * 1.02 / (60 * math.tan(math.radians(e0 + 10.3 / (e0 + 5.11))))
        e0 = e_app - refr
    e_geo = e0 + 8.794 / 3600 * math.cos(math.radians(e0))
    check(abs(az - 194.340241) < 0.02, "NOAA azimuth vs NREL SPA published", f"{az:.5f} vs 194.34024, "
          f"diff {az - 194.340241:+.5f}")
    check(abs(el - e_geo) < 0.02, "NOAA elevation vs NREL SPA published (refraction + parallax removed)",
          f"{el:.5f} vs {e_geo:.5f}, diff {el - e_geo:+.5f}")


def test_noaa_known_shot():
    """DSCF2977 at Burning Palms, 6:39 pm AEDT on 25 Dec 2025 (2026-10-28 relight check): 252.4 az, 16.1 up."""
    t = solar.parse_utc("2025-12-25T07:39:25Z")
    az, el = solar.sun_at(-34.1906, 151.0425, t)
    check(abs(az - 252.4) < 0.05 and abs(el - 16.1) < 0.05, "DSCF2977 sun 252.4 az / 16.1 el", (round(az, 3), round(el, 3)))
    t2 = solar.parse_utc("2025-12-25T18:39:25+11:00")
    check(t2 == t, "offset times parse to the same UTC instant")
    for bad in ("2025-12-25T07:39:25", "", "yesterday"):
        try:
            solar.parse_utc(bad)
            check(False, f"capture time {bad!r} refused")
        except ValueError as e:
            check(True, f"capture time {bad!r} refused", str(e)[:70])


def test_noaa_vs_atlas_scene():
    """Same instant, same place: our NOAA vs Atlas Scene's derive_sun wherever derive_sun accepts the elevation;
    below -4.2 deg derive_sun refuses and ours still answers (why atlas-vfx has its own)."""
    sun_mod = _atlas_scene.import_module("environment.sun")
    cam_mod = _atlas_scene.import_module("scene.camera")
    worst, n, refused = 0.0, 0, 0
    for lat in (-62.0, -34.1906, 0.5, 39.7, 64.1):
        for lon in (-122.4, 0.0, 151.0425):
            for day in ("2025-03-20", "2025-06-21", "2025-12-25"):
                for hour in range(0, 24, 3):
                    iso = f"{day}T{hour:02d}:17:42Z"
                    az, el = solar.sun_at(lat, lon, solar.parse_utc(iso))
                    try:
                        s = sun_mod.derive_sun(cam_mod.CaptureLocation(lat, lon), cam_mod.CaptureTime(utc=iso))
                    except Exception:  # noqa: BLE001 - Atlas Scene's SunPosition refuses < -4.2
                        refused += 1
                        if el >= -4.2:
                            check(False, f"derive_sun refused a valid sun {lat} {lon} {iso} el {el:.2f}")
                        continue
                    da = abs(((s.azimuth_deg - az + 180) % 360) - 180)
                    worst = max(worst, da, abs(s.elevation_deg - el))
                    n += 1
    check(n > 100 and worst < 0.01, "NOAA matches Atlas Scene derive_sun", f"{n} suns, worst {worst:.2e} deg")
    check(refused > 50, "derive_sun refuses suns below -4.2 deg; ours answers them", f"{refused} refused")


# ------------------------------------------------------------------------------------------------ conventions
def test_frames_and_camera():
    d = ps.azel_to_dir([0, 90, 180, 270], 0)
    check(np.allclose(d, [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0]], atol=1e-12), "sky frame: N -Z, E +X, S +Z, W -X")
    az, el = ps.dir_to_azel(ps.azel_to_dir(123.4, 21.0))
    check(abs(az - 123.4) < 1e-9 and abs(el - 21.0) < 1e-9, "azel <-> direction round trip")
    check(ps.sun_pixel(0, 0, 2048, 1024) == (1023.5, 511.5), "north on the horizon = centre of a north lat-long")
    check(ps.sun_pixel(90, 0, 2048, 1024)[0] == 1535.5, "east a quarter right of north (azimuth grows right)")
    check(ps.sun_pixel(200, 0, 2048, 1024, centre_deg=200)[0] == 1023.5, "camera-heading centre puts heading mid-frame")
    M = ps.latlong_to_atlas(0.0)
    check(abs(np.linalg.det(M) + 1) < 1e-12, "lat-long -> Atlas Scene axes is a mirror (det -1)")
    # Atlas Scene's Prague boundary reads Atlas (x, y, z) as Prague (x, -z, y), sun at (cos A, sin A) there
    A, e = math.radians(252.355), math.radians(16.12)
    atlas_sun = np.array([math.cos(A) * math.cos(e), math.sin(e), -math.sin(A) * math.cos(e)])
    check(np.allclose(M @ ps.azel_to_dir(252.355, 16.12), atlas_sun, atol=1e-12), "mapped sun = Atlas Scene's sun")
    # camera: heading 180 (south), sun at 252 (to the right of south seen from the southern hemisphere's west-ish)
    d_cam, meta = ps.sun_in_camera(252.355, 16.12, 180.0)
    check(d_cam[0][0] > 0 and d_cam[0][2] < 0 and d_cam[0][1] > 0, "sun 72 deg right of heading: camera x > 0, ahead, up",
          d_cam[0].round(3))
    d_ahead, _ = ps.sun_in_camera(180.0, 0.0, 180.0)
    check(np.allclose(d_ahead[0], [0, 0, -1], atol=1e-12), "sun at the heading on the horizon = straight down -Z")
    uv, front = ps.project_uv([[0.0, 0.0, -1.0]], 50.0, 16 / 9)
    check(np.allclose(uv[0], [0.5, 0.5]) and front[0], "sun ahead projects to frame centre")
    back = rl.sun_direction(0.8, 0.3, 50.0, 16 / 9, behind=1.0)
    uv, _ = ps.project_uv([back], 50.0, 16 / 9)
    check(np.allclose(uv[0], [0.8, 0.3], atol=1e-9), "project_uv inverts relight.sun_direction", uv[0].round(6))


def test_shot_camera_frames():
    """Frame 0 of a shot camera looks at `heading`; later frames follow the solve. A +10 deg yaw about +Y (a pan to
    the LEFT seen from above in a Y-up world) must read as heading - 10."""
    def ry(deg):
        a = math.radians(deg)
        M = np.eye(4)
        M[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        return M
    plate = {"path": "synthetic", "width": 4096, "height": 2160, "colorspace": "ACEScg", "range": "full"}
    sensor = {"width_mm": 23.5, "height_mm": 12.39}
    cam = sc.build(name="sky_t", frame_start=1001, fps=24.0, plate=plate, sensor=sensor, cam_to_world=[ry(37.0), ry(47.0)],
                   focal_mm=16.47, source={"kind": "synthetic", "solver": "tests/test_physical_sky.py"})
    dirs, meta = ps.sun_in_camera(252.355, 16.12, 200.0, shot_camera=cam.data)
    t0, _ = ps.sun_in_camera(252.355, 16.12, 200.0)
    t1, _ = ps.sun_in_camera(252.355, 16.12, 190.0)
    check(np.allclose(dirs[0], t0[0], atol=1e-9), "shot camera frame 0 = typed camera at heading")
    check(np.allclose(dirs[1], t1[0], atol=1e-9), "shot camera frame 1 (+10 yaw) = typed camera at heading - 10")
    fov = math.degrees(2 * math.atan(23.5 / (2 * 16.47)))
    check(abs(meta["fov_x"] - fov) < 1e-9 and meta["frames"] == 2, "shot camera gives fov_x and frames",
          round(meta["fov_x"], 4))


# ------------------------------------------------------------------------------------------------ colour
def test_colour_hop():
    """Linear Rec.709 -> ACEScg through the shipped config is the published 3x3 (Bradford D65 -> D60 white), linear
    and unclipped up to sun-disk values."""
    eye = np.eye(3, dtype=np.float32)
    M = ps.to_working(eye[None]).reshape(3, 3).T                         # columns = images of the basis
    want = np.array([[0.6131, 0.3395, 0.0474], [0.0702, 0.9164, 0.0134], [0.0206, 0.1096, 0.8698]])
    check(np.allclose(M, want, atol=2e-3), "OCIO Rec.709 -> ACEScg matrix", M.round(4).tolist())
    big = np.array([[[1.5e6, 9.0e5, 4.0e5]]], np.float32)
    out = ps.to_working(big)
    check(np.allclose(out[0, 0], M @ big[0, 0], rtol=1e-4) and out.dtype == np.float32,
          "sun-disk values pass unclipped in float32", out[0, 0])


# ------------------------------------------------------------------------------------------------ sky renders
RENDERS: dict = {}


def brightest(img):
    y, x = np.unravel_index(int(np.argmax(img.sum(-1))), img.shape[:2])
    return int(x), int(y)


def wrap_dx(a, b, W):
    return abs(((a - b + W / 2) % W) - W / 2)


def test_sun_pixel_north():
    hdri, rig, prev, info, exr = run_node(name="sky_north", latlong_centre="north")
    RENDERS["north"] = (hdri, rig, info, exr)
    H, W = hdri.shape[:2]
    px, py = ps.sun_pixel(info["sun"]["azimuth_deg"], info["sun"]["elevation_deg"], W, H, 0.0)
    bx, by = brightest(hdri)
    check(info["sky_rendered"] and (W, H) == (2048, 1024), "sky rendered at 2048 x 1024", info["sky_rendered"])
    check(wrap_dx(bx, px, W) <= 1.5 and abs(by - py) <= 1.5, "north lat-long: brightest pixel at the predicted sun",
          f"got ({bx}, {by}), predicted ({px:.2f}, {py:.2f}) for az {info['sun']['azimuth_deg']:.3f} "
          f"el {info['sun']['elevation_deg']:.3f}")
    mirrored = W - 1 - px
    check(wrap_dx(bx, mirrored, W) > 100, "and not at the mirrored column (the 2026-10-28 trap)", round(mirrored, 1))
    check(hdri.dtype == np.float32 and np.isfinite(hdri).all() and hdri.max() > 1e4, "HDRI float32, finite, unclipped",
          f"max {hdri.max():.4g}")
    check(hdri[H // 2 + 5:].mean() > 0 and hdri[: H // 2].mean() > 0, "sky above, Atlas Scene's ground fill below")
    e = info["provenance"]["hdri_sun_energy"]
    check(e is not None and 0.5 < e < 1.5, "HDRI holds the sun's energy (point-sampled disk)", round(e, 4))
    print("     info:", json.dumps({k: info[k] for k in ("sun", "light_rig")})[:600])


def test_sun_pixel_camera_heading():
    hdri, rig, prev, info, exr = run_node(name="sky_heading", latlong_centre="camera heading", heading_deg=200.0,
                                          sun_mode="manual", sun_azimuth=45.0, sun_elevation=30.0)
    H, W = hdri.shape[:2]
    px, py = ps.sun_pixel(45.0, 30.0, W, H, 200.0)
    bx, by = brightest(hdri)
    check(wrap_dx(bx, px, W) <= 1.5 and abs(by - py) <= 1.5, "camera-heading lat-long, manual sun: brightest pixel "
          "at the predicted sun", f"got ({bx}, {by}), predicted ({px:.2f}, {py:.2f})")
    check(info["sun"]["mode"] == "manual" and info["inputs"]["latlong_centre_deg"] == 200.0, "info records mode and centre")


def test_files_and_provenance():
    hdri, rig, info, exr = RENDERS["north"]
    exr = Path(exr)
    side = exr.with_name("sky_north.sidecar.json")
    check(exr.is_file() and side.is_file(), "EXR + sidecar written", exr.parent)
    check(Path(os.environ["ATLAS_VFX_WORK"]).resolve() in exr.resolve().parents, "default out_dir under WORK_ROOT")
    h = exr_io.header(exr)
    check(h["format"] == "float" and h["ocio_colorspace"] == "ACEScg" and h["colorInteropID"] == "lin_ap1_scene",
          "EXR is float32, ACEScg in the header", h)
    px, _ = exr_io.read_rgba(exr)
    check(np.allclose(px[..., :3], hdri, rtol=1e-6, atol=0), "EXR pixels = the IMAGE output")
    prov = info["provenance"]["atlas_scene"]
    check(prov["root"] and prov["version"] and prov["commit"] and len(prov["commit"]) == 40,
          "provenance names atlas_scene path, version and git commit", f"{prov['root']} {prov['version']} "
          f"{prov['commit'][:10]} dirty={prov['dirty']}")
    check(prov["prague"]["upstream_commit"] == "2385c912e9051c1258013ff8c3ce2e19e10fb917" and prov["prague"]["native"],
          "provenance names the Prague upstream commit and the compiled module", prov["prague"]["native"])
    chk = info["provenance"].get("atlas_scene_derive_sun", {})
    check(chk.get("ok") and max(abs(x) for x in chk["diff_deg"]) < 0.01, "info cross-checks derive_sun", chk.get("diff_deg"))
    doc = json.loads(side.read_text(encoding="utf-8"))
    check(doc["node"] == "AtlasVFXPhysicalSky" and doc["outputs"]["sky_rendered"], "sidecar holds the info")


def test_rig_into_relight():
    """The physical sun (72 deg right of the camera) replaces the plate's sun (up-left) in Relight; unconnected,
    Relight is unchanged; 'physical sky' strength = sun_strength x plate sky luminance x trim."""
    import torch
    from atlas_vfx.layers.node_relight import AtlasVFXRelight
    H, W, T = 120, 160, 4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = (xx - W / 2) / 40, -(yy - H / 2) / 40
    al = (x * x + y * y < 1).astype(np.float32)
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    n = np.stack([x, y, z], -1) * al[..., None] + np.array([0, 0, 1], np.float32) * (1 - al[..., None])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    bg = np.full((T, H, W, 3), 0.05, np.float32)
    bg[:, : H // 3] = (0.30, 0.25, 0.18)
    d2 = (xx - 0.15 * W) ** 2 + (yy - 0.1 * H) ** 2
    bg += (np.exp(-d2 / 30.0) * 8.0)[None, ..., None] * np.array([1.0, 0.6, 0.3], np.float32)
    plate_rig = rl.light_from_plate(bg, smooth=1)
    fg = np.full((T, H, W, 3), 0.4, np.float32) * al[None, ..., None]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))     # noqa: E731
    args = dict(element=t(fg), matte=t(np.repeat(al[None], T, 0)), normals=t(np.repeat(((n + 1) / 2)[None], T, 0)),
                light_rig=plate_rig, encoding="linear", sun_gain=4.0, ambient_gain=1.0, studio_flat=0.45,
                facing_power=2.0, fresnel_power=4.0, edge_rim=1.5, edge_px=8.0, rim_max=0.0, exposure_match=False,
                exposure_bias=0.9, exposure_smooth=1)
    node = AtlasVFXRelight()
    base = node.run(**args)
    relit, passes, _ = rl.relight(fg, np.repeat(al[None], T, 0), np.repeat(n[None], T, 0), plate_rig, None, 4.0, 1.0,
                                  rim_max=0.0, exposure_match=False, exposure_smooth=1)
    check(np.allclose(base[0].numpy(), relit, atol=1e-5), "Relight without sun_rig = the old behaviour")
    side = lambda key: (float(key[:, : W // 2][al[:, : W // 2] > 0].mean()),            # noqa: E731
                        float(key[:, W // 2:][al[:, W // 2:] > 0].mean()))
    l0, r0 = side(base[2].numpy()[0] @ rl.LUM)
    check(l0 > 5 * max(r0, 1e-6), "plate sun up-left: rim on the left", (round(l0, 4), round(r0, 4)))
    sun_rig = RENDERS["north"][1]
    out = node.run(**args, sun_rig=sun_rig)
    l1, r1 = side(out[2].numpy()[0] @ rl.LUM)
    check(r1 > 5 * max(l1, 1e-6), "physical sun 72 deg right: rim moves to the right", (round(l1, 4), round(r1, 4)))
    rep = json.loads(out[3])
    check(rep["sun_source"].startswith("sun_rig") and rep["sun_dir_first"] == sun_rig["sun_dir"][0], "report names the sun source")
    phys = json.loads(node.run(**{**args, "sun_gain": 1.0}, sun_rig=sun_rig, sun_strength_from="physical sky")[3])
    sky = float(np.median(np.asarray(plate_rig["sky"]).reshape(-1, 3) @ rl.LUM))
    check(abs(phys["sun_gain"] - sun_rig["sun_strength"] * sky) < 1e-9, "physical strength = sun_strength x plate sky",
          f"{phys['sun_gain']:.4f} = {sun_rig['sun_strength']} x {sky:.4f}")
    merged = rl.override_sun(plate_rig, sun_rig)
    check(merged["frames"] == T and len(merged["sun_dir"]) == T and merged["sky"] == plate_rig["sky"],
          "one-frame sun rig held over the plate's frames, ambient kept")
    try:
        node.run(**args, sun_strength_from="physical sky")
        check(False, "physical strength without a sun_rig refused")
    except ValueError:
        check(True, "physical strength without a sun_rig refused")


def test_below_horizon():
    """2 am AEDT: the sun is far below -4.2 deg. No sky is faked: black HDRI, a warning, sky_rendered false; the rig
    still points at the real (under-horizon) sun with zero strength."""
    hdri, rig, prev, info, exr = run_node(name="sky_night", capture_utc="2025-12-25T15:00:00Z", width=256)
    el = info["sun"]["elevation_deg"]
    check(el < -4.2 and not info["sun"]["renderable"], "night sun below Prague's domain", round(el, 2))
    check(not info["sky_rendered"] and info["warnings"] and float(np.abs(hdri).max()) == 0.0,
          "HDRI black, warning raised, sky_rendered false", info["warnings"][:1])
    check(rig["sun_dir"][0][1] < 0 and rig["sun_strength"] == 0 and not rig["sun_visible"],
          "rig: sun direction below the horizon, zero strength", rig["sun_dir"][0])
    _, el_ref = solar.sun_at(-34.1906, 151.0425, solar.parse_utc("2025-12-25T15:00:00Z"))
    check(abs(el - el_ref) < 1e-9 and Path(exr).is_file(), "direction is the exact NOAA sun; EXR still written")
    hdri2, _, _, info2, _ = run_node(name="sky_offset_t", capture_utc="2025-12-25T15:00:00Z", time_offset_h=-7.0,
                                     width=256)
    check(info2["sky_rendered"] and info2["sun"]["sun_utc"] == "2025-12-25T08:00:00Z", "time_offset_h relights to "
          "another hour (2 am -> 7 pm AEDT renders)", round(info2["sun"]["elevation_deg"], 2))


def test_outputs_confined():
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    node = AtlasVFXPhysicalSky()
    outside = Path(tempfile.gettempdir()).parent / "not_a_work_root_sky"
    for label, over in (("out_dir outside the work roots", {"out_dir": str(outside)}),
                        ("out_dir inside the repo", {"out_dir": str(ROOT / "workflows")}),
                        ("name with a path in it", {"name": "..\\escape"}),
                        ("name with quotes", {"name": "a'''b"})):
        try:
            node.run(**node_args(width=256, capture_utc="2025-12-25T15:00:00Z", **over))
            check(False, f"refused: {label}")
        except ValueError as e:
            check(True, f"refused: {label}", str(e)[:80])
    check(not outside.exists(), "nothing written outside the roots")
    for bad in ({"width": 1026}, {"visibility_km": 10.0, "capture_utc": "2025-12-25T07:39:25Z"}):
        try:
            node.run(**node_args(**{"width": 256, **bad}))
            check(False, f"refused: {bad}")
        except ValueError as e:
            check(True, f"refused: {bad}", str(e)[:80])


def test_adapter_missing():
    """A Python with no Atlas Scene anywhere: the node still registers, and running it says exactly how to fix it."""
    empty = TMP / "no_home"
    empty.mkdir(exist_ok=True)
    fake_ds = TMP / "fake.dat"
    fake_ds.write_bytes(b"0")
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from atlas_vfx.registry import load_nodes
        from atlas_vfx import _atlas_scene
        classes, names, disabled = load_nodes(("atlas_vfx.layers",))
        assert "AtlasVFXPhysicalSky" in classes and "AtlasVFXPhysicalSky" in names, disabled
        assert not _atlas_scene.available(), _atlas_scene.report()
        from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
        req = AtlasVFXPhysicalSky.INPUT_TYPES()["required"]
        d = {{k: (v[1]["default"] if len(v) > 1 and "default" in v[1] else v[0][0]) for k, v in req.items()}}
        d["width"] = 256
        try:
            AtlasVFXPhysicalSky().run(**d)
        except _atlas_scene.AtlasSceneMissing as e:
            print("MESSAGE", e)
            sys.exit(0)
        sys.exit("node ran without Atlas Scene")
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ATLAS_SCENE")}
    env.update(USERPROFILE=str(empty), HOME=str(empty), ATLAS_SCENE_ROOT=str(empty), ATLAS_PRAGUE_DATASET=str(fake_ds),
               ATLAS_VFX_WORK=str(TMP / "work2"), ATLAS_VFX_OUT_ROOTS=str(TMP))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(TMP))
    msg = r.stdout + r.stderr
    check(r.returncode == 0, "no Atlas Scene: node registers, run raises AtlasSceneMissing", msg.strip()[-300:])
    check(all(s in msg for s in ("ATLAS_SCENE_ROOT", "pip install -e", "ATLAS_SCENE_PRAGUE_NATIVE")),
          "the message says how to fix it", msg.strip()[:160])


if __name__ == "__main__":
    print("dataset:", dataset())
    if not _atlas_scene.available():
        raise SystemExit("FAIL: Atlas Scene not found: " + _atlas_scene.install_message("(atlas_scene)"))
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - one broken test must not hide the others
                import traceback
                traceback.print_exc()
                check(False, f"{name} raised {type(e).__name__}: {e}")
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
