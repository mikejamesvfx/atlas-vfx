"""Sensor pass: lens, apply order, crop windows, map files and the nodes (synthetic).

- Brown-Conrady: distort then undistort round trip (strong barrel + tangential) < 1e-6 px; a hand-computed point
- lens maps: no distortion -> G identity in both directions, R / B radially scaled by CA; redistort then undistort per
  channel round trip; vignette gain 1 at the centre, 1 + v1 + v2 + v3 in the corners
- apply order: vignette is applied BEFORE distortion (a flat image comes out with gain(undistort(p)), not gain(p));
  apply then remove gives the image back
- crop windows (LTX crop mode): maps cut to a window and resized give the same pixels as cropping the full result
- map EXRs: float, ZIP, colorInteropID data (OCIO Raw); Nuke Reads name every sequence
- end to end from a synthetic tracked plate (camera + depth folder) through Sensor RS Maps, Sensor Lens Maps and
  Sensor Apply nodes (ATLAS_VFX_LTX_PREP windows), sidecars record where the readout and lens came from

Run: <V141 python> tests\\test_sensor_lens.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("ATLAS_VFX_OUT_ROOTS", tempfile.gettempdir())     # sensor nodes may write under temp

from atlas_vfx.camera_side import plate_depth as pdp  # noqa: E402
from atlas_vfx.sensor import apply as ap  # noqa: E402
from atlas_vfx.sensor import jobs  # noqa: E402
from atlas_vfx.sensor import lens as ln  # noqa: E402
from atlas_vfx.sensor import maps_io as mio  # noqa: E402
from atlas_vfx.sensor import rs  # noqa: E402
from test_sensor_rs import FPS, H, W, XH2_HQ_S, make_cam, scene_depth  # noqa: E402

FAILS = []
STRONG = dict(k1=-0.12, k2=0.03, k3=-0.004, p1=0.001, p2=-0.0008)


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def lens_for(w=W, h=H, **k):
    f = 16.47 * w / 23.5
    return ln.BrownConrady(f, f, w / 2.0, h / 2.0, **k)


def smooth_image(w=W, h=H):
    X, Y = ln.pixel_grid(w, h)
    return np.stack([0.5 + 0.3 * np.sin(X / 37.0) * np.cos(Y / 23.0), 0.4 + 0.2 * np.cos(X / 51.0 + Y / 29.0),
                     0.3 + 0.25 * np.sin((X + Y) / 41.0)], -1).astype(np.float32)


def test_brown_roundtrip():
    L = lens_for(**STRONG)
    X, Y = ln.pixel_grid(W, H)
    xd, yd = L.distort(X, Y)
    xu, yu = L.undistort(xd, yd)
    e = np.hypot(xu - X, yu - Y).max()
    corner = np.hypot(xd[0, 0] - X[0, 0], yd[0, 0] - Y[0, 0])
    check(e < 1e-6, "Brown-Conrady: distort then undistort round trip < 1e-6 px",
          f"max {e:.2e} px; corner moves {corner:.1f} px")
    k1 = lens_for(k1=-0.1)
    u = k1.cx + 0.5 * k1.fx
    ud, vd = k1.distort(np.array([u]), np.array([k1.cy]))
    want = k1.cx + 0.5 * (1 - 0.1 * 0.25) * k1.fx
    check(abs(ud[0] - want) < 1e-9 and abs(vd[0] - k1.cy) < 1e-9, "Brown-Conrady: hand-computed point",
          f"{ud[0]:.6f} vs {want:.6f}")


def test_lens_maps_ca():
    L = lens_for()
    m = ln.lens_maps(L, W, H, ca_r=0.002, ca_b=-0.0015)
    ident = rs.identity_map(W, H)
    g_err = max(np.abs(m[k]["G"] - ident).max() for k in ("undistort", "redistort"))
    check(g_err < 1e-6, "CA maps leave G identity (no distortion)", f"max {g_err:.1e} (s/t)")
    X, Y = ln.pixel_grid(W, H)
    x, y = ln.from_st(m["undistort"]["R"], W, H)
    ratio = np.hypot(x[0, 0] - L.cx, y[0, 0] - L.cy) / np.hypot(X[0, 0] - L.cx, Y[0, 0] - L.cy)
    check(abs(ratio - 1.002) < 1e-6, "CA: R undistort map is a 1.002 radial scale about the principal point",
          f"{ratio:.7f}")
    m2 = ln.lens_maps(L.__class__(**{**L.to_dict(), **STRONG}), W, H, ca_r=0.002, ca_b=-0.0015)
    worst = 0.0
    for c in "RGB":
        px, py = ln.from_st(m2["redistort"][c], W, H)
        st = rs.sample(m2["undistort"][c].astype(np.float64), px, py, W, H)
        qx, qy = ln.from_st(st, W, H)
        ok = (px > 2) & (px < W - 2) & (py > 2) & (py < H - 2)
        worst = max(worst, float(np.median(np.hypot(qx - X, qy - Y)[ok])))
    check(worst < 0.01, "lens maps: redistort then undistort per channel round trip", f"worst median {worst:.4f} px")
    g = ln.vignette_gain(W, H, -0.3, 0.05, -0.01)
    check(abs(g[H // 2, W // 2] - 1) < 1e-3 and abs(g[0, 0] - 0.74) < 0.01,
          "vignette gain: 1 at the centre, 1 + v1 + v2 + v3 in the corner", f"{g[H // 2, W // 2]:.4f}, {g[0, 0]:.4f}")


def test_apply_order_and_inverse():
    L = lens_for(**STRONG)
    m = ln.lens_maps(L, W, H)
    gain = ln.vignette_gain(W, H, -0.4)
    flat = np.ones((H, W, 3), np.float32)
    out = ap.apply(flat, gain=gain, lens=m["redistort"])
    X, Y = ln.pixel_grid(W, H)
    xu, yu = L.undistort(X, Y)
    want = rs.sample(gain.astype(np.float64), xu, yu, W, H)
    ok = (xu > 1) & (xu < W - 1) & (yu > 1) & (yu < H - 1)
    e_right = np.abs(out[..., 1] - want)[ok].max()
    e_wrong = np.abs(out[..., 1] - gain)[ok].max()
    check(e_right < 2e-3 and e_wrong > 10 * e_right, "apply order: vignette before distortion",
          f"vs gain(undistort(p)) {e_right:.1e}, vs gain(p) {e_wrong:.1e}")
    cam = make_cam(yaw_deg=1.5, pitch_deg=0.4, move=(0.05, 0, -0.2), wobble=0.3)
    f, T = 1004, XH2_HQ_S * FPS
    m_rs, m_de = rs.rs_maps(cam, f, scene_depth(cam, f, T), XH2_HQ_S, FPS, lens=ln.BrownConrady.from_camera(cam, f, **STRONG))
    img = smooth_image()
    there = ap.apply(img, gain=gain, lens=m["redistort"], rs=[m_rs])
    back = ap.remove(there, gain=gain, lens=m["undistort"], ders=m_de)
    inner = (slice(40, H - 40), slice(60, W - 60))
    err = np.abs(back - img)[inner]
    check(np.median(err) < 2e-3 and np.percentile(err, 99) < 0.02, "apply then remove gives the image back",
          f"median {np.median(err):.1e}, p99 {np.percentile(err, 99):.1e}")


def test_rs_with_lens_uses_sensor_rows():
    cam = make_cam(yaw_deg=2.0)
    f = 1004
    depth = np.full((H, W), 8.0)
    L = ln.BrownConrady.from_camera(cam, f, **STRONG)
    m_lens, _ = rs.rs_maps(cam, f, depth, XH2_HQ_S, FPS, lens=L, want_ders=False)
    # independent: sensor pixel p -> its row time from p_y; undistort; the pure-rotation camera term; distort
    p = np.array([[60.5, 20.5], [450.5, 250.5], [256.5, 135.5]])
    R0, C0 = rs.pose_at(cam, f)
    k = cam.intrinsics(f)
    st = rs.sample(m_lens.astype(np.float64), p[:, 0], p[:, 1], W, H)
    gx, gy = ln.from_st(st, W, H)
    for (px, py), g in zip(p, np.c_[gx, gy]):
        R, C = rs.pose_at(cam, f + float(rs.row_tau(py, H, XH2_HQ_S * FPS)))
        xu, yu = L.undistort(np.array(px), np.array(py))
        Xc = np.array([(xu - k.cx) / k.fx * 8, -(yu - k.cy) / k.fy * 8, -8.0])
        Xw = R @ Xc + C
        uv, _ = cam.project(f, Xw[None])
        qd = np.array(L.distort(uv[0, 0], uv[0, 1]))
        check(np.hypot(*(qd - g)) < 1e-3, f"RS with lens: row time from the sensor row at ({px:.0f},{py:.0f})",
              f"{np.hypot(*(qd - g)):.1e} px")


def test_windows():
    cam = make_cam(yaw_deg=1.8, wobble=0.4)
    f = 1004
    m_rs, _ = rs.rs_maps(cam, f, scene_depth(cam, f, XH2_HQ_S * FPS), XH2_HQ_S, FPS, want_ders=False)
    img = smooth_image()
    full = ap.apply(img, rs=[m_rs])
    win = [130, 60, 256, 136]
    x0, y0, w, h = win
    crop = img[y0:y0 + h, x0:x0 + w]
    small = ap.resize_map(ap.window_map(m_rs, win, (W, H)), w, h)
    got = ap.apply(crop, rs=[small])
    ref = full[y0:y0 + h, x0:x0 + w]
    inner = (slice(12, h - 12), slice(12, w - 12))
    e = np.abs(got - ref)[inner].max()
    check(e < 1e-4, "crop window: windowed map on the crop = crop of the full result", f"max {e:.1e}")
    import cv2
    half = cv2.resize(crop, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    got2 = ap.apply(half, rs=[ap.window_map(m_rs, win, (W, H))])           # map resized to the image inside
    ref2 = cv2.resize(ref, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    e2 = np.median(np.abs(got2 - ref2)[6:-6, 6:-6])
    check(e2 < 2e-3, "crop window at half size (LTX work size): matches the resized crop", f"median {e2:.1e}")


def test_map_files():
    with tempfile.TemporaryDirectory() as td:
        st = rs.identity_map(64, 34) + np.float32(0.001)
        p = mio.write_stmap(Path(td) / "rs.1001.exr", st, {"frame": 1001})
        back = mio.read_stmap(p)
        _, names, attrs = mio.read_exr(p)
        import OpenImageIO as oiio
        fmt = str(oiio.ImageInput.open(str(p)).spec().format)
        check(np.array_equal(back, st) and fmt == "float", "STMap EXR round trip is exact float32", fmt)
        check(attrs["colorInteropID"] == "data" and attrs["ocio_colorspace"] == "Raw" and attrs["compression"] == "zip",
              "STMap EXR: ZIP, colorInteropID data, OCIO Raw", f"{attrs['compression']}, {attrs['colorInteropID']}")
        nk = mio.nuke_rs(td, 1001, 1010, blur=2)
        check(all(s in nk for s in ("rs.####.exr", "ders.####.exr", "rs_blur1.####.exr", 'colorspace "Raw"')),
              "Nuke Reads name every RS sequence, Raw")


def synthetic_shot(td: Path):
    from atlas_vfx.contracts import shot_camera as sc
    cam = make_cam(n=7, yaw_deg=1.6, pitch_deg=0.3, move=(0.04, 0, -0.15), wobble=0.3, f0=1130)
    cam.data["plate"]["path"] = str(td / "SYN.mov")
    depth_dir = td / "depth"
    T = XH2_HQ_S * FPS
    for f in cam.frames:
        z = scene_depth(cam, f, T, 256, 135)                       # proxy-size depth like Plate Depth writes
        pdp.write_exr(depth_dir / f"depth.{f:04d}.exr", z.astype(np.float32), np.ones_like(z, bool), {"frame": f})
    return sc.ShotCamera(cam.data), depth_dir


def test_nodes_end_to_end():
    import torch
    from atlas_vfx.sensor.node_apply import AtlasVFXSensorApply
    from atlas_vfx.sensor.node_lens_maps import AtlasVFXSensorLensMaps
    from atlas_vfx.sensor.node_rs_maps import AtlasVFXSensorRSMaps
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam, depth_dir = synthetic_shot(td)
        sl, lens_dir, _, lrep = AtlasVFXSensorLensMaps().run("typed", -0.05, 0.01, 0, 0, 0, 0.001, -0.001, -0.3, 0,
                                                             0, 1.0, str(td / "lens"), shot_camera=cam.data)
        res = AtlasVFXSensorRSMaps().run(cam.data, str(depth_dir), 30.9, "measured if connected", "both", 0.0, 5,
                                         1.0, 3, 0, 0, str(td / "rs"), readout_measured_ms=29.5, sensor_lens=sl)
        rs_dir, _, rep = res["result"]
        rep = json.loads(rep)
        check(rep["readout_ms"] == 29.5 and rep["readout_from"].startswith("measured"),
              "RS Maps: measured readout wins over the typed one, and the sidecar says so", rep["readout_from"])
        check(rep["lens_from"].startswith("Sensor Lens Maps") and rep["lens"]["k1"] == -0.05,
              "RS Maps: lens from Sensor Lens Maps (row times on sensor rows)", rep["lens_from"])
        check(rep["roundtrip_median_px"] is not None and rep["roundtrip_median_px"] < 0.05,
              "RS Maps: round trip reported < 0.05 px", f"{rep['roundtrip_median_px']}")
        files = sorted(p.name for p in Path(rs_dir).iterdir())
        check(len([f for f in files if f.startswith("rs.")]) == 7 and "sensor_rs.nk" in files
              and "sensor_rs.sidecar.json" in files, "RS Maps: 7 rs + 7 ders EXRs, Nuke Reads, sidecar")
        res_t = AtlasVFXSensorRSMaps().run(cam.data, str(depth_dir), 15.3, "typed", "apply RS", 180.0, 3, 0.5, 3,
                                           1131, 1132, str(td / "rs_t"), readout_measured_ms=29.5)
        rep_t = json.loads(res_t["result"][2])
        check(rep_t["readout_ms"] == 15.3 and rep_t["readout_from"] == "typed" and rep_t["blur_samples"] == 3
              and (td / "rs_t" / "rs_blur2.1132.exr").is_file() and not (td / "rs_t" / "ders.1131.exr").exists(),
              "RS Maps: typed readout, apply-only, blur sub-samples, half-size maps")
        # render side: a full-plate "LTX render" at work size, then a crop-mode one through ATLAS_VFX_LTX_PREP windows
        img = np.stack([smooth_image() for _ in cam.frames])
        work = np.stack([__import__("cv2").resize(i, (320, 168)) for i in img])
        out, r = AtlasVFXSensorApply().run(torch.from_numpy(work), "apply camera", rs_dir, lens_dir, 1130, "",
                                           jobs.DISPLAY_VIEW, True, True, True, "edge")
        r = json.loads(r)
        check(tuple(out.shape) == work.shape and r["rs"] and r["lens"] and r["vignette"],
              "Sensor Apply (display view, work size): vignette + lens + RS applied", r["order"])
        back, _ = AtlasVFXSensorApply().run(out, "remove camera", rs_dir, lens_dir, 1130, "", jobs.DISPLAY_VIEW,
                                            True, True, True, "edge")
        e = np.median(np.abs(back.numpy() - work)[:, 20:-20, 30:-30])
        check(e < 0.01, "Sensor Apply: apply then remove camera at work size", f"median {e:.1e}")
        wins = [[100 + 4 * i, 50, 256, 135] for i in range(len(cam.frames))]
        prep = {"frames": [1130, 1136], "windows": wins, "plate_ref": {"ocio_config": str(jobs.pl.DEFAULT_OCIO)}}
        crops = np.stack([i[w[1]:w[1] + w[3], w[0]:w[0] + w[2]] for i, w in zip(img, wins)])
        full, _ = AtlasVFXSensorApply().run(torch.from_numpy(img), "apply camera", rs_dir, "", 1130, "", "ACEScg",
                                            False, False, True, "edge")
        crop_out, rc = AtlasVFXSensorApply().run(torch.from_numpy(crops), "apply camera", rs_dir, "", 0, "",
                                                 "ACEScg", False, False, True, "edge", ltx_prep=prep)
        ref = np.stack([o[w[1]:w[1] + w[3], w[0]:w[0] + w[2]] for o, w in zip(full.numpy(), wins)])
        e = np.abs(crop_out.numpy() - ref)[:, 12:-12, 12:-12].max()
        check(e < 1e-4 and json.loads(rc)["frames_from"] == "ltx_prep",
              "Sensor Apply with ATLAS_VFX_LTX_PREP crop windows = crop of the full-plate result", f"max {e:.1e}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
