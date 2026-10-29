"""Do the Sensor maps match the real plate? A test that needs no LTX.

The deck is a known plane (y = 0 in the .abc world). Take the deck from a reference frame and reproject it into a
target frame through the camera via the plane (a homography, global shutter):
  plain : plate[ref] -> plane -> frame t                      (ignores rolling shutter both ends)
  sensor: remove_camera(plate[ref]) -> plane -> frame t -> apply_camera(t)
Compare each with plate[t] on the deck boards (pixels whose ray hits the deck inside a box, away from the element and the
edges). If the RS maps are right, `sensor` should line up better, most of all where the predicted skew is large.

Run: <V141 python> tools\\rs_deck_proof.py [ref_frame] [rs_maps_dir] [shot_camera.json]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

WORK = Path(r"C:\Users\miike\atlas-vfx-work\DSCF3000")
CAM = WORK / "camera" / "DSCF3000_S2S3_abc.shot_camera.json"
SENSOR = (WORK / "sensor" / "DSCF3000_S2S3_abc_rs", WORK / "sensor" / "DSCF3000_S2S3_abc_lens")
PLATE = r"C:\Users\miike\Videos\atlas-vfx_footage\DSCF3000.MOV"
WIDTH = 1024
DECK_X, DECK_Z = (-0.3, 2.2), (-9.25, -7.4)      # deck boards right of the element spot (x -1.32), in front of the wall


def rays_to_deck(cam, n, w, h):
    """World points on y = 0 for every pixel of a w x h image of frame n, and a mask of usable deck pixels."""
    C2W = cam.cam_to_world(n)
    k = cam.intrinsics(n)
    s = w / cam.data["plate"]["width"]
    u = (np.arange(w) + 0.5) / s
    v = (np.arange(h) + 0.5) / s
    dx = (u[None, :] - k.cx) / k.fx
    dy = -(v[:, None] - k.cy) / k.fy
    d = np.stack(np.broadcast_arrays(dx, dy, -np.ones((h, w))), -1) @ C2W[:3, :3].T
    o = C2W[:3, 3]
    t = -o[1] / np.where(np.abs(d[..., 1]) > 1e-9, d[..., 1], np.nan)
    X = o + t[..., None] * d
    ok = (t > 0) & (X[..., 0] > DECK_X[0]) & (X[..., 0] < DECK_X[1]) & (X[..., 2] > DECK_Z[0]) & (X[..., 2] < DECK_Z[1])
    return X, ok


def project(cam, n, X, w, h):
    W2C = cam.world_to_cam(n)
    k = cam.intrinsics(n)
    s = w / cam.data["plate"]["width"]
    p = X @ W2C[:3, :3].T + W2C[:3, 3]
    z = -p[..., 2]
    u = (k.fx * p[..., 0] / z + k.cx) * s - 0.5
    v = (-k.fy * p[..., 1] / z + k.cy) * s - 0.5
    return u.astype(np.float32), v.astype(np.float32), z


def main() -> int:
    import cv2

    ref_n = int(sys.argv[1]) if len(sys.argv) > 1 else 1170
    rs_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else SENSOR[0]   # e.g. a test readout's maps
    cam = sc.load(sys.argv[3] if len(sys.argv) > 3 else CAM)     # e.g. the RS-aware solve; maps must match it
    ref = pl.plate_ref(PLATE, 129, 296)
    sensor = li.sensor_maps(str(rs_dir), str(SENSOR[1]))
    rs_rep = json.loads((rs_dir / "sensor_rs.json").read_text())
    off = ref["frame_offset"]
    fwd, _ = li.view_processors(ref["ocio_config"])
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)

    def frame(n):
        _, img = next(pl.decode_rgb(ref, n - off, n - off, width=WIDTH))
        return pl.to_working(img, ref)

    src = frame(ref_n)
    h, w = src.shape[:2]
    src_clean = li.remove_camera(src, sensor, ref_n)
    # control: the same two extra resamples with no net geometry (remove + apply at the reference frame)
    src_ctrl = li.apply_camera(np.dstack([src_clean, np.ones((h, w), np.float32)]), sensor, ref_n)[..., :3]
    targets = [n for n in range(1150, 1215, 3) if abs(n - ref_n) >= 6]
    rows = []
    for n in targets:
        X, ok = rays_to_deck(cam, n, w, h)
        u, v, z = project(cam, ref_n, X, w, h)
        ok &= np.isfinite(u) & (z > 0) & (u > 8) & (u < w - 8) & (v > 8) & (v < h - 8)
        ok = cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if ok.sum() < 2000:
            continue
        um, vm = np.nan_to_num(u, nan=-1), np.nan_to_num(v, nan=-1)
        plain = cv2.remap(src, um, vm, cv2.INTER_LINEAR)
        clean = cv2.remap(src_clean, um, vm, cv2.INTER_LINEAR)
        ctrl0 = cv2.remap(src_ctrl, um, vm, cv2.INTER_LINEAR)
        sens = li.apply_camera(np.dstack([clean, np.ones((h, w), np.float32)]), sensor, n)[..., :3]
        # control gets one more identity-geometry resample at frame n to match the sensor path's resample count
        ctrl = li.apply_camera(np.dstack([li.remove_camera(ctrl0, sensor, n), np.ones((h, w), np.float32)]),
                               sensor, n)[..., :3]
        tgt = frame(n)
        g = lambda a: cv2.GaussianBlur(li.to_view(a, fwd) @ lum, (0, 0), 1.0)
        gt, gpn, gs, gc = g(tgt), g(plain), g(sens), g(ctrl)
        # gradient-domain error is robust to the small exposure drift between frames
        grad = lambda a: np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
        e_plain = float(np.median(np.abs(grad(gpn) - grad(gt))[ok]))
        e_sens = float(np.median(np.abs(grad(gs) - grad(gt))[ok]))
        e_ctrl = float(np.median(np.abs(grad(gc) - grad(gt))[ok]))
        rows.append({"frame": n, "ref": ref_n, "deck_px": int(ok.sum()), "err_plain": round(e_plain, 5),
                     "err_control": round(e_ctrl, 5), "err_sensor": round(e_sens, 5),
                     "better_pct": round(100 * (e_plain - e_sens) / e_plain, 1),
                     "better_vs_control_pct": round(100 * (e_ctrl - e_sens) / e_ctrl, 1)})
        print(rows[-1])
    out = WORK / "review" / f"rs_deck_proof_ref{ref_n}_{rs_dir.name}_{cam.name}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    b = np.array([r["better_pct"] for r in rows])
    bc = np.array([r["better_vs_control_pct"] for r in rows])
    print(f"vs plain: sensor better in {int((b > 0).sum())} of {len(b)} frames, median {np.median(b):+.1f} %")
    print(f"vs control (same resampling, no RS geometry): sensor better in {int((bc > 0).sum())} of {len(bc)}, "
          f"median {np.median(bc):+.1f} %; readout {rs_rep.get('readout_ms')} ms ({rs_rep.get('readout_from')}), "
          f"maps {rs_dir.name}")
    print("csv", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
