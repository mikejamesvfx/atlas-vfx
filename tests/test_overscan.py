"""Overscan: a clean (undistorted) frame bigger than the plate, so barrel corners survive the lens round trip.

  1. Sensor Lens Maps 'auto' overscan on a strong barrel lens: the clean frame grows, the maps' own inverse check
     stays sub-pixel; 'none' keeps the old plate-size maps.
  2. plate -> remove camera -> clean (clean size) -> apply camera -> plate: the CORNERS come back (with overscan),
     where without it they are lost to edge smear.
  3. clean_space.clean_camera: a 3D point projected through the clean camera lands where the lensed camera's
     undistort_pixels puts it plus the overscan offset; Card Project's clean render, redistorted, puts the card's
     corner where the lensed camera projects it.
  4. LTX Prep / Return helpers: remove_camera gives the clean size, apply_camera the plate size.
Run: <V141 python> tests/test_overscan.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TMP = Path(tempfile.mkdtemp(prefix="atlas_overscan_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import clean_space as cs  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import card_project as cp  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402
from atlas_vfx.sensor import jobs  # noqa: E402
from atlas_vfx.sensor import lens as ln  # noqa: E402

W, H, F = 480, 270, 300.0
K1, K2 = -0.18, 0.03
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def plate_image():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 0.2 + 0.15 * np.sin(x / 7.0) * np.cos(y / 9.0)
    g = 0.2 + 0.1 * np.cos((x + y) / 13.0)
    b = 0.2 + 0.08 * np.sin(y / 5.0)
    return np.dstack([r, g, b]).astype(np.float32)


def camera():
    M = np.eye(4)
    M[:3, 3] = [0.0, 1.5, 0.0]
    plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
    return sc.build(name="os_cam", frame_start=1001, fps=24.0, plate=plate,
                    sensor={"width_mm": 23.5, "height_mm": 23.5 * H / W}, cam_to_world=[M, M],
                    focal_mm=F * 23.5 / W, source={"kind": "synthetic"},
                    lens={"distortion_model": "brown", "distortion_params": [K1, K2, 0, 0, 0]})


def main() -> int:
    try:
        lens = ln.BrownConrady(F, F, W / 2, H / 2, k1=K1, k2=K2)
        d_over, d_none = TMP / "over", TMP / "none"
        rep = jobs.make_lens_maps(lens, (W, H), d_over, overscan="auto")
        rep0 = jobs.make_lens_maps(lens, (W, H), d_none)
        Wc, Hc = rep["clean_size"]
        check(Wc > W and Hc > H and rep0["clean_size"] == [W, H],
              f"auto overscan grows the clean frame to {Wc}x{Hc} (overscan {rep['overscan_px']}); none keeps {W}x{H}")
        check(max(rep["roundtrip_median_px"].values()) < 0.05,
              f"overscanned maps invert sub-pixel ({rep['roundtrip_median_px']})")

        m_over, m_none = jobs.Maps(lens_folder=str(d_over)), jobs.Maps(lens_folder=str(d_none))
        plate = plate_image()
        clean = li.remove_camera(plate, m_over, 1001)
        check(clean.shape[:2] == (Hc, Wc), f"remove camera gives the clean frame {clean.shape[1]}x{clean.shape[0]}")
        back = li.apply_camera(np.dstack([clean, np.ones(clean.shape[:2], np.float32)]), m_over, 1001)
        check(back.shape[:2] == (H, W), "apply camera gives the plate size back")
        corner = (slice(2, 30), slice(2, 50))
        e_over = float(np.median(np.abs(back[..., :3] - plate)[corner]))
        clean0 = li.remove_camera(plate, m_none, 1001)
        back0 = li.apply_camera(np.dstack([clean0, np.ones(clean0.shape[:2], np.float32)]), m_none, 1001)
        e_none = float(np.median(np.abs(back0[..., :3] - plate)[corner]))
        check(e_over < 3e-3 and e_none > 5 * e_over,
              f"corners survive with overscan (median {e_over:.4f}), lost without it ({e_none:.4f})")

        cam = camera()
        ccam = cs.clean_camera(cam, d_over)
        check((ccam.data["plate"]["width"], ccam.data["plate"]["height"]) == (Wc, Hc)
              and ccam.data["lens"]["distortion_model"] == "none"
              and abs(ccam.intrinsics(1001).fx - cam.intrinsics(1001).fx) < 1e-9,
              "clean camera: clean frame, no lens, same focal in pixels")
        pts = np.array([[-2.0, 2.2, -4.0], [1.8, 0.4, -3.0], [0.0, 1.5, -6.0], [2.6, 2.8, -3.5]])
        uv_lens, _ = cam.project(1001, pts)
        uv_clean, _ = ccam.project(1001, pts)
        ox, oy = (Wc - W) / 2, (Hc - H) / 2
        want = cam.undistort_pixels(1001, uv_lens) + [ox, oy]
        check(np.allclose(uv_clean, want, atol=1e-6), "clean camera = lensed camera's undistort_pixels + overscan")

        # a tiny card rendered through the clean camera, redistorted, lands where the lensed camera projects it
        # (a small card, near the corner of the frame where the lens bends most; compare the alpha centroid)
        X = np.array([-1.9, 2.45, -3.0])
        h = 0.01
        corners = [X + [-h, h, 0], X + [h, h, 0], X + [h, -h, 0], X + [-h, -h, 0]]
        img = cp.render(ccam, 1001, corners, np.ones((8, 8, 4), np.float32))
        on_plate = li.apply_camera(img, m_over, 1001)
        a = on_plate[..., 3]
        ys, xs = np.mgrid[0:H, 0:W]
        got = np.array([(a * (xs + 0.5)).sum(), (a * (ys + 0.5)).sum()]) / max(a.sum(), 1e-9)
        uv, _ = cam.project(1001, X[None])
        check(a.sum() > 0.5 and np.abs(got - uv[0]).max() < 0.25,
              f"clean card redistorted lands on the lensed projection ({got.round(2)} vs {uv[0].round(2)})")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
