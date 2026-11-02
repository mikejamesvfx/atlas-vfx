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


def node_card_project(cam, lens_folder):
    """The Card Project NODE with a lens folder on a real (synthetic HEVC) plate: clean frames at the clean size,
    plate frames at the plate size, render.json versions and sizes, and the preview (which once picked up the
    decoded plate under the same variable name and crashed)."""
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_plate import make_plate

    from atlas_vfx.camera_side import exr_io
    from atlas_vfx.camera_side import plate as pl
    from atlas_vfx.layers.node_card_project import AtlasVFXCardProject

    planes = []
    for k in range(2):
        planes.append((np.full((H, W), 400 + 20 * k, np.uint16), np.full((H, W // 2), 512, np.uint16),
                       np.full((H, W // 2), 512, np.uint16)))
    ref = pl.plate_ref(make_plate(TMP / "plates" / "syn.mov", "hevc", "pc", planes), frame_offset=1001)
    tex = TMP / "tex.exr"
    t = np.zeros((32, 32, 4), np.float32)
    t[..., 1], t[..., 3] = 0.5, 1.0
    exr_io.write_rgba(tex, t)
    card = {"name": "card", "corners": [[-1.6, 2.4, -3.2], [-0.6, 2.4, -3.2], [-0.6, 1.4, -3.2], [-1.6, 1.4, -3.2]],
            "texture": str(tex)}
    out, prev = AtlasVFXCardProject().run(ref, cam.data, card, 1.0, 1, 1, 256, str(TMP / "cards"),
                                          sensor_lens_folder=str(lens_folder))
    rj = json.loads(next(Path(out).glob("*.render.json")).read_text(encoding="utf-8"))
    Wc, Hc = cs.clean_size(lens_folder)
    clean = exr_io.read_rgba(Path(out) / "card.1001.exr")[0]
    plate = exr_io.read_rgba(Path(out) / "plate" / "card.1001.exr")[0]
    check(clean.shape[:2] == (Hc, Wc) and plate.shape[:2] == (H, W) and plate.shape[2] == 4,
          f"node: clean frames {clean.shape[1]}x{clean.shape[0]}, plate frames {plate.shape[1]}x{plate.shape[0]}")
    check(rj["versions"] == {"clean": "card.%04d.exr", "plate": "plate/card.%04d.exr"}
          and rj["sizes"] == {"clean": [Wc, Hc], "plate": [W, H]}, f"node: render.json versions + sizes {rj['sizes']}")
    check(prev.shape[0] == 2 and prev.shape[-1] == 3, f"node: preview over the plate {tuple(prev.shape)}")


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
        node_card_project(cam, d_over)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
