"""T3: Solve Compare.

- a camera vs itself scaled 7x, rotated and moved: scale 7, residual 0, view angle 0 (and 1/7 the other way)
- noise on the copy shows up as residual and angle of the right size
- acceptance on real data, if present: Nuke Camera1 vs solve_s1s3_f16 reproduces the prototype (align_nuke.py,
  2026-10-18): median centre residual 0.97 % of path, median view angle 2.94 deg, scale 0.10488 Nuke per solve unit

Run: <V141 python> tests\\test_compare.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import compare as cmp  # noqa: E402
from atlas_vfx.camera_side.sensors import sensor_block  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_camera_io import PLATE, SENSOR, rot, synthetic  # noqa: E402

NUKE_JSON = Path(r"C:\Users\miike\howling-yowies\nuke\export\Camera1.json")
F16 = Path(r"C:\Users\miike\ComfyUI_V135\ComfyUI\output\_pick\vfx_probe\solve_s1s3_f16")
PROTO = {"residual_pct_of_path": 0.970970721865577, "view_angle_median": 2.9385329189968328,
         "scale_nuke_per_mine": 0.10487544183598306, "frames": 297}


def transformed(cam: sc.ShotCamera, s, R, t, noise_pos=0.0, noise_deg=0.0, seed=0) -> sc.ShotCamera:
    rng = np.random.default_rng(seed)
    mats = []
    for f in cam.frames:
        M = cam.cam_to_world(f)
        N = np.eye(4)
        Rn = R @ M[:3, :3]
        if noise_deg:
            ax = rng.normal(size=3)
            Rn = rot(ax / np.linalg.norm(ax), noise_deg) @ Rn
        N[:3, :3] = Rn
        N[:3, 3] = s * R @ M[:3, 3] + t + rng.normal(0, noise_pos, 3)
        mats.append(N)
    return sc.build(name=f"{cam.name}_x{s}", frame_start=cam.frames[0], fps=24.0, plate=PLATE, sensor=SENSOR,
                    cam_to_world=mats, focal_mm=[cam.intrinsics(f).focal_mm for f in cam.frames],
                    source={"kind": "synthetic"})


def test_exact():
    a = synthetic()
    R = rot((0.3, 0.8, -0.52) / np.linalg.norm((0.3, 0.8, -0.52)), 73.0)
    b = transformed(a, 7.0, R, np.array([12.0, -3.0, 40.0]))
    r = cmp.compare(a, b)
    assert abs(r["scale_b_per_a"] - 7.0) < 1e-9, r["scale_b_per_a"]
    assert r["centre_residual_b_units"]["max"] < 1e-9 and r["view_angle_deg"]["max"] < 1e-5
    assert r["rotation_angle_deg"]["max"] < 1e-5
    assert np.allclose(r["rotation_b_from_a"], R, atol=1e-9)
    back = cmp.compare(b, a)
    assert abs(back["scale_b_per_a"] - 1 / 7) < 1e-9
    print(f"ok exact: scale {r['scale_b_per_a']:.9f}, residual max {r['centre_residual_b_units']['max']:.1e}, "
          f"view angle max {r['view_angle_deg']['max']:.1e} deg; reverse scale {back['scale_b_per_a']:.9f}")


def test_noise():
    a = synthetic()
    b = transformed(a, 2.0, np.eye(3), np.zeros(3), noise_pos=0.01, noise_deg=1.0, seed=5)
    r = cmp.compare(a, b)
    assert abs(r["scale_b_per_a"] - 2.0) < 0.01
    assert 0.003 < r["centre_residual_b_units"]["median"] < 0.03
    # 1 deg of rotation noise per frame plus the fit's own rotation error from the 1 cm position noise
    assert 0.5 < r["rotation_angle_deg"]["median"] < 3.0, r["rotation_angle_deg"]
    print(f"ok noise: 1 cm / 1 deg noise -> residual median {r['centre_residual_b_units']['median'] * 100:.2f} cm, "
          f"rotation median {r['rotation_angle_deg']['median']:.2f} deg, view {r['view_angle_deg']['median']:.2f}")


def test_real():
    if not (NUKE_JSON.is_file() and F16.is_dir()):
        print("skip real acceptance: Camera1.json or solve_s1s3_f16 missing")
        return
    nk = cio.import_camera(NUKE_JSON, plate=PLATE, sensor=SENSOR)
    so = cio.import_camera(F16, plate=PLATE, sensor=SENSOR, frame_offset=1001)
    r = cmp.compare(so, nk)
    got = {"residual_pct_of_path": r["residual_pct_of_path"], "view_angle_median": r["view_angle_deg"]["median"],
           "scale_nuke_per_mine": r["scale_b_per_a"], "frames": r["frames_compared"]}
    assert got["frames"] == PROTO["frames"]
    for k in ("residual_pct_of_path", "view_angle_median", "scale_nuke_per_mine"):
        assert abs(got[k] - PROTO[k]) <= 1e-6 * max(1, abs(PROTO[k])), (k, got[k], PROTO[k])
    assert 0.8 < got["residual_pct_of_path"] < 1.2 and 2.5 < got["view_angle_median"] < 3.5
    print("ok real acceptance: " + cmp.summary_line(r))
    print(f"    prototype: residual {PROTO['residual_pct_of_path']:.4f} %, view {PROTO['view_angle_median']:.4f} deg, "
          f"scale {PROTO['scale_nuke_per_mine']:.6f}; now {got['residual_pct_of_path']:.4f} %, "
          f"{got['view_angle_median']:.4f} deg, {got['scale_nuke_per_mine']:.6f}")


def main() -> int:
    test_exact()
    test_noise()
    test_real()
    return 0


if __name__ == "__main__":
    sys.exit(main())
