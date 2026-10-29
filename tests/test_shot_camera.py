"""T1: contract shot-camera@0.1.

- save/load round trip is exact
- validation rejects a missing frame, a duplicate frame, wrong units, a non-orthonormal rotation, a mirrored one
- intrinsics and project agree with the pinhole maths by hand
- to_nuke_chan of a synthetic camera matches a hand-built chan to 1e-6 (ZXY order, non-trivial angles, a ry that
  crosses 180 so unwrapping is exercised)
- real data, if present: Mike's Camera1_metres.chan -> shot camera -> chan reproduces every value to 1e-6

Run: <V141 python> tests\\test_shot_camera.py
"""
from __future__ import annotations

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

REAL_CHAN = Path(r"C:\Users\miike\howling-yowies\nuke\export\Camera1_metres.chan")

PLATE = {"path": "synthetic", "width": 4096, "height": 2160, "colorspace": "ACEScg", "range": "full"}
SENSOR = {"width_mm": 23.5, "height_mm": 12.39, "preset": "fujifilm_xh2_dci4k_24p_dis_off"}
SOURCE = {"kind": "synthetic", "solver": "tests/test_shot_camera.py"}

# hand-built chan: frame tx ty tz rx ry rz focal (degrees, Nuke ZXY), written out by hand, not by the module
HAND = [
    (1001, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 16.47),
    (1002, 0.1, 1.61, -0.25, 12.5, -37.25, 4.0, 16.47),
    (1003, 0.25, 1.63, -0.5, -8.0, 95.5, 150.0, 18.0),   # frame-to-frame steps stay under 180 (a real move)
    (1004, -1.234567, 2.0, -3.5, 33.3, 170.0, 179.0, 24.0),
    (1005, -1.5, 2.1, -4.0, 30.0, 185.0, 185.0, 24.0),    # ry/rz cross 180: must stay continuous, not wrap to -175
    (1006, -1.75, 2.2, -4.5, -60.0, 200.0, 200.0, 35.0),
]


def R_zxy_independent(rx, ry, rz):
    """Built from axis-angle (Rodrigues), not from the module's matrices: Y(ry) * X(rx) * Z(rz)."""
    def rot(axis, deg):
        a = math.radians(deg)
        k = np.array(axis, float)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return rot((0, 1, 0), ry) @ rot((1, 0, 0), rx) @ rot((0, 0, 1), rz)


def c2w(rx, ry, rz, tx, ty, tz):
    M = np.eye(4)
    M[:3, :3] = R_zxy_independent(rx, ry, rz)
    M[:3, 3] = [tx, ty, tz]
    return M


def synthetic() -> sc.ShotCamera:
    mats = [c2w(r[4], r[5], r[6], r[1], r[2], r[3]) for r in HAND]
    return sc.build(name="synthetic_t1", frame_start=1001, fps=24.0, plate=PLATE, sensor=SENSOR,
                    cam_to_world=mats, focal_mm=[r[7] for r in HAND], source=SOURCE, plate_frame_offset=1001)


def expect_reject(data: dict, needle: str, label: str) -> str:
    errs = sc.errors(data)
    hit = [e for e in errs if needle in e]
    assert hit, f"{label}: expected an error containing {needle!r}, got {errs}"
    try:
        sc.validate(data)
    except sc.ShotCameraError:
        pass
    else:
        raise AssertionError(f"{label}: validate() did not raise")
    return hit[0]


def test_round_trip(cam):
    with tempfile.TemporaryDirectory() as td:
        p = sc.save(cam, Path(td) / "cam.json")
        back = sc.load(p)
        assert back.data == cam.data, "round trip changed the document"
        for n in cam.frames:
            assert np.array_equal(back.cam_to_world(n), cam.cam_to_world(n))
        # saved file is valid JSON with one frame per line
        lines = p.read_text(encoding="utf-8").splitlines()
        assert sum(1 for ln in lines if ln.strip().startswith('{"frame"')) == len(HAND)
    print("ok round trip: save/load exact, one frame per line")


def test_rejects(cam):
    shown = []
    d = copy.deepcopy(cam.data)
    del d["frames"][2]
    shown.append(expect_reject(d, "missing", "missing frame"))

    d = copy.deepcopy(cam.data)
    d["frames"][3]["frame"] = 1002
    shown.append(expect_reject(d, "appears twice", "duplicate frame"))

    d = copy.deepcopy(cam.data)
    d["units"] = "cm"
    shown.append(expect_reject(d, "units", "units cm"))

    d = copy.deepcopy(cam.data)
    d["sensor"]["width_mm"], d["sensor"]["height_mm"] = 0.0235, 0.01239
    shown.append(expect_reject(d, "wrong units", "sensor in metres"))

    d = copy.deepcopy(cam.data)
    d["frames"][0]["focal_mm"] = 0.01647
    shown.append(expect_reject(d, "wrong units", "focal in metres"))

    d = copy.deepcopy(cam.data)
    M = np.array(d["frames"][1]["cam_to_world"])
    M[:3, :3] *= 1.01                      # scaled rotation
    d["frames"][1]["cam_to_world"] = M.tolist()
    shown.append(expect_reject(d, "not orthonormal", "scaled rotation"))

    d = copy.deepcopy(cam.data)
    M = np.array(d["frames"][1]["cam_to_world"])
    M[0, 1] += 1e-4                        # small shear, still above tolerance
    d["frames"][1]["cam_to_world"] = M.tolist()
    shown.append(expect_reject(d, "not orthonormal", "sheared rotation"))

    d = copy.deepcopy(cam.data)
    M = np.array(d["frames"][1]["cam_to_world"])
    M[:3, 0] *= -1                         # mirrored: orthonormal but det -1
    d["frames"][1]["cam_to_world"] = M.tolist()
    shown.append(expect_reject(d, "det", "mirrored rotation"))

    d = copy.deepcopy(cam.data)
    d["sensor"]["height_mm"] = 15.7       # Nuke default vaperture
    shown.append(expect_reject(d, "pixel aspect", "Nuke default vaperture"))

    d = copy.deepcopy(cam.data)
    d["shot"]["frame_rate"] = 24          # typo'd key
    shown.append(expect_reject(d, "frame_rate", "unknown key"))

    assert sc.errors(cam.data) == []
    print(f"ok validation: {len(shown)} bad documents rejected, e.g.")
    for s in shown:
        print("    " + s)


def test_intrinsics_project(cam):
    k = cam.intrinsics(1001)
    fx = 16.47 * 4096 / 23.5
    assert abs(k.fx - fx) < 1e-9 and k.fy == k.fx      # square pixels: fy from fx, not from sensor height
    ana = sc.ShotCamera({**cam.data, "plate": {**cam.data["plate"], "width": 2048, "pixel_aspect": 2.0}})
    assert abs(ana.intrinsics(1001).fy - 2 * ana.intrinsics(1001).fx) < 1e-9  # anamorphic 2x
    assert (k.cx, k.cy) == (2048.0, 1080.0)
    # frame 1001: camera at (0, 1.6, 0), identity rotation, looking down -Z
    px, depth = cam.project(1001, [[0, 1.6, -10], [1, 1.6, -10], [0, 2.6, -10], [0, 1.6, 10]])
    assert np.allclose(px[0], [2048, 1080]) and abs(depth[0] - 10) < 1e-12
    assert np.allclose(px[1], [2048 + fx * 0.1, 1080])          # +X is right in the image
    assert np.allclose(px[2], [2048, 1080 - k.fy * 0.1])        # +Y is up, image y down
    assert depth[3] < 0                                          # behind the camera
    # moving frame: a point straight ahead along the camera's -Z lands on the principal point
    for n in cam.frames:
        M = cam.cam_to_world(n)
        p = M[:3, 3] - 7.0 * M[:3, 2]
        q, d = cam.project(n, p)
        assert np.allclose(q[0], [2048, 1080], atol=1e-6) and abs(d[0] - 7.0) < 1e-9
    print(f"ok intrinsics/project: fx {k.fx:.3f} px, axes and principal point checked on {len(cam.frames)} frames")


def parse_chan(text):
    rows = []
    for ln in text.strip().splitlines():
        parts = ln.split("\t")
        assert len(parts) == 8, f"chan row has {len(parts)} columns: {ln!r}"
        rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def test_chan(cam):
    text = cam.to_nuke_chan()
    got = parse_chan(text)
    assert [r[0] for r in got] == [r[0] for r in HAND]
    worst = 0.0
    for g, h in zip(got, HAND):
        worst = max(worst, max(abs(a - b) for a, b in zip(g[1:], h[1:])))
    assert worst <= 1e-6, f"chan differs from hand-built by {worst}"
    # the order really is ZXY: an XYZ build of the same numbers gives a different matrix
    rx, ry, rz = HAND[1][4:7]
    assert not np.allclose(R_zxy_independent(rx, ry, rz), sc.euler_zxy_to_matrix(rz, ry, rx))
    assert np.allclose(R_zxy_independent(rx, ry, rz), sc.euler_zxy_to_matrix(rx, ry, rz), atol=1e-12)
    # scaled to Nuke units
    nk = parse_chan(cam.to_nuke_chan(units_per_m=1 / 10.33))
    assert abs(nk[3][1] - HAND[3][1] / 10.33) <= 1e-6 and nk[3][4:] == got[3][4:]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cam.chan"
        cam.to_nuke_chan(p)
        assert p.read_text(encoding="utf-8") == text
    print(f"ok chan: matches hand-built to {worst:.1e} (ZXY, ry/rz through 180 unwrapped)")
    print("    " + text.strip().splitlines()[4])


def test_real_chan():
    if not REAL_CHAN.is_file():
        print(f"skip real chan: {REAL_CHAN} not found")
        return
    rows = parse_chan(REAL_CHAN.read_text(encoding="utf-8"))
    mats = [c2w(r[4], r[5], r[6], r[1], r[2], r[3]) for r in rows]
    cam = sc.build(name="DSCF3000_Camera1_metres", frame_start=rows[0][0], fps=24.0,
                   plate={**PLATE, "path": r"C:\Users\miike\Videos\atlas-vfx_footage\DSCF3000.MOV",
                          "colorspace": "F-Log2 F-Gamut", "camera_body": "FUJIFILM X-H2"},
                   sensor=SENSOR, cam_to_world=mats, focal_mm=[r[7] for r in rows],
                   source={"kind": "import", "solver": "Nuke 17.1 CameraTracker", "path": str(REAL_CHAN)},
                   scale={"method": "known_lengths", "m_per_source_unit": 10.33}, plate_frame_offset=1001)
    back = parse_chan(cam.to_nuke_chan())
    worst = max(max(abs(a - b) for a, b in zip(g[1:], h[1:])) for g, h in zip(back, rows))
    assert [r[0] for r in back] == [r[0] for r in rows]
    assert worst <= 1e-6, f"real chan round trip off by {worst}"
    rmin = min(min(r[4:7]) for r in rows)
    rmax = max(max(r[4:7]) for r in rows)
    print(f"ok real chan: {len(rows)} frames {rows[0][0]}-{rows[-1][0]} round trip within {worst:.1e} "
          f"(angles span {rmin:.1f} to {rmax:.1f} deg)")


def main() -> int:
    cam = synthetic()
    test_round_trip(cam)
    test_rejects(cam)
    test_intrinsics_project(cam)
    test_chan(cam)
    test_real_chan()
    return 0


if __name__ == "__main__":
    sys.exit(main())
