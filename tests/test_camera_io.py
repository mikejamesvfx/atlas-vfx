"""T3: Camera Import.

- synthetic camera -> .chan -> import -> identical within 1e-6 (ZXY with non-trivial rotations; a wrong rot_order
  does not match)
- synthetic camera -> Nuke Camera JSON (as the export snippet writes it) -> import -> identical; metres-per-unit
  applied; haperture different from the sensor rescales focal so the horizontal FOV is kept; vaperture ignored;
  lens shift and odd xform orders refused
- synthetic COLMAP text model at proxy size -> import -> projecting points with COLMAP's own formula (x = K (R X + t),
  y down, z forward) matches ShotCamera.project at plate size; principal point scaled; a gap in the frames refused
- real data, if present: Nuke Camera1.json and solve_s1s3_f16 import and validate

Run: <V141 python> tests\\test_camera_io.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side.sensors import sensor_block  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402

NUKE_JSON = Path(r"C:\Users\miike\howling-yowies\nuke\export\Camera1.json")
F16 = Path(r"C:\Users\miike\ComfyUI_V135\ComfyUI\output\_pick\vfx_probe\solve_s1s3_f16")

PLATE = cio.plate_block(None, 4096, 2160)
SENSOR = sensor_block("fujifilm_xh2_dci4k_24p_dis_off")
N = 12


def rot(axis, deg):
    """Rodrigues, independent of the module."""
    a = math.radians(deg)
    k = np.array(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def zxy(rx, ry, rz):
    return rot((0, 1, 0), ry) @ rot((1, 0, 0), rx) @ rot((0, 0, 1), rz)


def motion():
    """N frames of a walking, panning, tilting, rolling camera: (frame, rx, ry, rz, tx, ty, tz, focal)."""
    out = []
    for i in range(N):
        out.append((1001 + i, -6 + 2.5 * i, 20 - 7.25 * i, 3 * math.sin(i), 0.3 * i, 1.6 + 0.05 * math.sin(i),
                    -0.8 * i, 16.47 + 0.1 * i))
    return out


def synthetic() -> sc.ShotCamera:
    mats = []
    for f, rx, ry, rz, tx, ty, tz, _ in motion():
        M = np.eye(4)
        M[:3, :3] = zxy(rx, ry, rz)
        M[:3, 3] = [tx, ty, tz]
        mats.append(M)
    return sc.build(name="syn", frame_start=1001, fps=24.0, plate=PLATE, sensor=SENSOR, cam_to_world=mats,
                    focal_mm=[m[7] for m in motion()], source={"kind": "synthetic"}, plate_frame_offset=1001)


def max_diff(a: sc.ShotCamera, b: sc.ShotCamera, m_per_unit_b: float = 1.0):
    assert a.frames == b.frames, (a.frames[:3], b.frames[:3])
    dm = df = 0.0
    for f in a.frames:
        A, B = a.cam_to_world(f), b.cam_to_world(f)
        dm = max(dm, float(np.max(np.abs(A[:3, :3] - B[:3, :3]))),
                 float(np.max(np.abs(A[:3, 3] - B[:3, 3] / m_per_unit_b))))
        df = max(df, abs(a.intrinsics(f).focal_mm - b.intrinsics(f).focal_mm))
    return dm, df


def test_chan(td: Path, cam):
    p = td / "syn.chan"
    cam.to_nuke_chan(p)
    back = cio.import_camera(p, plate=PLATE, sensor=SENSOR, name="syn")
    dm, df = max_diff(cam, back)
    assert dm < 1e-6 and df < 1e-6, (dm, df)
    wrong = cio.from_nuke_chan(p, plate=PLATE, sensor=SENSOR, rot_order="XYZ")
    dw, _ = max_diff(cam, wrong)
    assert dw > 1e-2, "XYZ import of a ZXY chan should not match"
    print(f"ok chan round trip: {N} frames within {dm:.1e} (matrix/translation), focal {df:.1e}; "
          f"rot_order XYZ off by {dw:.2f}")


def nuke_json_doc(haperture=23.5, focal_scale=1.0, unit=1.0, **over):
    frames = [{"frame": f, "translate": [tx / unit, ty / unit, tz / unit], "rotate": [rx, ry, rz],
               "focal": foc * focal_scale, "haperture": haperture, "vaperture": 15.7, "win_translate": [0.0, 0.0]}
              for f, rx, ry, rz, tx, ty, tz, foc in motion()]
    d = {"name": "Camera1", "class": "Camera3", "rot_order": "ZXY", "xform_order": "SRT",
         "animated_translate": True, "frames": frames}
    d.update(over)
    return d


def test_nuke_json(td: Path, cam):
    p = td / "Camera1.json"
    p.write_text(json.dumps(nuke_json_doc()), encoding="utf-8")
    back = cio.import_camera(p, plate=PLATE, sensor=SENSOR)
    dm, df = max_diff(cam, back)
    assert dm < 1e-12 and df < 1e-12, (dm, df)
    ext = back.data["extensions"]["atlas_vfx_import"]
    assert ext["nuke_vaperture_mm_ignored"] == [15.7] and back.data["sensor"]["height_mm"] == 12.39

    # Nuke units, 10.33 m per unit
    p.write_text(json.dumps(nuke_json_doc(unit=10.33)), encoding="utf-8")
    back = cio.import_camera(p, plate=PLATE, sensor=SENSOR, m_per_unit=10.33)
    dm, _ = max_diff(cam, back)
    assert dm < 1e-12 and back.data["scale"] == {"method": "known_lengths", "m_per_source_unit": 10.33}

    # film back left at 36 mm in Nuke: focal 36/23.5 times bigger there, same horizontal FOV
    p.write_text(json.dumps(nuke_json_doc(haperture=36.0, focal_scale=36 / 23.5)), encoding="utf-8")
    back = cio.import_camera(p, plate=PLATE, sensor=SENSOR)
    _, df = max_diff(cam, back)
    assert df < 1e-9 and "focal_rescaled" in back.data["extensions"]["atlas_vfx_import"], df
    hfov_nuke = 2 * math.degrees(math.atan(36 / (2 * motion()[0][7] * 36 / 23.5)))
    hfov_ours = 2 * math.degrees(math.atan(23.5 / (2 * back.intrinsics(1001).focal_mm)))
    assert abs(hfov_nuke - hfov_ours) < 1e-9

    for over, needle in (({"xform_order": "TRS"}, "xform_order"),):
        p.write_text(json.dumps(nuke_json_doc(**over)), encoding="utf-8")
        try:
            cio.import_camera(p, plate=PLATE, sensor=SENSOR)
        except cio.CameraImportError as e:
            assert needle in str(e)
        else:
            raise AssertionError(f"accepted {over}")
    d = nuke_json_doc()
    d["frames"][3]["win_translate"] = [0.1, 0.0]
    p.write_text(json.dumps(d), encoding="utf-8")
    try:
        cio.import_camera(p, plate=PLATE, sensor=SENSOR)
    except cio.CameraImportError as e:
        assert "win_translate" in str(e)
    else:
        raise AssertionError("accepted lens shift")
    print(f"ok Nuke JSON: exact, metres per unit applied, haperture 36 -> focal rescaled (hFOV {hfov_ours:.3f} deg "
          f"kept), vaperture 15.7 ignored, TRS and lens shift refused")


def quat_wxyz(R):
    """Rotation matrix -> Hamilton quaternion (w, x, y, z), COLMAP order."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((0.25 * s), (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def write_colmap_text(folder: Path, poses, f_px, cx, cy, size=(1024, 540), skip=()):
    """poses: {plate index: (R_cam_from_world, t)} in COLMAP axes."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cameras.txt").write_text(f"1 SIMPLE_RADIAL {size[0]} {size[1]} {f_px} {cx} {cy} 0\n")
    lines = []
    for k, (i, (R, t)) in enumerate(sorted(poses.items()), 1):
        if i in skip:
            continue
        q = quat_wxyz(R)
        lines.append(f"{k} {' '.join(repr(float(v)) for v in q)} {' '.join(repr(float(v)) for v in t)} 1 "
                     f"plate_{i:04d}.png\n\n")
    (folder / "images.txt").write_text("".join(lines))
    (folder / "points3D.txt").write_text("")


def colmap_poses(rng):
    """Random-ish camera path in COLMAP world: cam_from_world (R, t) per plate index 0..N-1."""
    poses = {}
    for i in range(N):
        R_c2w = rot((0, 1, 0), 10 * i - 30) @ rot((1, 0, 0), 15 + i) @ rot((0, 0, 1), 4 * math.cos(i))
        C = np.array([0.4 * i, -0.2 + 0.02 * i, 3.0 + 0.3 * i]) + rng.normal(0, 0.01, 3)
        R = R_c2w.T
        poses[i] = (R, -R @ C)
    return poses


def test_colmap(td: Path):
    rng = np.random.default_rng(3)
    poses = colmap_poses(rng)
    f_px, cx, cy = 717.7, 515.25, 268.5          # off-centre principal point, proxy pixels
    model = td / "solve" / "sparse" / "0"
    write_colmap_text(model, poses, f_px, cx, cy)
    cam = cio.import_camera(td / "solve", plate=PLATE, sensor=SENSOR, frame_offset=1001)
    assert cam.frames == list(range(1001, 1001 + N))
    assert abs(cam.intrinsics(1001).focal_mm - f_px * 23.5 / 1024) < 1e-12
    assert cam.data["lens"]["principal_point_px"] == [cx * 4, cy * 4]
    assert cam.data["lens"]["distortion_model"] == "none"

    # world points in front of every camera, projected both ways
    worst = 0.0
    for i, (R, t) in poses.items():
        C = -R.T @ t
        fwd = R.T @ np.array([0, 0, 1.0])
        pts = C + 6 * fwd + rng.normal(0, 1.5, (50, 3))
        Xc = pts @ R.T + t
        assert np.all(Xc[:, 2] > 0)
        uv_colmap = np.stack([f_px * Xc[:, 0] / Xc[:, 2] + cx, f_px * Xc[:, 1] / Xc[:, 2] + cy], 1) * 4.0
        uv_ours, depth = cam.project(1001 + i, pts)
        assert np.allclose(depth, Xc[:, 2], atol=1e-9)
        worst = max(worst, float(np.max(np.abs(uv_ours - uv_colmap))))
    assert worst < 1e-6, f"projection mismatch {worst} px"

    gap = td / "gap"
    write_colmap_text(gap, poses, f_px, cx, cy, skip=(5,))
    try:
        cio.import_camera(gap, plate=PLATE, sensor=SENSOR)
    except cio.CameraImportError as e:
        assert "1006" in str(e), e
    else:
        raise AssertionError("imported a solve with a gap")
    part = cio.import_camera(gap, plate=PLATE, sensor=SENSOR, frames=(1001, 1005))
    assert part.frames == list(range(1001, 1006))
    print(f"ok COLMAP: axis conversion checked by projection, 600 points within {worst:.1e} px at 4K; principal "
          f"point x4; gap at 1006 refused, range without it imports")


def test_real():
    if NUKE_JSON.is_file():
        nk = cio.import_camera(NUKE_JSON, plate=PLATE, sensor=SENSOR)
        ext = nk.data["extensions"]["atlas_vfx_import"]
        print(f"ok real Nuke Camera1: {len(nk.frames)} frames {nk.frames[0]}-{nk.frames[-1]}, focal "
              f"{nk.intrinsics(1001).focal_mm:.4f} mm, haperture {ext['nuke_haperture_mm']}, vaperture "
              f"{ext['nuke_vaperture_mm_ignored']} ignored")
    else:
        print(f"skip real Nuke: {NUKE_JSON} missing")
    if F16.is_dir():
        so = cio.import_camera(F16, plate=PLATE, sensor=SENSOR, frame_offset=1001)
        print(f"ok real f16 solve: {len(so.frames)} frames {so.frames[0]}-{so.frames[-1]}, focal "
              f"{so.intrinsics(1001).focal_mm:.4f} mm, lens {so.data['lens']}, rmse "
              f"{so.data['source']['rmse_px']:.3f} px @ {so.data['source']['rmse_px_at_width']}")
    else:
        print(f"skip real solve: {F16} missing")


def test_shot_camera(td: Path, cam):
    """A shot camera JSON is cropped and renamed like every other kind (crop_to_plate was ignored before
    2026-10-26); world, source and extensions come through unchanged."""
    d = json.loads(json.dumps(cam.data))
    d.setdefault("extensions", {})["atlas_vfx_measure"] = {"m_per_unit": 1.0}
    p = sc.save(d, td / "syn.shot_camera.json")
    out = cio.import_camera(p, "shot_camera", plate=PLATE, sensor=SENSOR, frames=(1003, 1008), name="syn_crop")
    assert out.frames == list(range(1003, 1009)), out.frames
    assert out.data["shot"]["frame_start"] == 1003 and out.data["shot"]["frame_end"] == 1008
    assert out.name == "syn_crop" and out.data["extensions"]["atlas_vfx_import"]["renamed_from"] == "syn"
    assert "atlas_vfx_measure" in out.data["extensions"] and out.data["source"] == cam.data["source"]
    assert np.array_equal(out.cam_to_world(1005), cam.cam_to_world(1005))
    whole = cio.import_camera(p, "shot_camera", plate=PLATE, sensor=SENSOR, frames=None, name="")
    assert whole.frames == cam.frames and whole.name == "syn"
    print("ok   shot camera JSON: cropped to 1003-1008, renamed, world and extensions kept")


def main() -> int:
    cam = synthetic()
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        test_chan(td, cam)
        test_nuke_json(td, cam)
        test_colmap(td)
        test_shot_camera(td, cam)
    test_real()
    return 0


if __name__ == "__main__":
    sys.exit(main())
