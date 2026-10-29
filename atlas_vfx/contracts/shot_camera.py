"""contract shot-camera@0.1: load, validate, save, and the camera helpers every consumer needs.

Pure python + numpy + jsonschema. No ComfyUI, no Atlas Camera: this file moves with the contract at the split.
Conventions are in the schema description; the short version: metres, right-handed, Y up, camera looks down -Z,
cam_to_world row-major for column vectors, pixel origin at the top-left corner, Nuke Euler order ZXY
(R = Ry @ Rx @ Rz, degrees).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTRACT = "shot-camera@0.1"
SCHEMA_PATH = Path(__file__).with_name("shot_camera_v0_1.schema.json")

ORTHO_TOL = 1e-6        # max |R^T R - I| and |det R - 1|
BOTTOM_TOL = 1e-9
SENSOR_MM_RANGE = (1.0, 100.0)   # outside this, the sensor is almost certainly not in mm
FOCAL_MM_RANGE = (1.0, 5000.0)
ASPECT_TOL = 0.01       # sensor aspect vs pixel aspect, relative


class ShotCameraError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__(f"{CONTRACT} invalid ({len(errors)}):\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------------------------- Euler (Nuke ZXY)

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Nuke rot_order ZXY, degrees: rotate about Z, then X, then Y, so R = Ry @ Rx @ Rz."""
    return _ry(math.radians(ry)) @ _rx(math.radians(rx)) @ _rz(math.radians(rz))


def matrix_to_euler_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_zxy_to_matrix, degrees, rx in [-90, 90]. At rx = +-90 (camera straight up/down) rz is set
    to 0 and the whole roll goes into ry."""
    R = np.asarray(R, float)
    sx = -R[1, 2]
    rx = math.asin(max(-1.0, min(1.0, sx)))
    if abs(math.cos(rx)) > 1e-9:
        ry = math.atan2(R[0, 2], R[2, 2])
        rz = math.atan2(R[1, 0], R[1, 1])
    else:
        ry = math.atan2(-R[2, 0], R[0, 0])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _unwrap_deg(prev: float, cur: float) -> float:
    """cur moved by whole turns to sit within 180 degrees of prev, so animation curves do not jump."""
    return cur - 360.0 * round((cur - prev) / 360.0)


# --------------------------------------------------------------------------------------------- schema + rules

_validator = None


def _schema_validator():
    global _validator
    if _validator is None:
        import jsonschema

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        _validator = cls(schema)
    return _validator


def errors(data: dict) -> list[str]:
    """Every problem with data as a list of readable strings (empty means valid)."""
    out = []
    for e in sorted(_schema_validator().iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "(root)"
        out.append(f"{where}: {e.message}")
    if out:
        return out   # the rules below assume the shape is right

    shot, plate, sensor = data["shot"], data["plate"], data["sensor"]
    f0, f1 = shot["frame_start"], shot["frame_end"]
    if f1 < f0:
        out.append(f"shot: frame_end {f1} is before frame_start {f0}")

    seen: dict[int, int] = {}
    for i, fr in enumerate(data["frames"]):
        n = fr["frame"]
        if n in seen:
            out.append(f"frames[{i}]: frame {n} appears twice (also frames[{seen[n]}])")
        seen[n] = i
        if not (f0 <= n <= f1):
            out.append(f"frames[{i}]: frame {n} outside shot range {f0}-{f1}")
        M = np.asarray(fr["cam_to_world"], float)
        if not np.all(np.isfinite(M)):
            out.append(f"frames[{i}] (frame {n}): cam_to_world has non-finite values")
            continue
        if np.max(np.abs(M[3] - [0, 0, 0, 1])) > BOTTOM_TOL:
            out.append(f"frames[{i}] (frame {n}): cam_to_world bottom row is {M[3].tolist()}, must be [0,0,0,1]")
        R = M[:3, :3]
        ortho = float(np.max(np.abs(R.T @ R - np.eye(3))))
        det = float(np.linalg.det(R))
        if ortho > ORTHO_TOL:
            out.append(f"frames[{i}] (frame {n}): rotation not orthonormal (max |RtR - I| = {ortho:.3g})")
        elif abs(det - 1.0) > ORTHO_TOL:
            out.append(f"frames[{i}] (frame {n}): rotation det {det:.6f}, must be +1 (mirrored axes?)")
        fm = fr["focal_mm"]
        if not (FOCAL_MM_RANGE[0] <= fm <= FOCAL_MM_RANGE[1]):
            out.append(f"frames[{i}] (frame {n}): focal_mm {fm} outside {FOCAL_MM_RANGE}, wrong units?")
    missing = [n for n in range(f0, f1 + 1) if n not in seen]
    if missing:
        show = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
        out.append(f"frames: {len(missing)} missing from {f0}-{f1}: {show}")

    for k in ("width_mm", "height_mm"):
        v = sensor[k]
        if not (SENSOR_MM_RANGE[0] <= v <= SENSOR_MM_RANGE[1]):
            out.append(f"sensor/{k}: {v} outside {SENSOR_MM_RANGE} mm, wrong units?")
    pa = plate.get("pixel_aspect", 1.0)
    implied = (sensor["width_mm"] / plate["width"]) / (sensor["height_mm"] / plate["height"])
    if abs(implied / pa - 1.0) > ASPECT_TOL:
        out.append(
            f"sensor: {sensor['width_mm']} x {sensor['height_mm']} mm over {plate['width']} x {plate['height']} px "
            f"implies pixel aspect {implied:.4f}, plate says {pa} (sensor height wrong? Nuke's default vaperture "
            f"15.7 is not the real one)"
        )
    return out


def validate(data: dict) -> dict:
    """Raises ShotCameraError listing every problem; returns data when valid."""
    errs = errors(data)
    if errs:
        raise ShotCameraError(errs)
    return data


# --------------------------------------------------------------------------------------------- the camera

@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    focal_mm: float
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        """OpenCV-style K (camera x right, y down, z forward). Use ShotCamera.project for -Z-forward cameras."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)


class ShotCamera:
    """A validated shot-camera@0.1 document with the helpers consumers need."""

    def __init__(self, data: dict, check: bool = True):
        if check:
            validate(data)
        self.data = data
        self._by_frame = {fr["frame"]: fr for fr in data["frames"]}

    @property
    def frames(self) -> list[int]:
        return sorted(self._by_frame)

    @property
    def name(self) -> str:
        return self.data["shot"]["name"]

    def _frame(self, frame: int) -> dict:
        try:
            return self._by_frame[int(frame)]
        except KeyError:
            raise KeyError(f"frame {frame} not in {self.name} ({self.frames[0]}-{self.frames[-1]})") from None

    def cam_to_world(self, frame: int) -> np.ndarray:
        return np.array(self._frame(frame)["cam_to_world"], float)

    def world_to_cam(self, frame: int) -> np.ndarray:
        M = self.cam_to_world(frame)
        R, t = M[:3, :3], M[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    def position(self, frame: int) -> np.ndarray:
        return self.cam_to_world(frame)[:3, 3]

    def intrinsics(self, frame: int) -> Intrinsics:
        plate, sensor = self.data["plate"], self.data["sensor"]
        w, h = plate["width"], plate["height"]
        f = float(self._frame(frame)["focal_mm"])
        pp = self.data.get("lens", {}).get("principal_point_px") or [w / 2.0, h / 2.0]
        # Horizontal aperture + pixel aspect, like Nuke. sensor height_mm is only checked (validate), never used:
        # measured heights are rounded (12.39 vs 12.3926 for square pixels) and would skew fy by ~0.02 %.
        fx = f * w / sensor["width_mm"]
        return Intrinsics(
            width=w, height=h, focal_mm=f,
            fx=fx, fy=fx * float(plate.get("pixel_aspect", 1.0)),
            cx=float(pp[0]), cy=float(pp[1]),
        )

    def _check_distortion(self, ignore: bool):
        lens = self.data.get("lens", {})
        model = lens.get("distortion_model", "none")
        if not ignore and model != "none" and any(abs(p) > 0 for p in lens.get("distortion_params", [])):
            raise NotImplementedError(f"project() is pinhole only; lens has {model} distortion. "
                                      "Undistort first or pass ignore_distortion=True.")

    def project(self, frame: int, points_world, ignore_distortion: bool = False):
        """World points (N,3) metres -> (pixels (N,2), depth (N,) metres in front of the camera; <= 0 is behind)."""
        self._check_distortion(ignore_distortion)
        P = np.atleast_2d(np.asarray(points_world, float))
        W2C = self.world_to_cam(frame)
        pc = P @ W2C[:3, :3].T + W2C[:3, 3]
        depth = -pc[:, 2]
        k = self.intrinsics(frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = k.cx + k.fx * pc[:, 0] / depth
            v = k.cy - k.fy * pc[:, 1] / depth
        return np.stack([u, v], axis=1), depth

    def nuke_chan_rows(self, units_per_m: float = 1.0) -> list[tuple]:
        """(frame, tx, ty, tz, rx, ry, rz, focal) per frame, Euler ZXY in degrees, unwrapped frame to frame."""
        rows, prev = [], None
        for n in self.frames:
            M = self.cam_to_world(n)
            r = list(matrix_to_euler_zxy(M[:3, :3]))
            if prev is not None:
                r = [_unwrap_deg(p, c) for p, c in zip(prev, r)]
            prev = r
            t = M[:3, 3] * units_per_m
            rows.append((n, *t.tolist(), *r, float(self._frame(n)["focal_mm"])))
        return rows

    def to_nuke_chan(self, path: str | os.PathLike | None = None, units_per_m: float = 1.0,
                     precision: int = 6) -> str:
        """Nuke Camera .chan text (tab separated: frame tx ty tz rx ry rz focal), matching Nuke's own export.
        Import on a Camera with rot_order ZXY. Translations are metres times units_per_m."""
        lines = []
        for row in self.nuke_chan_rows(units_per_m):
            vals = [f"{v:.{precision}f}" for v in row[1:]]
            lines.append("\t".join([str(row[0]), *vals]))
        text = "\n".join(lines) + "\n"
        if path is not None:
            _atomic_write(Path(path), text)
        return text


# --------------------------------------------------------------------------------------------- I/O

def load(path: str | os.PathLike) -> ShotCamera:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ShotCamera(data)


def dumps(data: dict) -> str:
    """JSON with one frame per line (readable, diffable); floats at full precision so load(save(x)) == x."""
    validate(data)
    body = dict(data)
    frames = sorted(data["frames"], key=lambda fr: fr["frame"])
    body["frames"] = "__FRAMES__"
    text = json.dumps(body, indent=2, allow_nan=False)
    rows = ",\n    ".join(json.dumps(fr, allow_nan=False) for fr in frames)
    return text.replace('"__FRAMES__"', "[\n    " + rows + "\n  ]") + "\n"


def save(cam: ShotCamera | dict, path: str | os.PathLike) -> Path:
    data = cam.data if isinstance(cam, ShotCamera) else cam
    p = Path(path)
    _atomic_write(p, dumps(data))
    return p


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def build(*, name: str, frame_start: int, fps: float, plate: dict, sensor: dict, cam_to_world, focal_mm,
          source: dict, scale: dict | None = None, lens: dict | None = None, plate_frame_offset: int | None = None,
          reproj_px=None, extensions: dict | None = None) -> ShotCamera:
    """Make a validated ShotCamera from arrays: cam_to_world (N,4,4), focal_mm scalar or (N,)."""
    mats = np.asarray(cam_to_world, float)
    n = len(mats)
    focal = np.broadcast_to(np.asarray(focal_mm, float), (n,))
    frames = []
    for i in range(n):
        fr = {"frame": frame_start + i, "cam_to_world": mats[i].tolist(), "focal_mm": float(focal[i])}
        if reproj_px is not None:
            fr["reproj_px"] = float(reproj_px[i])
        frames.append(fr)
    shot = {"name": name, "frame_start": frame_start, "frame_end": frame_start + n - 1, "fps": fps}
    if plate_frame_offset is not None:
        shot["plate_frame_offset"] = plate_frame_offset
    data = {
        "contract": CONTRACT, "units": "m", "axes": "y_up_rh_minus_z_forward",
        "shot": shot, "plate": dict(plate), "sensor": dict(sensor), "frames": frames,
        "scale": dict(scale or {"method": "unscaled", "m_per_source_unit": 1.0}),
        "source": dict(source),
    }
    if lens is not None:
        data["lens"] = dict(lens)
    if extensions is not None:
        data["extensions"] = dict(extensions)
    return ShotCamera(data)
