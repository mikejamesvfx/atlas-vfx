"""Lens: Brown-Conrady distortion, lateral CA and vignette, as STMaps and a gain image.

Coordinates are the shot-camera contract's pixels: origin at the top-left corner of the image, pixel centres at
i + 0.5, y down. A map is built for the output pixel centres. STMaps follow Nuke (s = x / W, t = 1 - y / H, origin
bottom-left), the same convention as platelock warp.to_stmap / from_stmap.

Distortion (OpenCV convention, normalised by focal in pixels, y down):
    x, y   = (u - cx) / fx, (v - cy) / fy
    r2     = x^2 + y^2
    radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
    xd     = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
    yd     = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
`distort` takes undistorted (pinhole) pixels to sensor pixels; `undistort` is its exact inverse (Newton, to 1e-10
in normalised units). The contract's lens block for model "brown" stores distortion_params in OpenCV order
[k1, k2, p1, p2, k3] so OpenCV / COLMAP OPENCV calibrations drop straight in; "simple_radial" (COLMAP) is [k1].

Lateral CA: each channel is the G image scaled radially about the principal point in sensor space,
p_c = c + (p_G - c) * (1 + ca_c), with ca_g = 0. So G is the reference and its CA map is the identity.

Vignette, applied to LINEAR ACEScg only: Cd = Cs * (1 + v1 r^2 + v2 r^4 + v3 r^6), r = distance from the
principal point over the half diagonal (r = 1 in the corners).

Map names (like Nuke's LensDistortion):
  undistort   output = clean (rectilinear) image; samples the distorted plate.     Plate -> clean.
  redistort   output = sensor (distorted) image; samples the clean image.          Clean/AI -> plate look.
Each comes as R, G, B maps (identical when there is no CA).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BrownConrady:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_camera(cls, cam, frame: int | None = None, **override) -> "BrownConrady":
        """From a ShotCamera's intrinsics and lens block (none / simple_radial / brown)."""
        frame = cam.frames[0] if frame is None else frame
        k = cam.intrinsics(frame)
        lens = cam.data.get("lens", {}) or {}
        model = lens.get("distortion_model", "none")
        prm = [float(v) for v in lens.get("distortion_params", [])]
        coeffs = {}
        if model == "simple_radial" and prm:
            coeffs = {"k1": prm[0]}
        elif model == "brown":
            names = ("k1", "k2", "p1", "p2", "k3")
            coeffs = {n: v for n, v in zip(names, prm)}
        elif model not in ("none",) and any(abs(v) > 0 for v in prm):
            raise NotImplementedError(f"lens model {model} not supported by Sensor Lens (use brown)")
        out = cls(k.fx, k.fy, k.cx, k.cy, **coeffs)
        return replace(out, **override) if override else out

    def scaled(self, s: float) -> "BrownConrady":
        """The same lens for an image resized by s (coefficients are in normalised units, so they stay)."""
        return replace(self, fx=self.fx * s, fy=self.fy * s, cx=self.cx * s, cy=self.cy * s)

    @property
    def is_identity(self) -> bool:
        return all(abs(v) == 0 for v in (self.k1, self.k2, self.k3, self.p1, self.p2))

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ model

    def _fwd(self, x, y):
        r2 = x * x + y * y
        rad = 1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * rad + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * rad + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def distort(self, u, v):
        """Undistorted pixels -> distorted (sensor) pixels."""
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        if self.is_identity:
            return u.copy(), v.copy()
        xd, yd = self._fwd((u - self.cx) / self.fx, (v - self.cy) / self.fy)
        return xd * self.fx + self.cx, yd * self.fy + self.cy

    def undistort(self, ud, vd, iters: int = 30, tol: float = 1e-12):
        """Distorted (sensor) pixels -> undistorted pixels, exact inverse of distort (Newton)."""
        ud = np.asarray(ud, np.float64)
        vd = np.asarray(vd, np.float64)
        if self.is_identity:
            return ud.copy(), vd.copy()
        xd = (ud - self.cx) / self.fx
        yd = (vd - self.cy) / self.fy
        r2 = xd * xd + yd * yd
        x = xd / (1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3)))
        y = yd / (1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3)))
        k1, k2, k3, p1, p2 = self.k1, self.k2, self.k3, self.p1, self.p2
        for _ in range(iters):
            r2 = x * x + y * y
            rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            drad = 2 * (k1 + r2 * (2 * k2 + 3 * k3 * r2))     # d rad / d r2 * 2 -> d rad/dx = drad * x
            fx_, fy_ = self._fwd(x, y)
            ex, ey = fx_ - xd, fy_ - yd
            a = rad + x * x * drad + 2 * p1 * y + 6 * p2 * x
            b = x * y * drad + 2 * p1 * x + 2 * p2 * y
            c = x * y * drad + 2 * p1 * x + 2 * p2 * y
            d = rad + y * y * drad + 6 * p1 * y + 2 * p2 * x
            det = a * d - b * c
            dx = (d * ex - b * ey) / det
            dy = (a * ey - c * ex) / det
            x = x - dx
            y = y - dy
            if float(np.max(np.abs(dx) + np.abs(dy), initial=0.0)) < tol:
                break
        return x * self.fx + self.cx, y * self.fy + self.cy


# ---------------------------------------------------------------------------------- maps

def pixel_grid(w: int, h: int):
    """Contract pixel centres (x, y), each (h, w) float64."""
    return np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)


def to_st(x, y, w: int, h: int) -> np.ndarray:
    """Contract pixel positions -> Nuke STMap (h, w, 2) float32. Same as platelock warp.to_stmap(mx = x - 0.5)."""
    return np.stack([x / w, 1.0 - y / h], -1).astype(np.float32)


def from_st(st: np.ndarray, w: int, h: int):
    """Nuke STMap -> contract pixel positions (x, y) float64."""
    st = np.asarray(st, np.float64)
    return st[..., 0] * w, (1.0 - st[..., 1]) * h


CHANNELS = ("R", "G", "B")


def _ca_scale(ca: dict, c: str) -> float:
    return 1.0 + float(ca.get(c, 0.0)) if c != "G" else 1.0


def lens_maps(lens: BrownConrady, w: int, h: int, ca_r: float = 0.0, ca_b: float = 0.0) -> dict:
    """{"undistort": {R, G, B}, "redistort": {R, G, B}} STMaps (h, w, 2) float32 at w x h.

    `lens` must already be at this resolution (BrownConrady.scaled). ca_r / ca_b are radial scale - 1 of R and B
    relative to G (e.g. 0.0004 = R is 0.04 % larger than G at the sensor)."""
    ca = {"R": ca_r, "B": ca_b}
    X, Y = pixel_grid(w, h)
    out = {"undistort": {}, "redistort": {}}
    xd, yd = lens.distort(X, Y)                                         # clean pixel -> where G lands on the sensor
    for c in CHANNELS:
        s = _ca_scale(ca, c)
        out["undistort"][c] = to_st(lens.cx + (xd - lens.cx) * s, lens.cy + (yd - lens.cy) * s, w, h)
        xg, yg = lens.cx + (X - lens.cx) / s, lens.cy + (Y - lens.cy) / s   # sensor pixel of channel c -> G position
        xu, yu = lens.undistort(xg, yg)
        out["redistort"][c] = to_st(xu, yu, w, h)
    return out


def vignette_gain(w: int, h: int, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0,
                  cx: float | None = None, cy: float | None = None) -> np.ndarray:
    """(h, w) float32 gain 1 + v1 r^2 + v2 r^4 + v3 r^6, r = 1 at the corners (half diagonal)."""
    cx = w / 2.0 if cx is None else cx
    cy = h / 2.0 if cy is None else cy
    X, Y = pixel_grid(w, h)
    r2 = ((X - cx) ** 2 + (Y - cy) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)
    return (1 + r2 * (v1 + r2 * (v2 + r2 * v3))).astype(np.float32)
