"""Hosek-Wilkie clear sky (2012) + solar disc (2013): Physical Sky's fallback when Atlas Scene's Prague sky is not
reachable. Numpy and the standard library only; never imports Atlas Scene or the rest of atlas_vfx, so Atlas Scene
can register it later through its own `register_evaluator` hook (`factory` below).

The formulas are a port of the authors' BSD-3 sample implementation v1.4a (ArHosekSkyModel.c: CookConfiguration,
CookRadianceConfiguration, GetRadianceInternal, arhosekskymodel_radiance, sr_internal, solar_radiance_internal2).
The coefficients are the authors' data, converted by tools/hosek_wilkie_data.py into data/hosek_wilkie_spectral.npz
and hash-checked against data/hosek_wilkie_spectral.json on load. Licence: data/THIRD_PARTY_HOSEK_WILKIE.md.

Evaluator contract (Atlas Scene's AtmosphereEvaluator): directions (N, 3) unit vectors, right-handed, Y up, in the
same frame as the sun direction; sky_radiance and sun_radiance return (N, 3) unclipped Linear Rec.709 radiance in
radiometric CIE-weighted units, the same scale as Atlas Scene's Prague evaluator: spectral W/(m^2 sr nm) through the
CIE 1931 observer, no 683 lm/W (multiply Y by LM_PER_W for cd/m^2). The lower hemisphere is black. sun_radiance is the disc alone (sky not added), zero outside the 0.255 deg radius.

Domain: sun elevation 0-90 deg (no twilight), turbidity 1-10, albedo 0-1, sea level. Approximate near the horizon.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
NPZ = DATA_DIR / "hosek_wilkie_spectral.npz"
MANIFEST = DATA_DIR / "hosek_wilkie_spectral.json"
VERSION = "1.0"
SOLAR_RADIUS_RAD = math.radians(0.51) / 2.0      # TERRESTRIAL_SOLAR_RADIUS in the reference source
SOLAR_RADIUS_DEG = math.degrees(SOLAR_RADIUS_RAD)
MIN_ELEVATION_DEG = 0.0
TURBIDITY_RANGE = (1.0, 10.0)
LM_PER_W = 683.0
CHUNK = 1 << 18


# ------------------------------------------------------------------------------------------------ data
class DataError(RuntimeError):
    """The shipped coefficient data is missing or does not match its manifest."""


@lru_cache(maxsize=1)
def load_data() -> dict[str, np.ndarray]:
    """The coefficient arrays, checked array by array against the manifest's sha256. No pickle."""
    if not NPZ.is_file() or not MANIFEST.is_file():
        raise DataError(f"Hosek-Wilkie data missing ({NPZ.name} / {MANIFEST.name}); rebuild with "
                        "tools/hosek_wilkie_data.py")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    with np.load(NPZ, allow_pickle=False) as z:
        for name, meta in manifest["arrays"].items():
            a = np.ascontiguousarray(z[name], np.float64)
            if list(a.shape) != meta["shape"] or hashlib.sha256(a.tobytes()).hexdigest() != meta["sha256"]:
                raise DataError(f"Hosek-Wilkie data: {name} does not match its manifest")
            out[name] = a
    return out


@lru_cache(maxsize=1)
def data_id() -> str:
    """A short id of the data (hash of the manifest's array hashes), for provenance and cache keys."""
    arrays = json.loads(MANIFEST.read_text(encoding="utf-8"))["arrays"]
    return hashlib.sha256("".join(arrays[k]["sha256"] for k in sorted(arrays)).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ colour
def _lobe(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf_wyman2013(lam_nm) -> np.ndarray:
    """CIE 1931 2-degree colour matching functions, (N, 3): the multi-lobe fit of Wyman, Sloan and Shirley, JCGT
    2(2) 2013 (written from the paper's equations)."""
    x = np.asarray(lam_nm, np.float64)
    xb = 1.056 * _lobe(x, 599.8, 37.9, 31.0) + 0.362 * _lobe(x, 442.0, 16.0, 26.7) - 0.065 * _lobe(x, 501.1, 20.4, 26.2)
    yb = 0.821 * _lobe(x, 568.8, 46.9, 40.5) + 0.286 * _lobe(x, 530.9, 16.3, 31.1)
    zb = 1.217 * _lobe(x, 437.0, 11.8, 36.0) + 0.681 * _lobe(x, 459.0, 26.0, 13.8)
    return np.stack([xb, yb, zb], -1)


def _rgb_to_xyz(prim, white) -> np.ndarray:
    P = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in prim]).T
    W = np.array([white[0] / white[1], 1.0, (1 - white[0] - white[1]) / white[1]])
    return P * np.linalg.solve(P, W)


REC709_TO_XYZ = _rgb_to_xyz([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], (0.3127, 0.3290))
XYZ_TO_REC709 = np.linalg.inv(REC709_TO_XYZ)


@lru_cache(maxsize=1)
def band_weights() -> np.ndarray:
    """(11, 3): Linear Rec.709 (radiometric, no 683) from the 11 band values, the bands joined linearly between their
    centres (as arhosekskymodel_radiance interpolates) and integrated against the CMFs at 1 nm, 320-720 nm."""
    bands = load_data()["bands_nm"]
    lam = np.arange(bands[0], bands[-1] + 0.5, 1.0)
    hats = np.stack([np.interp(lam, bands, np.eye(len(bands))[i]) for i in range(len(bands))])   # (11, L)
    xyz = hats @ cmf_wyman2013(lam) * 1.0                                                         # (11, 3), 1 nm steps
    return xyz @ XYZ_TO_REC709.T


# ------------------------------------------------------------------------------------------------ model
def _bezier(x: float) -> np.ndarray:
    return np.array([(1 - x) ** 5, 5 * (1 - x) ** 4 * x, 10 * (1 - x) ** 3 * x ** 2, 10 * (1 - x) ** 2 * x ** 3,
                     5 * (1 - x) * x ** 4, x ** 5])


def _turbidity_albedo_weights(turbidity: float, albedo: float) -> np.ndarray:
    """(2, 10) weights over (albedo 0/1, turbidity 1..10) as CookConfiguration mixes them."""
    t_int = int(turbidity)
    rem = turbidity - t_int
    w = np.zeros((2, 10))
    for a_i, a_w in ((0, 1.0 - albedo), (1, albedo)):
        w[a_i, t_int - 1] += a_w * (1.0 - rem)
        if t_int < 10:
            w[a_i, t_int] += a_w * rem
    return w


def cook(turbidity: float, albedo: float, sun_el_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """(configs (11, 9), radiances (11,)) for the sky formula: CookConfiguration / CookRadianceConfiguration."""
    d = load_data()
    x = (sun_el_rad / (math.pi / 2.0)) ** (1.0 / 3.0)
    B = _bezier(x)
    W = _turbidity_albedo_weights(turbidity, albedo)
    configs = np.einsum("at,p,batpk->bk", W, B, d["sky_config"])
    rads = np.einsum("at,p,batp->b", W, B, d["sky_radiance"])
    return configs, rads


def sky_spectral(configs, rads, cos_theta, cos_gamma) -> np.ndarray:
    """(N, 11) spectral sky radiance, GetRadianceInternal x the band radiance, for view zenith / sun angles given
    as cosines (cos_theta >= 0)."""
    c = configs[:, None, :]                          # (11, 1, 9)
    ct, cg = cos_theta[None], cos_gamma[None]        # (1, N)
    gamma = np.arccos(np.clip(cg, -1.0, 1.0))
    exp_m = np.exp(c[..., 4] * gamma)
    ray_m = cg * cg
    mie_m = (1.0 + cg * cg) / (1.0 + c[..., 8] ** 2 - 2.0 * c[..., 8] * cg) ** 1.5
    zenith = np.sqrt(ct)
    f = (1.0 + c[..., 0] * np.exp(c[..., 1] / (ct + 0.01))) * \
        (c[..., 2] + c[..., 3] * exp_m + c[..., 5] * ray_m + c[..., 6] * mie_m + c[..., 7] * zenith)
    return (f * rads[:, None]).T


def solar_spectral(turbidity: float, ray_el_rad, sin_gamma) -> np.ndarray:
    """(N, 11) direct solar radiance (no sky): solar_radiance_internal2 at each band centre, for rays at elevation
    ray_el_rad (>= 0) and angle gamma from the sun's centre."""
    d = load_data()
    sol_sin = math.sin(SOLAR_RADIUS_RAD)
    sc2 = np.clip(1.0 - sin_gamma ** 2 / (sol_sin * sol_sin), 0.0, None)
    sample_cos = np.sqrt(sc2)                                            # (N,)
    t_low = int(turbidity) - 1
    t_frac = turbidity - (t_low + 1)
    if t_low == 9:
        t_low, t_frac = 8, 1.0
    pos = np.minimum((np.power(2.0 * ray_el_rad / math.pi, 1.0 / 3.0) * 45).astype(np.int64), 44)
    x = ray_el_rad - (pos / 45.0) ** 3 * (math.pi * 0.5)
    sol = d["solar"]                                                     # (11, 10, 45, 4), file order
    out = np.empty((ray_el_rad.size, sol.shape[0]))
    for b in range(sol.shape[0]):
        def sr(t):
            c = sol[b, t][pos]                                           # (N, 4); constant term last
            return c[:, 3] + x * (c[:, 2] + x * (c[:, 1] + x * c[:, 0]))
        direct = (1.0 - t_frac) * sr(t_low) + t_frac * sr(t_low + 1)
        ld = d["limb"][b]
        dark = ld[0] + sample_cos * (ld[1] + sample_cos * (ld[2] + sample_cos * (ld[3] + sample_cos * (
            ld[4] + sample_cos * ld[5]))))
        out[:, b] = np.where(sample_cos > 0.0, direct * dark, 0.0)
    return out


# ------------------------------------------------------------------------------------------------ evaluator
@dataclass(frozen=True)
class Capabilities:
    altitude: bool = False
    polarisation: bool = False
    transmittance: bool = False
    spectral: bool = False
    finite_distance: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def compass_to_dir(az_deg: float, el_deg: float) -> np.ndarray:
    """Atlas Scene's world direction of a compass azimuth / elevation: (cos A cos e, sin e, -sin A cos e)."""
    A, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(A) * math.cos(e), math.sin(e), -math.sin(A) * math.cos(e)])


class HosekWilkieEvaluator:
    """Hosek-Wilkie sky + sun for one sun direction, turbidity and ground albedo, in the frame of `sun_dir`."""

    def __init__(self, sun_dir, turbidity: float, albedo: float):
        s = np.asarray(sun_dir, np.float64).reshape(3)
        self.sun_dir = s / np.linalg.norm(s)
        self.sun_el_rad = math.asin(max(-1.0, min(1.0, float(self.sun_dir[1]))))
        if self.sun_el_rad < 0.0:
            raise ValueError(f"sun elevation {math.degrees(self.sun_el_rad):.3f} deg: Hosek-Wilkie covers 0-90 deg")
        if not TURBIDITY_RANGE[0] <= turbidity <= TURBIDITY_RANGE[1]:
            raise ValueError(f"turbidity {turbidity} outside Hosek-Wilkie's {TURBIDITY_RANGE}")
        if not 0.0 <= albedo <= 1.0:
            raise ValueError(f"ground albedo {albedo} outside 0-1")
        self.turbidity, self.albedo = float(turbidity), float(albedo)
        self.configs, self.rads = cook(self.turbidity, self.albedo, self.sun_el_rad)
        self.W = band_weights()
        self.implementation_version = (f"atlas_vfx.sky_hosek/{VERSION} hosek-wilkie-1.4a data={data_id()} "
                                       f"cmf=wyman2013 T={self.turbidity:.4f} albedo={self.albedo:.4f}")

    @classmethod
    def from_compass(cls, az_deg: float, el_deg: float, turbidity: float, albedo: float):
        return cls(compass_to_dir(az_deg, el_deg), turbidity, albedo)

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _chunks(self, directions, fn):
        d = np.asarray(directions, np.float64).reshape(-1, 3)
        out = np.zeros((d.shape[0], 3))
        for i in range(0, d.shape[0], CHUNK):
            c = d[i:i + CHUNK]
            up = c[:, 1] > 0.0
            if up.any():
                out[i:i + CHUNK][up] = fn(c[up])
        return out

    def sky_radiance(self, directions) -> np.ndarray:
        def f(c):
            return sky_spectral(self.configs, self.rads, np.clip(c[:, 1], 0.0, 1.0),
                                np.clip(c @ self.sun_dir, -1.0, 1.0)) @ self.W
        return self._chunks(directions, f)

    def sun_radiance(self, directions) -> np.ndarray:
        def f(c):
            cg = np.clip(c @ self.sun_dir, -1.0, 1.0)
            out = np.zeros((c.shape[0], 3))
            near = cg > math.cos(SOLAR_RADIUS_RAD)
            if near.any():
                cn = cg[near]
                ray_el = np.arcsin(np.clip(c[near, 1], 0.0, 1.0))
                out[near] = solar_spectral(self.turbidity, ray_el, np.sqrt(np.clip(1.0 - cn * cn, 0.0, 1.0))) @ self.W
            return out
        return self._chunks(directions, f)

    def transmittance(self, directions, distance_m):
        raise NotImplementedError("Hosek-Wilkie has no transmittance")

    def in_scatter(self, directions, distance_m):
        raise NotImplementedError("Hosek-Wilkie has no in-scatter")


def factory(env):
    """For Atlas Scene's register_evaluator(name, factory): an evaluator from its Environment (env.sun: azimuth /
    elevation in degrees; env.atmosphere: turbidity if it has one, ground_albedo). Duck-typed, no Atlas Scene import."""
    sun = env.sun
    atm = env.atmosphere
    turb = getattr(getattr(atm, "turbidity", None), "value", None) or 2.0
    return HosekWilkieEvaluator.from_compass(float(sun.azimuth_deg), float(sun.elevation_deg), float(turb),
                                             float(atm.ground_albedo.value))
