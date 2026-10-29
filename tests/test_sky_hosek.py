"""Hosek-Wilkie fallback sky (atlas_vfx/layers/sky_hosek.py): the shipped data matches its manifest and fails loudly
when it doesn't, the numpy port matches the authors' compiled reference (tests/fixtures/hosek_golden.json, made by
`tools/hosek_wilkie_data.py --golden`; SKIP when it isn't there), the evaluator keeps Atlas Scene's contract, the sun
disc and limb darkening behave, the result is physically plausible, the colour path is right, and the module imports
nothing but numpy and the standard library. Runs everywhere: no Atlas Scene, no dataset.

Run: <V141 python> tests\\test_sky_hosek.py   (plain script, exits non-zero on failure; no pytest)
"""
from __future__ import annotations

import ast
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import sky_hosek as sh  # noqa: E402

FAILS: list[str] = []
GOLDEN = ROOT / "tests" / "fixtures" / "hosek_golden.json"


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if str(detail) != "" else ""), flush=True)
    if not cond:
        FAILS.append(label)


def test_data():
    d = sh.load_data()
    shapes = {k: v.shape for k, v in d.items()}
    check(shapes == {"bands_nm": (11,), "sky_config": (11, 2, 10, 6, 9), "sky_radiance": (11, 2, 10, 6),
                     "solar": (11, 10, 45, 4), "limb": (11, 6)}, "data arrays and shapes", shapes)
    check(np.allclose(d["bands_nm"], np.arange(320, 721, 40)), "11 bands, 320-720 nm every 40 nm")
    m = json.loads(sh.MANIFEST.read_text(encoding="utf-8"))
    check("BSD-3-Clause" in m["licence"] and m["source"]["commit"] and len(m["source"]["commit"]) == 40,
          "manifest names the licence and the pinned source commit", m["source"]["commit"][:10])
    notice = (sh.DATA_DIR / "THIRD_PARTY_HOSEK_WILKIE.md").read_text(encoding="utf-8")
    check("Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie" in notice and "3-clause BSD" in notice,
          "the BSD notice ships next to the data")
    # a changed coefficient must be refused
    tmp = Path(tempfile.mkdtemp(prefix="hosek_tamper_"))
    with np.load(sh.NPZ, allow_pickle=False) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["limb"][3, 2] += 1e-9
    np.savez_compressed(tmp / sh.NPZ.name, **arrs)
    shutil.copy(sh.MANIFEST, tmp / sh.MANIFEST.name)
    old = sh.NPZ
    sh.NPZ = tmp / sh.NPZ.name
    sh.load_data.cache_clear()
    try:
        sh.load_data()
        check(False, "a tampered coefficient is refused")
    except sh.DataError as e:
        check(True, "a tampered coefficient is refused", str(e)[:70])
    finally:
        sh.NPZ = old
        sh.load_data.cache_clear()


def _interp_bands(values, wl):
    return float(np.interp(wl, sh.load_data()["bands_nm"], values))


def test_golden():
    if not GOLDEN.is_file():
        print(f"SKIP golden: {GOLDEN.relative_to(ROOT)} not built (run tools/hosek_wilkie_data.py --golden on a "
              "machine with a C compiler)")
        return
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    worst = 0.0
    for t, a, se, th, ga, wl, want in g["sky"]:
        cfg, rad = sh.cook(t, a, se)
        bands = sh.sky_spectral(cfg, rad, np.array([math.cos(th)]), np.array([math.cos(ga)]))[0]
        got = _interp_bands(bands, wl)
        worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    check(worst < 1e-9, "sky radiance = the reference C", f"{len(g['sky'])} cases, worst rel {worst:.2e}")
    worst = 0.0
    for t, a, el, ga, wl, want in g["sun"]:
        direct, dark = sh.solar_terms(t, np.array([el]), np.array([math.sin(ga)]))
        got = _interp_bands(direct[0], wl) * _interp_bands(dark[0], wl)
        worst = max(worst, abs(got - want) / max(abs(want), 1e-300) if want else abs(got))
    check(worst < 1e-9, "solar radiance = the reference C", f"{len(g['sun'])} cases, worst {worst:.2e}")


def test_contract():
    ev = sh.HosekWilkieEvaluator.from_compass(250.0, 30.0, 2.5, 0.25)
    caps = ev.capabilities().to_dict()
    check(set(caps) == {"altitude", "polarisation", "transmittance", "spectral", "finite_distance"}
          and all(v is False for v in caps.values()), "capabilities: Atlas Scene's five flags, all False", caps)
    check(ev.implementation_version.startswith("atlas_vfx.sky_hosek/") and sh.data_id() in ev.implementation_version,
          "implementation_version names the port and the data", ev.implementation_version)
    rng = np.random.default_rng(7)
    d = rng.normal(size=(5000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sky, sun = ev.sky_radiance(d), ev.sun_radiance(d)
    check(sky.shape == (5000, 3) and sky.dtype == np.float64 and np.isfinite(sky).all() and np.isfinite(sun).all(),
          "radiance (N, 3) float64, finite")
    below = d[:, 1] <= 0
    check(not sky[below].any() and not sun[below].any() and (sky[~below] > 0).all(), "lower hemisphere black, sky "
          "above positive")
    for name in ("transmittance", "in_scatter"):
        try:
            getattr(ev, name)(d[:3], 10.0)
            check(False, f"{name} raises NotImplementedError")
        except NotImplementedError:
            check(True, f"{name} raises NotImplementedError")
    for bad in ({"turbidity": 0.5, "albedo": 0.2}, {"turbidity": 2.0, "albedo": 1.5}):
        try:
            sh.HosekWilkieEvaluator.from_compass(250.0, 30.0, **bad)
            check(False, f"refused {bad}")
        except ValueError:
            check(True, f"refused {bad}")
    try:
        sh.HosekWilkieEvaluator.from_compass(250.0, -1.0, 2.0, 0.2)
        check(False, "refused a sun below the horizon")
    except ValueError:
        check(True, "refused a sun below the horizon")

    class Box:
        def __init__(self, **k):
            self.__dict__.update(k)
    env = Box(sun=Box(azimuth_deg=250.0, elevation_deg=30.0), atmosphere=Box(ground_albedo=Box(value=0.25)))
    ev2 = sh.factory(env)
    check(np.allclose(ev2.sky_radiance(d[:50]), sh.HosekWilkieEvaluator.from_compass(250, 30, 2.0, 0.25)
                      .sky_radiance(d[:50])), "factory(env) builds the same evaluator (turbidity 2 when unset)")


def test_sun_disc():
    ev = sh.HosekWilkieEvaluator.from_compass(120.0, 40.0, 3.0, 0.1)
    s = ev.sun_dir
    a = np.cross(s, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)

    def at(deg):
        r = math.radians(deg)
        return (math.cos(r) * s + math.sin(r) * a)[None]
    centre = float((ev.sun_radiance(at(0.0)) @ [0.2126, 0.7152, 0.0722])[0])
    edge = float((ev.sun_radiance(at(0.95 * sh.SOLAR_RADIUS_DEG)) @ [0.2126, 0.7152, 0.0722])[0])
    out = ev.sun_radiance(at(0.27))
    check(centre > 0 and not out.any(), "sun > 0 at the centre, 0 outside the 0.255 deg disc", f"{centre:.4g}")
    check(0 < edge < centre, "limb darkening: the edge is darker than the centre", f"{edge / centre:.3f}")
    check(centre > 100 * float((ev.sky_radiance(at(0.0)) @ [0.2126, 0.7152, 0.0722])[0]), "the disc outshines the sky "
          "behind it")


def test_rotation():
    """Only the angles to the zenith and to the sun matter: turning sun and directions together about Y changes
    nothing."""
    rng = np.random.default_rng(3)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c, s = math.cos(1.1), math.sin(1.1)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    sun = sh.compass_to_dir(80.0, 25.0)
    a = sh.HosekWilkieEvaluator(sun, 2.2, 0.3)
    b = sh.HosekWilkieEvaluator(R @ sun, 2.2, 0.3)
    check(np.allclose(a.sky_radiance(d), b.sky_radiance(d @ R.T), rtol=1e-12, atol=0), "rotation about Y invariant")


def test_plausible():
    """Clear sky, sun at 60 deg, turbidity 2: direct normal illuminance and diffuse sky illuminance in the range of
    measured clear days; hazier air dims the sun and brightens the sky relative to it."""
    from atlas_vfx.layers import physical_sky as ps

    def light(t):
        ev = sh.HosekWilkieEvaluator.from_compass(180.0, 60.0, t, 0.25)
        e_sun = ps._sun_irradiance(ev, sh.compass_to_dir(180.0, 60.0))
        e_sky = ps._sky_irradiance_up(ev)
        return float(ps.LUM709 @ e_sun) * sh.LM_PER_W, float(ps.LUM709 @ e_sky) * sh.LM_PER_W
    sun2, sky2 = light(2.0)
    sun6, sky6 = light(6.0)
    check(70e3 <= sun2 <= 115e3 and 8e3 <= sky2 <= 30e3, "el 60, T 2: E_sun 70-115 klx, E_sky 8-30 klx",
          f"{sun2 / 1e3:.1f} / {sky2 / 1e3:.1f} klx")
    check(sun6 < sun2 and sky6 / sun6 > sky2 / sun2, "T 6: dimmer sun, more sky per sun", f"{sun6 / 1e3:.1f} / "
          f"{sky6 / 1e3:.1f} klx")
    t20, _ = ps.turbidity_for_visibility(20.0)
    t131, _ = ps.turbidity_for_visibility(131.8)
    check(1.0 <= t131 < t20 <= 10.0, "visibility -> turbidity: hazier air, higher turbidity", (round(t20, 3),
                                                                                               round(t131, 3)))


def test_colour():
    lam = np.arange(320.0, 831.0, 1.0)
    c = sh.cmf_wyman2013(lam)
    X, Y, Z = c.sum(0)
    check(abs(X / (X + Y + Z) - 1 / 3) < 0.003 and abs(Y / (X + Y + Z) - 1 / 3) < 0.003,
          "equal-energy spectrum -> xy (1/3, 1/3)", (round(X / (X + Y + Z), 4), round(Y / (X + Y + Z), 4)))
    i = int(c[:, 1].argmax())
    check(553 <= lam[i] <= 557 and abs(c[i, 1] - 1) < 0.01, "ybar peaks at ~555 nm with ~1", (lam[i], round(c[i, 1], 4)))
    d65 = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
    check(np.allclose(sh.XYZ_TO_REC709 @ d65, 1.0, atol=1e-4), "D65 -> Rec.709 white (1, 1, 1)")
    check(np.allclose(sh.XYZ_TO_REC709[0], [3.2409699, -1.5373832, -0.4986108], atol=1e-6),
          "XYZ -> Rec.709 = the BT.709 chromaticities matrix (as colour-science)", sh.XYZ_TO_REC709[0].round(7))


def test_imports():
    src = Path(sh.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add("." if n.level else n.module.split(".")[0])
    check(mods <= set(sys.stdlib_module_names) | {"numpy", "__future__"} and "atlas_scene" not in src.replace(
        "Atlas Scene", ""), "imports numpy and the standard library only, no atlas_scene", sorted(mods))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "load"]
    check(loads and all(any(k.arg == "allow_pickle" and getattr(k.value, "value", True) is False for k in n.keywords)
                        for n in loads), "np.load always with allow_pickle=False")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                check(False, f"{name} raised {type(e).__name__}: {e}")
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
