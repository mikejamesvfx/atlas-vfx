"""Fit Physical Sky's Hosek-Wilkie turbidity table to Prague (owner's machine: needs Atlas Scene + a Prague dataset).

For each visibility V in the table, Prague is rendered at sun elevations 10, 20, 40 and 60 deg (width 256, albedo
0.25; Hosek-Wilkie is known to disagree near the horizon, so lower suns are left out of the fit). A golden-section
search over turbidity 1-10 then minimises the summed squared log error of the light a comp gets:
  sun_strength      the light rig's key against a sky of 1 (Relight's 'physical sky' mode)
  E_sky_up          sky irradiance on an up-facing surface (absolute scale)
and reports, without fitting them, the zenith / horizon luminance ratio and the sky's xy 90 deg round from the sun
at 30 deg elevation (Hosek-Wilkie's sky hue differs from Prague's in a way turbidity does not fix; fitting it pulled
turbidity away from the light, 2026-10-29).
It writes atlas_vfx/layers/data/hosek_visibility_turbidity.json with the fit and its provenance (Atlas Scene
version and commit, the Prague upstream commit, the dataset's size and quick hash; no paths), and prints the
absolute Prague/Hosek scale for the record.

Run: <V141 python> tools/hosek_calibrate.py [--dataset FILE]   (else ATLAS_PRAGUE_DATASET, as the node does)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="hosek_cal_"))
os.environ.setdefault("ATLAS_VFX_WORK", str(TMP))

from atlas_vfx import _atlas_scene  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.layers import physical_sky as ps  # noqa: E402
from atlas_vfx.layers import sky_hosek as sh  # noqa: E402
from atlas_vfx.layers.node_physical_sky import dataset_path  # noqa: E402

TABLE = ROOT / "atlas_vfx" / "layers" / "data" / "hosek_visibility_turbidity.json"
VIS = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 131.8)
ELEV = (10.0, 20.0, 40.0, 60.0)
AZ, W, ALB = 250.0, 256, 0.25


def features(img, e_sun, e_sky, el):
    H = img.shape[0]
    lum = img @ ps.LUM709
    row = lambda e: min(H - 1, max(0, int(round((90.0 - e) / 180.0 * H - 0.5))))       # noqa: E731
    x = int(round(((AZ + 90.0 + 180.0) % 360.0) / 360.0 * W - 0.5)) % W
    c = img[row(30.0), x]
    X = sh.REC709_TO_XYZ @ c
    rig = ps.light_rig([[0, 0, -1]], e_sun, e_sky, ALB, el, 50.0, 16 / 9)
    return {"sun_strength": rig["sun_strength"], "e_sky": float(ps.LUM709 @ e_sky),
            "zen_hor": float(lum[0].mean() / lum[row(5.0)].mean()), "xy": X[:2] / X.sum(),
            "e_sun": float(ps.LUM709 @ e_sun)}


def hosek(el, t):
    ev = ps._Oriented(sh.HosekWilkieEvaluator.from_compass(AZ, el, t, ALB), 0.0)
    e_sun = ps._sun_irradiance(ev, ps.azel_to_dir(AZ, el))
    e_sky = ps._sky_irradiance_up(ev)
    return features(ps.render_latlong(ev, W, ALB, 0.0, e_sun, e_sky, el), e_sun, e_sky, el)


def error(p, h):
    return math.log(h["sun_strength"] / p["sun_strength"]) ** 2 + math.log(h["e_sky"] / p["e_sky"]) ** 2


def golden(f, a, b, tol=1e-3):
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="")
    a = ap.parse_args(argv)
    ds = dataset_path(a.dataset)
    _atlas_scene.prague()
    points, rows, scale = [], [], []
    for v in VIS:
        prague = {}
        for el in ELEV:
            r = ps.render_sky(AZ, el, width=W, visibility_km=v, ground_albedo=ALB, dataset=str(ds))
            prague[el] = features(r["lin_rec709"], r["e_sun"], r["e_sky_up"], el)
        t = golden(lambda t: sum(error(prague[el], hosek(el, t)) for el in ELEV), 1.0, 10.0)
        fit = {el: hosek(el, t) for el in ELEV}
        res = sum(error(prague[el], fit[el]) for el in ELEV)
        for el in ELEV:
            scale += [fit[el]["e_sky"] / prague[el]["e_sky"], fit[el]["e_sun"] / prague[el]["e_sun"]]
        points.append([v, round(t, 4)])
        rows.append({"visibility_km": v, "turbidity": round(t, 4), "residual": round(res, 5),
                     "sun_strength_ratio": [round(fit[el]["sun_strength"] / prague[el]["sun_strength"], 3)
                                            for el in ELEV],
                     "e_sky_ratio": [round(fit[el]["e_sky"] / prague[el]["e_sky"], 3) for el in ELEV],
                     "zenith_horizon_ratio": [round(fit[el]["zen_hor"] / prague[el]["zen_hor"], 3) for el in ELEV],
                     "dxy_90deg": [round(float(np.abs(fit[el]["xy"] - prague[el]["xy"]).max()), 4) for el in ELEV]})
        print(f"V {v:6.1f} km -> T {t:.3f}  residual {res:.4f}  sun_strength H/P {rows[-1]['sun_strength_ratio']}  "
              f"E_sky H/P {rows[-1]['e_sky_ratio']}  dxy {rows[-1]['dxy_90deg']}")
    rep = _atlas_scene.report()
    table = {
        "about": "Hosek-Wilkie turbidity for Physical Sky's visibility_km (Prague's input), piecewise linear in log "
                 "visibility.",
        "method": "fitted to Prague renders (tools/hosek_calibrate.py): golden-section on turbidity per visibility, "
                  "summed squared log error of sun_strength and E_sky_up (colour and zenith/horizon reported in fit, "
                  f"not fitted), sun elevations {list(ELEV)} deg, azimuth {AZ}, albedo {ALB}, width {W}",
        "provenance": {"date": date.today().isoformat(), "atlas_scene_version": rep.get("version"),
                       "atlas_scene_commit": rep.get("commit"),
                       "prague_upstream_commit": (rep.get("prague") or {}).get("upstream_commit"),
                       "dataset": {"name": ds.name, "bytes": ds.stat().st_size, "quick_sha256": pl.quick_sha256(ds)},
                       "hosek_data": sh.data_id(),
                       "absolute_scale_hosek_over_prague_median": round(float(np.median(scale)), 4)},
        "fit": rows,
        "points": points,
    }
    TABLE.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TABLE.relative_to(ROOT)}; absolute scale Hosek/Prague median {np.median(scale):.3f} "
          f"(range {min(scale):.3f}-{max(scale):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
