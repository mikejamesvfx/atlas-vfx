"""Sensor presets: the area of the sensor a recording mode actually uses, its rolling-shutter readout, and the focal of
the lenses shot on it, each value marked with where it came from. Upstream TODO: belongs in Atlas Camera.

Every readout and focal is a list of evidence; the flat fields (`readout_ms`, `readout_source`, `readout_status`,
`readout_date`, and the same for focal) are filled from that list by one rule at import, so they can't drift apart:

    chart measurement > RS-aware solve > preset spec

and never a Readout Solve residual from a global-shutter camera (kind `gs_residual`): that number is the GS solve's
own bend, not the sensor (on the owner's test plate it read 23.5 ms against 26.6 ms RS-aware).
Within one kind the first entry listed wins, so list the more specific one first. Status per kind: chart =
measured, RS-aware solve = solved, spec = unmeasured.

Video files carry no focal length (exiftool on five X-H2 clips: lens range only), so focal is kept per lens and zoom
mark the same way. It matters: on one test clip the solved focal moved the solved readout 18 -> 23 ms.
"""
from __future__ import annotations

import copy

STATUS = {"chart": "measured", "rs_solve": "solved", "solve": "solved", "spec": "unmeasured"}
READOUT_RANK = {"chart": 0, "rs_solve": 1, "spec": 2}
FOCAL_RANK = {"chart": 0, "solve": 1, "spec": 2}
NEVER = {"gs_residual": "Readout Solve residual from a global-shutter camera: the GS solve's bend, not the sensor"}
STATUSES = ("unmeasured", "solved", "measured")


def evidence(kind: str, value: float, source: str, date: str, **extra) -> dict:
    """One piece of evidence for a readout (ms) or a focal (mm)."""
    return {"kind": kind, "value": float(value), "source": source, "date": date, **extra}


def pick(candidates, rank: dict) -> dict | None:
    """The winning entry by `rank` (lower wins, first listed wins a tie), plus what was passed over and why.
    Entries without a value, or with a value <= 0, are skipped. None when nothing usable is left."""
    usable, rejected = [], []
    for i, c in enumerate(candidates):
        kind = c.get("kind")
        if kind in NEVER:
            rejected.append({**c, "why": NEVER[kind]})
        elif kind not in rank:
            raise ValueError(f"unknown evidence kind {kind!r}; have {sorted(rank)} (and never {sorted(NEVER)})")
        elif c.get("value") is None or not c["value"] > 0:
            rejected.append({**c, "why": "no value"})
        else:
            usable.append((rank[kind], i, c))
    if not usable:
        return None
    usable.sort(key=lambda t: t[:2])
    best = usable[0][2]
    passed = [{**c, "why": "lower in the rule" if r > usable[0][0] else "listed after the winner"}
              for r, _, c in usable[1:]]
    return {"value": best["value"], "kind": best["kind"], "status": STATUS[best["kind"]], "source": best["source"],
            "date": best["date"], "passed_over": passed, "rejected": rejected}


def pick_readout(candidates) -> dict | None:
    """chart > RS-aware solve > spec; a GS residual is never picked."""
    return pick(candidates, READOUT_RANK)


def pick_focal(candidates) -> dict | None:
    """chart > solve > the nominal zoom mark."""
    return pick(candidates, FOCAL_RANK)


# 30.9 ms is DPReview's full-width 8K figure over 4320 rows; DCI 4K HQ reads about 4076 rows.
_XH2_DCI4K_SPEC_MS = round(30.9 * 4076 / 4320, 1)       # 29.2

# Presets and lenses are generic: published specs for the body or lens, true for anyone's copy. A solve or chart
# measurement from one person's footage is evidence about that shot, not about the camera model, so it does not go
# here. It reaches Sensor RS Maps with the shot (the camera's extensions.atlas_vfx_rs_solve, or the chart input).
PRESETS = {
    "fujifilm_xh2_dci4k_24p_dis_off": {
        "body": "FUJIFILM X-H2", "width_mm": 23.5, "height_mm": 12.39, "resolution": [4096, 2160],
        "note": "DCI 4K HQ 24p, DIS off: 23.5 mm APS-C width, height from the DCI 4K aspect (23.5 x 2160 / 4096)",
        "readout_evidence": [
            evidence("spec", _XH2_DCI4K_SPEC_MS, "DPReview 30.9 ms (8K full width, 4320 rows) scaled to the ~4076 "
                     "rows DCI 4K HQ reads; scaled, not measured", "2026-10-22"),
        ],
    },
}

LENSES = {
    "fujifilm_xf16-55_f2.8": {
        "name": "FUJINON XF16-55mm F2.8", "zoom_mm": [16.0, 55.0], "sensor_preset": "fujifilm_xh2_dci4k_24p_dis_off",
        "marks": {
            "16": {
                "focal_evidence": [evidence("spec", 16.0, "zoom mark (nominal)", "2026-10-22")],
                "distortion": {"status": "unmeasured", "source": "no published k1 k2; measure on a chart",
                               "date": "2026-10-22"},
            },
        },
    },
}


def _settle(entry: dict, what: str, pick_fn, unit: str) -> None:
    won = pick_fn(entry.get(f"{what}_evidence", []))
    entry[f"{what}_{unit}"] = won["value"] if won else None
    entry[f"{what}_source"] = f"{won['kind']}: {won['source']}" if won else "none"
    entry[f"{what}_status"] = won["status"] if won else "unmeasured"
    entry[f"{what}_date"] = won["date"] if won else None


for _p in PRESETS.values():
    _settle(_p, "readout", pick_readout, "ms")
for _l in LENSES.values():
    for _m in _l["marks"].values():
        _settle(_m, "focal", pick_focal, "mm")


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown sensor preset {name!r}; have {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def readout_candidates(preset_name: str | None = None, *, chart_ms: float | None = None, camera: dict | None = None,
                       gs_residual_ms: float | None = None) -> list:
    """Evidence for one shot, most specific first: a chart measurement passed in, the shot camera's own RS-aware solve
    (extensions.atlas_vfx_rs_solve), a GS residual (kept only to be rejected on the record), then the preset's list."""
    out = []
    if chart_ms:
        out.append(evidence("chart", chart_ms, "chart measurement connected to the node", "this run"))
    ext = ((camera or {}).get("extensions") or {}).get("atlas_vfx_rs_solve")
    if ext and ext.get("readout_ms"):
        how = "readout free" if ext.get("readout_solved") else "readout held"
        out.append(evidence("rs_solve", ext["readout_ms"], f"this shot camera's RS-aware solve ({how}, input "
                            f"{ext.get('input_camera', '?')})", "camera"))
    if gs_residual_ms:
        out.append(evidence("gs_residual", gs_residual_ms, "Sensor Readout Solve on a GS camera", "this run"))
    if preset_name:
        out.extend({**e, "source": f"preset {preset_name}: {e['source']}"}
                   for e in get_preset(preset_name)["readout_evidence"])
    return out


def sensor_block(preset: str | None = None, width_mm: float | None = None, height_mm: float | None = None) -> dict:
    """shot-camera@0.1 'sensor' block from a preset, or from explicit mm values (preset 'custom' or None)."""
    if preset and preset != "custom":
        if preset not in PRESETS:
            raise KeyError(f"unknown sensor preset {preset!r}; have {sorted(PRESETS)}")
        p = PRESETS[preset]
        return {"width_mm": p["width_mm"], "height_mm": p["height_mm"], "preset": preset}
    if not (width_mm and height_mm):
        raise ValueError("custom sensor needs width_mm and height_mm")
    return {"width_mm": float(width_mm), "height_mm": float(height_mm)}


def preset_for_body(body: str | None, width: int, height: int) -> str | None:
    for name, p in PRESETS.items():
        if p["body"] == body and p["resolution"] == [width, height]:
            return name
    return None


def preset_for_camera(camera: dict) -> str | None:
    """The sensor preset a shot camera was imported with, else the one its plate's body and size match."""
    name = (camera.get("sensor") or {}).get("preset")
    if name in PRESETS:
        return name
    plate = camera.get("plate") or {}
    return preset_for_body(plate.get("camera_body"), plate.get("width"), plate.get("height"))
