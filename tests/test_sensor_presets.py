"""Sensor presets carry readout and focal with a source (plan, Phase 1b S2).

- every sensor preset has readout_ms, readout_source, readout_status (unmeasured / solved / measured) and a date,
  and the flat fields agree with the rule over its evidence; every lens zoom mark has the same for focal
- the rule: chart + solve + spec picks the chart, solve + spec picks the solve, spec alone picks the spec, and a GS
  residual (Sensor Readout Solve on a global-shutter camera) is never picked, wherever it is listed or alone
- presets are generic: the X-H2 DCI 4K HQ preset holds the published spec only (29.2 ms, unmeasured), the XF16-55
  the nominal mark; the owner's test-plate solves are shot evidence and stay out of the table
- Sensor RS Maps on 'preset rule': the camera's preset gives the readout, a connected Readout Solve value is
  rejected on the record, a chart input wins, the camera's own RS-aware solve beats the spec, a camera with no
  preset falls back to the typed value; the sidecar records which source won. Old modes still listed.
- Sensor Readout Solve says DEPRECATED and logs a warning when it runs

Run: <V141 python> tests\\test_sensor_presets.py
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("ATLAS_VFX_OUT_ROOTS", tempfile.gettempdir())     # RS Maps writes under temp

from atlas_vfx.camera_side import sensors  # noqa: E402
from atlas_vfx.sensor import jobs  # noqa: E402

FAILS = []
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
XH2 = "fujifilm_xh2_dci4k_24p_dis_off"
SPEC = round(30.9 * 4076 / 4320, 1)                  # the X-H2 DCI 4K HQ spec readout, 29.2 ms


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def ev(kind, ms):
    return sensors.evidence(kind, ms, f"test {kind}", "2026-10-26")


def test_fields():
    for name, p in sensors.PRESETS.items():
        have = all(k in p for k in ("readout_ms", "readout_source", "readout_status", "readout_date"))
        check(have and isinstance(p["readout_ms"], float) and p["readout_ms"] > 0, f"{name}: readout fields",
              f"{p.get('readout_ms')} ms, {p.get('readout_status')}")
        check(p.get("readout_status") in sensors.STATUSES, f"{name}: status is one of {sensors.STATUSES}")
        check(bool(DATE.match(str(p.get("readout_date")))), f"{name}: readout date", str(p.get("readout_date")))
        check(all(e["kind"] in sensors.READOUT_RANK and DATE.match(e["date"]) and e["source"]
                  for e in p["readout_evidence"]), f"{name}: every readout evidence has a kind, source and date")
        won = sensors.pick_readout(p["readout_evidence"])
        check(won["value"] == p["readout_ms"] and won["status"] == p["readout_status"],
              f"{name}: flat fields are the rule's pick over the evidence")
    for name, lens in sensors.LENSES.items():
        check(lens["sensor_preset"] in sensors.PRESETS, f"{name}: names a known sensor preset")
        for mark, m in lens["marks"].items():
            ok = (all(k in m for k in ("focal_mm", "focal_source", "focal_status", "focal_date"))
                  and m["focal_mm"] > 0 and m["focal_status"] in sensors.STATUSES and DATE.match(str(m["focal_date"])))
            check(ok, f"{name} @ {mark} mm: focal fields", f"{m.get('focal_mm')} mm, {m.get('focal_status')}")
            d = m.get("distortion", {})
            check(d.get("status") in sensors.STATUSES and "source" in d, f"{name} @ {mark} mm: distortion marked",
                  d.get("status", ""))


def test_xh2_values():
    """Presets are generic (published specs); the owner's test-plate solves are shot evidence, not preset data."""
    p = sensors.PRESETS[XH2]
    check([e["kind"] for e in p["readout_evidence"]] == ["spec"], "X-H2 DCI 4K HQ: spec only, no test-plate solve")
    check(abs(p["readout_ms"] - 30.9 * 4076 / 4320) < 0.05 and p["readout_status"] == "unmeasured",
          "X-H2 DCI 4K HQ: 30.9 ms scaled to 4076 rows, unmeasured", f"{p['readout_ms']} ms")
    m16 = sensors.LENSES["fujifilm_xf16-55_f2.8"]["marks"]["16"]
    check([e["kind"] for e in m16["focal_evidence"]] == ["spec"] and m16["focal_mm"] == 16.0
          and m16["focal_status"] == "unmeasured", "XF16-55 at 16: the nominal mark, unmeasured", m16["focal_source"])
    check(m16["distortion"]["status"] == "unmeasured" and "k1" not in m16["distortion"],
          "XF16-55 at 16: distortion unmeasured, no made-up k1 k2")


def test_rule():
    chart, solve, spec, gs = ev("chart", 28.8), ev("rs_solve", 26.6), ev("spec", 29.2), ev("gs_residual", 23.5)
    w = sensors.pick_readout([spec, solve, chart])
    check(w["kind"] == "chart" and w["value"] == 28.8 and w["status"] == "measured", "chart + solve + spec: chart")
    check(len(w["passed_over"]) == 2, "chart + solve + spec: the other two are on the record")
    w = sensors.pick_readout([spec, solve])
    check(w["kind"] == "rs_solve" and w["status"] == "solved", "solve + spec: solve")
    w = sensors.pick_readout([spec])
    check(w["kind"] == "spec" and w["status"] == "unmeasured", "spec alone: spec")
    for order in ([gs, spec, solve], [solve, gs], [gs, spec], [spec, gs]):
        w = sensors.pick_readout(order)
        check(w["kind"] != "gs_residual" and any(r["kind"] == "gs_residual" for r in w["rejected"]),
              f"GS residual never picked: {[c['kind'] for c in order]} -> {w['kind']}")
    check(sensors.pick_readout([gs]) is None, "GS residual alone: nothing picked")
    check(sensors.pick_readout([ev("chart", 0), spec])["kind"] == "spec", "a zero chart value is skipped")
    s2 = sensors.evidence("rs_solve", 27.7, "second", "2026-10-26")
    check(sensors.pick_readout([solve, s2])["value"] == 26.6, "same kind: the first listed wins")
    try:
        sensors.pick_readout([ev("guess", 30.0)])
        check(False, "unknown kind refused")
    except ValueError:
        check(True, "unknown kind refused")


def camera_data(preset=XH2, rs_solve_ms=None):
    """Just the parts of a shot camera the rule reads."""
    d = {"sensor": {"width_mm": 23.5, "height_mm": 12.39}, "plate": {"width": 4096, "height": 2160}}
    if preset:
        d["sensor"]["preset"] = preset
    if rs_solve_ms:
        d["extensions"] = {"atlas_vfx_rs_solve": {"readout_ms": rs_solve_ms, "readout_solved": True,
                                                  "input_camera": "gs"}}
    return d


def test_decide():
    d = jobs.decide_readout(camera_data(), 30.9, jobs.PRESET_RULE, measured_ms=23.5)
    check(d["readout_ms"] == SPEC and d["winner"]["kind"] == "spec", "preset rule, GS camera: the preset's spec",
          d["readout_from"][:60])
    check(any(r["value"] == 23.5 and r["kind"] == "gs_residual" for r in d["rejected"]),
          "preset rule: connected Readout Solve value rejected on the record")
    check("warning" in d, "preset rule on a GS camera: warns")
    d = jobs.decide_readout(camera_data(), 30.9, jobs.PRESET_RULE, measured_ms=23.5, chart_ms=28.8)
    check(d["readout_ms"] == 28.8 and d["winner"]["status"] == "measured", "preset rule: chart input wins")
    d = jobs.decide_readout(camera_data(rs_solve_ms=27.7), 30.9, jobs.PRESET_RULE)
    check(d["readout_ms"] == 27.7 and "warning" not in d, "preset rule: the camera's own RS solve beats the spec")
    d = jobs.decide_readout(camera_data(preset=None) | {"plate": {"width": 1920, "height": 1080}}, 19.8,
                            jobs.PRESET_RULE)
    check(d["readout_ms"] == 19.8 and d["readout_from"].startswith("typed"), "preset rule, no preset: typed value")
    d = jobs.decide_readout(camera_data(preset=None) | {"plate": {"width": 4096, "height": 2160,
                                                                  "camera_body": "FUJIFILM X-H2"}},
                            30.9, jobs.PRESET_RULE)
    check(d["preset"] == XH2 and d["readout_ms"] == SPEC, "preset rule: preset found from the plate's body and size")
    d = jobs.decide_readout(camera_data(), 30.9, "measured if connected", measured_ms=29.5)
    check(d["readout_ms"] == 29.5, "old mode 'measured if connected' unchanged")
    d = jobs.decide_readout(camera_data(), 15.3, "typed", measured_ms=29.5, chart_ms=28.8)
    check(d["readout_ms"] == 15.3 and d["readout_from"] == "typed", "old mode 'typed' unchanged")
    before = copy.deepcopy(sensors.PRESETS)
    sensors.readout_candidates(XH2, chart_ms=28.8)[0]["value"] = -1
    sensors.get_preset(XH2)["readout_evidence"].clear()
    check(sensors.PRESETS == before, "candidates and get_preset hand out copies; the table stays as written")


def test_nodes():
    from atlas_vfx.sensor.node_readout_solve import AtlasVFXSensorReadoutSolve
    from atlas_vfx.sensor.node_rs_maps import AtlasVFXSensorRSMaps
    from test_sensor_lens import synthetic_shot
    req = AtlasVFXSensorRSMaps.INPUT_TYPES()
    opts, cfg = req["required"]["readout_source"]
    check(opts[0] == jobs.PRESET_RULE and cfg["default"] == jobs.PRESET_RULE
          and {"measured if connected", "typed"} <= set(opts), "RS Maps: default preset rule, old modes still listed")
    check("readout_measured_ms" in req["optional"] and "readout_chart_ms" in req["optional"],
          "RS Maps: Readout Solve input kept, chart input added")
    check(AtlasVFXSensorReadoutSolve.DESCRIPTION.startswith("DEPRECATED"), "Readout Solve: DESCRIPTION says DEPRECATED")

    seen = []

    class Grab(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = logging.getLogger("atlas_vfx")
    log.addHandler(Grab())
    try:
        AtlasVFXSensorReadoutSolve().run({"not": "a camera"}, "", "", 2, True, 50.0, "")
    except Exception:
        pass                                          # the warning comes first, the bad camera fails after
    check(any("deprecated" in m for m in seen), "Readout Solve: logs a deprecation warning when it runs")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam, depth_dir = synthetic_shot(td)
        res = AtlasVFXSensorRSMaps().run(cam.data, str(depth_dir), 30.9, jobs.PRESET_RULE, "apply RS", 0.0, 3, 0.5,
                                         1, 1131, 1132, str(td / "rs"), readout_measured_ms=23.5)
        rep = json.loads(res["result"][2])
        side = json.loads((td / "rs" / "sensor_rs.sidecar.json").read_text(encoding="utf-8"))
        check(rep["readout_ms"] == SPEC and rep["readout_from"].startswith("unmeasured (spec)"),
              "RS Maps node, preset rule: the preset's spec", rep["readout_from"][:50])
        win = side["inputs"].get("readout_winner") or {}
        check(side["inputs"].get("readout_mode") == jobs.PRESET_RULE and win.get("kind") == "spec",
              "RS Maps sidecar records the mode and which source won")
        rej = (side["outputs"].get("readout_decision") or {}).get("rejected", [])
        check(any(r["kind"] == "gs_residual" and r["value"] == 23.5 for r in rej),
              "RS Maps sidecar records the rejected Readout Solve value")
        check(any("not using 23.50 ms" in m for m in seen), "RS Maps logs the rejected value")
        res = AtlasVFXSensorRSMaps().run(cam.data, str(depth_dir), 30.9, jobs.PRESET_RULE, "apply RS", 0.0, 3, 0.5,
                                         1, 1131, 1131, str(td / "rs_c"), readout_chart_ms=28.8)
        rep = json.loads(res["result"][2])
        check(rep["readout_ms"] == 28.8 and rep["readout_from"].startswith("measured (chart)"),
              "RS Maps node: chart input wins", rep["readout_from"][:40])


if __name__ == "__main__":
    test_fields()
    test_xh2_values()
    test_rule()
    test_decide()
    test_nodes()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
