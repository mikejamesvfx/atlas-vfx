"""Sun Globe (web/atlas_vfx_sun_globe.js), the Physical Sky node's panel (catalogue 48).

What it pins:
1. The JS imports as an ES module with a stubbed ComfyUI `app` (node), registers one extension, leaves other node
   types alone, and a failed panel build leaves every native widget visible.
2. The JS / Python mirror, run for real under node: the NOAA sun (layers/solar.py), sky directions, the sun the node
   renders for each mode (physical_sky.effective_sun, errors included), the day's sun path and its events
   (physical_sky.sun_path), the camera basis and projection (camera_to_sky / project_uv), and the drag mapping
   (a dropped sun's widget values give that sun back through effective_sun).
3. The panel only drives native widgets: the names it hides and reads exist on the Python node, the node's inputs
   are unchanged (headless runs are identical), nothing queues from a pointermove, auto-queue defaults off, the
   renderer is disposed in a chained onRemoved, no CDN or cross-pack import.
4. The ui payload (node_physical_sky.globe_ui): shape, a sane sun path (sunrise / sunset at 0 deg, the noon max
   matches solar.py), the DSCF2977 sun, and a real run() below the horizon.
5. The vendored three.js matches its licence register entry.

Needs node (PATH or the default Windows install); FAILS without it (never skips).
Run: <V141 python> tests\\test_sun_globe.py   (plain script, exits non-zero on failure; run from a neutral cwd)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="atlas_vfx_globe_"))
os.environ["ATLAS_VFX_WORK"] = str(TMP / "work")          # before atlas_vfx is imported: plate.WORK_ROOT reads it
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)
sys.path.insert(0, str(ROOT))

from atlas_vfx.layers import physical_sky as ps  # noqa: E402
from atlas_vfx.layers import solar  # noqa: E402

WEB = ROOT / "web"
JS = WEB / "atlas_vfx_sun_globe.js"
SRC = JS.read_text(encoding="utf-8")
FAILS: list[str] = []

DSCF = (-34.1906, 151.0425, "2025-12-25T07:39:25Z")       # Burning Palms, 6:39 pm AEDT; sun 252.355 / 16.121


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if str(detail) != "" else ""), flush=True)
    if not cond:
        FAILS.append(label)


def node_exe() -> str:
    exe = shutil.which("node")
    if not exe and Path(r"C:\Program Files\nodejs\node.exe").is_file():
        exe = r"C:\Program Files\nodejs\node.exe"
    if not exe:
        raise SystemExit("FAIL: node not found (PATH or C:\\Program Files\\nodejs); the JS mirror cannot be tested")
    return exe


def stage() -> Path:
    """The extension as ComfyUI serves it (extensions/atlas-vfx/...) next to a stub scripts/app.js."""
    d = TMP / "esm"
    (d / "extensions" / "atlas-vfx" / "lib").mkdir(parents=True)
    (d / "scripts").mkdir()
    shutil.copy2(JS, d / "extensions" / "atlas-vfx" / JS.name)
    shutil.copy2(WEB / "lib" / "atlas-three.bundle.js", d / "extensions" / "atlas-vfx" / "lib" / "atlas-three.bundle.js")
    (d / "scripts" / "app.js").write_text(
        "export const app = { registerExtension(e) { (globalThis.__exts ||= []).push(e); },\n"
        "  queuePrompt() { globalThis.__queued = (globalThis.__queued || 0) + 1; } };\n", encoding="utf-8")
    (d / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    return d


RUNNER = r"""
import { readFileSync } from "node:fs";
const G = await import("./extensions/atlas-vfx/atlas_vfx_sun_globe.js");
const C = JSON.parse(readFileSync(process.argv[2], "utf-8"));
const out = {};
out.exports = Object.keys(G).sort();
out.exts = (globalThis.__exts || []).map((e) => ({ name: e.name, hook: typeof e.beforeRegisterNodeDef }));
Object.assign(out, { owned: G.OWNED_WIDGETS, read: G.READ_WIDGETS, modes: G.SUN_MODES, node: G.NODE_NAME,
  autoDefault: G.AUTO_QUEUE_DEFAULT, minEl: G.PRAGUE_MIN_ELEVATION });
out.sun = C.sun.map(([la, lo, iso]) => G.sunAt(la, lo, Date.parse(iso)));
out.dir = C.dir.map(([a, e]) => G.azelToDir(a, e));
out.inv = C.dir.map(([a, e]) => G.dirToAzel(G.azelToDir(a, e)));
out.eff = C.eff.map((c) => G.effectiveSun(...c));
out.path = C.path.map(([la, lo, cap]) => G.sunPath(la, lo, cap));
out.basis = C.basis.map(([h, t]) => G.cameraBasis(h, t));
out.proj = C.proj.map(([a, e, h, t, f, asp]) => G.projectToCamera(G.azelToDir(a, e), h, t, f, asp));
out.rel = C.proj.map(([a, e, h, t, f, asp]) => G.sunRelation(a, e, h, t, f, asp));
out.drag = C.drag.map(([m, a, e, d]) => G.dragWidgets(m, a, e, d));
out.parse = C.parse.map((s) => G.parseUtc(s));
out.clock = C.parse.map((s) => G.clockOffsetMin(s));
const p0 = G.sunPath(...C.path[0]);
const cap0 = G.parseUtc(C.path[0][2]);
out.snap = [0, 0.25, 0.5, 0.731, 1].map((x) => [5, 1].map((m) => G.offsetAtX(x, p0, cap0, null, m)));
out.snapClock = [0.25, 0.5].map((x) => G.offsetAtX(x, p0, cap0, 660, 5));
// the dome: three's own SphereGeometry uvs through the texture transform the panel sets
const THREE = await import("./extensions/atlas-vfx/lib/atlas-three.bundle.js");
out.dome = C.dome.map((c) => {
  const geo = new THREE.SphereGeometry(...G.domeGeometryArgs(1));
  const t = new THREE.Texture();
  const u = G.domeUvTransform(c);
  t.repeat.set(...u.repeat); t.offset.set(...u.offset); t.updateMatrix();
  const pos = geo.attributes.position, uv = geo.attributes.uv, rows = [];
  for (let i = 0; i < pos.count; i += 7) {
    const q = new THREE.Vector2(uv.getX(i), uv.getY(i)).applyMatrix3(t.matrix);
    rows.push([[pos.getX(i), pos.getY(i), pos.getZ(i)], [q.x, q.y]]);
  }
  return rows;
});
// node-type hooks: another node is untouched; the sky node gets onNodeCreated / onExecuted (three loads in node)
const ext = (globalThis.__exts || [])[0];
class Other {}
await ext.beforeRegisterNodeDef(Other, { name: "SomethingElse" });
out.otherUntouched = Object.getOwnPropertyNames(Other.prototype).join(",") === "constructor";
class Sky {}
await ext.beforeRegisterNodeDef(Sky, { name: G.NODE_NAME });
out.skyHooks = [typeof Sky.prototype.onNodeCreated, typeof Sky.prototype.onExecuted];
const n = new Sky();
n.onExecuted({ atlas_sun_globe: [{ version: 1 }] });
out.stored = n._atlasSunGlobePayload?.version === 1;
// no DOM here, so the panel cannot build: onNodeCreated must not throw and must hide nothing
const m = new Sky();
m.widgets = C.widgets.map((name) => ({ name, type: "number", value: 0, options: {} }));
try { m.onNodeCreated(); out.createdNoThrow = true; } catch (e) { out.createdNoThrow = String(e); }
out.hiddenAfterFailure = m.widgets.filter((w) => w.hidden || w.options.hidden || w.type === "atlas_hidden").map((w) => w.name);
out.queued = globalThis.__queued || 0;
console.log(JSON.stringify(out));
"""


def ang(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


# ------------------------------------------------------------------------------------------------ cases
SUN_CASES = [list(DSCF), [DSCF[0], DSCF[1], "2025-12-25T13:39:25Z"], [51.5, -0.12, "2026-03-20T12:00:00Z"],
             [78.2, 15.6, "2026-06-21T23:30:00Z"], [-33.9, 151.2, "1998-07-01T03:15:42.5Z"],
             [0.0, -170.0, "2030-01-01T00:00:00+11:00"], [64.1, -21.9, "2026-12-21T15:00:00-01:00"],
             [-89.0, 179.9, "2026-01-01T06:00:00Z"]]
DIR_CASES = [[0, 0], [90, 0], [180, 10], [270, -20], [252.355, 16.121], [33.3, 89.0], [359.9, -89.0]]
EFF_CASES = [  # mode, lat, lon, capture, time_offset_h, az_off, el_off, manual_az, manual_el
    ["metadata", *DSCF, 0, 0, 0, 0, 0], ["metadata", *DSCF, 6, 0, 0, 0, 0], ["offset", *DSCF, 0, 10, -3, 0, 0],
    ["offset", *DSCF, -2.5, -170, 20, 0, 0], ["manual", *DSCF, 0, 0, 0, 45, 30], ["manual", DSCF[0], DSCF[1], "", 0, 0, 0, 359.5, -10],
    ["metadata", DSCF[0], DSCF[1], "2025-12-25T07:39:25", 0, 0, 0, 0, 0], ["manual", DSCF[0], DSCF[1], "not a time", 0, 0, 0, 1, 1],
    ["offset", *DSCF, -4.5, 0, 85, 0, 0], ["metadata", DSCF[0], DSCF[1], "", 0, 0, 0, 0, 0]]
PATH_CASES = [list(DSCF), [51.5, -0.12, "2026-03-20T12:00:00Z"], [78.2, 15.6, "2026-06-21T23:30:00Z"],
              [78.2, 15.6, "2026-12-21T12:00:00Z"], [0.0, -170.0, "2030-01-01T00:00:00+11:00"],
              [DSCF[0], DSCF[1], "2025-12-25T18:39:25+11:00"]]
BASIS_CASES = [[0, 0], [90, 0], [180, 30], [252.4, -15], [359, 80], [45, -89]]
PROJ_CASES = [[252.4, 16.1, 180, 0, 69.5, 16 / 9], [252.4, 16.1, 252.4, 16.1, 69.5, 16 / 9], [252.4, 16.1, 72.4, 0, 69.5, 16 / 9],
              [200, 10, 180, 5, 69.5, 16 / 9], [181, 2, 180, -10, 30, 2.39], [90, 45, 100, 30, 90, 1.5]]
PARSE_CASES = ["2025-12-25T07:39:25Z", "2025-12-25T18:39:25+11:00", "2025-12-25 07:39:25Z", "2025-12-25T07:39:25",
               "garbage", "", "2026-06-01T00:00:00.250-03:30"]


def drag_cases():
    """(mode, az, el, derived, time_offset_h): derived = Python's NOAA sun for the DSCF2977 capture at that hour
    (None = no capture time). The JS gets the first four."""
    d0 = ps.effective_sun("metadata", *DSCF)["derived"]
    dn = ps.effective_sun("metadata", *DSCF, 9.0)["derived"]
    j0 = {"azimuth_deg": d0["azimuth_deg"], "elevation_deg": d0["elevation_deg"]}
    jn = {"azimuth_deg": dn["azimuth_deg"], "elevation_deg": dn["elevation_deg"]}
    return [["metadata", 100.04, 33.33, j0, 0.0], ["manual", 12.0, -50.0, j0, 0.0], ["offset", 262.36, 13.0, j0, 0.0],
            ["offset", 120.0, 85.0, jn, 9.0], ["metadata", 359.97, 0.04, j0, 0.0], ["offset", 5.0, 10.0, j0, 0.0],
            ["manual", -20.0, 95.0, None, 0.0]]


def run_js(cases: dict) -> dict:
    d = stage()
    (d / "run.mjs").write_text(RUNNER, encoding="utf-8")
    cf = TMP / "cases.json"
    cf.write_text(json.dumps(cases), encoding="utf-8")
    r = subprocess.run([node_exe(), str(d / "run.mjs"), str(cf)], capture_output=True, text=True, encoding="utf-8",
                       cwd=str(TMP), timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: node run failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def py_eff(c):
    try:
        return ps.effective_sun(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8])
    except ValueError as e:
        return {"error": str(e)}


# ------------------------------------------------------------------------------------------------ tests
def test_module_and_mirror():
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    widgets = list(AtlasVFXPhysicalSky.INPUT_TYPES()["required"])
    drags = drag_cases()
    cases = {"sun": SUN_CASES, "dir": DIR_CASES, "eff": EFF_CASES, "path": PATH_CASES, "basis": BASIS_CASES,
             "proj": PROJ_CASES, "drag": [c[:4] for c in drags], "parse": PARSE_CASES, "widgets": widgets,
             "dome": [0.0, 252.4, 180.0]}
    js, stderr = run_js(cases)

    # 1. module, extension, hooks
    want = {"NODE_NAME", "EXTENSION_NAME", "SUN_MODES", "OWNED_WIDGETS", "READ_WIDGETS", "PRAGUE_MIN_ELEVATION",
            "AUTO_QUEUE_PROPERTY", "AUTO_QUEUE_DEFAULT", "parseUtc", "clockOffsetMin", "sunAt", "azelToDir",
            "dirToAzel", "effectiveSun", "sunPath", "cameraBasis", "projectToCamera", "sunRelation", "dragWidgets",
            "offsetAtX", "fmtClock", "wrap180"}
    check(want <= set(js["exports"]), "ES-module import with a stubbed app: the mirror functions are exported",
          sorted(want - set(js["exports"])))
    check(js["exts"] == [{"name": "AtlasVFX.SunGlobe", "hook": "function"}], "one extension registered", js["exts"])
    check(js["otherUntouched"], "another node type is left alone")
    check(js["skyHooks"] == ["function", "function"], "Physical Sky gets onNodeCreated / onExecuted (three.js loads)",
          js["skyHooks"])
    check(js["stored"], "onExecuted keeps ui.atlas_sun_globe[0] before any panel exists")
    check(js["createdNoThrow"] is True, "no DOM: onNodeCreated does not throw", js["createdNoThrow"])
    check(js["hiddenAfterFailure"] == [], "no DOM: a failed panel hides nothing (native widgets keep working)",
          js["hiddenAfterFailure"])
    check("Sun Globe could not start" in stderr, "the failed build is reported on the console")
    check(js["queued"] == 0, "nothing queued by loading or creating")
    check(js["autoDefault"] is False, "auto-queue defaults OFF")

    # 3. widget contract against the Python node
    it = AtlasVFXPhysicalSky.INPUT_TYPES()
    allw = {**it["required"], **it.get("optional", {})}
    check(js["node"] == "AtlasVFXPhysicalSky", "the panel targets AtlasVFXPhysicalSky")
    check(all(n in it["required"] for n in js["owned"]), "every hidden widget exists on the Python node", js["owned"])
    check(all(n in allw for n in js["read"]), "every widget it reads exists", js["read"])
    check(not set(js["owned"]) & set(js["read"]), "it hides only what it writes (heading / fov / tilt stay visible)")
    check(sorted(js["owned"]) == sorted(["time_offset_h", "sun_mode", "sun_azimuth_offset", "sun_elevation_offset",
                                         "sun_azimuth", "sun_elevation"]), "the owned set is the six sun widgets")
    check(js["modes"] == list(it["required"]["sun_mode"][0]), "sun modes match the node's combo", js["modes"])
    check(js["minEl"] == ps.PRAGUE_MIN_ELEVATION, "below-horizon threshold = Prague's -4.2")
    committed = ["latitude", "longitude", "altitude_m", "capture_utc", "time_offset_h", "heading_deg", "fov_x",
                 "tilt_deg", "visibility_km", "ground_albedo", "exposure_ev", "sun_mode", "sun_azimuth_offset",
                 "sun_elevation_offset", "sun_azimuth", "sun_elevation", "width", "latlong_centre", "name", "out_dir"]
    check(list(it["required"]) == committed and list(it["optional"]) == ["shot_camera", "dataset", "ocio_config",
                                                                          "display", "view", "sky_model"],
          "node inputs unchanged from 8723e35 plus sky_model last (headless graphs identical; a new input goes last)")

    # 2. mirror
    worst = 0.0
    for c, (az, el) in zip(SUN_CASES, js["sun"]):
        paz, pel = solar.sun_at(c[0], c[1], solar.parse_utc(c[2]))
        worst = max(worst, ang(az, paz), abs(el - pel))
    check(worst < 1e-6, "sunAt (JS) = solar.sun_at (Python) on 8 suns incl. pre-2000, west of -150, polar",
          f"worst {worst:.2e} deg")
    check(ang(js["sun"][0][0], 252.355) < 0.001 and abs(js["sun"][0][1] - 16.121) < 0.001,
          "DSCF2977 in JS: 252.355 az / 16.121 el", js["sun"][0])
    dmax = max(float(np.abs(np.array(d) - ps.azel_to_dir(a, e)).max()) for (a, e), d in zip(DIR_CASES, js["dir"]))
    imax = max(max(ang(i[0], a) if abs(e) < 89.5 else 0.0, abs(i[1] - e)) for (a, e), i in zip(DIR_CASES, js["inv"]))
    check(dmax < 1e-12 and imax < 1e-9, "azelToDir = azel_to_dir (north -Z, east +X, Y up); dirToAzel inverts it",
          f"{dmax:.1e} / {imax:.1e}")
    bad = []
    for c, j in zip(EFF_CASES, js["eff"]):
        p = py_eff(c)
        if "error" in p or j.get("error"):
            if not ("error" in p and j.get("error")):
                bad.append((c[0], c[3], c[4], c[6], p.get("error"), j.get("error")))
            continue
        if ang(p["azimuth_deg"], j["azimuth_deg"]) > 1e-6 or abs(p["elevation_deg"] - j["elevation_deg"]) > 1e-6 \
                or p["renderable"] != j["renderable"]:
            bad.append((c[0], c[4], p["azimuth_deg"], j["azimuth_deg"], p["elevation_deg"], j["elevation_deg"]))
    check(not bad, "effectiveSun (JS) = effective_sun (Python), errors where the node raises, 10 cases", bad)
    errs = [bool(j.get("error")) for j in js["eff"]]
    check(errs == [False] * 6 + [True, True, True, True], "errors: bare time, bad time, elevation > 90, metadata "
          "without a time", errs)

    bad = []
    for c, j in zip(PATH_CASES, js["path"]):
        p = ps.sun_path(*c)
        if len(p["offset_h"]) != len(j["offset_h"]):
            bad.append((c, "length")); continue
        e1 = max(abs(a - b) for a, b in zip(p["offset_h"], j["offset_h"]))
        e2 = max(ang(a, b) for a, b in zip(p["azimuth_deg"], j["azimuth_deg"]))
        e3 = max(abs(a - b) for a, b in zip(p["elevation_deg"], j["elevation_deg"]))
        if e1 > 1e-6 or e2 > 1e-4 or e3 > 1e-4:
            bad.append((c, e1, e2, e3))
        for k in ("sunrise", "sunset", "noon", "golden_morning", "golden_evening"):
            if (p[k] is None) != (j[k] is None):
                bad.append((c, k, "presence")); continue
            if p[k] and (abs(p[k]["offset_h"] - j[k]["offset_h"]) > 1e-5
                         or abs(p[k]["elevation_deg"] - j[k]["elevation_deg"]) > 1e-3):
                bad.append((c, k, p[k], j[k]))
        if p["polar"] != j["polar"]:
            bad.append((c, "polar", p["polar"], j["polar"]))
    check(not bad, "sunPath (JS) = sun_path (Python): samples, rise / set / noon / golden, polar day and night", bad)
    check([j["polar"] for j in js["path"]] == [None, None, "day", "night", None, None], "polar cases recognised",
          [j["polar"] for j in js["path"]])

    worst, n = 0.0, 0
    for c, rows in zip([0.0, 252.4, 180.0], js["dome"]):
        for p, (u, v) in rows:
            az, el = ps.dir_to_azel(np.array(p))
            if el > 89.9:
                continue                                        # azimuth is undefined at the zenith
            x, y = ps.sun_pixel(az, el, 2048, 1024, c)          # the node's lat-long, pixel centres
            eu, ev = (x + 0.5) / 2048, 1.0 - (y + 0.5) / 1024   # as texture uv (flipY: v = 1 at the top row)
            worst = max(worst, abs(((u - eu + 0.5) % 1.0) - 0.5), abs(v - ev))
            n += 1
    check(n > 300 and worst < 1e-6, "sky dome: three's sphere uvs through domeUvTransform land on the node's lat-long "
          "pixel for every vertex (centre north, 252.4, 180)", f"{n} vertices, worst {worst:.1e}")

    bad = []
    for (h, t), b in zip(BASIS_CASES, js["basis"]):
        R = ps.camera_to_sky(h, t)
        J = np.stack([b["r"], b["u"], -np.array(b["f"])], 1)
        if float(np.abs(R - J).max()) > 1e-12:
            bad.append((h, t, float(np.abs(R - J).max())))
    check(not bad, "cameraBasis = camera_to_sky (right = forward x up, level roll)", bad)
    bad = []
    for c, j, rel in zip(PROJ_CASES, js["proj"], js["rel"]):
        a, e, h, t, f, asp = c
        d_cam, _ = ps.sun_in_camera(a, e, h, t)
        uv, front = ps.project_uv(d_cam[0], f, asp)
        if bool(front[0]) != j["front"] or (front[0] and float(np.abs(uv[0] - [j["u"], j["v"]]).max()) > 1e-9):
            bad.append((c, uv[0].tolist(), bool(front[0]), j))
        node_rel = ((a - h + 180.0) % 360.0) - 180.0                   # node_physical_sky's relative_to_heading_deg
        if abs(node_rel - rel["rel"]) > 1e-9:
            bad.append((c, "rel", node_rel, rel["rel"]))
    check(not bad, "projectToCamera / sunRelation = sun_in_camera + project_uv and the node's relative heading", bad)
    sides = [(r["side"], r["hand"], r["in_frame"]) for r in js["rel"]]
    check(sides[:3] == [("beside", "right", False), ("behind_actor", "centre", True), ("in_front_of_actor", "left", False)],
          "DSCF2977 sun vs camera: heading 180 beside (right), facing it behind the actor and in frame, opposite "
          "in front", sides[:3])

    bad = []
    for (mode, az, el, der, t_off), w in zip(drags, js["drag"]):
        base = {"sun_mode": mode, "sun_azimuth_offset": 0.0, "sun_elevation_offset": 0.0, "sun_azimuth": 0.0,
                "sun_elevation": 0.0, **w}
        if der is None:
            got = ps.effective_sun(base["sun_mode"], DSCF[0], DSCF[1], "", 0.0, base["sun_azimuth_offset"],
                                   base["sun_elevation_offset"], base["sun_azimuth"], base["sun_elevation"])
        else:
            got = ps.effective_sun(base["sun_mode"], *DSCF, t_off, base["sun_azimuth_offset"],
                                   base["sun_elevation_offset"], base["sun_azimuth"], base["sun_elevation"])
        want_el = max(-90.0, min(90.0, el))
        if ang(got["azimuth_deg"], az) > 0.0501 + 1e-9 or abs(got["elevation_deg"] - want_el) > 0.0501 + 1e-9:
            bad.append((mode, az, el, w, got["azimuth_deg"], got["elevation_deg"]))
    check(not bad, "drag mapping round-trips: the widgets it writes give the dropped sun back (to the 0.1 deg step)", bad)
    modes = [w["sun_mode"] for w in js["drag"]]
    check(modes == ["manual", "manual", "offset", "manual", "manual", "offset", "manual"],
          "drag writes manual from metadata / manual; offset stays offset unless the elevation offset leaves -90..90",
          modes)
    check(js["drag"][4].get("sun_azimuth") == 0.0, "azimuth 359.97 wraps to 0.0", js["drag"][4])

    want_ms = []
    for s in PARSE_CASES:
        try:
            want_ms.append(round(solar.parse_utc(s).timestamp() * 1000))
        except ValueError:
            want_ms.append(None)
    check(js["parse"] == want_ms, "parseUtc agrees with solar.parse_utc (offset required)", list(zip(js["parse"], want_ms)))
    check(js["clock"] == [None, 660, None, None, None, None, -210], "clockOffsetMin reads the written offset", js["clock"])
    cap = solar.parse_utc(DSCF[2]).timestamp()
    p0 = ps.sun_path(*DSCF)
    ok = True
    for row in js["snap"]:
        for o, step in zip(row, (5, 1)):
            t = cap + o * 3600.0
            ok &= abs(round(t / 60.0) - t / 60.0) < 1e-3 and round(t / 60.0) % step == 0     # offsets keep 1e-6 h
            ok &= p0["start_offset_h"] - step / 60 <= o <= p0["end_offset_h"] + step / 60
    for o in js["snapClock"]:
        t = cap + o * 3600.0 + 660 * 60
        ok &= round(t / 60.0) % 5 == 0
    check(ok, "the scrubber snaps to whole 5 / 1 minutes of the shown clock and stays on the day", js["snap"])


def test_source_rules():
    imports = re.findall(r"""import\s*(?:\{[^}]*\}\s*from\s*|\(\s*)["']([^"']+)["']""", SRC)
    check(imports == ["../../scripts/app.js", "./lib/atlas-three.bundle.js"],
          "imports: ComfyUI's app and the vendored three.js only (no CDN, no other pack)", imports)
    check("http://" not in SRC and "https://" not in SRC, "no URL in the panel")
    for s in ("widget.hidden = true", "widget.options.hidden = true", "serialize: false", "domWidget.serialize = false",
              "priorRemoved?.apply(this, arguments)", "renderer.dispose()", "if (disposed) return;",
              "canvas.setPointerCapture", "ev.shiftKey", "widget.callback?.(value)", "node.addDOMWidget("):
        check(s in SRC, f"pattern kept from the Gravity Compass: {s}")
    check(SRC.count("app.queuePrompt") == 1 and "function queueIfAuto()" in SRC, "one queuePrompt, in queueIfAuto")
    check(SRC.count("queueIfAuto(") == 2, "queueIfAuto is called from one place", SRC.count("queueIfAuto("))
    end = SRC.split("function endInteraction(", 1)[1].split("\n  }", 1)[0]
    check("if (changed) queueIfAuto();" in end, "it is called when an interaction ends, and only if something changed")
    moves = [SRC.split('canvas.addEventListener("pointermove"', 1)[1].split("const endDrag", 1)[0],
             SRC.split('scrub.addEventListener("pointermove"', 1)[1].split("\n", 1)[0],
             SRC.split("function scrubTo(", 1)[1].split("\n  }", 1)[0]]
    check(all("queue" not in m.lower() and "endInteraction" not in m for m in moves), "nothing queues while dragging")
    hide_at = SRC.index("OWNED_WIDGETS.forEach((n) => hideNativeWidget(owned[n]))")
    check(SRC.index("node.addDOMWidget(") < hide_at and SRC.index("new THREE.WebGLRenderer(") < hide_at,
          "widgets are hidden only after the renderer and the DOM widget exist")
    js_files = sorted(p.relative_to(WEB).as_posix() for p in WEB.rglob("*.js"))
    check(js_files == ["atlas_vfx_sun_globe.js", "lib/atlas-three.bundle.js"], "web/ holds only these scripts "
          "(ComfyUI loads every .js under WEB_DIRECTORY)", js_files)
    init = (ROOT / "__init__.py").read_text(encoding="utf-8")
    check('WEB_DIRECTORY = "./web"' in init and '"WEB_DIRECTORY"' in init, "root __init__ exports WEB_DIRECTORY")


def test_vendored_three():
    reg = json.loads((ROOT / "atlas_components.json").read_text(encoding="utf-8"))
    entry = next((c for c in reg["components"] if c["id"] == "three-js"), None)
    check(entry is not None and "MIT" in entry["note"], "licence register has three-js (MIT)")
    if entry is None:
        return
    files = {f["name"]: f for f in entry["files"]}
    raw = (WEB / "lib" / "atlas-three.bundle.js").read_bytes().replace(b"\r\n", b"\n")
    check(raw.startswith(b"/*! atlas-vfx provenance") and b"e2f8d38" in raw[:2000], "provenance header (Atlas Camera "
          "at e2f8d38) on the bundle")
    body = raw.split(b"*/\n", 1)[1]
    sha = hashlib.sha256(body).hexdigest()
    check(sha == files["web/lib/atlas-three.bundle.js"]["sha256"] and len(body) == files["web/lib/atlas-three.bundle.js"]["bytes_upstream"],
          "bundle = the registered Atlas Camera file (LF, header removed)", sha)
    check(b'var cp="185"' in body[:200] and b"SPDX-License-Identifier: MIT" in body, "three r185, MIT inside the bundle")
    lic = (WEB / "lib" / "three.LICENSE").read_bytes().replace(b"\r\n", b"\n")
    check(hashlib.sha256(lic).hexdigest() == files["web/lib/three.LICENSE"]["sha256"] and b"The MIT License" in lic,
          "three.LICENSE matches the register")


def test_payload():
    from atlas_vfx.layers import node_physical_sky as nps
    sun = ps.effective_sun("metadata", *DSCF)
    dirs, cam = ps.sun_in_camera(sun["azimuth_deg"], sun["elevation_deg"], 180.0, 0.0)
    prev = np.dstack(list(np.meshgrid(np.linspace(0, 1, 2048), np.linspace(0, 1, 1024))) + [np.full((1024, 2048), 0.5)])
    g = nps.globe_ui(sun, cam, 69.5, 16 / 9, 0.0, DSCF[0], DSCF[1], DSCF[2], 0.0, "metadata", 180.0, True, prev)
    keys = {"version", "sun", "inputs", "path", "clock", "camera", "latlong_centre_deg", "sky_rendered",
            "min_elevation_deg", "sky_model_used", "preview", "convention"}
    check(set(g) == keys and g["version"] == 1, "payload keys", sorted(set(g) ^ keys))
    check(json.loads(json.dumps(g)) == g, "payload is plain JSON")
    check(len(json.dumps(g)) < 400_000, "payload stays small", len(json.dumps(g)))
    s = g["sun"]
    check(ang(s["azimuth_deg"], 252.355) < 0.001 and abs(s["elevation_deg"] - 16.121) < 0.001 and s["mode"] == "metadata"
          and s["renderable"] and s["derived"] is not None, "DSCF2977 sun 252.355 / 16.121 in the payload",
          (s["azimuth_deg"], s["elevation_deg"]))
    check(g["inputs"] == {"latitude": DSCF[0], "longitude": DSCF[1], "capture_utc": DSCF[2], "time_offset_h": 0.0,
                          "sun_mode": "metadata", "heading_deg": 180.0}, "inputs echo the widgets the path needs")
    check(g["camera"] == {"source": "typed", "heading_deg": 180.0, "fov_x": 69.5, "aspect": 16 / 9, "tilt_deg": 0.0,
                          "frames": 1}, "typed camera block", g["camera"])
    check(g["clock"] == {"utc_offset_min": None, "label": "UTC"}, "a Z capture time shows UTC")
    check(ps.capture_clock("2025-12-25T18:39:25+11:00") == {"utc_offset_min": 660, "label": "+11:00"}, "+11:00 clock")
    check("north -Z, east +X" in g["convention"] and g["min_elevation_deg"] == -4.2, "convention and -4.2 stated")
    import base64

    import cv2
    pv = g["preview"]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(pv.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR) \
        if pv.startswith("data:image/jpeg;base64,") else None
    check(img is not None and img.shape[:2] == (384, 768), "preview: a 768 x 384 JPEG data URL",
          None if img is None else img.shape)
    if img is not None:
        check(abs(int(img[200, 700, 2]) - round(700 / 767 * 255)) < 12 and abs(int(img[200, 700, 0]) - 128) < 12,
              "preview keeps RGB order (R grows left to right, B flat)", img[200, 700].tolist())

    p = g["path"]
    check(p["step_min"] == 10.0 and len(p["offset_h"]) == 145 and abs(p["end_offset_h"] - p["start_offset_h"] - 24) < 1e-9
          and p["start_offset_h"] <= 0 <= p["end_offset_h"], "path: one solar day in 10 minute steps around the shot")
    rise, sset, noon = p["sunrise"], p["sunset"], p["noon"]
    check(rise and sset and abs(rise["elevation_deg"]) < 1e-3 and abs(sset["elevation_deg"]) < 1e-3,
          "sunrise / sunset sit at 0 deg", (rise and rise["elevation_deg"], sset and sset["elevation_deg"]))
    check(rise["offset_h"] < noon["offset_h"] < sset["offset_h"], "rise < noon < set")
    check(-14.0 < rise["offset_h"] < -12.0 and 1.0 < sset["offset_h"] < 2.0, "Sydney 25 Dec: rise near 05:40, set near "
          "20:05 AEDT (geometric)", (rise["utc"], sset["utc"]))
    t0 = solar.parse_utc(DSCF[2])
    fine = [solar.sun_at(DSCF[0], DSCF[1], solar.shifted(t0, p["start_offset_h"] + i / 60.0))[1] for i in range(24 * 60 + 1)]
    check(noon["elevation_deg"] >= max(p["elevation_deg"]) and abs(noon["elevation_deg"] - max(fine)) < 0.002,
          "solar noon = the highest sun of solar.py (1 minute sampling)", (noon["elevation_deg"], round(max(fine), 4)))
    check(78.5 < noon["elevation_deg"] < 80.0 and ang(noon["azimuth_deg"], 0.0) < 2.0,
          "noon sun north of the zenith, about 79 deg up (lat -34, declination -23.4)", (noon["azimuth_deg"], noon["elevation_deg"]))
    ge = p["golden_evening"]
    check(ge and noon["offset_h"] < ge["offset_h"] < sset["offset_h"] and abs(ge["elevation_deg"] - 4.0) < 1e-3,
          "golden evening at 4 deg, between noon and sunset", ge)
    check(nps.globe_ui(ps.effective_sun("manual", -34.0, 151.0, "", 0, 0, 0, 10, 20), {"camera": "typed", "tilt_deg": 0.0},
                       50, 1.5, 0.0, -34.0, 151.0, "", 0.0, "manual", 0.0, False, prev[:8, :16])["path"] is None,
          "no capture time: no path (the panel draws a manual sun only)")
    shot = nps.globe_ui(sun, {"camera": "shot_camera", "first_frame_tilt_deg": 4.5, "frames": 12}, 61.0, 2.0, 180.0,
                        DSCF[0], DSCF[1], DSCF[2], 0.0, "metadata", 180.0, True, prev[:8, :16])["camera"]
    check(shot == {"source": "shot_camera", "heading_deg": 180.0, "fov_x": 61.0, "aspect": 2.0, "tilt_deg": 4.5, "frames": 12},
          "shot camera block carries its fov, aspect and first-frame tilt", shot)
    R0 = ps.camera_to_sky(123.0, 17.5)
    check(abs(math.degrees(math.asin(-R0[1, 2])) - 17.5) < 1e-9, "first_frame_tilt_deg formula reads the tilt back")

    # a real run, below the horizon (no sky render, so no dataset needed)
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    req = AtlasVFXPhysicalSky.INPUT_TYPES()["required"]
    args = {k: (v[1]["default"] if len(v) > 1 and "default" in v[1] else v[0][0]) for k, v in req.items()}
    args.update(latitude=DSCF[0], longitude=DSCF[1], capture_utc=DSCF[2], time_offset_h=6.0, width=256, out_dir="",
                name="globe_night")
    out = AtlasVFXPhysicalSky().run(**args)
    gn = out["ui"]["atlas_sun_globe"][0]
    check(len(out["result"]) == 5 and out["ui"]["text"], "run() keeps its five outputs and the text summary")
    check(not gn["sky_rendered"] and not gn["sun"]["renderable"] and gn["sun"]["elevation_deg"] < -4.2
          and gn["inputs"]["time_offset_h"] == 6.0, "night run: payload says the sky was not rendered",
          (gn["sky_rendered"], gn["sun"]["elevation_deg"]))
    check(gn["preview"].startswith("data:image/jpeg;base64,") and gn["path"]["sunset"] is not None,
          "night run still sends the preview and the day's path")


def main() -> int:
    print(f"temp {TMP}")
    for t in (test_source_rules, test_vendored_three, test_module_and_mirror, test_payload):
        print(f"--- {t.__name__}", flush=True)
        try:
            t()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(False, f"{t.__name__} raised", repr(e))
    shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
        return 1
    print("\nall sun globe checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
