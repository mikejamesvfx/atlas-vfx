"""Visual review of the Sun Globe (web/atlas_vfx_sun_globe.js) outside ComfyUI: a headless browser opens
tools/sun_globe_harness.html (a fake Physical Sky node) and screenshots the panel for the DSCF2977 check shot.

Two steps, two pythons:
  <V141 python> tools\\sun_globe_review.py payload
      runs the real node (DSCF2977 metadata with the Prague render, and the same shot 6 h later, under the horizon)
      and writes the ui payloads + the node's widget list to ~/atlas-vfx-work/physical_sky/globe_review/.
  <a python with playwright> tools\\sun_globe_review.py shots
      serves the harness, the repo's web/ and a stub ComfyUI app through Playwright routes (no server, no files
      copied), then shoots: dscf2977_metadata, dscf2977_manual_drag (a real mouse drag of the sun), night,
      dscf2977_view_camera, plus review.json (widget values, panel text, console messages).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATLAS_VFX_WORK") or Path.home() / "atlas-vfx-work") / "physical_sky" / "globe_review"
DSCF = {"latitude": -34.1906, "longitude": 151.0425, "capture_utc": "2025-12-25T07:39:25Z", "heading_deg": 180.0,
        "fov_x": 69.5, "tilt_deg": 0.0}


def dataset() -> str:
    p = os.environ.get("ATLAS_PRAGUE_DATASET", "")
    if not p and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                p = winreg.QueryValueEx(k, "ATLAS_PRAGUE_DATASET")[0]
        except OSError:
            p = ""
    return p


def payload() -> int:
    sys.path.insert(0, str(ROOT))
    from atlas_vfx.layers.node_physical_sky import AtlasVFXPhysicalSky
    it = AtlasVFXPhysicalSky.INPUT_TYPES()
    spec = []
    for group in ("required", "optional"):
        for name, v in it[group].items():
            opts = v[1] if len(v) > 1 else {}
            if isinstance(v[0], list):
                spec.append([name, "combo", opts.get("default", v[0][0]), {"values": v[0]}])
            elif v[0] in ("FLOAT", "INT", "STRING"):
                spec.append([name, v[0].lower(), opts.get("default", "" if v[0] == "STRING" else 0),
                             {k: opts[k] for k in ("min", "max", "step") if k in opts}])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "widgets.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    base = {name: value for name, _t, value, _o in spec if name in it["required"]}
    base.update(DSCF, width=1024, name="globe_review", out_dir="", dataset=dataset())
    for tag, over in (("dscf2977", {}), ("night", {"time_offset_h": 6.0})):
        r = AtlasVFXPhysicalSky().run(**{**base, **over})
        g = r["ui"]["atlas_sun_globe"][0]
        (OUT / f"payload_{tag}.json").write_text(json.dumps(g), encoding="utf-8")
        print(tag, "sun", round(g["sun"]["azimuth_deg"], 3), round(g["sun"]["elevation_deg"], 3), "sky_rendered",
              g["sky_rendered"])
    print("wrote", OUT)
    return 0


STUB_APP = ("export const app = { registerExtension(e) { (window.__exts ||= []).push(e); },\n"
            "  queuePrompt() { window.__queued = (window.__queued || 0) + 1; } };\n")


def shots() -> int:
    from playwright.sync_api import sync_playwright

    host = "http://atlas.test"
    types = {".html": "text/html", ".js": "text/javascript", ".json": "application/json"}

    def route(r):
        path = r.request.url[len(host):].split("?", 1)[0]
        if path == "/harness.html":
            f = ROOT / "tools" / "sun_globe_harness.html"
        elif path == "/scripts/app.js":
            return r.fulfill(status=200, content_type="text/javascript", body=STUB_APP)
        elif path == "/harness/widgets.json":
            f = OUT / "widgets.json"
        elif path.startswith("/extensions/atlas-vfx/"):
            f = (ROOT / "web" / path[len("/extensions/atlas-vfx/"):]).resolve()
            if ROOT / "web" not in f.parents:
                return r.fulfill(status=404, body="")
        else:
            return r.fulfill(status=404, body="")
        if not f.is_file():
            return r.fulfill(status=404, body="")
        r.fulfill(status=200, content_type=types.get(f.suffix, "application/octet-stream"), body=f.read_bytes())

    meta = json.loads((OUT / "payload_dscf2977.json").read_text(encoding="utf-8"))
    night = json.loads((OUT / "payload_night.json").read_text(encoding="utf-8"))
    report: dict = {"console": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 520, "height": 1200}, device_scale_factor=2)
        page.on("console", lambda m: report["console"].append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: report["console"].append(f"pageerror: {e}"))
        page.route(f"{host}/**", route)
        page.goto(f"{host}/harness.html")
        page.wait_for_function("window.harnessReady === true", timeout=30000)
        report["hidden"] = page.evaluate("harness.hidden()")
        panel = page.locator("#dom")

        def settle(ms=900):
            page.wait_for_timeout(ms)

        def snap(tag):
            panel.screenshot(path=str(OUT / f"{tag}.png"))
            report[tag] = {"values": page.evaluate("harness.values()"), "text": page.evaluate("harness.text()"),
                           "queued": page.evaluate("harness.queued()")}
            print(tag, report[tag]["text"].replace("\n", " | "))

        # 1. DSCF2977 as the node ran it: metadata sun, the rendered sky on the dome
        page.evaluate("v => harness.set(v)", {**DSCF, "sun_mode": "metadata", "time_offset_h": 0.0})
        page.evaluate("p => harness.executed(p)", meta)
        settle(1500)
        snap("dscf2977_metadata")
        page.locator("#node").screenshot(path=str(OUT / "dscf2977_metadata_node.png"))

        # 2. drag the sun with the mouse (writes manual; nothing queues, auto-queue is off)
        sx, sy = page.evaluate("harness.sun()")
        page.mouse.move(sx, sy)
        page.mouse.down()
        for i in range(1, 21):
            page.mouse.move(sx + 7.0 * i, sy - 2.5 * i)
        page.mouse.up()
        settle()
        snap("dscf2977_manual_drag")

        # 3. the scrubber: one click near the middle of the day (writes time_offset_h, back to metadata)
        box = page.locator("#dom canvas").nth(1).bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.55, box["y"] + 15)
        settle(300)
        report["scrub_click"] = page.evaluate("harness.values()")

        # 4. night: the same shot 6 h later, as the node ran it
        page.evaluate("v => harness.set(v)", {**DSCF, "sun_mode": "metadata", "time_offset_h": 6.0})
        page.evaluate("p => harness.executed(p)", night)
        settle()
        snap("night")

        # 5. DSCF2977 again, seen from behind the camera
        page.evaluate("v => harness.set(v)", {**DSCF, "sun_mode": "metadata", "time_offset_h": 0.0})
        page.evaluate("p => harness.executed(p)", meta)
        page.get_by_text("VIEW NORTH").click()
        settle(1500)
        snap("dscf2977_view_camera")

        # 6. dome check: back to north, then orbit with the mouse until the viewer stands opposite the sun (compass
        #    72, yaw 108 deg = 236 px of drag), so the sun sits on the far wall of the dome: the preview's own sun
        #    (Physical Sky rings it) must sit right behind the marker
        page.get_by_text("VIEW CAMERA").click()
        settle(300)
        cbox = page.locator("#dom canvas").nth(0).bounding_box()
        x0, y0 = cbox["x"] + cbox["width"] - 20, cbox["y"] + 20
        page.mouse.move(x0, y0)
        page.mouse.down()
        for i in range(1, 25):
            page.mouse.move(x0 - 236 * i / 24, y0 + 30 * i / 24)
        page.mouse.up()
        settle(1200)
        snap("dscf2977_dome_check")
        browser.close()
    (OUT / "review.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("console:", *report["console"], sep="\n  ")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in ("payload", "shots"):
        raise SystemExit(__doc__)
    sys.exit(payload() if step == "payload" else shots())
