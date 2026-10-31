"""Atlas Comp's browser app (app/): the files exist, load as ES modules from this server only, never build HTML from
server strings, and every API path the app calls is a real route.

Run: <V141 python> tests\\test_comp_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP = ROOT / "app"
FILES = ("index.html", "style.css", "api.js", "main.js")
FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def route_regexes() -> list[tuple[str, re.Pattern]]:
    from atlas_vfx.comp.server import build_routes

    out = []
    for r in build_routes():
        rx = re.sub(r"\\\{[^}]+\\\}", "[^/?]+", re.escape(r.path))
        out.append((r.method, re.compile("^" + rx + "$")))
    return out


def main():
    missing = [f for f in FILES if not (APP / f).is_file()]
    check(not missing, "app: the four files exist", str(missing))
    if missing:
        print(f"\n{len(FAILS)} failed")
        return 1
    text = {f: (APP / f).read_text(encoding="utf-8") for f in FILES}
    html = text["index.html"]
    check(re.search(r'<script[^>]+type="module"[^>]+src="/atlas/app/main\.js"', html) is not None,
          "app: index.html loads main.js as a module from /atlas/app")
    hosts = [m for f in FILES for m in re.findall(r"https?://[^\s'\"`)]+", text[f])
             if m != "http://www.w3.org/2000/svg"]          # the SVG namespace is a name, never fetched
    check(not hosts, "app: no other hosts (no CDN, nothing leaves the machine)", str(hosts[:3]))
    js = text["api.js"] + text["main.js"]
    check("eval(" not in js and "new Function" not in js, "app: no eval")
    bad_html = re.findall(r"innerHTML\s*[+]?=\s*(?!['\"]\s*['\"])", js)
    check(not bad_html, "app: no innerHTML from data (textContent and createElement only)", str(bad_html[:2]))
    paths = re.findall(r"[`'\"](/atlas/api/[^`'\"?]*)", text["api.js"])
    rx = route_regexes()
    unknown = []
    for p in paths:
        concrete = re.sub(r"\$\{[^}]+\}", "X", p)
        if not any(r.match(concrete) for _, r in rx):
            unknown.append(p)
    check(paths and not unknown, "app: every API path in api.js is a real route", f"{len(paths)} paths, unknown {unknown}")
    check("n: ++S.nonce" in text["main.js"] and "frameURL(" in text["main.js"],
          "app: every frame request has its own URL (a reused URL never reloads, and the lock would never clear)")
    check('preview_method: "taesd"' in text["api.js"],
          "app: asks ComfyUI for tiny-VAE live previews with each prompt (its default is none)")
    main = text["main.js"]
    check('"execution_interrupted"' in main and re.search(r'type === "execution_success"\)\s*\{', main) is not None
          and 'd.node === null' not in main,
          "app: a cancelled job is reported as cancelled, and only a success switches the layer's version")
    for form in ("shot-form", "layer-form"):
        block = html.split(f'id="{form}"', 1)[1].split("</form>", 1)[0]
        first = re.search(r"<button[^>]*>", block)
        check(first is not None and 'value="ok"' in first.group(0),
              f"app: Enter in {form} submits it (its first button is OK, not Cancel)", first.group(0) if first else "")
    check("plate_proxy_error" in main and "retryProxy" in text["api.js"],
          "app: a plate that failed to decode shows why, with a retry")
    css = text["style.css"]
    check(all(t in css for t in ("#171411", "#1a1714", "#eaa03a", "#f5f2ed", "#a09890")) and "2px" in css,
          "design: the Atlas Camera tokens (warm ground, one amber, bone, ash, 2px corners)")
    check("COMPASS" in main and "compassField" in main and "segmentedField" in main,
          "design: the sun picker is a compass grid and short choices are segmented buttons")
    check('id="rooms"' in html and "01 CAMERA" in html and "02 SCENE" in html and "03 COMP" in html,
          "design: the rooms bar (Camera and Scene marked as not built yet)")
    check('id="job-thumb"' in html and 'id="play"' in html,
          "design: the live preview sits in the job box, and the timeline has a play button")
    check("Light mix" in main and "lightMix" in main and "{ mix:" in main and "Mix off" in main,
          "design: a recipe layer with renders gets Light mix sliders, one per version, saved as mix")
    check('id="presets-btn"' in html and 'id="presets-dialog"' in html and "renderPresets" in main
          and "presets:" in text["api.js"], "design: a Presets gallery opens from the top bar")
    check("/prompt" in text["api.js"] and "/ws?clientId=" in text["api.js"],
          "app: queues on ComfyUI's /prompt and listens on its websocket")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
