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
    hosts = [m for f in FILES for m in re.findall(r"https?://[^\s'\"`)]+", text[f])]
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
    check("/prompt" in text["api.js"] and "/ws?clientId=" in text["api.js"],
          "app: queues on ComfyUI's /prompt and listens on its websocket")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
