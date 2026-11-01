"""The Light room viewer is correct (Light Mix plan phase 4): app/lightmix.js in a real browser (headless Chromium
through Playwright, WebGL2) against the Python reference.

Three takes (HDR ramps up to 8x display white, negatives, alpha) plus under / over layers go through
  - the browser: LightMixer with OCIO's own GLSL ES 3.0 view shader (comp/cache.view_shader), read back as 8 bit
  - Python: comp/mix.mix_frame -> frames.blend under / over -> exposure -> OCIO CPU view -> 8 bit
They must agree within 1 code value on nearly every pixel (GPU float maths vs OCIO's CPU path).

Playwright runs in whichever Python on this machine has it (the V141 venv does not); the test skips (passes) when
none does. Run: <V141 python> tests/test_light_view_browser.py
"""
from __future__ import annotations

import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.comp import cache as cc  # noqa: E402
from atlas_vfx.comp import frames as fr  # noqa: E402
from atlas_vfx.comp import mix as mx  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

FAILS: list[str] = []
W, H = 256, 96

PAGE = """<!doctype html><meta charset="utf-8"><canvas id="c"></canvas>
<script type="module">
import { LightMixer, parseFrame } from "./lightmix.js";
const get = async (n) => (await fetch(n)).arrayBuffer();
try {
  const shader = await (await fetch("shader.json")).json();
  const cfg = await (await fetch("case.json")).json();
  const m = new LightMixer(document.getElementById("c"));
  m.setView(shader);
  const takes = [];
  for (const v of cfg.versions) takes.push(parseFrame(await get(`take_${v}.f16`)));
  m.upload({ under: parseFrame(await get("under.f16")), over: parseFrame(await get("over.f16")), takes });
  m.set({ versions: cfg.versions, mix: cfg.mix, blend: "over", opacity: cfg.opacity, exposure: cfg.exposure });
  m.draw();
  const gl = m.gl, w = m.canvas.width, h = m.canvas.height;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  window.RESULT = { w, h, px: Array.from(px), renderer: gl.getParameter(gl.RENDERER) };
} catch (e) { window.RESULT = { error: String(e && e.stack || e) }; }
</script>"""

DRIVER = r"""
import json, sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto(url)
    pg.wait_for_function("window.RESULT !== undefined", timeout=60000)
    r = pg.evaluate("window.RESULT")
    r["logs"] = logs
    b.close()
open(out, "w").write(json.dumps(r))
"""


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def playwright_python() -> str | None:
    cands = [sys.executable] + [shutil.which(n) for n in ("python", "python3", "py")]
    for c in dict.fromkeys(x for x in cands if x):
        try:
            r = subprocess.run([c, "-c", "import playwright.sync_api"], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return c
    return None


def scene():
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    u, v = xx / (W - 1), yy / (H - 1)
    takes = {}
    for k, ver in enumerate(("v001", "v002", "v003")):
        t = np.zeros((H, W, 4), np.float32)
        t[..., 0] = (u * 8.0) ** (1 + 0.3 * k)                       # to 8x display white
        t[..., 1] = (v * 3.0) * (0.5 + 0.25 * k)
        t[..., 2] = 0.05 + 0.4 * k * u * v - 0.02 * (k == 2)         # a little negative blue on one take
        t[..., 3] = np.clip(0.4 + u, 0, 1)
        t[..., :3] *= t[..., 3:4]                                     # premultiplied
        takes[ver] = t
    under = np.zeros((H, W, 4), np.float32)
    under[..., :3] = [0.18, 0.12, 0.08]
    under[..., 3] = 1.0
    over = np.zeros((H, W, 4), np.float32)
    over[20:40, 30:120] = [0.5, 0.05, 0.02, 0.6]
    # everything through float16, as the cache sends it
    q = lambda a: a.astype("<f2").astype(np.float32)                  # noqa: E731
    return {k: q(t) for k, t in takes.items()}, q(under), q(over)


def reference(takes, under, over, mix, opacity, exposure):
    m = mx.mix_frame(takes, mix)
    c = under.copy()
    if m is not None:
        c = fr.blend(c, m, "over", opacity)
    c = fr.blend(c, over, "over", 1.0)
    rgb = np.ascontiguousarray(c[..., :3] * np.float32(2.0 ** exposure))
    fwd, _ = li.view_processors("")
    return np.round(li.to_view(rgb, fwd) * 255.0).astype(np.int32)


def main() -> int:
    py = playwright_python()
    if py is None:
        print("skip: no Python with Playwright on this machine")
        return 0
    takes, under, over = scene()
    t = mx.tint_unit([1.0, 0.85, 0.6])
    mix = {"mode": "blend", "takes": {"v001": {"weight": 0.5, "stops": 0.0, "tint": [1, 1, 1], "mute": False, "solo": False},
                                      "v002": {"weight": 0.3, "stops": -1.0, "tint": t, "mute": False, "solo": False},
                                      "v003": {"weight": 0.2, "stops": 1.5, "tint": [1, 1, 1], "mute": True, "solo": False}}}
    case = {"versions": ["v001", "v002", "v003"], "mix": mix, "opacity": 0.9, "exposure": 0.5}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy(ROOT / "app" / "lightmix.js", d / "lightmix.js")
        (d / "index.html").write_text(PAGE, encoding="utf-8")
        (d / "shader.json").write_text(json.dumps(cc.view_shader("", fr.DEFAULT_VIEW)), encoding="utf-8")
        (d / "case.json").write_text(json.dumps(case), encoding="utf-8")
        for ver, img in takes.items():
            (d / f"take_{ver}.f16").write_bytes(cc.encode(img, {"version": ver}))
        (d / "under.f16").write_bytes(cc.encode(under, {"side": "under"}))
        (d / "over.f16").write_bytes(cc.encode(over, {"side": "over"}))
        (d / "driver.py").write_text(DRIVER, encoding="utf-8")
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass
        handler = functools.partial(Quiet, directory=str(d))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = subprocess.run([py, str(d / "driver.py"), f"http://127.0.0.1:{srv.server_port}/index.html",
                                str(d / "out.json")], capture_output=True, text=True, timeout=180)
        finally:
            srv.shutdown()
        if r.returncode != 0:
            check(False, "browser ran the page", (r.stderr or r.stdout)[-400:])
            return 1
        res = json.loads((d / "out.json").read_text(encoding="utf-8"))
    if "error" in res:
        check(False, "lightmix.js compiled OCIO's shader and drew", res["error"][:500])
        return 1
    got = np.array(res["px"], np.int32).reshape(res["h"], res["w"], 4)[::-1, :, :3]   # GL rows are bottom up
    want = reference(takes, under, over, mix, case["opacity"], case["exposure"])
    diff = np.abs(got - want)
    within = float((diff.max(-1) <= 1).mean())
    check(got.shape == want.shape, "canvas is the frame's size", f"{got.shape}")
    check(within >= 0.999 and int(diff.max()) <= 3,
          "GPU mix + OCIO GPU view = mix_frame + OCIO CPU view (8 bit)",
          f"{within * 100:.3f} % within 1 code, max {int(diff.max())}, {res.get('renderer')}")
    print("PASS" if not FAILS else f"FAILED {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
