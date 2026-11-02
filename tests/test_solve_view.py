"""Solve View (catalogue 69): node_solve_view.payload and web/atlas_vfx_solve_view.js.

  1. Payload from a synthetic COLMAP solve (text model written here with a RADIAL lens, observations = the true
     projections): every frame, every point, errors near zero through the camera's lens, per-frame errors.
  2. A camera whose extensions point the proxies at a folder outside the work roots is refused (graph data).
  3. The JS imports under node with stubbed ComfyUI app / api, registers one extension for the node, and its
     seed helpers append labelled seeds to the widget JSON.
Needs node (PATH or the default Windows install). Run: <V141 python> tests/test_solve_view.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="atlas_solve_view_"))
os.environ["ATLAS_VFX_OUT_ROOTS"] = str(TMP)

from atlas_vfx.camera_side import camera_io as cio  # noqa: E402
from atlas_vfx.camera_side import node_solve_view as nsv  # noqa: E402

W, H, F, K1, K2 = 1024, 540, 700.0, -0.06, 0.01
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def write_model(d: Path, n_frames=8, n_pts=300):
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(-4, 4, n_pts), rng.uniform(-2, 2, n_pts), rng.uniform(6, 14, n_pts)])
    d.mkdir(parents=True)
    (d / "cameras.txt").write_text(f"1 RADIAL {W} {H} {F} {W / 2} {H / 2} {K1} {K2}\n")
    imgs, obs = [], {j: [] for j in range(n_pts)}
    for i in range(n_frames):
        R = Rotation.from_euler("y", 2 * i, degrees=True).as_matrix()         # cam_from_world
        C = np.array([0.3 * i, 0, 0])
        t = -R @ C
        Xc = X @ R.T + t
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        r2 = x * x + y * y
        s = 1 + K1 * r2 + K2 * r2 * r2
        u, v = F * x * s + W / 2, F * y * s + H / 2
        q = Rotation.from_matrix(R).as_quat()
        line, k = [], 0
        for j in range(n_pts):
            if 0 <= u[j] < W and 0 <= v[j] < H:
                line.append(f"{u[j]:.6f} {v[j]:.6f} {j + 1}")
                obs[j].append(f"{i + 1} {k}")
                k += 1
        imgs.append(f"{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 plate_{i:04d}.png\n" + " ".join(line))
    (d / "images.txt").write_text("\n".join(imgs) + "\n")
    (d / "points3D.txt").write_text("\n".join(f"{j + 1} {X[j, 0]} {X[j, 1]} {X[j, 2]} 128 128 128 0 " + " ".join(o)
                                              for j, o in obs.items() if len(o) >= 2) + "\n")
    return n_frames


def node_exe() -> str:
    exe = shutil.which("node")
    if not exe and Path(r"C:\Program Files\nodejs\node.exe").is_file():
        exe = r"C:\Program Files\nodejs\node.exe"
    if not exe:
        raise SystemExit("FAIL: node not found; the panel cannot be tested")
    return exe


def js_checks():
    d = TMP / "esm"
    (d / "extensions" / "atlas-vfx" / "lib").mkdir(parents=True)
    (d / "scripts").mkdir()
    shutil.copy2(ROOT / "web" / "atlas_vfx_solve_view.js", d / "extensions" / "atlas-vfx")
    shutil.copy2(ROOT / "web" / "lib" / "atlas-three.bundle.js", d / "extensions" / "atlas-vfx" / "lib")
    (d / "scripts" / "app.js").write_text("export const app = { registerExtension(e) { (globalThis.__exts ||= [])"
                                          ".push(e); } };\n")
    (d / "scripts" / "api.js").write_text("export const api = { apiURL: (p) => '/api' + p };\n")
    (d / "package.json").write_text('{"type": "module"}')
    (d / "run.mjs").write_text("""
const G = await import("./extensions/atlas-vfx/atlas_vfx_solve_view.js");
const exts = (globalThis.__exts || []).map((e) => e.name);
let s = G.addSeed("[]", 12, 100.04, 50.06);
s = G.addSeed(s, 13, 1, 2);
const bad = G.addSeed("not json", 3, 4, 5);
console.log(JSON.stringify({ exts, node: G.NODE_NAME, key: G.UI_KEY, s: JSON.parse(s), bad: JSON.parse(bad),
  colors: [0.2, 1.0, 5.0, -1].map(G.errColor) }));
""")
    r = subprocess.run([node_exe(), "run.mjs"], cwd=d, capture_output=True, text=True)
    if r.returncode:
        check(False, f"panel imports under node: {r.stderr[-400:]}")
        return
    o = json.loads(r.stdout)
    check(o["exts"] == ["AtlasVFX.SolveView"] and o["node"] == "AtlasVFXSolveView" and o["key"] == nsv.UI_KEY,
          f"panel registers one extension for the node, same ui key as Python ({o['exts']})")
    check([x["label"] for x in o["s"]] == ["s1", "s2"] and o["s"][0] == {"label": "s1", "frame": 12, "xy": [100, 50.1]},
          f"seeds append with labels and plate xy ({o['s']})")
    check(len(o["bad"]) == 1, "a broken seeds widget starts a fresh list")
    check(o["colors"][0][1] > 0.8 and o["colors"][2][0] > 0.9, "error colours green / red")


def main() -> int:
    try:
        n = write_model(TMP / "model")
        plate = {"path": "synthetic", "width": W, "height": H, "colorspace": "ACEScg", "range": "full"}
        cam = cio.from_colmap(TMP / "model", plate=plate, sensor={"width_mm": 23.5, "height_mm": 23.5 * H / W},
                              frame_offset=1001)
        lens = cam.data["lens"]
        check(lens["distortion_model"] == "brown" and lens["distortion_params"] == [K1, K2, 0.0, 0.0, 0.0],
              f"from_colmap maps RADIAL k1 k2 to brown ({lens})")
        p = nsv.payload(cam, max_points=10000, images=0)
        check(len(p["frames"]) == n and len(p["points"]) == 3 * len(p["err"]) and len(p["err"]) > 200,
              f"payload: {len(p['frames'])} frames, {len(p['err'])} points")
        check(max(p["err"]) < 0.01, f"points reproject through the lens at {max(p['err']):.4f} px worst")
        check(len(p["frame_err"]) == n and p["frame_offset"] == 1001, "error per frame and the plate frame offset")
        json.dumps(p, allow_nan=False)
        check(True, "payload is plain JSON")
        small = nsv.payload(cam, max_points=50, images=0)
        check(len(small["err"]) == 50, "max_points subsamples")
        bad = cio.sc.ShotCamera({**cam.data, "extensions": {**cam.data.get("extensions", {}),
                                                            "atlas_vfx_solve_plate": {"proxy_dir": str(ROOT)}}})
        try:
            nsv._proxy_images(bad, None, 3)
            check(False, "proxies outside the work roots refused")
        except ValueError:
            check(True, "proxies outside the work roots refused")
        js_checks()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
