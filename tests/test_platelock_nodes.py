"""Plate lock nodes: they register, and run as a graph would on torch tensors.

- the root registry loads all eight Plate Lock nodes
- shot-camera@0.1 -> platelock track: projecting world points matches ShotCamera.project (corner vs centre
  pixel origin accounted for)
- Read EXR -> Screen Key -> Camera -> Person x2 -> QC -> Lock -> Layer -> Write, on a small synthetic shot,
  with the AI take at half resolution (the lock resizes it and its camera)

Run: <V141 python> tests\\test_platelock_nodes.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ATLAS_VFX_OUT_ROOTS", tempfile.gettempdir())     # Write node may write under temp

import torch  # noqa: E402

from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.platelock import adapters  # noqa: E402
from atlas_vfx.platelock.core import exr, synth  # noqa: E402
from atlas_vfx.registry import load_nodes  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


NODES, NAMES, DISABLED = load_nodes()
WANT = ["AtlasVFXPlateLockReadEXR", "AtlasVFXPlateLockScreenKey", "AtlasVFXPlateLockCamera",
        "AtlasVFXPlateLockPerson", "AtlasVFXPlateLockQC", "AtlasVFXPlateLock", "AtlasVFXPlateLockLayer",
        "AtlasVFXPlateLockWrite"]


def test_registry():
    missing = [n for n in WANT if n not in NODES]
    check(not missing, "all Plate Lock nodes registered", f"missing {missing}, disabled {DISABLED}")
    check(all(NODES[n].CATEGORY.startswith("Atlas VFX/Layers") for n in WANT if n in NODES), "category")


def test_shot_camera_adapter():
    rng = np.random.default_rng(2)
    mats, foc = [], []
    for f in range(4):
        R = sc.euler_zxy_to_matrix(5 * f - 3, 20 * f - 30, 2 * f)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = [0.2 * f, 1.6, -0.3 * f]
        mats.append(m.tolist())
        foc.append(16.47)
    cam = sc.build(name="pl_adapter", frame_start=1001, fps=24.0,
                   plate={"path": "synthetic", "width": 4096, "height": 2160, "colorspace": "ACEScg", "range": "full"},
                   sensor={"width_mm": 23.5, "height_mm": 12.39}, cam_to_world=mats, focal_mm=foc,
                   source={"kind": "synthetic", "solver": "tests/test_platelock_nodes.py"})
    tr = adapters.track_from_shot_camera(cam)
    worst = 0.0
    for n, f in enumerate(cam.frames):
        c2w = np.asarray(cam.cam_to_world(f))
        pts_cam = np.stack([rng.uniform(-3, 3, 50), rng.uniform(-2, 2, 50), -rng.uniform(2, 30, 50)], 1)
        Pw = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
        uv_ref, _ = cam.project(f, Pw)
        u, v, z = tr.project(Pw, n)
        worst = max(worst, float(np.abs(np.stack([u, v], 1) - (uv_ref - 0.5)).max()))
        check(bool((z > 0).all()), f"points in front stay in front, frame {f}") if n == 0 else None
    check(worst < 1e-6, "shot-camera adapter projects like ShotCamera.project", f"max {worst:.1e} px")


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))


def test_graph():
    s = synth.make_shot(n_plate=10, n_ai=14)
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(s.plate):
            exr.write_rgba(f"{d}/plate.{1001 + i:04d}.exr", f, attrs={"framesPerSecond": ("rational", (24, 1))})
        frames, meta, first = NODES["AtlasVFXPlateLockReadEXR"]().run(f"{d}/plate.####.exr", 0, -1, 1)
        check(tuple(frames.shape) == s.plate.shape and first == 1001 and json.loads(meta)["first"]["fps"] == 24,
              "Read EXR node", f"{tuple(frames.shape)}, first {first}")
        (matte,) = NODES["AtlasVFXPlateLockScreenKey"]().run(frames, "blue", 0.04, 0.2)
        check(tuple(matte.shape) == s.plate.shape[:3], "Screen Key node")

        # plate camera from a platelock.track/1 file, AI take at half resolution with its camera scaled to match
        s.plate_track.save(f"{d}/plate_track.json")
        s.ai_track.save(f"{d}/ai_track.json")
        cam_node = NODES["AtlasVFXPlateLockCamera"]()
        ro_ms = s.readout / 24.0 * 1000
        plate_cam, _ = cam_node.run(frames, "platelock_json", f"{d}/plate_track.json", 35, 23.1, ro_ms, 24.0, "centre")
        H, W = s.plate.shape[1:3]
        ai_half = np.stack([__import__("cv2").resize(f, (W // 2, H // 2), interpolation=__import__("cv2").INTER_AREA)
                            for f in s.ai])
        ai_cam, _ = cam_node.run(T(ai_half), "platelock_json", f"{d}/ai_track.json", 35, 23.1, 0.0, 24.0, "centre")
        check(abs(ai_cam.intr.fx - s.ai_track.intr.fx / 2) < 1e-9 and abs(plate_cam.readout - s.readout) < 1e-9,
              "Camera node: scaled to the frames it belongs to, readout from ms")

        person = NODES["AtlasVFXPlateLockPerson"]()
        (ppl,) = person.run("left", T(s.plate_masks[0]), T(s.ai_masks[0][:, ::2, ::2]), 3.0)
        (ppl,) = person.run("right", T(s.plate_masks[1]), T(s.ai_masks[1][:, ::2, ::2]), 3.6, people=ppl)
        check([p.name for p in ppl] == ["left", "right"], "Person nodes chain")

        out = NODES["AtlasVFXPlateLockQC"]().run(ppl)
        check("needs:" in out["result"][0], "QC node", out["result"][0].splitlines()[-1] if out["result"][0] else "")

        lock = NODES["AtlasVFXPlateLock"]()
        comp, bg, bgv, res, rep = lock.run(frames, T(ai_half), plate_cam, ai_cam, ppl, 2.0, "metric", 10.0, 180.0, 1,
                                           2, True, 8.0, ai_depth=T(s.ai_depth[:, ::2, ::2]), plate_matte=matte)
        after = res.report_after["people"]
        e = {n: round(v["mean_edge_px"], 2) for n, v in after.items()}
        check(tuple(comp.shape) == s.plate.shape and max(e.values()) < 1.0,
              "Lock node at half-res AI (silhouette only)", f"edge {e}")
        rgb, alpha, names = NODES["AtlasVFXPlateLockLayer"]().run(res, "right")
        check(tuple(alpha.shape) == s.plate.shape[:3] and names == "left, right", "Layer node")
        od = Path(d) / "out"
        NODES["AtlasVFXPlateLockWrite"]().run(res, str(od), 1001, 1001, f"{d}/ai.####.exr", True)
        js = json.loads((od / "shot.json").read_text())
        check((od / "platelock_lock.nk").exists() and "sidecar" in js and js["sidecar"]["core_hash"],
              "Write node: maps, shot.json with sidecar, Nuke script")
        try:
            NODES["AtlasVFXPlateLockWrite"]().run(res, str(ROOT / "out_in_repo"), 1001, 1001, "", False)
            check(False, "Write refuses an out_dir inside the repo")
        except ValueError:
            check(True, "Write refuses an out_dir inside the repo")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
