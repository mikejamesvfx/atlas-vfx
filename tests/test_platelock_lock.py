"""Plate lock end to end on the synthetic shot (two performers, rolling-shutter plate, drifting relit AI take).

- QC before says the shot needs retime, camera re-shoot and people warp
- with exact correspondences (a perfect body solve) both performers land within 0.1 px of the plate edge
- with sparse, noisy, biased correspondences the edge snap still pulls the edge in, and beats no snap
- with no correspondences (silhouette only) the edge still lands under 0.5 px
- background on real AI pixels is within ~1% of the answer and far better than the raw AI frame
- delivery: STMap EXRs re-applied to the AI frames reproduce the baked layers; shot.json and .nk written

Run: <V141 python> tests\\test_platelock_lock.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.platelock.core import deliver, exr, key, pipeline, synth, warp  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


S = synth.make_shot()
KEY = np.stack([key.screen_key(f) for f in S.plate])


def noisy_corr(k, n=60, noise=0.8, bias=(1.2, -0.8)):
    rng = np.random.default_rng(7 + k)

    def fn(i, j):
        p, a = synth.gt_correspondences(S, k, i, j, n=n, seed=100)
        return p, a + rng.normal(0, noise, a.shape) + np.asarray(bias)
    return fn


def run(corr_kind, snap=True):
    people = []
    for k, p in enumerate(S.performers):
        if corr_kind == "exact":
            c = (lambda i, j, k=k: synth.gt_correspondences(S, k, i, j))
        elif corr_kind == "noisy":
            c = noisy_corr(k)
        else:
            c = None
        people.append(pipeline.PersonInput(p.name, S.plate_masks[k], S.ai_masks[k], p.depth, c))
    return pipeline.lock_shot(S.plate, S.ai, S.ai_depth, S.plate_track, S.ai_track, people,
                              1 / S.damage.gauge_scale, pipeline.LockConfig(snap=snap), plate_key=KEY)


def edges(r):
    return {n: v["mean_edge_px"] for n, v in r.report_after["people"].items()}


def main():
    exact = run("exact")
    needs = exact.report_before["needs"]
    check({"retime", "camera_reshoot", "people_warp"} <= set(needs), "QC before diagnoses the damage", ", ".join(needs))
    e = edges(exact)
    ious = {n: v["mean_iou"] for n, v in exact.report_after["people"].items()}
    check(max(e.values()) < 0.1 and min(ious.values()) > 0.99, "exact correspondences lock both performers",
          f"edge {e}, IoU {ious}")
    check(exact.report_after["needs"] == [], "QC after: nothing left to fix")

    noisy_snap, noisy_raw = run("noisy", True), run("noisy", False)
    es, er = edges(noisy_snap), edges(noisy_raw)
    check(all(es[n] < er[n] for n in es) and max(es.values()) < 0.6,
          "edge snap rescues sparse noisy biased correspondences", f"no snap {er} -> snap {es}")

    sil = edges(run("none"))
    check(max(sil.values()) < 0.5, "silhouette-only lock", f"edge {sil}")

    real = exact.bg_valid >= 1.0
    err = np.abs(exact.bg - S.plate_bg_gt).mean(-1)[real].mean()
    naive = np.abs(S.ai_bg[:S.plate.shape[0]] - S.plate_bg_gt).mean()
    check(err < 0.012 and err < naive / 5, "re-shot background vs answer", f"{err:.4f} vs raw AI {naive:.4f}")
    check(real.mean() > 0.9, "most background comes from real AI pixels", f"{real.mean():.3f}")

    with tempfile.TemporaryDirectory() as d:
        shot = deliver.write_lock(exact, d)
        nk = Path(d, "platelock_lock.nk").read_text()
        check("TimeWarp" in nk and "Read_left_stmap" in nk, "Nuke script written")
        js = json.loads(Path(d, "shot.json").read_text())
        check(js["schema"] == "platelock.shot/1" and set(js["people"]) == {"left", "right"}, "shot.json written")
        worst = 0.0
        for name in ("left", "right"):
            for i in (0, 11, 23):
                fn = 1001 + i
                st = exr.read_exr(f"{d}/{name}/{name}_stmap.{fn:04d}.exr")
                src = shot["people"][name]["source_frames"][i] - 1001
                m = warp.from_stmap(np.stack([st["R"], st["G"]], -1), S.ai.shape[2], S.ai.shape[1])
                re = warp.remap(S.ai[src], *m, border="replicate")
                bk = exr.read_exr(f"{d}/{name}/{name}.{fn:04d}.exr")
                a = bk["A"] > 0.5
                worst = max(worst, float(np.abs(re[..., 1] - bk["G"])[a].max()))
        check(worst < 1e-4, "STMap re-applied to the AI frame reproduces the baked layer", f"max {worst:.1e}")


if __name__ == "__main__":
    main()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
