"""Plate Lock cross-view warp (frame for frame, per ID).

- known warp: a plate performer warped by a known non-rigid field (scale, shift, sine bends) and relit,
  over invented background -> the solved map matches the field (median, p95, and in the 0-5px edge band)
- full synthetic take: relit, resized, offset performers -> ground-truth texture points land within a px
- solving at half resolution and scaling the maps up keeps the accuracy
- the EXR job (V141 / OpenImageIO): ACEScg plate + AI + ID mattes in, warped layers tagged with their
  colourspace, float STMaps that reproduce the baked layer, comp, sidecar JSON and Nuke Reads out

Run: <V141 python> tests\\test_platelock_crossview.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.platelock.core import crossview as cv  # noqa: E402
from atlas_vfx.platelock.core import synth, warp  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print(("ok   " if cond else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


S = synth.make_shot(n_plate=8, n_ai=10, px_scale=6)      # 960 x 576, performers ~150 px tall
H, W = S.plate.shape[1:3]
YS, XS = np.mgrid[0:H, 0:W].astype(np.float32)


def known_case(i, k):
    pm = S.plate_masks[k, i]
    cy, cx = [c.mean() for c in np.nonzero(pm > 0.5)]
    tx = cx + (XS - cx) * 1.05 + 7 + 3 * np.sin((YS - cy) / 40)
    ty = cy + (YS - cy) * 1.03 - 5 + 2 * np.sin((XS - cx) / 35)
    ix, iy = XS.copy(), YS.copy()
    for _ in range(30):  # invert the field so ai(t(x)) == plate(x)
        ix += XS - warp.remap(tx, ix, iy, border="replicate")
        iy += YS - warp.remap(ty, ix, iy, border="replicate")
    ai = warp.remap(S.plate[i] * np.array([1.15, 1.0, 0.85], np.float32), ix, iy, border="replicate")
    am = warp.remap(pm, ix, iy)
    ai = ai * am[..., None] + S.ai_bg[i] * (1 - am[..., None])
    return pm, am, ai, tx, ty


def bands(e, d, m):
    return {f"{a}-{b}": float(np.median(e[m & (d >= a) & (d < b)])) for a, b in ((0, 5), (5, 15), (15, 999))}


def test_known_warp():
    E, D = [], []
    for i, k in ((2, 0), (5, 1)):
        pm, am, ai, tx, ty = known_case(i, k)
        r = cv.crossview_frame(cv.acescct_from_acescg(S.plate[i]), cv.acescct_from_acescg(ai), {"p": pm}, {"p": am},
                               cv.CVConfig(proc_width=0))
        mx, my = r.ids["p"].maps
        m = pm > 0.5
        E.append(np.hypot(mx - tx, my - ty)[m])
        D.append(distance_transform_edt(m)[m])
    E, D = np.concatenate(E), np.concatenate(D)
    b = bands(E, D, np.ones_like(E, bool))
    check(np.median(E) < 0.6 and np.percentile(E, 95) < 2.0 and b["0-5"] < 1.5,
          "known non-rigid warp recovered", f"median {np.median(E):.2f}, p95 {np.percentile(E, 95):.2f}, bands {b}")
    mom = cv.moments_map(*known_case(2, 0)[:2], (H, W))
    pm, am, ai, tx, ty = known_case(2, 0)
    e0 = np.median(np.hypot(mom[0] - tx, mom[1] - ty)[pm > 0.5])
    check(e0 > 3 * np.median(E), "dense flow beats the moments step alone", f"moments {e0:.2f}px")


def gt_error(i, k, maps):
    j = int(round(S.tau_true[k, i]))
    P, A = synth.gt_correspondences(S, k, i, j, n=500, seed=5)
    f = lambda m: warp.remap(m, P[:, 0][None].astype(np.float32), P[:, 1][None].astype(np.float32))[0]  # noqa: E731
    return np.hypot(f(maps[0]) - A[:, 0], f(maps[1]) - A[:, 1]), np.hypot(P[:, 0] - A[:, 0], P[:, 1] - A[:, 1])


def test_full_take():
    E, E0 = [], []
    for i in (2, 5):
        for k, p in enumerate(S.performers):
            j = int(round(S.tau_true[k, i]))
            r = cv.crossview_frame(cv.acescct_from_acescg(S.plate[i]), cv.acescct_from_acescg(S.ai[j]),
                                   {p.name: S.plate_masks[k, i]}, {p.name: S.ai_masks[k, j]}, cv.CVConfig(proc_width=0))
            e, e0 = gt_error(i, k, r.ids[p.name].maps)
            E.append(e)
            E0.append(e0)
    E, E0 = np.concatenate(E), np.concatenate(E0)
    check(np.median(E) < 0.8 and np.percentile(E, 90) < 3.5, "relit resized offset performers lock",
          f"raw median {np.median(E0):.1f}px -> {np.median(E):.2f}px, p90 {np.percentile(E, 90):.2f}px")


def test_half_res_solve():
    pm, am, ai, tx, ty = known_case(2, 0)
    r = cv.crossview_frame(cv.acescct_from_acescg(S.plate[2]), cv.acescct_from_acescg(ai), {"p": pm}, {"p": am},
                           cv.CVConfig(proc_width=W // 2))
    maps = cv.scale_maps(r.ids["p"].maps, r.proc_wh, (W, H), (W, H))
    e = np.hypot(maps[0] - tx, maps[1] - ty)[pm > 0.5]
    # half the detail, so expect about twice the full-res error: solve at delivery res for 4K/6K finals
    check(r.proc_wh == (W // 2, H // 2) and np.median(e) < 1.5, "half-res solve scaled to full res",
          f"median {np.median(e):.2f}px at full res, p95 {np.percentile(e, 95):.2f}")
    ix, iy = warp.identity_maps(H // 2, W // 2)
    mx, my = cv.scale_maps((ix, iy), (W // 2, H // 2), (W, H), (W, H))
    jx, jy = warp.identity_maps(H, W)
    check(max(np.abs(mx - jx).max(), np.abs(my - jy).max()) < 1e-4, "map scaling is exact for identity, borders too")


def test_repo_guard():
    from atlas_vfx.platelock.crossview_job import inside_repo
    check(inside_repo(ROOT) and inside_repo(ROOT / "a" / "b") and not inside_repo(ROOT.parent / (ROOT.name + "-work"))
          and not inside_repo(ROOT.parent), "out_dir guard: repo refused, sibling <repo>-work allowed")


def test_job():
    try:
        import OpenImageIO  # noqa: F401
    except ImportError:
        print("skip EXR job test (no OpenImageIO here)")
        return
    from atlas_vfx.camera_side import exr_io
    from atlas_vfx.platelock import crossview_job
    from atlas_vfx.platelock.core import seqio
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        frames = (2, 3)
        for i in frames:
            fn = 1001 + i
            j = int(round(S.tau_true[0, i]))
            exr_io.write_rgba(d / "plate" / f"plate.{fn:04d}.exr", np.dstack([S.plate[i], np.ones((H, W))]), "ACEScg")
            exr_io.write_rgba(d / "ai" / f"ai.{fn:04d}.exr", np.dstack([S.ai[j], np.ones((H, W))]), "ACEScg")
            for k, p in enumerate(S.performers):
                m = S.plate_masks[k, i]
                exr_io.write_rgba(d / "m" / f"{p.name}.{fn:04d}.exr", np.dstack([m, m, m, m]), "Raw")
                am = S.ai_masks[k, j]
                exr_io.write_rgba(d / "am" / f"{p.name}.{fn:04d}.exr", np.dstack([am, am, am, am]), "Raw")
        out = d / "out"
        side = crossview_job.run(str(d / "plate" / "plate.####.exr"), str(d / "ai" / "ai.####.exr"),
                                 {p.name: str(d / "m" / f"{p.name}.####.exr") for p in S.performers}, str(out),
                                 ai_ids={p.name: str(d / "am" / f"{p.name}.####.exr") for p in S.performers},
                                 cfg=cv.CVConfig(proc_width=W // 2))
        fn = 1001 + frames[0]
        hd = exr_io.header(out / "left" / f"left.{fn:04d}.exr")
        check(hd["ocio_colorspace"] == "ACEScg" and "half" in hd["format"].lower(),
              "warped layer is half EXR tagged with its colourspace", f"{hd['ocio_colorspace']}, {hd['format']}")
        hs = exr_io.header(out / "left" / f"left_stmap.{fn:04d}.exr")
        check("float" in hs["format"].lower(), "STMap is float", hs["format"])
        st = seqio._read_pixels(str(out / "left" / f"left_stmap.{fn:04d}.exr"), ("R", "G"))
        baked = seqio._read_pixels(str(out / "left" / f"left.{fn:04d}.exr"), ("R", "G", "B", "A"))
        ai_full = seqio._read_pixels(str(d / "ai" / f"ai.{fn:04d}.exr"))
        re = warp.remap(ai_full, *warp.from_stmap(st, W, H), border="replicate")
        m = baked[..., 3] > 0.5
        err = float(np.abs(re - baked[..., :3])[m].max())
        check(err < 4e-3, "STMap re-applied to the AI frame reproduces the warped layer", f"max {err:.1e} (half)")
        js = json.loads((out / "crossview.json").read_text())
        check(js["schema"] == "platelock.crossview/1" and set(js["summary"]) == {"left", "right"}
              and js["inputs"]["match_space"] == "ACEScct", "sidecar JSON")
        check((out / "comp" / f"comp.{fn:04d}.exr").exists() and (out / "platelock_crossview.nk").exists(),
              "comp and Nuke Reads written")
        try:
            crossview_job.run(str(d / "plate" / "plate.####.exr"), str(d / "ai" / "ai.####.exr"),
                              {"left": str(d / "m" / "left.####.exr")}, str(ROOT / "nope"))
            check(False, "refuses an out_dir inside the repo")
        except ValueError:
            check(True, "refuses an out_dir inside the repo")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
