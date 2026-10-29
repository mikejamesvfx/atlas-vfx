"""Phase 1b S5: does RS correction help against the real plate? Any shot camera + plate + rigid plane.

The tools/rs_deck_proof.py method (2026-10-20), generalised (core in atlas_vfx/sensor/rs_proof.py): reproject a
rigid plane from each reference frame into the other frames through the camera, RS on (Sensor RS maps made from the
plane's depth at the camera's readout, removed at the reference and applied at the target) vs RS off (a control with
the same resampling and no net RS geometry), gradient error on the plane against the real plate.

The camera should be an RS-aware solve: its poses are frame-time (centre row) poses and its readout is in
extensions.atlas_vfx_rs_solve (RS Solve, catalogue 47). --readout-ms overrides.

Surface, one of:
  --plane cx,cy,cz,nx,ny,nz [--box x0,x1,y0,y1,z0,z1]   a known plane (e.g. the deck, y = 0), box in world units,
                                                        'none' for an open side
  --tracks GS_CAMERA.json --fit-frames A-B:S --fit-region fx0,fy0,fx1,fy1
        a plane fitted to the shot's own tracks (the GS camera's COLMAP solve gives the 2D tracks), triangulated
        with THIS camera at each observation's row time; region as fractions of the frame. Pixels are kept only
        where the nearest tracked point is on the plane (rocks standing on the beach drop out).

RS off here is the RS camera's own poses with no readout (the plan's measure). With --baseline GS.json the same
targets are also scored against the GS camera the RS solve started from (its own plane from the same tracks at
readout 0, its control image), on the pixels both cameras keep: RS on vs the best camera without a readout, so a
pose change in the RS solve can't pass for a readout gain. It also reports the GS camera with the readout pasted
on (the double count).

Writes <out>/<tag>.csv (every pair), <tag>.summary.json (per reference, overall, the plan's acceptance check, and
vs_gs_camera with --baseline), review/<tag>_ref<R>_t<T>[_vs_gs].jpg. Plates are only read.

Run: <V141 python> tools\\rs_s5_proof.py --camera CAM.json --refs 1004,1010 [--targets all|A-B:S] [--min-gap 1]
     [--width 1024] [--out DIR] [--tag NAME] (--plane ... | --tracks ... [--baseline GS.json])
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import measure  # noqa: E402
from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402
from atlas_vfx.sensor import rs_proof as rp  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def frange(text, frames):
    if text in ("", "all"):
        return list(frames)
    span, _, step = text.partition(":")
    a, _, b = span.partition("-")
    return [f for f in range(int(a), int(b) + 1, int(step or 1)) if f in frames]


def floats(text, n, allow_none=False):
    v = [None if allow_none and s.strip().lower() == "none" else float(s) for s in text.split(",")]
    if len(v) != n:
        raise SystemExit(f"expected {n} numbers, got {text!r}")
    return v


def review_image(path, imgs, gray, title, second="plain"):
    """Six tiles: target, imgs['plain'] (labelled `second`), RS off, RS on, and the gradient error of the last two."""
    import cv2

    def view(a):
        g = gray(a)
        lo, hi = np.percentile(g[imgs["mask"]], [1, 99]) if imgs["mask"].any() else (g.min(), g.max())
        return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)

    gt = rp.grad_mag(cv2.GaussianBlur(gray(imgs["target"]), (0, 0), 1.0))
    tiles = []
    edge = cv2.morphologyEx(imgs["mask"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    for name in ("target", "plain", "control", "sensor"):
        t = (view(imgs[name]) * 255).astype(np.uint8)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t[edge] = (0, 200, 255)
        tiles.append(t)
    scale = None
    for name in ("control", "sensor"):
        e = np.abs(rp.grad_mag(cv2.GaussianBlur(gray(imgs[name]), (0, 0), 1.0)) - gt)
        e[~imgs["mask"]] = 0
        scale = scale or max(np.percentile(e[imgs["mask"]], 99), 1e-6)
        t = cv2.applyColorMap((np.clip(e / scale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        tiles.append(t)
    ys, xs = np.nonzero(imgs["mask"])
    h, w = tiles[0].shape[:2]
    pad = 24
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, h), max(xs.min() - pad, 0), min(xs.max() + pad, w)
    k = min(2.0, 900 / max(x1 - x0, 1), 500 / max(y1 - y0, 1))
    crop = [cv2.resize(t[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) for t in tiles]
    for t, (name) in zip(crop, ("target", second, "RS off (control)", "RS on (sensor)", "grad error RS off",
                                "grad error RS on")):
        cv2.rectangle(t, (0, 0), (min(t.shape[1], 300), 26), (0, 0, 0), -1)
        cv2.putText(t, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(80, 80, 80)) for t in crop]
    sheet = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])
    bar = np.zeros((30, sheet.shape[1], 3), np.uint8)
    fs = min(0.6, 0.6 * sheet.shape[1] / max(cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 20, 1))
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    sheet = np.vstack([bar, sheet])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--readout-ms", type=float, default=None)
    ap.add_argument("--plate", default="")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--targets", default="all")
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--plane", default="")
    ap.add_argument("--box", default="")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--fit-frames", default="all")
    ap.add_argument("--fit-region", default="0,0.62,1,1")
    ap.add_argument("--mask-radius", type=float, default=0.03, help="fraction of the image width")
    ap.add_argument("--eval-region", default="fit",
                    help="fx0,fy0,fx1,fy1: only pixels inside this part of the frame, at the target and at the "
                         "reference (keeps e.g. foliage at the top of frame out); 'fit' = --fit-region with --tracks, "
                         "'all' = whole frame")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--baseline", default="",
                    help="GS shot camera of the same plate (with --tracks: usually the same file). Also scores the "
                         "best camera without a readout: its own plane from the same tracks triangulated at readout 0, "
                         "its control image, on the pixels both cameras keep (RS on vs GS off)")
    ap.add_argument("--baseline-max-px", type=float, default=6.0,
                    help="ray gate (px at plate size) for the baseline camera's triangulated tracks")
    ap.add_argument("--criterion", choices=["fast pan", "not worse"], default="fast pan",
                    help="the plan's S5 check: 'fast pan' (>= 20 %% median, no ref worse than -5 %%) or 'not worse'")
    a = ap.parse_args()

    cam = sc.load(pl.clean_path(a.camera))
    ext = (cam.data.get("extensions") or {}).get("atlas_vfx_rs_solve") or {}
    readout = a.readout_ms if a.readout_ms is not None else ext.get("readout_ms")
    if readout is None:
        raise SystemExit("camera has no RS-solve readout (extensions.atlas_vfx_rs_solve); give --readout-ms")
    readout_from = "--readout-ms" if a.readout_ms is not None else (
        f"camera RS solve ({'solved' if ext.get('readout_solved') else 'held'})")
    frames = list(cam.frames)
    refs = [int(r) for r in a.refs.split(",")]
    targets = frange(a.targets, frames)
    tag = pl.safe_name(a.tag or f"{cam.name}_s5")
    out = pl.resolve_out_dir(a.out, pl.WORK_ROOT / "rs_clips" / tag)
    out.mkdir(parents=True, exist_ok=True)
    plate_path = pl.clean_path(a.plate) or cam.data["plate"]["path"]
    log(cam.name, f"readout {readout:.2f} ms ({readout_from}), refs {refs}, {len(targets)} targets, out {out}")

    W = a.width
    H = int(round(W * cam.data["plate"]["height"] / cam.data["plate"]["width"] / 2)) * 2
    PW = cam.data["plate"]["width"]
    surface = {}
    box, obs, tri, plane = None, None, None, None
    if a.plane:
        v = floats(a.plane, 6)
        c, nrm = np.array(v[:3]), np.array(v[3:]) / np.linalg.norm(v[3:])
        if a.box:
            b = floats(a.box, 6, allow_none=True)
            box = [None if b[i] is None and b[i + 1] is None else
                   (-np.inf if b[i] is None else b[i], np.inf if b[i + 1] is None else b[i + 1]) for i in (0, 2, 4)]
        surface = {"kind": "plane given", "centre": c.tolist(), "normal": nrm.tolist(), "box": a.box or None}
    elif a.tracks:
        gs = sc.load(pl.clean_path(a.tracks))
        pts, note = measure.points_for_camera(gs, max_reproj_px=6.0)
        obs = pts["observations"]
        # tracks are triangulated at the camera's own readout (the one its poses were solved with); --readout-ms
        # changes only the maps, so a wrong-readout check runs on the same surface and mask
        r_fr = float(ext.get("readout_ms", readout)) / 1000.0 * float(cam.data["shot"]["fps"])
        tri = rp.triangulate_rs(cam, obs, r_fr)
        fx0, fy0, fx1, fy1 = floats(a.fit_region, 4)
        region = (fx0 * PW, fy0 * cam.data["plate"]["height"], fx1 * PW, fy1 * cam.data["plate"]["height"])
        plane = rp.fit_plane_from_tracks(cam, obs, tri, frange(a.fit_frames, frames), region)
        c, nrm = plane["centre"], plane["normal"]
        dist = np.abs((tri["xyz"] - c) @ nrm)
        surface = {"kind": "plane fitted to the shot's tracks", "tracks_from": gs.name, "tracks_note": note,
                   "triangulated": int(len(tri["ids"])), "tri_err_px_median": float(np.median(tri["err_px"])),
                   "fit_frames": a.fit_frames, "fit_region_frac": [fx0, fy0, fx1, fy1],
                   "centre": c.tolist(), "normal": nrm.tolist(), "thresh": plane["thresh"], "rms": plane["rms"],
                   "inliers": plane["points"], "of": plane["of"],
                   "all_tracks_on_plane_pct": float(100 * np.mean(dist < plane["thresh"])),
                   "normal_vs_view_deg": float(np.degrees(np.arccos(abs(np.mean(
                       [-cam.cam_to_world(f)[:3, 2] for f in frames], 0) @ nrm / np.linalg.norm(np.mean(
                           [-cam.cam_to_world(f)[:3, 2] for f in frames], 0)))))),
                   "median_depth": float(np.median([np.median(-(tri["xyz"] @ cam.world_to_cam(f)[:3, :3].T
                                                                 + cam.world_to_cam(f)[:3, 3])[:, 2])
                                                     for f in frames[::8]])),
                   "camera_path_length": float(np.sum(np.linalg.norm(np.diff(
                       [cam.cam_to_world(f)[:3, 3] for f in frames], axis=0), axis=1))),
                   "mask_radius_px": a.mask_radius * W}
        log("surface", json.dumps({k: v for k, v in surface.items() if k != "tracks_note"}))
    else:
        raise SystemExit("give --plane or --tracks")
    base = None
    if a.baseline:
        if not a.tracks:
            raise SystemExit("--baseline needs --tracks (each camera gets its own plane from the same tracks)")
        bcam = sc.load(pl.clean_path(a.baseline))
        if (bcam.data.get("extensions") or {}).get("atlas_vfx_rs_solve"):
            raise SystemExit("--baseline must be a global-shutter camera (it has an RS-solve readout)")
        # a GS solve of an RS plate keeps the skew in its residuals (DSCF2973: 5.3 px median vs 2.6 after the RS
        # solve), so the RS camera's 3 px ray gate would starve its plane; it gets the RS Solve node's 6 px
        # same-world gate for GS solves instead
        btri = rp.triangulate_rs(bcam, obs, 0.0, max_px=a.baseline_max_px)
        bplane = rp.fit_plane_from_tracks(bcam, obs, btri, frange(a.fit_frames, frames), region)
        base = {"cam": bcam, "tri": btri, "plane": bplane, "maps": {}, "labels": {}}
        surface["baseline"] = {"camera": bcam.name, "max_px": a.baseline_max_px, "triangulated": int(len(btri["ids"])),
                               "tri_err_px_median": float(np.median(btri["err_px"])),
                               "centre": bplane["centre"].tolist(), "normal": bplane["normal"].tolist(),
                               "thresh": bplane["thresh"], "rms": bplane["rms"], "inliers": bplane["points"],
                               "of": bplane["of"]}
        log("baseline surface", json.dumps(surface["baseline"]))

    need = sorted(set(refs) | set(targets))
    ref = pl.plate_ref(plate_path, need[0] - cam.data["shot"].get("plate_frame_offset", 1001),
                       need[-1] - cam.data["shot"].get("plate_frame_offset", 1001))
    off = ref["frame_offset"]
    imgs = {}
    for i, img in pl.decode_rgb(ref, need[0] - off, need[-1] - off, width=W):
        if i + off in need:
            imgs[i + off] = pl.to_working(img, ref)
    log(f"decoded {len(imgs)} frames at {W}x{H}")
    fwd, _ = li.view_processors(ref["ocio_config"])
    gray = (lambda img: (li.to_view(img, fwd) @ rp.LUMA).astype(np.float32))
    maps = {}

    def maps_for(n):
        if n not in maps:
            maps[n] = rp.rs_maps_for_plane(cam, n, W, H, readout, c, nrm)
        return maps[n]

    ev = a.eval_region if a.eval_region != "fit" else (a.fit_region if a.tracks else "all")
    region_mask = None
    if ev != "all":
        ex0, ey0, ex1, ey1 = floats(ev, 4)
        region_mask = np.zeros((H, W), bool)
        region_mask[int(round(ey0 * H)):int(round(ey1 * H)), int(round(ex0 * W)):int(round(ex1 * W))] = True
    surface["eval_region_frac"] = ev
    labels = {}

    def label(n):
        if obs is None:
            return region_mask
        if n not in labels:
            m = rp.track_label_map(obs, tri, plane, n, W, H, PW, a.mask_radius * W)
            labels[n] = m if region_mask is None else (m & region_mask)
        return labels[n]

    def base_maps(n):
        # the GS camera's maps at the same readout: only its control (remove + apply, no net geometry) and its
        # double-count image use them
        if n not in base["maps"]:
            base["maps"][n] = rp.rs_maps_for_plane(base["cam"], n, W, H, readout, base["plane"]["centre"],
                                                   base["plane"]["normal"])
        return base["maps"][n]

    def base_label(n):
        if n not in base["labels"]:
            m = rp.track_label_map(obs, base["tri"], base["plane"], n, W, H, PW, a.mask_radius * W)
            base["labels"][n] = m if region_mask is None else (m & region_mask)
        return base["labels"][n]

    rows, srcs, bsrcs = [], {}, {}
    for r in refs:
        if r not in imgs:
            log(f"ref {r} not in the camera's frames; skipped")
            continue
        s_clean, s_ctrl = rp.reference_sources(imgs[r], maps_for(r))
        srcs[r] = (s_clean, s_ctrl)
        if base:
            bsrcs[r] = rp.reference_sources(imgs[r], base_maps(r))
        for n in targets:
            if abs(n - r) < a.min_gap or n == r:
                continue
            row = rp.compare(cam, r, n, imgs[r], s_clean, s_ctrl, imgs[n], maps_for(n), c, nrm, gray=gray,
                             box=box, extra_mask=label(n), ref_mask=label(r), want_images=bool(base))
            if row and base:
                brow = rp.compare(base["cam"], r, n, imgs[r], *bsrcs[r], imgs[n], base_maps(n),
                                  base["plane"]["centre"], base["plane"]["normal"], gray=gray,
                                  extra_mask=base_label(n), ref_mask=base_label(r), want_images=True)
                both = rp.against_baseline(imgs[n], row, brow, gray=gray) if brow else None
                row.pop("_images")
                row.update(both or {"common_px": 0, "on_vs_gs_pct": None})
            if row:
                rows.append(row)
        rr = [x for x in rows if x["ref"] == r]
        if rr:
            msg = (f"ref {r}: {len(rr)} targets, RS on vs off median "
                   f"{np.median([x['better_vs_control_pct'] for x in rr]):+.1f} %")
            gb = [x["on_vs_gs_pct"] for x in rr if x.get("on_vs_gs_pct") is not None]
            if gb:
                msg += f", RS on vs GS camera off {np.median(gb):+.1f} % ({len(gb)} pairs)"
            log(msg)
    if not rows:
        raise SystemExit("no reference/target pair had enough surface pixels")
    with open(out / f"{tag}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(dict.fromkeys(k for x in rows for k in x)), restval="")
        wr.writeheader()
        wr.writerows(rows)
    summ = rp.summarise(rows)
    acc = rp.acceptance(summ, kind=a.criterion)
    vs_gs = None
    if base:
        s_gs = rp.summarise(rows, key="on_vs_gs_pct")
        dc = [x["gs_on_vs_gs_off_pct"] for x in rows if x.get("gs_on_vs_gs_off_pct") is not None]
        oc = [x["on_vs_off_common_pct"] for x in rows if x.get("on_vs_off_common_pct") is not None]
        vs_gs = {"camera": base["cam"].name, "summary": s_gs, "acceptance": rp.acceptance(s_gs, kind=a.criterion),
                 "on_vs_off_on_common_px_median_pct": float(np.median(oc)) if oc else None,
                 "gs_camera_plus_readout_vs_gs_off_median_pct": float(np.median(dc)) if dc else None}

    # review: first, middle and last reference, each with one target (display only, numbers come from every pair)
    rv = []
    used = sorted({x["ref"] for x in rows})
    for r in sorted({used[0], used[len(used) // 2], used[-1]}):
        cand = [x for x in rows if x["ref"] == r]
        wide = [x for x in cand if abs(x["frame"] - r) >= 8] or cand
        far = max(wide, key=lambda x: x["surface_px"])          # display only: the biggest surface 8+ frames away
        n = far["frame"]
        res = rp.compare(cam, r, n, imgs[r], *srcs[r], imgs[n], maps_for(n), c, nrm, gray=gray, box=box,
                         extra_mask=label(n), ref_mask=label(r), want_images=True)
        p = out / "review" / f"{tag}_ref{r}_t{n}.jpg"
        review_image(p, res["_images"], gray, f"{cam.name} ref {r} -> {n}, readout {readout:.2f} ms: RS on vs off "
                     f"{res['better_vs_control_pct']:+.1f} %, vs plain {res['better_pct']:+.1f} %")
        rv.append(str(p))
        if base:
            brow = rp.compare(base["cam"], r, n, imgs[r], *bsrcs[r], imgs[n], base_maps(n), base["plane"]["centre"],
                              base["plane"]["normal"], gray=gray, extra_mask=base_label(n), ref_mask=base_label(r),
                              want_images=True)
            both = rp.against_baseline(imgs[n], res, brow, gray=gray) if brow else None
            if both:
                im = dict(res["_images"], plain=brow["_images"]["control"],
                          mask=res["_images"]["mask"] & brow["_images"]["mask"])
                p = out / "review" / f"{tag}_ref{r}_t{n}_vs_gs.jpg"
                review_image(p, im, gray, f"ref {r} -> {n}, common pixels: RS on vs off "
                             f"{both['on_vs_off_common_pct']:+.1f} %, RS on vs GS camera off {both['on_vs_gs_pct']:+.1f} %",
                             second=f"GS camera off ({base['cam'].name})")
                rv.append(str(p))
    rs_skew = [{"frame": n, **rp.rs.skew_px_per_row(maps[n][0])} for n in sorted(maps)]
    full = {"camera": cam.name, "camera_path": str(a.camera), "plate": str(plate_path), "readout_ms": readout,
            "readout_from": readout_from, "image": [W, H], "refs": refs, "targets": a.targets, "min_gap": a.min_gap,
            "camera_rs_solve": {k: (ext.get("report") or {}).get(k) for k in
                                ("frames", "n_points", "reproj_input_gs", "reproj_solved", "align_to_input",
                                 "pose_change", "message")},
            "surface": surface, "summary": summ, "acceptance": acc, "vs_gs_camera": vs_gs, "review": rv,
            "skew_px_per_row_at_image_size": {"median_abs": float(np.median([abs(s["dx_per_row"]) for s in rs_skew])),
                                              "max_abs": float(np.max([abs(s["dx_per_row"]) for s in rs_skew]))},
            "method": "tools/rs_s5_proof.py + atlas_vfx/sensor/rs_proof.py (rs_deck_proof method, maps from the "
                      "plane's depth)", "seconds": round(time.time() - T0)}
    (out / f"{tag}.summary.json").write_text(json.dumps(full, indent=1, default=float), encoding="utf-8")
    for p in summ["per_ref"]:
        log(f"ref {p['ref']}: {p['targets']} targets, RS on vs off {p['on_vs_off_median_pct']:+.1f} % "
            f"(better in {p['on_better_in']}), vs plain {p['vs_plain_median_pct']:+.1f} %")
    log(f"median of refs {summ['median_of_refs_pct']:+.1f} %, all pairs {summ['median_all_pairs_pct']:+.1f} % "
        f"({summ['on_better_pairs']} of {summ['pairs']}), worst ref {summ['worst_ref_pct']:+.1f} %")
    log("acceptance", json.dumps(acc))
    if vs_gs:
        sg = vs_gs["summary"]
        for p in sg["per_ref"]:
            log(f"ref {p['ref']}: RS on vs GS camera off {p['on_vs_off_median_pct']:+.1f} % "
                f"(better in {p['on_better_in']} of {p['targets']})")
        log(f"vs GS camera: median of refs {sg['median_of_refs_pct']:+.1f} %, all pairs "
            f"{sg['median_all_pairs_pct']:+.1f} % ({sg['on_better_pairs']} of {sg['pairs']}), worst ref "
            f"{sg['worst_ref_pct']:+.1f} %; on common pixels RS on vs off "
            f"{vs_gs['on_vs_off_on_common_px_median_pct']:+.1f} %, GS camera + readout vs GS off "
            f"{vs_gs['gs_camera_plus_readout_vs_gs_off_median_pct']:+.1f} %")
        log("acceptance vs GS camera", json.dumps(vs_gs["acceptance"]))
    log("wrote", out / f"{tag}.summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
