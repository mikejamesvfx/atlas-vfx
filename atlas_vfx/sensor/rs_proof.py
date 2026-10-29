"""Does rolling-shutter correction help against the real plate? (Phase 1b S5.) A test that needs no LTX.

A rigid surface (a plane: the deck, a beach) is seen by a camera whose poses are frame-time (centre-row) poses
from an RS-aware solve, plus a readout. Take the surface from a reference frame and put it into a target frame
through the camera via the plane:
  plain  : plate[ref] -> plane -> frame t                                  (RS off: rows treated as one instant)
  sensor : de-RS(plate[ref]) -> plane -> frame t -> apply RS(t)            (RS on: Sensor Pass maps both ends)
  control: the plain geometry with the same extra resamples and no net RS geometry (remove + apply at ref, then
           remove + apply at t), so blur from resampling is not mistaken for alignment.
Each is compared with plate[t] on surface pixels, as the median absolute difference of gradient magnitude
(robust to small exposure drift). RS on vs off = sensor vs control. Same method as tools/rs_deck_proof.py
(2026-10-20), generalised from the deck's y = 0 plane to any plane, with the RS maps made in memory from the
plane's own depth by sensor/rs.py (the Sensor RS Maps code) and applied by sensor/apply.py (Sensor Apply).

Surface from a solve: the camera's own 2D tracks triangulated with the RS camera at each observation's row time
(`triangulate_rs`, one time model: camera_side/rs_time), a RANSAC plane through the tracks in a picked image region
(`fit_plane_from_tracks`), and a per-frame mask that keeps only pixels whose nearest tracked point lies on the
plane (rocks or posts standing on it drop out, `track_label_map`).

Pixels: contract (row i's centre at i + 0.5) for cameras and maps; OpenCV (centre at i) only for cv2.remap.
Pure numpy / scipy / OpenCV; imports camera_side (rs_time, measure.fit_plane) and this package only.
"""
from __future__ import annotations

import numpy as np

from ..camera_side import rs_time
from ..camera_side.measure import fit_plane
from . import apply as ap
from . import rs

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------------------------------- geometry

def _rays(cam, n, w, h):
    """Origin (3,) and world ray directions (h, w, 3) of every pixel centre of a w x h image of frame n
    (frame-time pose, global shutter)."""
    C2W = cam.cam_to_world(n)
    k = cam.intrinsics(n)
    s = w / cam.data["plate"]["width"]
    u = (np.arange(w) + 0.5) / s
    v = (np.arange(h) + 0.5) / s
    dx = (u[None, :] - k.cx) / k.fx
    dy = -(v[:, None] - k.cy) / k.fy
    d = np.stack(np.broadcast_arrays(dx, dy, -np.ones((h, w))), -1) @ C2W[:3, :3].T
    return C2W[:3, 3], d


def plane_hits(cam, n, w, h, centre, normal):
    """World points where each pixel's ray meets the plane (h, w, 3), and the z depth along the camera axis
    (h, w; NaN where the ray misses or points away). The ray direction has camera z = -1, so t is the z depth."""
    o, d = _rays(cam, n, w, h)
    nrm = np.asarray(normal, np.float64)
    den = d @ nrm
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((np.asarray(centre, np.float64) - o) @ nrm) / np.where(np.abs(den) > 1e-12, den, np.nan)
    t = np.where(t > 0, t, np.nan)
    return o + t[..., None] * d, t


def project(cam, n, X, w, h):
    """World points -> OpenCV pixel coordinates (float32) in a w x h image of frame n, and z depth."""
    W2C = cam.world_to_cam(n)
    k = cam.intrinsics(n)
    s = w / cam.data["plate"]["width"]
    p = X @ W2C[:3, :3].T + W2C[:3, 3]
    z = -p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (k.fx * p[..., 0] / z + k.cx) * s - 0.5
        v = (-k.fy * p[..., 1] / z + k.cy) * s - 0.5
    return u.astype(np.float32), v.astype(np.float32), z


def rs_maps_for_plane(cam, n, w, h, readout_ms, centre, normal, far=None):
    """(rs_map, ders_map) for frame n at w x h, depth from the plane (misses filled with `far`, default 50 x the
    median hit depth). Same function Sensor RS Maps calls per frame."""
    _, z = plane_hits(cam, n, w, h, centre, normal)
    ok = np.isfinite(z)
    if not ok.any():
        raise ValueError(f"frame {n}: no pixel sees the plane")
    far = float(far) if far else 50.0 * float(np.median(z[ok]))
    z = np.where(ok, np.minimum(z, far), far)
    return rs.rs_maps(cam, n, z, readout_ms / 1000.0, width=w, height=h)


# ------------------------------------------------------------------------------- surface from tracks

def row_poses(cam, frame, y, readout_fr):
    """Camera-to-world rotation (m, 3, 3) and centre (m, 3) at the capture time of sensor row y (contract px, plate
    size) of each observation in frame `frame` (Nuke frames, arrays), through rs_time (geodesic, linear centre,
    neighbour pairing, ends extrapolated from the inner pair)."""
    frames = list(cam.frames)
    R = np.array([cam.cam_to_world(f)[:3, :3] for f in frames])
    C = np.array([cam.cam_to_world(f)[:3, 3] for f in frames])
    H = int(cam.data["plate"]["height"])
    pos = {f: i for i, f in enumerate(frames)}
    fi = np.array([pos[int(f)] for f in np.atleast_1d(frame)])
    rf = rs_time.row_frac(np.atleast_1d(y), H)
    if len(frames) < 2 or readout_fr == 0:
        return R[fi], C[fi]
    rel = rs_time.rel_rotvecs(R)
    i0 = rs_time.pair_index(fi, rf, 0, len(frames) - 1)
    a = (fi - i0) + rf * float(readout_fr)
    return rs_time.interp_rot(R, rel, i0, a), rs_time.interp_centre(C, i0, a)


def triangulate_rs(cam, observations, readout_fr, *, min_obs=3, max_px=3.0):
    """Midpoint triangulation of every tracked point from its observations, each ray cast from the camera at that
    observation's row time. observations: {nuke frame: (point indices, xy contract px at plate size)} (what
    measure.points_for_camera gives). Returns {"ids", "xyz", "err_px" (median ray-to-point error in px), "n_obs"}
    for points with at least min_obs observations in the camera's frames and err under max_px."""
    k = cam.intrinsics(cam.frames[0])
    ids, xs, ys, fr = [], [], [], []
    for f, (idx, xy) in observations.items():
        if f not in cam._by_frame or not len(idx):
            continue
        ids.append(np.asarray(idx))
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
        fr.append(np.full(len(idx), f))
    ids, x, y, fr = (np.concatenate(a) for a in (ids, xs, ys, fr))
    R, C = row_poses(cam, fr, y, readout_fr)
    d = np.stack([(x - k.cx) / k.fx, -(y - k.cy) / k.fy, -np.ones_like(x)], -1)
    d = np.einsum("nij,nj->ni", R, d)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    uniq, inv = np.unique(ids, return_inverse=True)
    P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = np.zeros((len(uniq), 3, 3))
    b = np.zeros((len(uniq), 3))
    np.add.at(A, inv, P)
    np.add.at(b, inv, np.einsum("nij,nj->ni", P, C))
    cnt = np.bincount(inv, minlength=len(uniq))
    good = (cnt >= min_obs) & (np.linalg.cond(A) < 1e8)
    X = np.full((len(uniq), 3), np.nan)
    X[good] = np.linalg.solve(A[good], b[good][..., None])[..., 0]
    # error in px: distance from X to each ray over the depth along it, times the focal
    v = X[inv] - C
    along = np.einsum("ni,ni->n", v, d)
    perp = np.linalg.norm(v - along[:, None] * d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(along > 0, perp / along * k.fx, np.inf)
    err = np.full(len(uniq), np.inf)
    order = np.argsort(inv, kind="stable")
    splits = np.cumsum(cnt)[:-1]
    for j, ej in enumerate(np.split(e[order], splits)):
        if good[j] and len(ej):
            err[j] = float(np.median(ej))
    keep = good & (err < max_px)
    return {"ids": uniq[keep], "xyz": X[keep], "err_px": err[keep], "n_obs": cnt[keep]}


def fit_plane_from_tracks(cam, observations, tri, frames, region, thresh=None):
    """RANSAC plane (measure.fit_plane) through the triangulated tracks seen in `region` (x0, y0, x1, y1 contract px
    at plate size) in any of `frames`. Returns {"centre", "normal", "thresh", "rms", "points", "of"}; the normal
    points to the side the cameras are on."""
    pos = {int(i): j for j, i in enumerate(tri["ids"])}
    x0, y0, x1, y1 = region
    sel = set()
    for f in frames:
        if f not in observations:
            continue
        idx, xy = observations[f]
        m = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        sel.update(int(i) for i in np.asarray(idx)[m] if int(i) in pos)
    if len(sel) < 30:
        raise ValueError(f"only {len(sel)} triangulated tracks in region {region}")
    P = tri["xyz"][[pos[i] for i in sorted(sel)]]
    if thresh is None:
        thresh = 0.01 * float(np.linalg.norm(np.percentile(P, 95, 0) - np.percentile(P, 5, 0)))
    c, nrm, inl, rms = fit_plane(P, thresh=thresh)
    Cs = np.array([cam.cam_to_world(f)[:3, 3] for f in cam.frames])
    if np.median((Cs - c) @ nrm) < 0:
        nrm = -nrm
    return {"centre": c, "normal": nrm, "thresh": float(thresh), "rms": float(rms), "points": int(inl.sum()),
            "of": int(len(P))}


def track_label_map(observations, tri, plane, frame, w, h, plate_w, radius_px):
    """(h, w) bool: pixels whose nearest tracked point (observed in `frame`) is on the plane (|distance| under the
    plane's thresh) and lies within radius_px (image px). Pixels near no tracked point are False."""
    from scipy.spatial import cKDTree

    if frame not in observations:
        return np.zeros((h, w), bool)
    pos = {int(i): j for j, i in enumerate(tri["ids"])}
    idx, xy = observations[frame]
    keep = np.array([int(i) in pos for i in idx], bool)
    if keep.sum() < 3:
        return np.zeros((h, w), bool)
    j = np.array([pos[int(i)] for i in np.asarray(idx)[keep]])
    on = np.abs((tri["xyz"][j] - plane["centre"]) @ plane["normal"]) < plane["thresh"]
    s = w / plate_w
    tree = cKDTree(xy[keep] * s)
    gy, gx = np.mgrid[0:h, 0:w]
    dist, nn = tree.query(np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5]), distance_upper_bound=radius_px)
    ok = np.isfinite(dist)
    lab = np.zeros(h * w, bool)
    lab[ok] = on[nn[ok]]
    return lab.reshape(h, w)


# ------------------------------------------------------------------------------------------- the test

def grad_mag(g):
    import cv2
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def default_gray(img):
    """Luma of a linear image through a simple log-ish curve (for synthetic tests; the tools use the OCIO view)."""
    return np.log2(np.maximum(img[..., :3] @ LUMA, 1e-4)).astype(np.float32)


def grad_errors(tgt, imgs: dict, mask, *, gray=default_gray, blur=1.0) -> dict:
    """Median |gradient magnitude - target's| over mask for each image in imgs (the proof's error measure)."""
    import cv2

    g = (lambda a: cv2.GaussianBlur(gray(a), (0, 0), blur))
    gt = grad_mag(g(tgt))
    return {k: float(np.median(np.abs(grad_mag(g(a)) - gt)[mask])) for k, a in imgs.items()}


def against_baseline(tgt, row, base_row, *, gray=default_gray, blur=1.0, min_px=2000):
    """The same target through two cameras: `row` from the RS camera, `base_row` from a GS camera of the same plate
    (each from compare(..., want_images=True), each in its own world with its own plane). Errors on the pixels both
    keep: RS on (RS camera + maps) and RS off (its control), GS off (the GS camera's control: the best camera
    without a readout, the fair 'off') and GS on (GS camera + maps, the double count). None if too few pixels."""
    m = row["_images"]["mask"] & base_row["_images"]["mask"]
    if m.sum() < min_px:
        return None
    e = grad_errors(tgt, {"rs_on": row["_images"]["sensor"], "rs_off": row["_images"]["control"],
                          "gs_off": base_row["_images"]["control"], "gs_on": base_row["_images"]["sensor"]}, m,
                    gray=gray, blur=blur)
    return {"common_px": int(m.sum()), **{f"err_{k}": round(v, 6) for k, v in e.items()},
            "on_vs_off_common_pct": round(100 * (e["rs_off"] - e["rs_on"]) / e["rs_off"], 2),
            "on_vs_gs_pct": round(100 * (e["gs_off"] - e["rs_on"]) / e["gs_off"], 2),
            "gs_on_vs_gs_off_pct": round(100 * (e["gs_off"] - e["gs_on"]) / e["gs_off"], 2)}


def _remove(img, maps):
    return ap.remove(img, ders=maps[1])


def _apply(img, maps):
    return ap.apply(img, rs=[maps[0]])


def reference_sources(src, maps_ref):
    """(clean source, control source) for a reference frame: de-RS'd, and de-RS'd + RS'd back (no net geometry)."""
    clean = _remove(src, maps_ref)
    return clean, _apply(clean, maps_ref)


def compare(cam, ref_n, n, src, src_clean, src_ctrl, tgt, maps_n, centre, normal, *, gray=default_gray,
            box=None, extra_mask=None, ref_mask=None, erode_px=9, border_px=8, min_px=2000, blur=1.0,
            want_images=False):
    """One reference -> target pair. maps_n: (rs_map, ders_map) of frame n at the image size. box: optional world
    axis-aligned box ((x0, x1), (y0, y1), (z0, z1)), None entries unbounded. extra_mask: (h, w) bool in frame n;
    ref_mask: (h, w) bool in the reference frame (sampled at the reprojected position). Returns a row dict, or
    None when fewer than min_px surface pixels remain."""
    import cv2

    h, w = tgt.shape[:2]
    X, t = plane_hits(cam, n, w, h, centre, normal)
    ok = np.isfinite(t)
    if box is not None:
        for ax, lim in enumerate(box):
            if lim is None:
                continue
            with np.errstate(invalid="ignore"):
                ok &= (X[..., ax] > lim[0]) & (X[..., ax] < lim[1])
    u, v, z = project(cam, ref_n, np.where(ok[..., None], X, 0.0), w, h)
    ok &= np.isfinite(u) & (z > 0) & (u > border_px) & (u < w - border_px) & (v > border_px) & (v < h - border_px)
    if extra_mask is not None:
        ok &= extra_mask
    um, vm = np.where(ok, u, -1).astype(np.float32), np.where(ok, v, -1).astype(np.float32)
    if ref_mask is not None:
        ok &= cv2.remap(ref_mask.astype(np.uint8), um, vm, cv2.INTER_NEAREST) > 0
    ok = cv2.erode(ok.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)) > 0
    if ok.sum() < min_px:
        return None
    plain = cv2.remap(np.ascontiguousarray(src, np.float32), um, vm, cv2.INTER_LINEAR)
    clean = cv2.remap(np.ascontiguousarray(src_clean, np.float32), um, vm, cv2.INTER_LINEAR)
    ctrl0 = cv2.remap(np.ascontiguousarray(src_ctrl, np.float32), um, vm, cv2.INTER_LINEAR)
    sens = _apply(clean, maps_n)
    ctrl = _apply(_remove(ctrl0, maps_n), maps_n)
    e = grad_errors(tgt, {"plain": plain, "control": ctrl, "sensor": sens}, ok, gray=gray, blur=blur)
    row = {"frame": int(n), "ref": int(ref_n), "surface_px": int(ok.sum()),
           "err_plain": round(e["plain"], 6), "err_control": round(e["control"], 6),
           "err_sensor": round(e["sensor"], 6),
           "better_pct": round(100 * (e["plain"] - e["sensor"]) / e["plain"], 2),
           "better_vs_control_pct": round(100 * (e["control"] - e["sensor"]) / e["control"], 2)}
    if want_images:
        row["_images"] = {"target": tgt, "plain": plain, "control": ctrl, "sensor": sens, "mask": ok}
    return row


def summarise(rows, key="better_vs_control_pct"):
    """Per reference: median RS on vs off (key: sensor vs control by default; 'on_vs_gs_pct' for RS on vs the GS
    camera) over its targets; overall: the median of the per-reference medians and the median over all pairs."""
    rows = [r for r in rows if r.get(key) is not None]
    refs = sorted({r["ref"] for r in rows})
    per = []
    for ref in refs:
        rr = [r for r in rows if r["ref"] == ref]
        bc = np.array([r[key] for r in rr])
        bp = np.array([r["better_pct"] for r in rr])
        per.append({"ref": ref, "targets": len(rr), "on_vs_off_median_pct": float(np.median(bc)),
                    "on_better_in": int((bc > 0).sum()), "vs_plain_median_pct": float(np.median(bp)),
                    "err_control_median": float(np.median([r["err_control"] for r in rr])),
                    "err_sensor_median": float(np.median([r["err_sensor"] for r in rr]))})
    allbc = np.array([r[key] for r in rows])
    return {"per_ref": per,
            "median_of_refs_pct": float(np.median([p["on_vs_off_median_pct"] for p in per])) if per else None,
            "median_all_pairs_pct": float(np.median(allbc)) if len(allbc) else None,
            "worst_ref_pct": float(min(p["on_vs_off_median_pct"] for p in per)) if per else None,
            "pairs": int(len(rows)), "on_better_pairs": int((allbc > 0).sum())}


def acceptance(summary, *, kind="fast pan", min_gain_pct=20.0, max_ref_loss_pct=5.0):
    """Plan S5 (proposed margins). 'fast pan': RS on beats off by >= min_gain_pct (median of the references) and no
    reference is worse by more than max_ref_loss_pct. 'not worse' (the gentle deck): median of the references >= 0."""
    gain = summary["median_of_refs_pct"]
    worst = summary["worst_ref_pct"]
    if kind == "not worse":
        return {"kind": kind, "median_gain_pct": gain, "not_worse_pass": bool(gain is not None and gain >= 0.0),
                "worst_ref_pct": worst}
    return {"kind": kind, "median_gain_pct": gain, "median_gain_pass": bool(gain is not None and gain >= min_gain_pct),
            "worst_ref_pct": worst, "no_ref_worse_pass": bool(worst is not None and worst >= -max_ref_loss_pct),
            "margins": {"min_gain_pct": min_gain_pct, "max_ref_loss_pct": max_ref_loss_pct}}
