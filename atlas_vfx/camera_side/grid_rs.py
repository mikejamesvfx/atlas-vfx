"""Lens and rolling-shutter readout from a filmed checker grid (the VFX dept 96 x 48 in chart: 32 x 16 squares of
3 in, 31 x 15 inner corners).

Why a grid. The RS bundle adjustment (rs_solve.py) measures readout from a natural plate, but there the readout
trades against the solved structure and the focal (DSCF3004: 18 ms with the solve's 23 mm, 22.9 ms locked at
16.47 mm). On a flat chart the structure is known to the millimetre, so the only unknowns are the lens (from
static frames) and per-frame poses plus one readout (from a pan): the lean of the vertical lines over the frame
height is the readout, measured on sub-pixel corners.

Pieces:
  detect(gray8)            corners of the visible part of the chart, with local (col, row) indices and the
                           checker parity of each corner (so a 180 degree index flip can be told apart)
  calibrate(views, size)   OpenCV pinhole + k1, k2 (tangential and k3 held at 0, as the Sensor Lens Maps model);
                           drops a view far worse than the rest (a bogus detection) and solves again
  index_sequence(...)      local indices -> chart indices that stay the same across a pan (absolute when the whole
                           chart is in view; otherwise predicted from the previous frames' homographies)
  fit_readout(...)         per-frame pose + one readout, points fixed on the chart, row time from the sensor
                           (distorted) row, camera_side/rs_time.py, as rs_solve / sensor rs (OpenCV px shifted +0.5)

Conventions inside this module: OpenCV camera (x right, y down, z forward); chart plane Z = 0 with X along the
columns and Y along the rows, metres. Pure numpy / scipy / OpenCV; no imports from layers/ or deliver/.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from . import rs_time

CHART = {"name": "VFX dept lens grid 96x48 in", "cols": 32, "rows": 16, "square_m": 0.0762}
MIN_PATTERN = (7, 5)            # smallest visible block of inner corners accepted (cols, rows)


# ------------------------------------------------------------------------------------------------ detection

def detect(gray8: np.ndarray, min_pattern=MIN_PATTERN):
    """Visible inner corners of the chart. Returns None or dict(xy (N,2) px, ij (N,2) local col/row ints,
    parity (N,) 0/1 = whether the cell right-below the corner is black, shape (cols, rows))."""
    import cv2

    flags = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners, meta = cv2.findChessboardCornersSBWithMeta(gray8, min_pattern, flags)
    if not ok or corners is None:
        return None
    meta = np.asarray(meta)
    rows, cols = meta.shape[:2]
    xy = corners.reshape(-1, 2).astype(float)
    if len(xy) != rows * cols:
        return None
    jj, ii = np.mgrid[0:rows, 0:cols]
    m = meta.reshape(-1)
    parity = np.where(np.isin(m, (1, 3)), 1, np.where(np.isin(m, (2, 4)), 0, -1))
    return {"xy": xy, "ij": np.column_stack([ii.ravel(), jj.ravel()]), "parity": parity, "shape": (cols, rows)}


def chart_xyz(ij, square=CHART["square_m"]) -> np.ndarray:
    ij = np.asarray(ij, float)
    return np.column_stack([ij[:, 0] * square, ij[:, 1] * square, np.zeros(len(ij))])


# ---------------------------------------------------------------------------------------------- calibration

def calibrate(views, size, focal_guess_px: float | None = None, max_view_rms_px: float = 1.0, min_views: int = 4):
    """views: list of detect() results from static (or slow) frames; size (W, H). Local indices are fine here
    (each view gets its own pose). Returns dict(K, dist [k1, k2, p1, p2, k3], rms_px, n_views, view_rms_px, dropped).

    A view whose reprojection error is over max_view_rms_px AND 4x the median of the others is dropped and the
    lens solved again, one view at a time, keeping at least min_views. OpenCV's exhaustive chessboard search is
    not deterministic: now and then it returns a bogus patch (a 7 x 5 piece of an 18 x 10 board, 2026-10-29, CI at 4
    threads), and one such view moved the focal 7 %. dropped = [(index into views, its rms px)]."""
    import cv2

    W, H = size
    obj_all = [chart_xyz(v["ij"]).astype(np.float32) for v in views]
    img_all = [v["xy"].astype(np.float32).reshape(-1, 1, 2) for v in views]
    f0 = focal_guess_px or 0.7 * W
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K3)
    keep, dropped = list(range(len(views))), []
    while True:
        K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]], float)
        obj, img = [obj_all[i] for i in keep], [img_all[i] for i in keep]
        rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, (W, H), K0, np.zeros(5), flags=flags)
        per = []
        for o, im, r, t in zip(obj, img, rv, tv):
            pr, _ = cv2.projectPoints(o, r, t, K, dist)
            per.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
        worst = int(np.argmax(per))
        others = np.median([e for k, e in enumerate(per) if k != worst]) if len(per) > 1 else per[worst]
        if len(keep) <= min_views or per[worst] <= max_view_rms_px or per[worst] <= 4 * others:
            break
        dropped.append((keep.pop(worst), per[worst]))
    return {"K": K, "dist": dist.ravel()[:5], "rms_px": float(rms), "n_views": len(keep),
            "view_rms_px": dict(zip(keep, per)), "dropped": dropped}


def undistort_px(xy, K, dist) -> np.ndarray:
    """Distorted pixels -> pinhole pixels with the same K (so rows keep their meaning)."""
    import cv2

    if dist is None or not np.any(np.asarray(dist)):
        return np.asarray(xy, float)
    u = cv2.undistortPoints(np.asarray(xy, np.float64).reshape(-1, 1, 2), K, np.asarray(dist, float), P=K)
    return u.reshape(-1, 2)


# ------------------------------------------------------------------------------------------------- indexing

def _pnp(xyz, xy, K):
    """World -> camera pose of the chart. IPPE gives both planar solutions; each is refined (iterative LM) and the
    one with the lower reprojection wins (a rolling-shutter frame can make IPPE's first pick the mirror one)."""
    import cv2

    best = None
    try:
        n, rvs, ts, _ = cv2.solvePnPGeneric(xyz, xy, K, None, flags=cv2.SOLVEPNP_IPPE)
        starts = list(zip(rvs, ts))
    except cv2.error:
        starts = []
    if not starts:
        ok, rv, t = cv2.solvePnP(xyz, xy, K, None)
        starts = [(rv, t)]
    for rv, t in starts:
        rv, t = cv2.solvePnPRefineLM(xyz, xy, K, None, rv.copy(), t.copy())
        p, _ = cv2.projectPoints(xyz, rv, t, K, None)
        e = float(np.mean(np.sum((p.reshape(-1, 2) - xy) ** 2, 1)))
        if best is None or e < best[0]:
            best = (e, rv, t)
    R = Rotation.from_rotvec(best[1].ravel()).as_matrix()          # world -> camera
    return R, best[2].ravel()


def _homography(ij, xy):
    import cv2

    Hm, _ = cv2.findHomography(np.asarray(ij, np.float64), np.asarray(xy, np.float64), 0)
    return Hm


def _apply_h(Hm, ij):
    p = np.column_stack([np.asarray(ij, float), np.ones(len(ij))]) @ Hm.T
    return p[:, :2] / p[:, 2:3]


def index_sequence(dets, K, dist=None, max_err_squares: float = 0.5, ratio: float = 0.6):
    """dets: {frame: detect() or None}. Returns {frame: (xy_pinhole (N,2), xyz (N,3), xy_sensor (N,2))} in one
    chart labelling per run of consecutive detections (xy_sensor = the detected, distorted pixels: their row is the
    row time). With the whole chart in view the labelling is absolute (flip from the image direction). Otherwise it
    is carried frame to frame in image space: every chart corner's position is predicted from
    the previous frames' homographies (constant acceleration), and the detector's local indices take the flip and
    integer shift that land them on the prediction. A frame whose best labelling is off by more than max_err_squares,
    or isn't clearly better than the runner-up (ratio), starts a new run (fit_readout gives each run its own poses, so runs never need to agree)."""
    out = {}
    hist = []                                       # [(H chart->image)] of the previous consecutive frames
    last = None
    for f in sorted(dets):
        d = dets[f]
        if d is None or (last is not None and f != last + 1):
            hist = []
        if d is None:
            last = None
            continue
        xy = undistort_px(d["xy"], K, dist)
        cols, rows = d["shape"]
        cands = []
        for flip in (False, True):
            ij = d["ij"].copy()
            if flip:
                ij = np.column_stack([cols - 1 - ij[:, 0], rows - 1 - ij[:, 1]])
            cands.append(ij)
        full = (cols, rows) == (CHART["cols"] - 1, CHART["rows"] - 1)
        if full or not hist:
            # whole chart seen: local indices are chart indices up to a 180 degree flip; take the labelling whose
            # columns run left to right in the image (any roll under 90 deg). Same rule starts a partial run.
            chosen = max(cands, key=lambda c: np.polyfit(c[:, 0], xy[:, 0], 1)[0])
            if not full:
                hist = []
        else:
            Hp = hist[-1]
            pitch = np.median(np.linalg.norm(np.diff(_apply_h(Hp, [[0, 0], [1, 0], [0, 1]]), axis=0)[:1], axis=1))
            centre = np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1))
            errs = []
            for ij in cands:
                def pred(q):                        # constant acceleration / velocity / position
                    a = _apply_h(Hp, q)
                    if len(hist) > 2:
                        return 3 * a - 3 * _apply_h(hist[-2], q) + _apply_h(hist[-3], q)
                    return 2 * a - _apply_h(hist[-2], q) if len(hist) > 1 else a
                # chart index nearest to the centre corner: search a window of shifts around a coarse estimate
                g = np.mgrid[-40:41, -25:26].reshape(2, -1).T
                pg = pred(g)
                k = np.argmin(np.linalg.norm(pg - xy[centre], axis=1))
                shift = g[k] - ij[centre]
                for ds in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    s = shift + np.array(ds)
                    errs.append((float(np.median(np.linalg.norm(pred(ij + s) - xy, axis=1))), ij + s))
            errs.sort(key=lambda e: e[0])
            best = errs[0]
            if best[0] > max_err_squares * pitch or best[0] > ratio * errs[1][0]:
                hist = []                           # lost the labelling: start a new run here
                chosen = max(cands, key=lambda c: np.polyfit(c[:, 0], xy[:, 0], 1)[0])
            else:
                chosen = best[1]
        out[f] = (xy, chart_xyz(chosen), d["xy"])
        hist = (hist + [_homography(chosen, xy)])[-3:]
        last = f
    return out


# ------------------------------------------------------------------------------------------- readout fit

def _segments(frames):
    segs, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            segs.append(cur)
            cur = [b]
    segs.append(cur)
    return [s for s in segs if len(s) >= 3]


def fit_readout(obs, K, H, *, readout_start: float = 0.3, solve_readout: bool = True, soft_px: float = 1.0,
                fps: float = 24.0):
    """obs: index_sequence() output. Per-frame poses (contiguous runs, >= 3 frames) + one readout r (frames),
    chart points fixed. Pose at row time: rotation slerped and centre lerped between whole frames, as rs_solve.
    Returns dict(readout_frames, readout_ms, sigma_ms, rms_px gs/rs, n_frames, n_obs, per-frame residuals)."""
    frames = sorted(obs)
    segs = _segments(frames)
    if not segs:
        raise ValueError("readout fit needs >= 3 consecutive frames with the chart detected")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    allf = [f for s in segs for f in s]
    idx = {f: i for i, f in enumerate(allf)}
    seg_of = np.concatenate([[k] * len(s) for k, s in enumerate(segs)])
    R0, C0 = [], []
    for f in allf:
        xy, xyz = obs[f][:2]
        R, t = _pnp(xyz, xy, K)
        R0.append(R)
        C0.append(-R.T @ t)
    R0, C0 = np.array(R0), np.array(C0)

    fi, X, uv, vs = [], [], [], []
    for f in allf:
        xy, xyz = obs[f][:2]
        fi.append(np.full(len(xy), idx[f]))
        X.append(xyz)
        uv.append(xy)
        vs.append(obs[f][2][:, 1] if len(obs[f]) > 2 else xy[:, 1])
    fi, X, uv, vs = np.concatenate(fi), np.concatenate(X), np.concatenate(uv), np.concatenate(vs)
    rowfrac = rs_time.row_frac(vs, H, pixels="opencv")     # sensor row (distorted, OpenCV px) sets the time
    # neighbour to interpolate towards, inside the frame's own segment
    seg_start = {k: idx[s[0]] for k, s in enumerate(segs)}
    seg_end = {k: idx[s[-1]] for k, s in enumerate(segs)}
    lo = np.array([seg_start[seg_of[i]] for i in fi])
    hi = np.array([seg_end[seg_of[i]] for i in fi])
    i0 = rs_time.pair_index(fi, rowfrac, lo, hi)
    n_f, n_o = len(allf), len(fi)

    def unpack(x):
        r = x[0] if solve_readout else readout_start
        p = x[1 if solve_readout else 0:].reshape(n_f, 6)
        R = Rotation.from_rotvec(p[:, :3]).as_matrix() @ R0     # world -> camera
        C = C0 + p[:, 3:]
        return r, R, C

    def project(x, r=None):
        rr, R, C = unpack(x)
        r = rr if r is None else r
        a = (fi - i0) + rowfrac * r
        Rt = rs_time.interp_rot(R, rs_time.rel_rotvecs(R), i0, a)     # geodesic: same for world->camera R
        Ct = rs_time.interp_centre(C, i0, a)
        Xc = np.einsum("nij,nj->ni", Rt, X - Ct)
        return np.column_stack([cx + fx * Xc[:, 0] / Xc[:, 2], cy + fy * Xc[:, 1] / Xc[:, 2]])

    def fun(x):
        return (project(x) - uv).ravel()

    n_par = (1 if solve_readout else 0) + 6 * n_f
    sp = lil_matrix((2 * n_o, n_par), dtype=int)
    o = 1 if solve_readout else 0
    for k in range(n_o):
        for row in (2 * k, 2 * k + 1):
            if solve_readout:
                sp[row, 0] = 1
            for fr in (i0[k], i0[k] + 1, fi[k]):
                sp[row, o + 6 * fr:o + 6 * fr + 6] = 1
    x0 = np.zeros(n_par)
    if solve_readout:
        x0[0] = readout_start
    rms0 = float(np.sqrt(np.mean(np.sum((project(x0, 0.0) - uv) ** 2, 1))))
    lo, hi = np.full(n_par, -np.inf), np.full(n_par, np.inf)
    if solve_readout:
        lo[0], hi[0] = 0.0, 1.9                    # 0 .. 79 ms at 24p: any real sensor, and keeps the slerp pairing
    sol = least_squares(fun, x0, jac_sparsity=sp, loss="soft_l1", f_scale=soft_px, method="trf", x_scale="jac",
                        bounds=(lo, hi), max_nfev=200, xtol=1e-12, ftol=1e-12)
    res = sol.fun.reshape(-1, 2)
    e = np.linalg.norm(res, axis=1)
    out = {"n_frames": n_f, "n_obs": int(n_o), "segments": [[s[0], s[-1]] for s in segs],
           "rms_px_start": rms0, "rms_px": float(np.sqrt(np.mean(e ** 2))), "median_px": float(np.median(e)),
           "nfev": int(sol.nfev), "message": sol.message}
    if solve_readout:
        r = float(sol.x[0])
        J = sol.jac
        JtJ = (J.T @ J).toarray() if hasattr(J, "toarray") else J.T @ J
        sig2 = float(np.sum(res ** 2) / max(1, 2 * n_o - n_par))
        # readout variance from the Schur complement (poses marginalised); no motion -> not observable -> inf
        a00, b, A = JtJ[0, 0], JtJ[1:, 0], JtJ[1:, 1:]
        try:
            s00 = a00 - b @ np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            s00 = 0.0
        sig = float(np.sqrt(sig2 / s00)) if s00 > 1e-9 * max(a00, 1e-30) else float("inf")
        out.update({"readout_frames": r, "readout_ms": r / fps * 1000, "sigma_ms": sig / fps * 1000})
    # how fast the camera turned (what makes the readout measurable)
    _, R, _ = unpack(sol.x)
    rot = [np.degrees(Rotation.from_matrix(R[i + 1] @ R[i].T).magnitude()) for i in range(n_f - 1)
           if seg_of[i] == seg_of[i + 1]]
    out["rot_deg_per_frame_median"] = float(np.median(rot)) if rot else 0.0
    return out
