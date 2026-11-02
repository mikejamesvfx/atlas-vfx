"""Rotation-only (nodal) solve: when the camera turns about its lens and incremental SfM has no baseline to start
from, the camera is solved as rotations plus focal and distortion from the matches already in the database.

Each verified match (a in frame i, b in frame j) is a ray: undistort a, rotate it from camera i to camera j, project
and distort through camera j, compare with b (and the reverse). Unknowns: a rotation per frame (the first frame is
held), focal and the distortion coefficients of the solver's camera model. A pure rotation pins the focal well
(homography self-calibration), so the lens is solved here as in the SfM path.

Initialisation: a focal sweep. For each candidate focal the frame-to-frame rotations are fitted to the matched rays
(Kabsch) and chained; the focal with the lowest transfer error starts the least-squares solve.

The result is written as a COLMAP text model like the SfM path's: every camera centre at the origin, each track's
point placed along its mean ray at POINT_DISTANCE (unscaled; a nodal solve has no depth), so from_colmap and
colmap_points read it unchanged. COLMAP axes: x right, y down, z forward; images hold cam_from_world.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

POINT_DISTANCE = 100.0
MAX_MATCHES_PER_PAIR = 150


def _undistort(u, v, k):
    x, y = u.copy(), v.copy()
    for _ in range(12):
        r2 = x * x + y * y
        f = 1 + k[0] * r2 + k[1] * r2 * r2
        x, y = u / f, v / f
    return x, y


def _distort(x, y, k):
    r2 = x * x + y * y
    f = 1 + k[0] * r2 + k[1] * r2 * r2
    return x * f, y * f


def _kabsch(a, b):
    """R minimising |R a - b| over rows (unit rays)."""
    U, _s, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def read_matches(db_path, names: list[str], rng):
    """Keypoints per image (in `names` order) and the verified matches: [(i, j, idx_i, idx_j)]. The verified pairs
    come straight from the database's pair table (checking every image pair was O(n^2) calls)."""
    import sqlite3

    import pycolmap

    from .plate_solve import frame_of

    M = 2147483647                                           # COLMAP pair_id = id1 * M + id2, id1 < id2
    con = sqlite3.connect(str(db_path))
    try:
        pair_ids = [p for (p,) in con.execute("SELECT pair_id FROM two_view_geometries WHERE rows >= 15")]
    finally:
        con.close()
    db = pycolmap.Database.open(str(db_path))
    try:
        imgs = {img.name: img for img in db.read_all_images()}
        order = [imgs[n] for n in names if n in imgs]
        pos = {img.image_id: k for k, img in enumerate(order)}
        kps = [np.asarray(db.read_keypoints(img.image_id), np.float64)[:, :2] for img in order]
        cam = db.read_camera(order[0].camera_id)
        pairs = []
        for pid in sorted(pair_ids):
            a_id, b_id = pid // M, pid % M
            if a_id not in pos or b_id not in pos:
                continue
            if pos[a_id] > pos[b_id]:
                a_id, b_id = b_id, a_id
            m = np.asarray(db.read_two_view_geometry(a_id, b_id).inlier_matches, np.int64).reshape(-1, 2)
            if len(m) < 15:
                continue
            if len(m) > MAX_MATCHES_PER_PAIR:
                m = m[np.sort(rng.choice(len(m), MAX_MATCHES_PER_PAIR, replace=False))]
            pairs.append((pos[a_id], pos[b_id], m[:, 0], m[:, 1]))
    finally:
        db.close()
    pairs.sort(key=lambda p: (p[0], p[1]))
    return order, kps, cam, pairs, [frame_of(o.name) for o in order]


def _rays(kp, f, c, k):
    x, y = _undistort((kp[:, 0] - c[0]) / f, (kp[:, 1] - c[1]) / f, k)
    r = np.column_stack([x, y, np.ones_like(x)])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


class _Flat:
    """Every match of every pair in flat arrays, so a lens trial is a few vector operations, not a loop over pairs."""

    def __init__(self, kps, pairs):
        self.I = np.concatenate([np.full(len(a), i) for i, _j, a, _b in pairs])
        self.J = np.concatenate([np.full(len(a), j) for _i, j, a, _b in pairs])
        self.P = np.concatenate([np.full(len(a), p) for p, (_i, _j, a, _b) in enumerate(pairs)])
        self.XA = np.concatenate([kps[i][a] for i, _j, a, _b in pairs])
        self.XB = np.concatenate([kps[j][b] for _i, j, _a, b in pairs])
        self.n_pairs = len(pairs)
        self.pij = [(int(i), int(j)) for i, j, _a, _b in pairs]
        self.splits = np.cumsum([len(a) for _i, _j, a, _b in pairs])[:-1]
        # sums over matches as sparse matrix products (np.add.at is slow): rows = pair / frame, cols = match
        from scipy.sparse import csr_matrix

        m = len(self.I)
        cols, ones = np.arange(m), np.ones(m)
        self.SP = csr_matrix((ones, (self.P, cols)), shape=(self.n_pairs, m))
        n = int(max(self.I.max(), self.J.max())) + 1
        self.SI = csr_matrix((ones, (self.I, cols)), shape=(n, m))
        self.SJ = csr_matrix((ones, (self.J, cols)), shape=(n, m))

    def rays(self, f, c, k):
        return _rays(self.XA, f, c, k), _rays(self.XB, f, c, k)

    def per_pair(self, flat):
        return np.split(flat, self.splits)


def _kabsch_batch(H):
    """Rotations R maximising sum dst . (R src) for stacked H = sum outer(dst, src) (N, 3, 3)."""
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros_like(H)
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = 1.0, 1.0, np.where(d == 0, 1.0, d)
    return U @ D @ Vt


def _chain_flat(n, fl, RA, RB, use=None):
    """Rotations (cam_from_world, frame 0 = I) chained frame to frame from the nearest earlier frame with a fitted
    pair (one batched Kabsch over all pairs)."""
    w = np.ones(len(fl.I)) if use is None else use.astype(np.float64)
    outer = (w[:, None, None] * RB[:, :, None] * RA[:, None, :]).reshape(-1, 9)
    rel = _kabsch_batch(np.asarray(fl.SP @ outer).reshape(-1, 3, 3))
    best = {}
    for p, (i, j) in enumerate(fl.pij):
        if j not in best or i > best[j][0]:
            best[j] = (i, p)
    R = np.zeros((n, 3, 3))
    R[0] = np.eye(3)
    for j in range(1, n):
        if j in best:
            i, p = best[j]
            R[j] = rel[p] @ R[i]
        else:
            R[j] = R[j - 1]
    return R


def _transfer_flat(fl, R, f, c, k, RA, RB):
    """Per match transfer error (px), the worse of the two directions."""
    out = None
    for src, dst, rs, xt in ((fl.I, fl.J, RA, fl.XB), (fl.J, fl.I, RB, fl.XA)):
        q = np.einsum("mij,mj->mi", R[dst] @ np.transpose(R[src], (0, 2, 1)), rs)
        u, v = _distort(q[:, 0] / q[:, 2], q[:, 1] / q[:, 2], k)
        d = np.hypot(f * u + c[0] - xt[:, 0], f * v + c[1] - xt[:, 1])
        out = d if out is None else np.maximum(out, d)
    return out


def _average_flat(n, fl, RA, RB, use, R, sweeps: int = 20):
    """Rotation averaging, every frame at once per sweep (frame 0 held): each frame's rotation is the Kabsch fit of
    its rays to its neighbours' rays taken to the world through their current rotations.
    The rotations multiply in from outside, so each pair's match sum C_p = sum rb ra^T is formed once and a sweep
    works on pairs, not matches: frame j's fit is sum_p C_p R_i(p), frame i's is sum_p C_p^T R_j(p)."""
    w = use.astype(np.float64)
    C = np.asarray(fl.SP @ (w[:, None, None] * RB[:, :, None] * RA[:, None, :]).reshape(-1, 9)).reshape(-1, 3, 3)
    pi = np.array([i for i, _j in fl.pij])
    pj = np.array([j for _i, j in fl.pij])
    Ct = np.transpose(C, (0, 2, 1))
    R = np.asarray(R, np.float64).copy()
    for _ in range(sweeps):
        H = np.zeros((n, 3, 3))
        np.add.at(H, pj, C @ R[pi])
        np.add.at(H, pi, Ct @ R[pj])
        new = _kabsch_batch(H)
        has = np.abs(H).sum((1, 2)) > 0
        new[~has] = R[~has]
        new[0] = R[0]
        R = new
    return R


def _chain(n, kps, pairs, f, c, k):
    """Rotations chained frame to frame and the median transfer error under them (px)."""
    fl = _Flat(kps, pairs)
    RA, RB = fl.rays(f, c, k)
    R = _chain_flat(n, fl, RA, RB)
    return list(R), float(np.median(_transfer_flat(fl, R, f, c, k, RA, RB)))


def solve(db_path, names: list[str], *, focal_px: float | None, model: str = "RADIAL", refine_focal: bool = True,
          refine_distortion: bool = True, seed: int = 0, log=print) -> dict:
    """Lens (focal, k1[, k2]) by Nelder-Mead on the transfer error; for each lens the rotations are averaged
    exactly from the undistorted rays (a pure rotation makes rays agree up to R), so only 1-3 numbers are searched.
    All matches live in flat arrays (_Flat): a lens trial is a handful of vector operations."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    order, kps, cam, pairs, frames = read_matches(db_path, names, rng)
    n = len(order)
    if n < 2 or not pairs:
        raise ValueError("nodal solve: no verified matches")
    W, H = cam.width, cam.height
    c = np.array([W / 2.0, H / 2.0])
    nk = 1 if model == "SIMPLE_RADIAL" else 2
    fl = _Flat(kps, pairs)
    # focal and k1 trade against each other on a pan, so they are swept together (a focal-only sweep with no
    # distortion lands on the wrong side of the valley)
    if focal_px and not refine_focal:
        fs = [float(focal_px)]
    elif focal_px:
        fs = [float(focal_px) * s for s in np.geomspace(0.7, 1.4, 9)]
    else:
        fs = [W * s for s in np.geomspace(0.3, 4.0, 25)]
    k1s = np.linspace(-0.3, 0.3, 13) if refine_distortion else [0.0]

    def sweep_score(f, k1):
        k = np.array([k1, 0.0])
        RA, RB = fl.rays(f, c, k)
        R = _chain_flat(n, fl, RA, RB)
        return float(np.median(_transfer_flat(fl, R, f, c, k, RA, RB)))

    scores = sorted((sweep_score(f, k1), f, k1) for f in fs for k1 in k1s)
    starts = scores[:3]
    log("nodal: sweep starts " + ", ".join(f"f {f:.1f} k1 {k1:+.3f} ({e:.2f} px)" for e, f, k1 in starts))

    def lens(x):
        f = float(np.exp(x[0])) if refine_focal else fs[0]
        k = np.zeros(2)
        if refine_distortion:
            k[:nk] = x[1:1 + nk]
        return f, k

    state = {"use": np.ones(len(fl.I), bool), "last": None}

    def cost(x):
        f, k = lens(x)
        if not (0.1 * W < f < 20 * W) or np.any(np.abs(k) > 1.5):
            return 1e9
        RA, RB = fl.rays(f, c, k)
        # rotations from this lens's own rays: a warm start from another lens biases the cost towards that lens
        R = _average_flat(n, fl, RA, RB, state["use"], _chain_flat(n, fl, RA, RB, state["use"]), sweeps=12)
        e = _transfer_flat(fl, R, f, c, k, RA, RB)[state["use"]]
        state["last"] = R
        return float(np.mean(np.minimum(e, 3.0) ** 2))          # truncated: an outlier can't pull the lens

    def fit(f_start, k1_start):
        x0 = np.array([np.log(f_start), k1_start, 0.0][:1 + nk]) if refine_distortion else np.array([np.log(f_start)])
        res = None
        for _round in range(2):
            step = np.array([0.03, 0.03, 0.01][:len(x0)])     # log focal (3 %), k1, k2
            simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[d] * step[d] for d in range(len(x0))])
            # xatol 1e-5 in log focal is 0.001 % of the focal: far below what the data can tell apart
            res = minimize(cost, x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 400, "initial_simplex": simplex})
            x0 = res.x
            f, k = lens(x0)
            RA, RB = fl.rays(f, c, k)
            R = _average_flat(n, fl, RA, RB, state["use"], state["last"], sweeps=30)
            state["use"] = _transfer_flat(fl, R, f, c, k, RA, RB) < 3.0     # drop bad matches, refit
            state["last"] = R
        return x0, float(res.fun), state["last"], state["use"]

    best = None
    for _e, f_start, k1_start in starts:
        x, cst, R, use = fit(f_start, k1_start)
        f, k = lens(x)
        log(f"nodal: from f {f_start:.1f} k1 {k1_start:+.3f}: f {f:.2f}, k {k[:nk].round(5).tolist()}, "
            f"mean sq {cst:.4g} px2")
        if best is None or cst < best[1]:
            best = (x, cst, R, use)
        state["use"] = np.ones(len(fl.I), bool)
    x, _cst, R, use = best
    f, k = lens(x)
    RA, RB = fl.rays(f, c, k)
    e = _transfer_flat(fl, R, f, c, k, RA, RB)[use]
    rms = float(np.sqrt(np.mean(e ** 2)))
    e0, f0 = starts[0][0], starts[0][1]
    log(f"nodal: {n} frames, focal {f:.2f} px, k {k[:nk].round(5).tolist()}, rms {rms:.3f} px (start {e0:.2f} px median)")
    return {"order": order, "frames": frames, "kps": kps, "pairs": pairs, "use": fl.per_pair(use), "R": list(R),
            "f": float(f), "k": k[:nk].tolist(), "c": c.tolist(), "size": (W, H),
            "model": "SIMPLE_RADIAL" if nk == 1 else "RADIAL", "rms_px": rms, "focal_start_px": f0}


def write_model(sol: dict, out: Path) -> Path:
    """COLMAP text model (cameras, images, points3D) of a nodal solve; tracks are joined across pairs."""
    out.mkdir(parents=True, exist_ok=True)
    order, kps, R = sol["order"], sol["kps"], sol["R"]
    parent: dict = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j, a, b), keep in zip(sol["pairs"], sol["use"]):
        for ka, kb in zip(a[keep], b[keep]):
            ra, rb = find((i, int(ka))), find((j, int(kb)))
            if ra != rb:
                parent[ra] = rb
    tracks: dict = {}
    for node in list(parent):
        tracks.setdefault(find(node), []).append(node)
    f, c, k = sol["f"], sol["c"], np.array((sol["k"] + [0.0, 0.0])[:2])
    W, H = sol["size"]
    per_image: list[list] = [[] for _ in order]
    points = []
    for pid, members in enumerate((m for m in tracks.values() if len({fr for fr, _ in m}) >= 2), start=1):
        seen, dirs, els = set(), [], []
        for fr, kp in sorted(members):
            if fr in seen:
                continue                                  # one observation per image
            seen.add(fr)
            xy = kps[fr][kp]
            ray = _rays(xy[None], f, np.asarray(c), k)[0]
            dirs.append(R[fr].T @ ray)
            per_image[fr].append((xy, pid))
            els.append((fr, len(per_image[fr]) - 1))
        d = np.mean(dirs, axis=0)
        points.append((pid, d / np.linalg.norm(d) * POINT_DISTANCE, els))
    from scipy.spatial.transform import Rotation

    params = [f, c[0], c[1], *sol["k"]]
    (out / "cameras.txt").write_text(f"1 {sol['model']} {W} {H} " + " ".join(f"{v:.12g}" for v in params) + "\n",
                                     encoding="utf-8")
    lines = []
    for fr, img in enumerate(order):
        qx, qy, qz, qw = Rotation.from_matrix(R[fr]).as_quat()
        lines.append(f"{fr + 1} {qw:.15g} {qx:.15g} {qy:.15g} {qz:.15g} 0 0 0 1 {img.name}")
        lines.append(" ".join(f"{xy[0]:.4f} {xy[1]:.4f} {pid}" for xy, pid in per_image[fr]))
    (out / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pl = [f"{pid} {X[0]:.9g} {X[1]:.9g} {X[2]:.9g} 128 128 128 0 " + " ".join(f"{fr + 1} {ix}" for fr, ix in els)
          for pid, X, els in points]
    (out / "points3D.txt").write_text("\n".join(pl) + "\n", encoding="utf-8")
    return out


def to_binary(text_dir: Path, out: Path) -> Path:
    """Read the text model with pycolmap (checks it) and write it as the SfM path does."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(text_dir))
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    return out


def fov_deg(sol: dict) -> float:
    return math.degrees(2 * math.atan(sol["size"][0] / 2 / sol["f"]))
