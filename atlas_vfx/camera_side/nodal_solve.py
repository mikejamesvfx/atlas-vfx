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
    """Keypoints per image (in `names` order) and the verified sequential matches: [(i, j, idx_i, idx_j)]."""
    import pycolmap

    from .plate_solve import frame_of

    db = pycolmap.Database.open(str(db_path))
    try:
        imgs = {img.name: img for img in db.read_all_images()}
        order = [imgs[n] for n in names if n in imgs]
        pos = {img.image_id: k for k, img in enumerate(order)}
        kps = [np.asarray(db.read_keypoints(img.image_id), np.float64)[:, :2] for img in order]
        cam = db.read_camera(order[0].camera_id)
        pairs = []
        for a in order:
            for b in order:
                if pos[b.image_id] <= pos[a.image_id] or not db.exists_two_view_geometry(a.image_id, b.image_id):
                    continue
                m = np.asarray(db.read_two_view_geometry(a.image_id, b.image_id).inlier_matches, np.int64)
                m = m.reshape(-1, 2)
                if len(m) < 15:
                    continue
                if len(m) > MAX_MATCHES_PER_PAIR:
                    m = m[np.sort(rng.choice(len(m), MAX_MATCHES_PER_PAIR, replace=False))]
                pairs.append((pos[a.image_id], pos[b.image_id], m[:, 0], m[:, 1]))
    finally:
        db.close()
    return order, kps, cam, pairs, [frame_of(o.name) for o in order]


def _rays(kp, f, c, k):
    x, y = _undistort((kp[:, 0] - c[0]) / f, (kp[:, 1] - c[1]) / f, k)
    r = np.column_stack([x, y, np.ones_like(x)])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _chain(n, kps, pairs, f, c, k):
    """Rotations (cam_from_world, frame 0 = I) chained from the best-connected pair to each next frame, and the
    mean transfer error of all pairs under them (px)."""
    rel = {}
    for i, j, a, b in pairs:
        if j == i + 1 or (i, j) not in rel:
            rel[(i, j)] = _kabsch(_rays(kps[i][a], f, c, k), _rays(kps[j][b], f, c, k))
    R = [np.eye(3)]
    for j in range(1, n):
        cands = [i for i in range(j - 1, -1, -1) if (i, j) in rel]
        R.append(rel[(cands[0], j)] @ R[cands[0]] if cands else R[-1].copy())
    err = []
    for i, j, a, b in pairs:
        q = (R[j] @ R[i].T @ _rays(kps[i][a], f, c, k).T).T
        u, v = _distort(q[:, 0] / q[:, 2], q[:, 1] / q[:, 2], k)
        err.append(np.hypot(f * u + c[0] - kps[j][b][:, 0], f * v + c[1] - kps[j][b][:, 1]))
    return R, float(np.median(np.concatenate(err))) if err else float("inf")


def _average_rotations(n, rays, pairs, use, R, sweeps: int = 12):
    """Gauss-Seidel rotation averaging: each frame's rotation (frame 0 held) is the Kabsch fit of its rays to its
    neighbours' rays taken to the world through their current rotations. rays[(i, j)] = (rays in i, rays in j)."""
    by_frame = {j: [] for j in range(n)}
    for (i, j, _a, _b), keep in zip(pairs, use):
        ri, rj = rays[(i, j)]
        by_frame[j].append((i, ri[keep], rj[keep]))       # world ray R_i^T ri should map to rj via R_j
        by_frame[i].append((j, rj[keep], ri[keep]))
    R = [r.copy() for r in R]
    for _ in range(sweeps):
        for j in range(1, n):
            if not by_frame[j]:
                continue
            src = np.vstack([(R[i].T @ ri.T).T for i, ri, _rj in by_frame[j]])
            dst = np.vstack([rj for _i, _ri, rj in by_frame[j]])
            R[j] = _kabsch(src, dst)
    return R


def _transfer_px(pairs, use, kps, R, f, c, k):
    """Per match transfer error (px), the worse of the two directions, per pair."""
    out = []
    for (i, j, a, b), keep in zip(pairs, use):
        e = None
        for s, t, ia, ib in ((i, j, a[keep], b[keep]), (j, i, b[keep], a[keep])):
            q = (R[t] @ R[s].T @ _rays(kps[s][ia], f, c, k).T).T
            u, v = _distort(q[:, 0] / q[:, 2], q[:, 1] / q[:, 2], k)
            d = np.hypot(f * u + c[0] - kps[t][ib][:, 0], f * v + c[1] - kps[t][ib][:, 1])
            e = d if e is None else np.maximum(e, d)
        out.append(e)
    return out


def solve(db_path, names: list[str], *, focal_px: float | None, model: str = "RADIAL", refine_focal: bool = True,
          refine_distortion: bool = True, seed: int = 0, log=print) -> dict:
    """Lens (focal, k1[, k2]) by Nelder-Mead on the transfer error; for each lens the rotations are averaged
    exactly from the undistorted rays (a pure rotation makes rays agree up to R), so only 1-3 numbers are searched."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    order, kps, cam, pairs, frames = read_matches(db_path, names, rng)
    n = len(order)
    if n < 2 or not pairs:
        raise ValueError("nodal solve: no verified matches")
    W, H = cam.width, cam.height
    c = np.array([W / 2.0, H / 2.0])
    nk = 1 if model == "SIMPLE_RADIAL" else 2
    # focal and k1 trade against each other on a pan, so they are swept together (a focal-only sweep with no
    # distortion lands on the wrong side of the valley)
    if focal_px and not refine_focal:
        fs = [float(focal_px)]
    elif focal_px:
        fs = [float(focal_px) * s for s in np.geomspace(0.7, 1.4, 9)]
    else:
        fs = [W * s for s in np.geomspace(0.3, 4.0, 25)]
    k1s = np.linspace(-0.3, 0.3, 13) if refine_distortion else [0.0]
    scores = sorted((_chain(n, kps, pairs, f, c, np.array([k1, 0.0]))[1], f, k1) for f in fs for k1 in k1s)
    starts = scores[:3]
    log("nodal: sweep starts " + ", ".join(f"f {f:.1f} k1 {k1:+.3f} ({e:.2f} px)" for e, f, k1 in starts))

    def lens(x):
        f = float(np.exp(x[0])) if refine_focal else fs[0]
        k = np.zeros(2)
        if refine_distortion:
            k[:nk] = x[1:1 + nk]
        return f, k

    state = {"R": None, "use": [np.ones(len(a), bool) for _i, _j, a, _b in pairs]}

    def cost(x):
        f, k = lens(x)
        if not (0.1 * W < f < 20 * W) or np.any(np.abs(k) > 1.5):
            return 1e9
        rays = {(i, j): (_rays(kps[i][a], f, c, k), _rays(kps[j][b], f, c, k)) for i, j, a, b in pairs}
        # rotations from this lens's own rays: a warm start from another lens converges slowly along the shot
        # (Gauss-Seidel diffuses) and biases the cost towards that lens
        R = _average_rotations(n, rays, pairs, state["use"], _chain(n, kps, pairs, f, c, k)[0], sweeps=8)
        e = np.concatenate(_transfer_px(pairs, state["use"], kps, R, f, c, k))
        state["last"] = R
        return float(np.mean(np.minimum(e, 3.0) ** 2))          # truncated: an outlier can't pull the lens

    def fit(f_start, k1_start):
        x0 = np.array([np.log(f_start), k1_start, 0.0][:1 + nk]) if refine_distortion else np.array([np.log(f_start)])
        state["R"] = _chain(n, kps, pairs, f_start, c, np.array([k1_start, 0.0]))[0]
        for _round in range(2):
            step = np.array([0.03, 0.03, 0.01][:len(x0)])     # log focal (3 %), k1, k2
            simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[d] * step[d] for d in range(len(x0))])
            res = minimize(cost, x0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 600, "initial_simplex": simplex})
            x0 = res.x
            f, k = lens(x0)
            rays = {(i, j): (_rays(kps[i][a], f, c, k), _rays(kps[j][b], f, c, k)) for i, j, a, b in pairs}
            state["R"] = _average_rotations(n, rays, pairs, state["use"], state["last"], sweeps=20)
            e = _transfer_px(pairs, [np.ones(len(a), bool) for _i, _j, a, _b in pairs], kps, state["R"], f, c, k)
            state["use"] = [ei < 3.0 for ei in e]                 # drop bad matches, refit
        return x0, float(res.fun), state["R"], state["use"]

    best = None
    for _e, f_start, k1_start in starts:
        x, cst, R, use = fit(f_start, k1_start)
        f, k = lens(x)
        log(f"nodal: from f {f_start:.1f} k1 {k1_start:+.3f}: f {f:.2f}, k {k[:nk].round(5).tolist()}, "
            f"mean sq {cst:.4g} px2")
        if best is None or cst < best[1]:
            best = (x, cst, R, use)
        state["use"] = [np.ones(len(a), bool) for _i, _j, a, _b in pairs]
    x, _cst, R, use = best
    f, k = lens(x)
    e = np.concatenate(_transfer_px(pairs, use, kps, R, f, c, k))
    rms = float(np.sqrt(np.mean(e ** 2)))
    e0, f0 = starts[0][0], starts[0][1]
    log(f"nodal: {n} frames, focal {f:.2f} px, k {k[:nk].round(5).tolist()}, rms {rms:.3f} px (start {e0:.2f} px median)")
    return {"order": order, "frames": frames, "kps": kps, "pairs": pairs, "use": use, "R": R, "f": float(f),
            "k": k[:nk].tolist(), "c": c.tolist(), "size": (W, H), "model": "SIMPLE_RADIAL" if nk == 1 else "RADIAL",
            "rms_px": rms, "focal_start_px": f0}


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
