"""Set Proxy: the deck plane and the wall plane for this shot, trimmed to what the camera sees, as OBJ + USDA.

Input points are in a gravity-aligned shot camera world (Measure & Scale: Y up, ground y = 0), from the solve and from
unprojected Plate Depth. Constrained RANSAC:
  deck: horizontal (normal within max_tilt_deg of +Y), inliers near deck_y (the ground of T4 is the deck for S2-S3)
  wall: vertical (normal within max_tilt_deg of horizontal), facing the camera path, the largest such plane
Each plane is trimmed to the convex hull of its inliers (in the plane), so the proxy covers what was seen.
General meshing is out of scope for Phase 1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class SetProxyError(ValueError):
    pass


def _refine(pts, inl):
    c = pts[inl].mean(0)
    n = np.linalg.svd(pts[inl] - c)[2][-1]
    return c, n / np.linalg.norm(n)


def constrained_plane(pts: np.ndarray, axis_ok, thresh: float, iters: int = 2000, seed: int = 0,
                      score=None):
    """RANSAC over planes whose normal passes axis_ok(n). score(inlier mask, centre, normal) ranks candidates
    (default: inlier count). Returns (centre, normal, inlier mask, rms)."""
    rng = np.random.default_rng(seed)
    best, best_score = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        if not axis_ok(n):
            continue
        inl = np.abs((pts - s[0]) @ n) < thresh
        sc = score(inl, s[0], n) if score else inl.sum()
        if sc > best_score:
            best, best_score = inl, sc
    if best is None or best.sum() < 3:
        raise SetProxyError("no plane with the required orientation found")
    for _ in range(3):
        c, n = _refine(pts, best)
        if not axis_ok(n):
            break
        best = np.abs((pts - c) @ n) < thresh
    c, n = _refine(pts, best)
    rms = float(np.sqrt(np.mean(((pts[best] - c) @ n) ** 2)))
    return c, n, best, rms


def _hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise."""
    p = np.unique(np.round(xy, 9), axis=0)
    if len(p) < 3:
        raise SetProxyError("too few distinct points for a plane outline")
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def outline(pts: np.ndarray, c: np.ndarray, n: np.ndarray, cell: float = 0.1, min_share: float = 0.1) -> np.ndarray:
    """3D polygon on the plane around the inliers: points are binned in cell x cell metre squares on the plane,
    squares holding under min_share of the median occupied square are dropped (stray coplanar outliers), then the
    convex hull of what is left."""
    a = np.cross(n, [0, 1.0, 0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    xy = np.stack([(pts - c) @ a, (pts - c) @ b], 1)
    if len(xy) > 50:
        ij = np.floor(xy / cell).astype(np.int64)
        _, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
        keep = counts[inv.ravel()] >= max(2, min_share * np.median(counts))
        xy = xy[keep]
    h = _hull(xy)
    return c + h[:, :1] * a + h[:, 1:2] * b


def fit_set(points: np.ndarray, camera_positions: np.ndarray, deck_y: float = 0.0, deck_band: float = 0.3,
            max_tilt_deg: float = 3.0, thresh: float = 0.02, seed: int = 0) -> dict:
    """Deck + wall planes from points (metres, Y up). thresh is the inlier distance in metres."""
    pts = np.asarray(points, float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    up = np.array([0, 1.0, 0])
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    sin_tilt = math.sin(math.radians(max_tilt_deg))

    band = np.abs(pts[:, 1] - deck_y) < deck_band
    if band.sum() < 50:
        raise SetProxyError(f"only {band.sum()} points within {deck_band} m of deck_y {deck_y}")
    dp = pts[band]
    c, n, inl, rms = constrained_plane(dp, lambda n: abs(n @ up) >= cos_tilt, thresh, seed=seed)
    n = n if n @ up > 0 else -n
    deck = {"centre": c, "normal": n, "rms": rms, "inliers": int(inl.sum()), "outline": outline(dp[inl], c, n)}

    rest = pts[np.abs((pts - c) @ n) >= 3 * thresh]
    cam_mean = np.asarray(camera_positions, float).mean(0)

    def facing(inl, c0, n0):
        to_cam = cam_mean - c0
        return inl.sum() if abs(n0 @ to_cam) / (np.linalg.norm(to_cam) + 1e-12) > 0.5 else -1

    c2, n2, inl2, rms2 = constrained_plane(rest, lambda n: abs(n @ up) <= sin_tilt, thresh, seed=seed + 1,
                                           score=facing)
    n2 = n2 if n2 @ (cam_mean - c2) > 0 else -n2
    # trim each plane by the other: deck on the camera side of the wall (the floor through the doorway is the
    # same level but is not the deck), wall above the deck
    deck_pts = dp[inl]
    deck_pts = deck_pts[(deck_pts - c2) @ n2 > -3 * thresh]
    wall_pts = rest[inl2]
    wall_pts = wall_pts[(wall_pts - c) @ n > -3 * thresh]
    deck["outline"] = outline(deck_pts, c, n)
    deck["inliers_kept"] = int(len(deck_pts))
    wall = {"centre": c2, "normal": n2, "rms": rms2, "inliers": int(inl2.sum()), "inliers_kept": int(len(wall_pts)),
            "outline": outline(wall_pts, c2, n2)}
    for p in (deck, wall):
        p["d"] = float(-(p["normal"] @ p["centre"]))
        p["outline_centre"] = p["outline"].mean(0)
    deck["height_at_centre_m"] = float(deck["outline_centre"][1])
    deck["tilt_deg"] = math.degrees(math.acos(min(1.0, deck["normal"] @ up)))
    wall["tilt_deg"] = math.degrees(math.asin(min(1.0, abs(wall["normal"] @ up))))
    return {"deck": deck, "wall": wall, "points_used": int(len(pts))}


def _fan(poly_idx):
    return [(poly_idx[0], poly_idx[i], poly_idx[i + 1]) for i in range(1, len(poly_idx) - 1)]


def _ordered(poly: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Polygon wound counter-clockwise seen from the normal side (so the face faces the camera side)."""
    c = poly.mean(0)
    area_n = sum(np.cross(poly[i] - c, poly[(i + 1) % len(poly)] - c) for i in range(len(poly)))
    return poly if area_n @ normal > 0 else poly[::-1]


def write_obj(path, planes: dict) -> Path:
    lines, base = ["# atlas-vfx set proxy: metres, Y up (shot camera world)"], 1
    for name, p in planes.items():
        poly = _ordered(p["outline"], p["normal"])
        lines.append(f"o {name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in poly]
        lines.append(f"vn {p['normal'][0]:.6f} {p['normal'][1]:.6f} {p['normal'][2]:.6f}")
        nidx = sum(1 for ln in lines if ln.startswith("vn "))
        for a, b, c in _fan(list(range(base, base + len(poly)))):
            lines.append(f"f {a}//{nidx} {b}//{nidx} {c}//{nidx}")
        base += len(poly)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_usda(path, planes: dict) -> Path:
    out = ['#usda 1.0', '(', '    defaultPrim = "set_proxy"', '    metersPerUnit = 1', '    upAxis = "Y"', ')', '',
           'def Xform "set_proxy"', '{']
    for name, p in planes.items():
        poly = _ordered(p["outline"], p["normal"])
        pts = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in poly)
        out += [f'    def Mesh "{name}"', '    {',
                f'        int[] faceVertexCounts = [{len(poly)}]',
                f'        int[] faceVertexIndices = [{", ".join(str(i) for i in range(len(poly)))}]',
                f'        point3f[] points = [{pts}]',
                '        uniform token orientation = "rightHanded"',
                '        uniform token subdivisionScheme = "none"',
                '    }']
    out.append('}')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def to_json(result: dict) -> dict:
    def conv(p):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    return {"deck": conv(result["deck"]), "wall": conv(result["wall"]), "points_used": result["points_used"],
            "plane_equation": "normal . X + d = 0, metres, Y up"}
