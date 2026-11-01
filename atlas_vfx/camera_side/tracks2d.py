"""Manual 2D tracks: pyramidal Lucas-Kanade from seed points, forward-backward checked (ported from
tools/track_picks.py, with the plate path and frame range coming from the caller).

A track set is JSON-safe and travels as ATLAS_TRACKS_2D:
    {"kind": "atlas.tracks_2d", "version": 1, "width": plate px, "height": plate px,
     "tracks": {label: {frame index (str): [x, y]}}}          # plate pixels, top-left origin, frame = plate index
Solve Plate adds them to the COLMAP database as extra keypoints matched between consecutive frames
(add_track_matches), so a point the artist cares about is in the solve even where SIFT found nothing.
"""
from __future__ import annotations

import json
import re

import numpy as np

LK_DEFAULT = {"win": 41, "levels": 4, "iters": 50, "eps": 0.01}
_LABEL = re.compile(r"^[\w.\-]{1,64}$")


class TrackError(ValueError):
    pass


def parse_seeds(text: str) -> list[dict]:
    """Widget JSON [{"label": "rail_b", "frame": 95, "xy": [3266, 549]}, ...]; frame = plate index. Untrusted."""
    try:
        seeds = json.loads(text or "[]")
    except json.JSONDecodeError as e:
        raise TrackError(f"seeds are not JSON: {e}") from e
    if not isinstance(seeds, list) or len(seeds) > 500:
        raise TrackError("seeds must be a list of at most 500 {label, frame, xy}")
    out = []
    for i, s in enumerate(seeds):
        if not isinstance(s, dict):
            raise TrackError(f"seed {i} is not an object")
        label = str(s.get("label", f"t{i}"))
        if not _LABEL.match(label):
            raise TrackError(f"seed {i}: label {label!r} (letters, digits, _ . - only)")
        try:
            frame = int(s["frame"])
            x, y = (float(v) for v in s["xy"])
        except (KeyError, TypeError, ValueError) as e:
            raise TrackError(f"seed {i} needs frame and xy [x, y]") from e
        if not (np.isfinite(x) and np.isfinite(y)):
            raise TrackError(f"seed {i}: xy not finite")
        out.append({"label": label, "frame": frame, "xy": [x, y]})
    if len({s["label"] for s in out}) != len(out):
        raise TrackError("seed labels must be unique")
    return out


def _lk_params(win: int, levels: int, iters: int, eps: float) -> dict:
    import cv2

    return dict(winSize=(int(win), int(win)), maxLevel=int(levels),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps)))


def track_one(imgs: dict, seed_frame: int, xy, scale: float = 1.0, fb_px: float = 1.0, span: int | None = None,
              **lk) -> dict:
    """Sequential LK outwards from the seed. imgs: {frame: uint8 image at `scale` x plate}. Returns {frame: (x, y)}
    in plate px; a frame is kept while the forward-backward error stays under fb_px (plate px)."""
    import cv2

    p_lk = _lk_params(**{**LK_DEFAULT, **lk})
    res = {seed_frame: (float(xy[0]), float(xy[1]))}
    for direction in (1, -1):
        p = np.array([[xy]], np.float32) * scale
        f = seed_frame
        while True:
            g = f + direction
            if g not in imgs or (span is not None and abs(g - seed_frame) > span):
                break
            q, st, _ = cv2.calcOpticalFlowPyrLK(imgs[f], imgs[g], p, None, **p_lk)
            if not st[0, 0]:
                break
            back, st2, _ = cv2.calcOpticalFlowPyrLK(imgs[g], imgs[f], q, None, **p_lk)
            if not st2[0, 0] or np.linalg.norm(back - p) / scale > fb_px:
                break
            h, w = imgs[g].shape[:2]
            if not (0 <= q[0, 0, 0] < w and 0 <= q[0, 0, 1] < h):
                break
            res[g] = (float(q[0, 0, 0] / scale), float(q[0, 0, 1] / scale))
            p, f = q, g
    return res


def track_seeds(imgs: dict, seeds: list[dict], width: int, height: int, scale: float = 1.0, fb_px: float = 1.0,
                span: int | None = None, **lk) -> dict:
    tracks = {}
    for s in seeds:
        if s["frame"] not in imgs:
            raise TrackError(f"seed {s['label']}: frame {s['frame']} is not in the plate range")
        t = track_one(imgs, s["frame"], s["xy"], scale, fb_px, span, **lk)
        tracks[s["label"]] = {str(f): [round(x, 3), round(y, 3)] for f, (x, y) in sorted(t.items())}
    return {"kind": "atlas.tracks_2d", "version": 1, "width": int(width), "height": int(height), "tracks": tracks}


def validate(tr: dict) -> dict:
    if not isinstance(tr, dict) or tr.get("kind") != "atlas.tracks_2d":
        raise TrackError("not an ATLAS_TRACKS_2D payload")
    for label, t in tr.get("tracks", {}).items():
        if not _LABEL.match(str(label)) or not isinstance(t, dict):
            raise TrackError(f"bad track {label!r}")
    return tr


# --------------------------------------------------------------------------------------------- into the solve

def add_track_matches(db_path, names: list[str], tracks: dict, image_width: int | None = None) -> int:
    """Append each track's points as keypoints of their images and match them between consecutive tracked frames
    that the sequential matcher already verified (the pair's geometry stays; only inliers are added). Points are
    scaled from plate px to the database's image size. Returns the number of matches added."""
    import pycolmap

    from .plate_solve import frame_of

    validate(tracks)
    db = pycolmap.Database.open(str(db_path))
    try:
        by_frame = {}
        for img in db.read_all_images():
            by_frame[frame_of(img.name)] = img
        cam = db.read_camera(next(iter(by_frame.values())).camera_id)
        s = cam.width / float(tracks["width"])
        added_kp: dict[int, dict[str, int]] = {}         # image_id -> {label: keypoint index}
        kps = {}
        for label, t in tracks["tracks"].items():
            for f_str, xy in t.items():
                img = by_frame.get(int(f_str))
                if img is None:
                    continue
                if img.image_id not in kps:
                    kps[img.image_id] = [np.asarray(db.read_keypoints(img.image_id), np.float32)]
                    added_kp[img.image_id] = {}
                cur = sum(len(a) for a in kps[img.image_id])
                row = np.zeros((1, kps[img.image_id][0].shape[1]), np.float32)
                row[0, 0], row[0, 1] = xy[0] * s, xy[1] * s
                if row.shape[1] >= 6:
                    row[0, 2], row[0, 5] = 1.0, 1.0                  # identity affine shape
                elif row.shape[1] == 4:
                    row[0, 2] = 1.0                                  # scale 1, orientation 0
                kps[img.image_id].append(row)
                added_kp[img.image_id][label] = cur
        for iid, parts in kps.items():
            db.update_keypoints(iid, np.vstack(parts))
        n = 0
        for label, t in tracks["tracks"].items():
            fr = sorted(int(f) for f in t if int(f) in by_frame)
            for a, b in zip(fr, fr[1:]):
                ia, ib = by_frame[a].image_id, by_frame[b].image_id
                if not db.exists_two_view_geometry(ia, ib):
                    continue
                tvg = db.read_two_view_geometry(ia, ib)
                m = np.asarray(tvg.inlier_matches, np.uint32).reshape(-1, 2)
                # pycolmap reads and writes the pair in the order asked (it swaps to its canonical order itself)
                pair = [added_kp[ia][label], added_kp[ib][label]]
                tvg.inlier_matches = np.vstack([m, np.array([pair], np.uint32)])
                db.update_two_view_geometry(ia, ib, tvg)
                n += 1
    finally:
        db.close()
    return n
