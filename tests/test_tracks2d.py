"""camera_side/tracks2d.py (Track 2D, catalogue 68): LK tracks on a synthetic texture that moves by a known
sub-pixel amount each frame; seed parsing refuses hostile widget JSON; a point that disappears stops its track.
Run: <V141 python> tests/test_tracks2d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import tracks2d as t2  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    rng = np.random.default_rng(2)
    tex = gaussian_filter(rng.random((700, 900)), 2.0)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    H, W, n = 360, 640, 20
    vel = np.array([1.37, -0.62])                       # px per frame, x and y
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    imgs = {}
    for f in range(n):
        dx, dy = vel * f
        a = map_coordinates(tex, [yy + 100 - dy, xx + 100 - dx], order=3, mode="reflect")
        imgs[100 + f] = np.clip(a * 255, 0, 255).astype(np.uint8)
    seeds = t2.parse_seeds('[{"label": "a", "frame": 110, "xy": [300.0, 180.0]},'
                           ' {"label": "b.2", "frame": 100, "xy": [120.5, 90.25]}]')
    tr = t2.track_seeds(imgs, seeds, W, H, win=31, levels=3)
    t2.validate(tr)
    worst = 0.0
    for s in seeds:
        t = tr["tracks"][s["label"]]
        for fs, (x, y) in t.items():
            exp = np.array(s["xy"]) + vel * (int(fs) - s["frame"])
            worst = max(worst, float(np.hypot(x - exp[0], y - exp[1])))
    check(all(len(tr["tracks"][s["label"]]) == n for s in seeds), "every frame tracked both ways from the seed")
    check(worst < 0.1, f"LK tracks within 0.1 px of the true motion (worst {worst:.4f} px)")

    # occlusion: the texture under point a is replaced by flat grey from frame 115 on
    occ = dict(imgs)
    for f in range(115, 100 + n):
        a = occ[f].copy()
        cx, cy = (np.array([300.0, 180.0]) + vel * (f - 110)).astype(int)
        a[cy - 40:cy + 40, cx - 40:cx + 40] = 128
        occ[f] = a
    t = t2.track_one(occ, 110, [300.0, 180.0], win=31, levels=3)
    check(max(t) < 115 and min(t) == 100, f"track stops where its point disappears (kept {min(t)}-{max(t)})")

    bad = ['not json', '{"label": "a"}', '[{"label": "../x", "frame": 1, "xy": [1, 2]}]',
           '[{"label": "a", "frame": 1, "xy": [1]}]', '[{"label": "a", "frame": 1, "xy": [1, 2]},'
           ' {"label": "a", "frame": 2, "xy": [1, 2]}]', '[{"label": "a", "frame": 1, "xy": ["nan", 2]}]']
    refused = 0
    for b in bad:
        try:
            t2.parse_seeds(b)
        except t2.TrackError:
            refused += 1
    check(refused == len(bad), f"bad seeds refused ({refused}/{len(bad)})")
    try:
        t2.track_seeds(imgs, t2.parse_seeds('[{"label": "z", "frame": 5, "xy": [1, 2]}]'), W, H)
        check(False, "seed outside the plate range refused")
    except t2.TrackError:
        check(True, "seed outside the plate range refused")
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
