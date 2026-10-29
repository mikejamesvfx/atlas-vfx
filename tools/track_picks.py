"""Turn single manual picks into multi-frame picks by tracking them with pyramidal Lucas-Kanade.

Each seed is (label, frame, x, y) in plate pixels (top-left origin). The point is tracked frame by frame forwards and
backwards from its seed frame on CLAHE'd log-luma at half resolution; a frame is kept only if the forward-backward
error is under fb_px (plate pixels). Writes {label: {frame: [x, y]}} JSON (top-left plate pixels).

Run: <V141 python> tools\\track_picks.py seeds.json out.json [--span 24] [--every 4]
seeds.json: [{"label": "rail_b", "frame": 1095, "xy": [3266, 549]}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_vfx.camera_side import plate as pl  # noqa: E402

PLATE = r"C:\Users\miike\Videos\atlas-vfx_footage\DSCF3000.MOV"
LK = dict(winSize=(41, 41), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))


def frames_luma(ref, f0, f1, scale):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    out = {}
    W = int(ref["width"] * scale)
    for i, y16 in pl.decode_luma16(ref, f0 - ref["frame_offset"], f1 - ref["frame_offset"], width=W):
        out[i + ref["frame_offset"]] = (clahe.apply(y16) >> 8).astype(np.uint8)
    return out


def track(imgs, seed_frame, xy, frames, scale, fb_px):
    """Sequential LK from the seed outwards; returns {frame: (x, y)} plate px, FB-checked."""
    res = {seed_frame: tuple(xy)}
    for direction in (1, -1):
        p = np.array([[xy]], np.float32) * scale
        f = seed_frame
        while True:
            g = f + direction
            if g not in imgs or g not in frames:
                break
            q, st, _ = cv2.calcOpticalFlowPyrLK(imgs[f], imgs[g], p, None, **LK)
            if not st[0, 0]:
                break
            back, st2, _ = cv2.calcOpticalFlowPyrLK(imgs[g], imgs[f], q, None, **LK)
            if not st2[0, 0] or np.linalg.norm(back - p) / scale > fb_px:
                break
            res[g] = tuple((q[0, 0] / scale).tolist())
            p, f = q, g
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("out")
    ap.add_argument("--span", type=int, default=24)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--fb", type=float, default=1.0)
    a = ap.parse_args()
    seeds = json.loads(Path(a.seeds).read_text())
    ref = pl.plate_ref(PLATE, 0, 527)
    lo = max(1001, min(s["frame"] for s in seeds) - a.span)
    hi = min(1528, max(s["frame"] for s in seeds) + a.span)
    imgs = frames_luma(ref, lo, hi, a.scale)
    out = {}
    for s in seeds:
        rng = set(range(s["frame"] - a.span, s["frame"] + a.span + 1))
        t = track(imgs, s["frame"], s["xy"], rng, a.scale, a.fb)
        keep = {f: xy for f, xy in t.items() if (f - s["frame"]) % a.every == 0}
        out[s["label"]] = {str(f): [round(xy[0], 2), round(xy[1], 2)] for f, xy in sorted(keep.items())}
        print(f"{s['label']}: tracked {min(t)}-{max(t)} ({len(t)} frames), kept {len(keep)}")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
