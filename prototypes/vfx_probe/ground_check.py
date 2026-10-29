"""Independent check of the door scale: fit the lawn plane from 3D points seen in the lower part of S1 frames, then the
camera's height above it (handheld should be ~1.3-1.7 m) and the deck's height above it (3 steps up)."""
import os
import numpy as np
import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
rec = pycolmap.Reconstruction(os.path.join(HERE, "solve_s1s3_track", "sparse", "1"))
by_frame = {int(i.name[6:10]): i for i in rec.images.values()}
SCALE = 2.08 / 1.234  # m per unit, from the f0 door jamb (2.08 m frame assumed)


def C(f):
    p = by_frame[f].cam_from_world() if callable(getattr(by_frame[f], "cam_from_world", None)) else by_frame[f].cam_from_world
    R, t = p.rotation.matrix(), np.asarray(p.translation)
    return -R.T @ t


ids = set()
for f in range(0, 100, 5):
    for p in by_frame[f].points2D:
        if p.has_point3D() and p.xy[1] > 420:  # lawn in the bottom of the frame
            ids.add(p.point3D_id)
pts = np.array([rec.points3D[i].xyz for i in ids])
keep = np.ones(len(pts), bool)
for _ in range(4):
    c = pts[keep].mean(0); n = np.linalg.svd(pts[keep] - c)[2][-1]
    d = np.abs((pts - c) @ n); keep = d < max(np.median(d[keep]) * 2.5, 1e-6)
if (C(0) - c) @ n < 0:
    n = -n
rms = float(np.sqrt(np.mean(((pts[keep] - c) @ n) ** 2)))
print(f"lawn plane from {keep.sum()}/{len(pts)} points, rms {rms * SCALE * 100:.1f} cm")
for f in (0, 20, 40, 60, 80, 100, 120):
    print(f"  f{f}: camera {((C(f) - c) @ n) * SCALE:.2f} m above lawn")
# door threshold (deck) height above lawn, from the f0 ray-plane hit used in door_scale3 (bottom pick)
cam = next(iter(rec.cameras.values()))
p = by_frame[0].cam_from_world() if callable(getattr(by_frame[0], "cam_from_world", None)) else by_frame[0].cam_from_world
R, t = p.rotation.matrix(), np.asarray(p.translation)
for name, xy, dist_u in (("threshold", (538.7, 253.0), None), ("door head", (538.7, 81.3), None)):
    nn = np.asarray(cam.cam_from_img(np.array([xy], float))[0]); v = R.T @ np.array([nn[0], nn[1], 1.0]); v /= np.linalg.norm(v)
    # place it at the f0 door distance found by door_scale3 (9.32 units along the ray from the camera)
    X = C(0) + v * 9.32 / np.linalg.norm(v)
    print(f"  {name}: {((X - c) @ n) * SCALE:.2f} m above lawn")
print("  angle between lawn normal and door vertical (f0 picks):", end=" ")
