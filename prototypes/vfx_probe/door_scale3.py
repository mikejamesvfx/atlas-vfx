"""Door scale v3: the jamb itself has no SIFT points (a straight edge), so fit the clubhouse WALL plane to the solve's
3D points on the weatherboards beside the door (frame F, region WALL), then intersect the rays through the picked jamb
top and bottom with that plane. The door sits within a few cm of the wall plane, so the error that adds is small.
Done independently in frames 0, 40 and 60 as a consistency check."""
import os
import numpy as np
import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
SOLVE = sys.argv[1] if len(sys.argv) > 1 else "solve_s1s3_track"
_sp = os.path.join(HERE, SOLVE, "sparse")
rec = max((pycolmap.Reconstruction(os.path.join(_sp, m)) for m in os.listdir(_sp)), key=lambda r: r.num_reg_images())
print("solve", SOLVE, "| registered", rec.num_reg_images(), "| camera", list(map(float, next(iter(rec.cameras.values())).params)))
cam = next(iter(rec.cameras.values()))
by_frame = {int(i.name[6:10]): i for i in rec.images.values()}
# frame: (jamb top, jamb bottom, wall region x0,y0,x1,y1 = weatherboards right of the door, above the deck)
PICKS = {0: ((538.7, 81.3), (538.7, 253.0), (560, 60, 700, 240)),
         40: ((534.7, 71.7), (534.7, 256.7), (548, 40, 700, 250)),
         60: ((547.7, 63.7), (547.7, 258.7), (560, 30, 720, 250))}


def pose(img):
    p = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    return p.rotation.matrix(), np.asarray(p.translation)


def ray(f, xy):
    R, t = pose(by_frame[f])
    n = np.asarray(cam.cam_from_img(np.array([xy], float))[0], float)
    d = R.T @ np.array([n[0], n[1], 1.0])
    return -R.T @ t, d / np.linalg.norm(d)


res = []
for f, (tp, bp, (x0, y0, x1, y1)) in PICKS.items():
    img = by_frame[f]
    pts = np.array([rec.points3D[p.point3D_id].xyz for p in img.points2D
                    if p.has_point3D() and x0 <= p.xy[0] <= x1 and y0 <= p.xy[1] <= y1])
    if len(pts) < 15:
        print(f"f{f}: {len(pts)} wall points, skipped"); continue
    keep = np.ones(len(pts), bool)
    for _ in range(3):  # plane fit with outlier trimming
        c = pts[keep].mean(0)
        n = np.linalg.svd(pts[keep] - c)[2][-1]
        d = np.abs((pts - c) @ n)
        keep = d < max(np.median(d[keep]) * 3, 1e-6)
    rms = float(np.sqrt(np.mean(((pts[keep] - c) @ n) ** 2)))
    hits = []
    for xy in (tp, bp):
        o, v = ray(f, xy)
        hits.append(o + v * ((c - o) @ n) / (v @ n))
    top, bot = hits
    h = float(np.linalg.norm(top - bot))
    C = -pose(img)[0].T @ pose(img)[1]
    up = (top - bot) / h
    dist = float(np.linalg.norm(C - (top + bot) / 2))
    res.append((f, h, float((C - bot) @ up), dist))
    print(f"f{f}: {keep.sum()}/{len(pts)} wall pts, plane rms {rms:.4f} u, jamb {h:.4f} u, camera {float((C - bot) @ up):+.3f} u above threshold, {dist:.2f} u from door")
if res:
    hs = np.array([r[1] for r in res])
    print(f"jamb: median {np.median(hs):.4f} u, spread {hs.min():.4f}-{hs.max():.4f} ({(hs.max() - hs.min()) / np.median(hs) * 100:.1f}%)")
    for H in (2.04, 2.08, 2.10, 2.14):
        s = H / np.median(hs)
        print(f"  frame {H:.2f} m -> {s:.3f} m/unit | camera above deck " + ", ".join(f"f{r[0]} {r[2] * s:+.2f} m" for r in res) +
              " | distance to door " + ", ".join(f"f{r[0]} {r[3] * s:.1f} m" for r in res))
