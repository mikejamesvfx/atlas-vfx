"""Visual track check: a gridded card locked to the clubhouse wall (plane fitted to the solve's points on the weatherboards
beside the door in frame 0), projected through every solved frame onto the ACES SDR proxies, plus the door jamb line.
If the solve sticks, the card stays glued to the boards as the camera walks in. Writes card_check_<model>.mp4.
usage: card_check.py [solve_dir] (default solve_s1s3_track, model 1)"""
import os, subprocess, sys
import numpy as np
import pycolmap
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVE = sys.argv[1] if len(sys.argv) > 1 else "solve_s1s3_track"
sp = os.path.join(HERE, SOLVE, "sparse")
model = max(os.listdir(sp), key=lambda m: len(os.listdir(os.path.join(sp, m))))
best = max((pycolmap.Reconstruction(os.path.join(sp, m)) for m in os.listdir(sp)), key=lambda r: r.num_reg_images())
rec = best
cam = next(iter(rec.cameras.values()))
by = {int(i.name[6:10]): i for i in rec.images.values()}


def pose(img):
    p = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    return p.rotation.matrix(), np.asarray(p.translation)


def ray(f, xy):
    R, t = pose(by[f])
    n = np.asarray(cam.cam_from_img(np.array([xy], float))[0], float)
    d = R.T @ np.array([n[0], n[1], 1.0])
    return -R.T @ t, d / np.linalg.norm(d)


# wall plane from frame-0 points on the weatherboards right of the door
pts = np.array([rec.points3D[p.point3D_id].xyz for p in by[0].points2D
                if p.has_point3D() and 560 <= p.xy[0] <= 700 and 60 <= p.xy[1] <= 240])
keep = np.ones(len(pts), bool)
for _ in range(3):
    c = pts[keep].mean(0); n = np.linalg.svd(pts[keep] - c)[2][-1]
    d = np.abs((pts - c) @ n); keep = d < max(np.median(d[keep]) * 3, 1e-6)


def hit(f, xy):
    o, v = ray(f, xy)
    return o + v * ((c - o) @ n) / (v @ n)


top, bot = hit(0, (538.7, 81.3)), hit(0, (538.7, 253.0))     # door jamb (right side), frame 0 picks
up = top - bot; up -= n * (up @ n); h = np.linalg.norm(up); up /= h
right = np.cross(n, up); right /= np.linalg.norm(right)
if (hit(0, (650, 150)) - bot) @ right < 0:
    right = -right
# card: on the boards right of the jamb, 0.6 jamb-heights wide, 0.8 tall, starting 0.1 above the threshold
o0 = bot + right * 0.08 * h + up * 0.1 * h
U, V = right * 0.6 * h, up * 0.8 * h
lines = []
for k in range(7):
    a = k / 6
    lines.append((o0 + U * a, o0 + U * a + V))
    lines.append((o0 + V * a, o0 + V * a + U))
jamb = (bot, top)

PROXY = os.path.join(HERE, "DSCF3000", "proxy")
frames_dir = os.path.join(HERE, f"card_{SOLVE}")
os.makedirs(frames_dir, exist_ok=True)


def proj(R, t, X):
    Xc = R @ X + t
    if Xc[2] <= 0.05:
        return None
    return tuple(cam.img_from_cam(np.array([Xc]))[0])


for f in range(0, 297):
    im = Image.open(os.path.join(PROXY, f"plate_{f:04d}.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    if f in by:
        R, t = pose(by[f])
        for a, b in lines:
            pa, pb = proj(R, t, a), proj(R, t, b)
            if pa and pb:
                d.line([pa, pb], fill=(0, 255, 120), width=2)
        pa, pb = proj(R, t, jamb[0]), proj(R, t, jamb[1])
        if pa and pb:
            d.line([pa, pb], fill=(255, 60, 60), width=2)
    else:
        d.text((10, 30), "NOT SOLVED", fill="red")
    d.text((10, 10), f"{SOLVE}  f{f}", fill="yellow")
    im.save(os.path.join(frames_dir, f"c_{f:04d}.png"))
out = os.path.join(HERE, f"card_check_{SOLVE}.mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", os.path.join(frames_dir, "c_%04d.png"),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
print("wrote", out, "| wall pts", int(keep.sum()), "| jamb", round(float(h), 4), "units")
