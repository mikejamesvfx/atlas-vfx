"""Compare Mike's Nuke CameraTracker camera (Camera1.json, frames 1001-1528) with the pycolmap 16.47 mm solve
(solve_s1s3_f16, frames 0-296 = Nuke 1001-1297). Similarity fit (Umeyama) of camera centres, residuals, view-direction
agreement, then the door scale carried into Nuke units. Writes Camera1_metres.chan (whole Nuke range, scaled to metres)
and align_nuke.json next to the Nuke export."""
import json, math, os, sys
import numpy as np
import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = r"C:\Users\miike\howling-yowies\nuke\export"
SOLVE = os.path.join(HERE, "solve_s1s3_f16", "sparse")
DOOR_UNITS = 1.9202        # jamb height, frame 0, in solve_s1s3_f16 units (door_scale3.py)
DOOR_M = float(sys.argv[1]) if len(sys.argv) > 1 else 2.08

cam = json.load(open(os.path.join(EXPORT, "Camera1.json")))
nk = {r["frame"]: r for r in cam["frames"]}


def euler_zxy(rx, ry, rz):
    """Nuke rot_order ZXY: rotate Z first, then X, then Y (R = Ry @ Rx @ Rz for column vectors)."""
    x, y, z = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


rec = max((pycolmap.Reconstruction(os.path.join(SOLVE, m)) for m in os.listdir(SOLVE)), key=lambda r: r.num_reg_images())
mine = {}
for img in rec.images.values():
    p = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    R, t = p.rotation.matrix(), np.asarray(p.translation)
    mine[int(img.name[6:10])] = (-R.T @ t, R.T)      # centre, world-from-camera (COLMAP axes: x right, y down, z forward)

frames = sorted(f for f in mine if (f + 1001) in nk)
A = np.array([mine[f][0] for f in frames])
B = np.array([nk[f + 1001]["translate"] for f in frames])
# Umeyama similarity B ~ s R A + t
ma, mb = A.mean(0), B.mean(0)
Ac, Bc = A - ma, B - mb
U, S, Vt = np.linalg.svd(Bc.T @ Ac / len(A))
D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rg = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / (Ac ** 2).sum(1).mean())
tg = mb - s * Rg @ ma
res = np.linalg.norm((s * (Rg @ A.T)).T + tg - B, axis=1)
path_len = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))

# view direction agreement: camera forward (COLMAP +z) vs Nuke camera forward (-z in camera space)
ang = []
for f in frames:
    fwd_mine = Rg @ (mine[f][1] @ np.array([0, 0, 1.0]))
    r = nk[f + 1001]["rotate"]
    fwd_nk = euler_zxy(*r) @ np.array([0, 0, -1.0])
    ang.append(math.degrees(math.acos(np.clip(fwd_mine @ fwd_nk / np.linalg.norm(fwd_mine), -1, 1))))
ang = np.array(ang)

m_per_mine = DOOR_M / DOOR_UNITS
m_per_nuke = m_per_mine / s
out = {"frames_compared": len(frames), "scale_nuke_per_mine": s, "centre_residual_nuke_units": {"median": float(np.median(res)),
       "p95": float(np.percentile(res, 95)), "max": float(res.max())}, "residual_pct_of_path": float(np.median(res) / path_len * 100),
       "view_angle_deg": {"median": float(np.median(ang)), "p95": float(np.percentile(ang, 95)), "max": float(ang.max())},
       "door_frame_m": DOOR_M, "metres_per_nuke_unit": m_per_nuke, "nuke_path_len_units_1001_1297": path_len,
       "nuke_path_len_m": path_len * m_per_nuke, "nuke_focal_mm": nk[1001]["focal"], "nuke_haperture_mm": nk[1001]["haperture"]}
json.dump(out, open(os.path.join(EXPORT, "align_nuke.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k}: {v}")
# Camera1 scaled to metres (translation only; rotations unchanged), full Nuke range, chan layout + focal
with open(os.path.join(EXPORT, "Camera1_metres.chan"), "w") as fh:
    for f in sorted(nk):
        r = nk[f]
        t = [v * m_per_nuke for v in r["translate"]]
        fh.write(f"{f}\t" + "\t".join(f"{v:.6f}" for v in t + r["rotate"]) + f"\t{r['focal']:.6f}\n")
print("wrote Camera1_metres.chan")
# camera height above the Nuke ground (y up) for a sanity check, metres, relative to the lowest camera position
y = np.array([nk[f]["translate"][1] for f in sorted(nk)]) * m_per_nuke
print("camera y range (m):", round(float(y.min()), 2), "to", round(float(y.max()), 2))
