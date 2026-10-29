"""Re-map S1-S3 reusing the tracking-proxy features/matches, with lens distortion pinned at 0 (Fujinon XF video is
usually distortion-corrected in camera) so bundle adjustment can't trade a fake barrel for a bent scene.
Env: FOCAL (px @1024, default: let BA refine from 1287), FIX_FOCAL=1 to pin it too.
Writes solve_s1s3_nodist/ (database copy, sparse/, camera_path.json, report.json) and runs the lawn check."""
import json, os, shutil, sqlite3, sys
import numpy as np
import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "solve_s1s3_track", "database.db")
OUT = os.path.join(HERE, os.environ.get("OUT_NAME", "solve_s1s3_nodist"))
PROXY = os.path.join(HERE, "DSCF3000", "track")
os.makedirs(OUT, exist_ok=True)
db = os.path.join(OUT, "database.db")
shutil.copyfile(SRC, db)
focal = float(os.environ.get("FOCAL", "1287.47"))
con = sqlite3.connect(db)
for cid, params in con.execute("SELECT camera_id, params FROM cameras").fetchall():
    p = np.frombuffer(params, np.float64).copy()
    p[0], p[3] = focal, 0.0
    con.execute("UPDATE cameras SET params = ?, prior_focal_length = 1 WHERE camera_id = ?", (p.tobytes(), cid))
con.commit(); con.close()

opts = pycolmap.IncrementalPipelineOptions()
opts.ba_refine_extra_params = False
opts.ba_refine_principal_point = False
if os.environ.get("FIX_FOCAL") == "1":
    opts.ba_refine_focal_length = False
sparse = os.path.join(OUT, "sparse")
shutil.rmtree(sparse, ignore_errors=True); os.makedirs(sparse)
recs = pycolmap.incremental_mapping(db, PROXY, sparse, options=opts)
key, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
cam = next(iter(rec.cameras.values()))
print("models", {k: r.num_reg_images() for k, r in recs.items()}, "| chosen", key, "| params", list(map(float, cam.params)),
      "| hfov", float(np.degrees(2 * np.arctan(cam.width / (2 * cam.params[0])))), "| reproj", rec.compute_mean_reprojection_error(), flush=True)


def C(img):
    p = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    R, t = p.rotation.matrix(), np.asarray(p.translation)
    return -R.T @ t, R, t


by = {int(i.name[6:10]): i for i in rec.images.values()}
json.dump([{"frame": f, "center": C(by[f])[0].tolist()} for f in sorted(by)], open(os.path.join(OUT, "camera_path.json"), "w"))
# lawn planarity + camera height (relative), the check the distorted solve failed
ids = {p.point3D_id for f in range(0, 100, 5) if f in by for p in by[f].points2D if p.has_point3D() and p.xy[1] > 420}
pts = np.array([rec.points3D[i].xyz for i in ids]); keep = np.ones(len(pts), bool)
for _ in range(4):
    c = pts[keep].mean(0); n = np.linalg.svd(pts[keep] - c)[2][-1]
    d = np.abs((pts - c) @ n); keep = d < max(np.median(d[keep]) * 2.5, 1e-6)
if (C(by[0])[0] - c) @ n < 0:
    n = -n
spread = np.ptp(pts[keep] @ np.linalg.svd(pts[keep] - c)[2][0])
rms = float(np.sqrt(np.mean(((pts[keep] - c) @ n) ** 2)))
heights = {f: float((C(by[f])[0] - c) @ n) for f in (0, 20, 40, 60, 80, 100, 120) if f in by}
print(f"lawn rms / lawn extent = {rms / spread * 100:.2f}% (distorted solve was ~{0.439 / 1.686 / 15:.1%} rough)")
print("camera height above lawn (units):", {f: round(h, 3) for f, h in heights.items()})
json.dump({"params": list(map(float, cam.params)), "reproj": rec.compute_mean_reprojection_error(), "registered": len(by),
           "lawn_rms_over_extent": rms / spread, "cam_height_units": heights}, open(os.path.join(OUT, "report.json"), "w"), indent=1)
