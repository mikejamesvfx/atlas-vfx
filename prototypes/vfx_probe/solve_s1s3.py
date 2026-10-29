"""Test matchmove of DSCF3000 shots S1-S3 (frames 0-296) with pycolmap 4.2.1 (BSD-3), CPU SIFT.
Input: the ACES 2.0 SDR proxies (1024x540) made by plate_proxy.py from the F-Log2 plate. Nothing is written near the plate.
One continuous take, so it's solved as one sequence; the S1/S2/S3 split is reported per range.
Outputs in solve_s1s3/: database.db, sparse/<n>/ (COLMAP model), camera_path.json, report.json, solve_s1s3_plot.png."""
import json, os, shutil, sys, time
import numpy as np
import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.environ.get("SOLVE_PROXY") or os.path.join(HERE, "DSCF3000", "proxy")
OUT = os.environ.get("SOLVE_OUT") or os.path.join(HERE, "solve_s1s3")
PLOT = os.path.join(HERE, os.path.basename(OUT) + "_plot.png")
SHOTS = {"S1": (0, 129), "S2": (129, 201), "S3": (201, 297)}
FIRST, LAST = 0, 297
STEP = int(os.environ.get("SOLVE_STEP", "1"))

os.makedirs(OUT, exist_ok=True)
db = os.path.join(OUT, "database.db")
if os.path.exists(db) and "--fresh" in sys.argv:
    os.remove(db)
names = [f"plate_{i:04d}.png" for i in range(FIRST, LAST, STEP)]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

if not os.path.exists(db):
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    eo = pycolmap.FeatureExtractionOptions()
    eo.max_image_size = 1024
    eo.sift.max_num_features = 8192
    pycolmap.extract_features(db, PROXY, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=eo)
    log("features", len(names))
    po = pycolmap.SequentialPairingOptions()
    po.overlap = 15
    po.quadratic_overlap = True
    po.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=po)
    log("matched")

sparse = os.path.join(OUT, "sparse")
shutil.rmtree(sparse, ignore_errors=True)
os.makedirs(sparse)
opts = pycolmap.IncrementalPipelineOptions()
recs = pycolmap.incremental_mapping(db, PROXY, sparse, options=opts)
log("models", {k: r.num_reg_images() for k, r in recs.items()})
if not recs:
    sys.exit("no reconstruction")
key, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())

cam = next(iter(rec.cameras.values()))
path = []
for img in rec.images.values():
    fr = int(img.name[6:10])
    pose = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    R = pose.rotation.matrix()
    t = np.asarray(pose.translation)
    C = (-R.T @ t).tolist()
    errs = [rec.points3D[p.point3D_id].error for p in img.points2D if p.has_point3D()]
    path.append({"frame": fr, "R_cam_from_world": R.tolist(), "t": t.tolist(), "center": C,
                 "n_points": len(errs), "mean_err_px": float(np.mean(errs)) if errs else None})
path.sort(key=lambda d: d["frame"])
reg = {d["frame"] for d in path}

report = {"model": int(key), "all_models": {int(k): r.num_reg_images() for k, r in recs.items()},
          "frames_tried": len(names), "registered": len(reg), "points3D": rec.num_points3D(),
          "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
          "camera": {"model": str(cam.model), "width": cam.width, "height": cam.height, "params": list(map(float, cam.params)),
                     "focal_px": float(cam.params[0]), "hfov_deg": float(np.degrees(2 * np.arctan(cam.width / (2 * cam.params[0]))))},
          "shots": {}}
for s, (a, b) in SHOTS.items():
    want = set(range(a, b, STEP))
    got = sorted(want & reg)
    gaps, prev = [], None
    for f in sorted(want):
        if f not in reg:
            gaps.append(f)
    report["shots"][s] = {"frames": [a, b - 1], "registered": len(got), "of": len(want),
                          "missing": gaps[:40], "missing_count": len(gaps)}
json.dump(path, open(os.path.join(OUT, "camera_path.json"), "w"), indent=1)
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
log(json.dumps({k: v for k, v in report.items() if k != "shots"}))
for s, v in report["shots"].items():
    log(s, f"{v['registered']}/{v['of']}", "missing", v["missing_count"])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = np.array([d["center"] for d in path]); F = np.array([d["frame"] for d in path])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for s, (a, b) in SHOTS.items():
        m = (F >= a) & (F < b)
        ax[0].plot(C[m, 0], C[m, 2], ".-", ms=2, label=s)
        ax[1].plot(F[m], C[m, 1], ".", ms=2, label=s)
    ax[0].set_title("camera path, top-down (x, z), solve units"); ax[0].axis("equal"); ax[0].legend()
    ax[1].set_title("camera height (y) per frame"); ax[1].invert_yaxis()
    ax[2].plot([d["frame"] for d in path], [d["mean_err_px"] or 0 for d in path], ".", ms=2)
    ax[2].set_title("mean reprojection error per frame (px @1024)")
    fig.tight_layout(); fig.savefig(PLOT, dpi=110)
    log("plot written")
except Exception as e:
    log("no plot:", e)
