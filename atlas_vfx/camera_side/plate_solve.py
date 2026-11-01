"""Solve Plate engine: tracking images -> camera + lens + points (pycolmap incremental SfM).

Ported from tools/rs_clip_solve.py and prototypes/vfx_probe/resolve_nodist.py (the DSCF3000 S1-S3 solve), with the
lens solved rather than locked:

  extract     SIFT per image, single camera, RADIAL model (f, cx, cy, k1, k2), focal seeded from a guess when given,
              optional per-image masks (white in an Atlas mask = moving thing = no features there).
  match       sequential (overlap, quadratic), optional loop detection, plus manual 2D tracks as extra matches.
  map         incremental mapping, focal and k1/k2 refined, principal point held at the centre.
  bridge      when the mapper drops frames or splits the shot, map again starting from the biggest model so the rest
              can register onto it. What still fails is reported, never interpolated.
  reject      model free: points whose worst reprojection is far off (things that moved) are dropped, then a global
              bundle adjust; twice.
  degeneracy  how much of the shot two-view geometry calls panoramic/planar, the median triangulation angle and
              baseline over depth. A low-parallax shot gets a warning: its focal is weakly determined.

Everything is written under the out folder (database.db, sparse/, report). Pure python + pycolmap + numpy; no
ComfyUI imports. The feature front end sits behind FRONT_ENDS so a learned extractor / matcher can be added later.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FRAME_RX = re.compile(r"(\d+)(?!.*\d)")


class SolveError(RuntimeError):
    pass


@dataclass
class SolveOptions:
    camera_model: str = "RADIAL"            # RADIAL (f cx cy k1 k2), SIMPLE_RADIAL (f cx cy k1) or OPENCV
    focal_px: float | None = None           # guess at the image width the solver sees; None = COLMAP's 1.2 x max side
    refine_focal: bool = True
    refine_distortion: bool = True
    refine_principal_point: bool = False
    max_image_size: int = 1024
    max_features: int = 8192
    overlap: int = 15
    quadratic_overlap: bool = True
    loop_detection: bool = False
    vocab_tree_path: str = ""
    min_registered_frac: float = 0.5
    min_points: int = 300
    bridge_gaps: bool = True
    nodal_fallback: bool = True             # SfM too weak, or the shot is a pan -> rotation-only solve
    nodal_panoramic_frac: float = 0.5       # this share of rotation-only (panoramic) pairs makes it a pan
    reject_px: float = 2.0                  # drop points whose worst observation reprojects further than this
    reject_rounds: int = 2
    seed: int = 0
    num_threads: int = -1
    front_end: str = "pycolmap_sift"


@dataclass
class SolveResult:
    model_dir: Path
    frames: list[int]                       # registered frame indices (from image names), sorted
    run: tuple[int, int]                    # longest contiguous run of registered frames, inclusive
    report: dict = field(default_factory=dict)


def frame_of(name: str) -> int:
    m = FRAME_RX.search(Path(name).stem)
    if not m:
        raise SolveError(f"no frame number in image name {name!r}")
    return int(m.group(1))


def longest_run(frames: list[int]) -> tuple[int, int]:
    if not frames:
        raise SolveError("no registered frames")
    frames = sorted(frames)
    best, start = (frames[0], frames[0]), frames[0]
    for a, b in zip(frames, frames[1:] + [None]):
        if b != a + 1:
            if a - start > best[1] - best[0]:
                best = (start, a)
            start = b
    return best


def gaps(frames: list[int], first: int, last: int) -> list[int]:
    return sorted(set(range(first, last + 1)) - set(frames))


# --------------------------------------------------------------------------------------------- masks

def write_masks(masks, names: list[str], out: Path, dilate_px: int = 0) -> Path:
    """Atlas MASK frames (HxW float, 1 = moving thing) -> COLMAP mask PNGs (<image name>.png, 0 = no features).
    masks is a sequence aligned with names; a mask of another size is resized to the image (nearest)."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    if len(masks) != len(names):
        raise SolveError(f"{len(masks)} masks for {len(names)} images")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px > 0 else None
    for m, n in zip(masks, names):
        a = (np.asarray(m, np.float32) > 0.5).astype(np.uint8)
        if k is not None:
            a = cv2.dilate(a, k)
        if not cv2.imwrite(str(out / f"{n}.png"), (1 - a) * 255):
            raise SolveError(f"could not write mask for {n}")
    return out


def resize_masks(masks, size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    W, H = size
    return [cv2.resize(np.asarray(m, np.float32), (W, H), interpolation=cv2.INTER_NEAREST) for m in masks]


# --------------------------------------------------------------------------------------------- front end

class PycolmapSIFT:
    """SIFT + sequential matching inside the COLMAP database."""

    name = "pycolmap_sift"

    def extract(self, db: Path, image_dir: Path, names: list[str], opts: SolveOptions, mask_dir: Path | None,
                size: tuple[int, int]):
        import pycolmap

        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = opts.camera_model
        if opts.focal_px:
            W, H = size
            cx, cy = W / 2.0, H / 2.0
            prm = {"RADIAL": [opts.focal_px, cx, cy, 0.0, 0.0], "SIMPLE_RADIAL": [opts.focal_px, cx, cy, 0.0],
                   "OPENCV": [opts.focal_px, opts.focal_px, cx, cy, 0.0, 0.0, 0.0, 0.0]}[opts.camera_model]
            ro.camera_params = ",".join(f"{v:.6f}" for v in prm)
        if mask_dir is not None:
            ro.mask_path = str(mask_dir)
        eo = pycolmap.FeatureExtractionOptions()
        eo.max_image_size = opts.max_image_size
        eo.sift.max_num_features = opts.max_features
        if opts.num_threads > 0:
            eo.num_threads = opts.num_threads
        pycolmap.extract_features(str(db), str(image_dir), image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo)

    def match(self, db: Path, opts: SolveOptions):
        import pycolmap

        po = pycolmap.SequentialPairingOptions()
        po.overlap = opts.overlap
        po.quadratic_overlap = opts.quadratic_overlap
        po.loop_detection = bool(opts.loop_detection and opts.vocab_tree_path)
        if po.loop_detection:
            po.vocab_tree_path = opts.vocab_tree_path
        pycolmap.match_sequential(str(db), pairing_options=po)


FRONT_ENDS = {PycolmapSIFT.name: PycolmapSIFT}


# --------------------------------------------------------------------------------------------- mapping

def _mapper_options(opts: SolveOptions, relaxed: bool = False):
    import pycolmap

    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_focal_length = opts.refine_focal
    o.ba_refine_extra_params = opts.refine_distortion
    o.ba_refine_principal_point = opts.refine_principal_point
    o.random_seed = opts.seed
    if opts.num_threads > 0:
        o.num_threads = opts.num_threads
    if relaxed:                              # second pass: let weakly matched frames register onto the model
        o.mapper.abs_pose_min_num_inliers = 15
        o.mapper.abs_pose_min_inlier_ratio = 0.15
        o.mapper.max_reg_trials = 6
    return o


def _registered(rec) -> list[int]:
    return sorted(frame_of(i.name) for i in rec.images.values() if i.has_pose)


def _map(db: Path, image_dir: Path, out: Path, opts, input_path: Path | None = None):
    import pycolmap

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    recs = pycolmap.incremental_mapping(str(db), str(image_dir), str(out), options=opts,
                                        input_path=str(input_path) if input_path else "")
    if not recs:
        return None, {}
    key, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    return rec, {int(k): r.num_reg_images() for k, r in recs.items()}


def observation_errors(rec) -> dict[int, float]:
    """Worst reprojection error (px, at the solver's image size) per 3D point."""
    worst: dict[int, float] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        pose = img.cam_from_world()
        ids, xy = [], []
        for p in img.points2D:
            if p.has_point3D():
                ids.append(int(p.point3D_id))
                xy.append(p.xy)
        if not ids:
            continue
        X = np.array([rec.points3D[i].xyz for i in ids], float)
        Xc = (np.asarray(pose.rotation.matrix()) @ X.T).T + np.asarray(pose.translation)
        uv = np.array([cam.img_from_cam(x) for x in Xc], float)
        e = np.linalg.norm(uv - np.asarray(xy, float), axis=1)
        for i, v in zip(ids, e):
            if v > worst.get(i, -1.0):
                worst[i] = float(v)
    return worst


def reject_and_refine(rec, opts: SolveOptions) -> dict:
    import pycolmap

    out = {"rounds": []}
    for _ in range(max(0, opts.reject_rounds)):
        worst = observation_errors(rec)
        bad = [i for i, e in worst.items() if e > opts.reject_px]
        for i in bad:
            rec.delete_point3D(i)
        bo = pycolmap.BundleAdjustmentOptions()
        bo.refine_focal_length = opts.refine_focal
        bo.refine_extra_params = opts.refine_distortion
        bo.refine_principal_point = opts.refine_principal_point
        pycolmap.bundle_adjustment(rec, bo)
        out["rounds"].append({"dropped": len(bad), "points_left": rec.num_points3D(),
                              "mean_reproj_px": float(rec.compute_mean_reprojection_error())})
        if not bad:
            break
    return out


# --------------------------------------------------------------------------------------------- degeneracy

def degeneracy(db: Path, rec) -> dict:
    """Two-view configs from matching plus parallax from the model. warn=True means the focal is weakly determined."""
    import pycolmap

    C = pycolmap.TwoViewGeometryConfiguration
    con = sqlite3.connect(db)
    try:
        cfgs = [r[0] for r in con.execute("SELECT config FROM two_view_geometries WHERE rows > 0")]
    finally:
        con.close()
    n = max(len(cfgs), 1)
    rot = sum(c in (int(C.PANORAMIC), int(C.PLANAR_OR_PANORAMIC)) for c in cfgs) / n
    planar = sum(c == int(C.PLANAR) for c in cfgs) / n

    centres, imgs = {}, {}
    for img in rec.images.values():
        if img.has_pose:
            centres[img.image_id] = np.asarray(img.projection_center(), float)
            imgs[img.image_id] = img
    angles, depths = [], []
    for pid, pt in rec.points3D.items():
        els = [e.image_id for e in pt.track.elements if e.image_id in centres]
        if len(els) < 2:
            continue
        X = np.asarray(pt.xyz, float)
        a, b = centres[els[0]], centres[els[-1]]
        u, v = a - X, b - X
        c = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        angles.append(math.degrees(math.acos(min(1.0, max(-1.0, c)))))
        depths.append(np.linalg.norm(u))
    order = [centres[k] for k in sorted(centres, key=lambda k: frame_of(imgs[k].name))]
    path = float(sum(np.linalg.norm(b - a) for a, b in zip(order, order[1:])))
    med_depth = float(np.median(depths)) if depths else float("nan")
    med_angle = float(np.median(angles)) if angles else 0.0
    base_over_depth = path / med_depth if depths and med_depth > 0 else 0.0
    reasons = []
    if rot > 0.5:
        reasons.append(f"{rot:.0%} of matched pairs look rotation-only (panoramic)")
    if med_angle < 2.0:
        reasons.append(f"median triangulation angle {med_angle:.2f} deg < 2")
    if base_over_depth < 0.05:
        reasons.append(f"camera path {base_over_depth:.3f} x median depth < 0.05")
    return {"panoramic_pair_frac": round(rot, 3), "planar_pair_frac": round(planar, 3),
            "median_tri_angle_deg": round(med_angle, 3), "path_over_median_depth": round(base_over_depth, 4),
            "warn": bool(reasons), "reasons": reasons}


# --------------------------------------------------------------------------------------------- top level

def _image_size(image_dir: Path, name: str) -> tuple[int, int]:
    import cv2

    a = cv2.imread(str(image_dir / name), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SolveError(f"cannot read {image_dir / name}")
    return a.shape[1], a.shape[0]


def _camera_report(rec, size: tuple[int, int], opts: SolveOptions) -> dict:
    cam = next(iter(rec.cameras.values()))
    p = [float(v) for v in cam.params]
    info = {"model": cam.model.name, "width": cam.width, "height": cam.height, "params": p,
            "focal_px": float(cam.mean_focal_length())}
    if opts.focal_px:
        info["focal_seed_px"] = float(opts.focal_px)
        info["focal_delta_pct"] = round(100.0 * (info["focal_px"] / opts.focal_px - 1.0), 3)
    return info


def solve_folder(image_dir, names: list[str], out_dir, opts: SolveOptions | None = None, masks=None,
                 mask_dilate_px: int = 0, tracks: dict | None = None, log=print) -> SolveResult:
    """Solve the images `names` (in `image_dir`, frame index = last number in the name) into out_dir.
    masks: optional sequence aligned with names (1 = ignore). tracks: optional manual 2D tracks in the solver's
    image pixels, {"tracks": {id: {frame: [x, y]}}} (added as matches; see add_track_matches)."""
    opts = opts or SolveOptions()
    image_dir, out = Path(image_dir), Path(out_dir)
    names = sorted(names, key=frame_of)
    if len(names) < 3:
        raise SolveError("need at least 3 frames")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    size = _image_size(image_dir, names[0])
    db = out / "database.db"
    if db.exists():
        db.unlink()
    mask_dir = None
    if masks is not None:
        mask_dir = write_masks(resize_masks(masks, size), names, out / "masks", mask_dilate_px)
    fe = FRONT_ENDS[opts.front_end]()
    fe.extract(db, image_dir, names, opts, mask_dir, size)
    log(f"solve: features on {len(names)} images ({size[0]}x{size[1]})")
    fe.match(db, opts)
    n_tracks = 0
    if tracks:
        from . import tracks2d
        n_tracks = tracks2d.add_track_matches(db, names, tracks)
    log("solve: matched" + (f" (+{n_tracks} manual track matches)" if n_tracks else ""))

    first, last = frame_of(names[0]), frame_of(names[-1])
    rec, models = _map(db, image_dir, out / "sparse_1", _mapper_options(opts))
    passes = [{"pass": "map", "models": models, "registered": rec.num_reg_images() if rec else 0}]
    if rec is not None and opts.bridge_gaps and rec.num_reg_images() < len(names):
        seed_dir = out / "sparse_seed"
        shutil.rmtree(seed_dir, ignore_errors=True)
        seed_dir.mkdir()
        rec.write(str(seed_dir))
        rec2, models2 = _map(db, image_dir, out / "sparse_2", _mapper_options(opts, relaxed=True), seed_dir)
        if rec2 is not None and rec2.num_reg_images() > rec.num_reg_images():
            rec = rec2
        passes.append({"pass": "bridge", "models": models2, "registered": rec.num_reg_images()})
    rejected = reject_and_refine(rec, opts) if rec is not None else {"rounds": []}
    weak = rec is None or (len(_registered(rec)) < opts.min_registered_frac * len(names)
                           or rec.num_points3D() < opts.min_points)
    mode, nodal_info = "sfm", None
    model_dir = out / "sparse" / "0"
    shutil.rmtree(out / "sparse", ignore_errors=True)
    deg_sfm = degeneracy(db, rec) if rec is not None else None
    panoramic = deg_sfm is not None and deg_sfm["panoramic_pair_frac"] > opts.nodal_panoramic_frac
    if (weak or panoramic) and opts.nodal_fallback:
        # no baseline: SfM either can't start or trades focal against distortion freely. A camera turning about
        # its lens is solved as rotations instead, which pins the focal.
        from . import nodal_solve

        why = (f"{deg_sfm['panoramic_pair_frac']:.0%} of matched pairs are rotation-only" if panoramic else
               f"SfM registered {len(_registered(rec)) if rec else 0}/{len(names)} frames")
        log(f"solve: {why}; nodal solve")
        sol = nodal_solve.solve(db, names, focal_px=opts.focal_px,
                                model="SIMPLE_RADIAL" if opts.camera_model == "SIMPLE_RADIAL" else "RADIAL",
                                refine_focal=opts.refine_focal, refine_distortion=opts.refine_distortion,
                                seed=opts.seed, log=log)
        nodal_solve.to_binary(nodal_solve.write_model(sol, out / "sparse_nodal_txt"), model_dir)
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_dir))
        mode = "nodal"
        nodal_info = {"rms_px": sol["rms_px"], "focal_start_px": sol["focal_start_px"],
                      "fov_x_deg": nodal_solve.fov_deg(sol), "point_distance": nodal_solve.POINT_DISTANCE}
        passes.append({"pass": "nodal", "why": why, "registered": rec.num_reg_images()})
    reg = _registered(rec) if rec is not None else []
    if rec is None or len(reg) < opts.min_registered_frac * len(names) or rec.num_points3D() < opts.min_points:
        raise SolveError(f"solve too weak: {len(reg)}/{len(names)} frames, "
                         f"{rec.num_points3D() if rec else 0} points (passes {passes})")
    if mode == "sfm":
        model_dir.mkdir(parents=True)
        rec.write(str(model_dir))
    run = longest_run(reg)
    deg = deg_sfm if mode == "sfm" else {**(deg_sfm or {}), "nodal": True, "warn": False, "reasons": []}
    report = {
        "mode": mode, "nodal": nodal_info,
        "front_end": opts.front_end, "options": asdict(opts), "image_size": list(size),
        "frames": [first, last], "registered": len(reg), "of": len(names),
        "missing": gaps(reg, first, last), "longest_run": list(run),
        "points3D": rec.num_points3D(), "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "camera": _camera_report(rec, size, opts), "passes": passes, "reject": rejected,
        "degeneracy": deg, "manual_track_matches": n_tracks, "masked": mask_dir is not None,
        "seconds": round(time.time() - t0, 1),
    }
    if mode == "nodal":            # the model's points sit on their mean rays, so its own error says nothing
        report["mean_reproj_px"] = nodal_info["rms_px"]
    warn = []
    if mode == "nodal":
        warn.append("nodal solve: the shot has no usable baseline, so the camera turns about a fixed point and the "
                    f"points sit on a sphere {nodal_solve.POINT_DISTANCE:g} units out. Fine for a pan; there is no "
                    "depth, so Measure & Scale cannot set metres from these points.")
    if deg["warn"]:
        warn.append("low parallax: " + "; ".join(deg["reasons"]) + ". Focal and distortion are weakly determined; "
                    "check the focal against the lens or lock it from a chart.")
    if report["missing"]:
        warn.append(f"{len(report['missing'])} frames did not register; the camera keeps frames {run[0]}-{run[1]}")
    report["warnings"] = warn
    (out / "solve_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for w in warn:
        log("solve WARNING: " + w)
    log(f"solve: {len(reg)}/{len(names)} frames, {rec.num_points3D()} points, "
        f"{report['mean_reproj_px']:.3f} px, focal {report['camera']['focal_px']:.2f} px, {report['seconds']} s")
    return SolveResult(model_dir=model_dir, frames=reg, run=run, report=report)


# --------------------------------------------------------------------------------------------- preview

def preview(model_dir, image_dir, count: int = 4, max_px: float = 2.0) -> list[np.ndarray]:
    """A few tracking images with each solved point drawn where the solve reprojects it: green under 0.5 px,
    yellow up to max_px, red beyond. uint8 RGB."""
    import cv2
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    imgs = sorted((i for i in rec.images.values() if i.has_pose), key=lambda i: frame_of(i.name))
    if not imgs:
        return []
    pick = [imgs[round(k * (len(imgs) - 1) / max(count - 1, 1))] for k in range(min(count, len(imgs)))]
    out = []
    for img in pick:
        a = cv2.imread(str(Path(image_dir) / img.name), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        cam = rec.cameras[img.camera_id]
        pose = img.cam_from_world()
        R, t = np.asarray(pose.rotation.matrix()), np.asarray(pose.translation)
        for p in img.points2D:
            if not p.has_point3D():
                continue
            uv = np.asarray(cam.img_from_cam(R @ rec.points3D[p.point3D_id].xyz + t), float)
            e = float(np.linalg.norm(uv - np.asarray(p.xy)))
            col = (0, 220, 0) if e < 0.5 else (230, 200, 0) if e < max_px else (230, 0, 0)
            cv2.circle(rgb, (int(round(uv[0])), int(round(uv[1]))), 2, col, -1, cv2.LINE_AA)
        cv2.putText(rgb, f"{frame_of(img.name)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out.append(rgb)
    return out
