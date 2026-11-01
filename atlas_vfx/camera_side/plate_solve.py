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
from dataclasses import asdict, dataclass, field, replace
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
    nodal_fallback: bool = True             # SfM too weak, or a pan that SfM can't hold -> rotation-only solve
    nodal_panoramic_frac: float = 0.5       # this share of rotation-only (panoramic) pairs makes it a candidate
    camera_move: str = "auto"               # auto | free (always SfM) | nodal (tripod pan: always rotation-only)
    lens_k_limit: float = 0.25              # |k1| or |k2| past this is not a real lens: SfM has gone degenerate
    # SfM saw real parallax when points triangulate at this median angle and the camera path is at least this many
    # median depths long (dolly test: 6.4 deg, 0.40 depths; DSCF3002: 3.1 deg, 8.9 depths); then an implausible lens
    # is re-solved (lens rescue), never taken as a pan (DSCF3002, 2026-11-01)
    parallax_path_depth: float = 0.3
    parallax_tri_deg: float = 1.5
    lens_rescue: bool = True
    ba_global_ratio: float = 1.4            # global BA every 40 % of model growth (COLMAP default 1.1)
    ba_global_iterations: int = 30          # COLMAP default 50
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
    # global bundle adjust when the model grows 40 % (COLMAP: 10 %), at most 30 iterations: DSCF2977 (336 frames,
    # rotation heavy) mapped in 1031 s instead of 1822 s, the same camera against the pixels (0.164 vs 0.163 deg).
    # The motion test's own adjust afterwards refines the whole solve anyway.
    o.ba_global_frames_ratio = opts.ba_global_ratio
    o.ba_global_points_ratio = opts.ba_global_ratio
    o.ba_global_max_num_iterations = opts.ba_global_iterations
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


def _project(cam, x) -> tuple[float, float]:
    """pycolmap img_from_cam, or NaN when the point is behind the camera (pycolmap returns None there)."""
    if x[2] <= 1e-9:
        return (np.nan, np.nan)
    uv = cam.img_from_cam(x)
    return (np.nan, np.nan) if uv is None else (float(uv[0]), float(uv[1]))


def observation_errors(rec) -> dict[int, float]:
    """Worst reprojection error (px, at the solver's image size) per 3D point. One batched projection per image
    (pycolmap's img_from_cam takes arrays; a call per point was most of this function's time)."""
    ids_all, err_all = [], []
    xyz = {pid: pt.xyz for pid, pt in rec.points3D.items()}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        obs = [(int(p.point3D_id), p.xy) for p in img.points2D if p.has_point3D()]
        if not obs:
            continue
        ids = np.fromiter((i for i, _ in obs), np.int64, len(obs))
        xy = np.array([q for _, q in obs], float)
        X = np.array([xyz[i] for i in ids], float)
        pose = img.cam_from_world()
        Xc = X @ np.asarray(pose.rotation.matrix()).T + np.asarray(pose.translation)
        uv = np.asarray(rec.cameras[img.camera_id].img_from_cam(Xc), float).reshape(-1, 2)
        uv[Xc[:, 2] <= 1e-9] = np.nan                                 # behind the camera (see _project)
        e = np.linalg.norm(uv - xy, axis=1)
        ids_all.append(ids)
        err_all.append(np.where(np.isfinite(e), e, np.inf))           # behind the camera: as wrong as it gets
    if not ids_all:
        return {}
    ids, e = np.concatenate(ids_all), np.concatenate(err_all)
    order = np.lexsort((e, ids))                                       # by point, worst last
    last = np.r_[ids[order][1:] != ids[order][:-1], True]
    return {int(i): float(v) for i, v in zip(ids[order][last], e[order][last])}


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
    # the last adjust can still push a point behind its cameras (DSCF2977: one point made the mean error 1e170):
    # drop whatever is out of tolerance now, without another adjust, so the model holds no broken point
    worst = observation_errors(rec)
    late = [i for i, e in worst.items() if not e <= opts.reject_px]
    for i in late:
        rec.delete_point3D(i)
    out["final_dropped"] = len(late)
    out["final_mean_reproj_px"] = float(rec.compute_mean_reprojection_error()) if rec.num_points3D() else None
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


def _implausible_lens(rec, size: tuple[int, int], opts: SolveOptions) -> str | None:
    """Why SfM's solved lens can't be a real one (or None): a distortion term past lens_k_limit, or a focal more
    than 2x off the guess (no guess: outside a 10-120 degree horizontal field of view)."""
    cam = next(iter(rec.cameras.values()))
    p = [float(v) for v in cam.params]
    f = float(cam.mean_focal_length())
    k = p[3:5] if cam.model.name in ("RADIAL", "SIMPLE_RADIAL") else p[4:6]
    bad = [f"k{i + 1} {v:+.3f}" for i, v in enumerate(k) if abs(v) > opts.lens_k_limit]
    W = size[0]
    if opts.focal_px:
        if not 0.5 <= f / opts.focal_px <= 2.0:
            bad.append(f"focal {f:.0f} px vs guess {opts.focal_px:.0f}")
    else:
        fov = math.degrees(2 * math.atan(W / 2 / f))
        if not 10.0 <= fov <= 120.0:
            bad.append(f"field of view {fov:.0f} deg")
    return ", ".join(bad) or None


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
    stages, t_stage = {}, [time.time()]

    def stage(name):                                    # seconds per step, in the report
        now = time.time()
        stages[name] = round(now - t_stage[0], 1)
        t_stage[0] = now
    fe.extract(db, image_dir, names, opts, mask_dir, size)
    stage("features")
    log(f"solve: features on {len(names)} images ({size[0]}x{size[1]})")
    fe.match(db, opts)
    n_tracks = 0
    if tracks:
        from . import tracks2d
        n_tracks = tracks2d.add_track_matches(db, names, tracks)
    log("solve: matched" + (f" (+{n_tracks} manual track matches)" if n_tracks else ""))
    stage("matching")

    first, last = frame_of(names[0]), frame_of(names[-1])

    def map_and_bridge(o: SolveOptions, tag: str):
        """Incremental mapping, then a relaxed pass seeded with the model to bridge unregistered frames."""
        r, models = _map(db, image_dir, out / f"sparse{tag}_1", _mapper_options(o))
        ps_ = [{"pass": f"map{tag}", "models": models, "registered": r.num_reg_images() if r else 0}]
        stage(f"mapping{tag}")
        if r is not None and opts.bridge_gaps and r.num_reg_images() < len(names):
            seed_dir = out / f"sparse{tag}_seed"
            shutil.rmtree(seed_dir, ignore_errors=True)
            seed_dir.mkdir()
            r.write(str(seed_dir))
            r2, models2 = _map(db, image_dir, out / f"sparse{tag}_2", _mapper_options(o, relaxed=True), seed_dir)
            if r2 is not None and r2.num_reg_images() > r.num_reg_images():
                r = r2
            ps_.append({"pass": f"bridge{tag}", "models": models2, "registered": r.num_reg_images()})
            stage(f"bridge{tag}")
        return r, ps_

    rec, passes = map_and_bridge(opts, "")
    rejected = reject_and_refine(rec, opts) if rec is not None else {"rounds": []}
    stage("reject_refine")
    weak = rec is None or (len(_registered(rec)) < opts.min_registered_frac * len(names)
                           or rec.num_points3D() < opts.min_points)
    mode, nodal_info = "sfm", None
    model_dir = out / "sparse" / "0"
    shutil.rmtree(out / "sparse", ignore_errors=True)
    deg_sfm = degeneracy(db, rec) if rec is not None else None
    panoramic = deg_sfm is not None and deg_sfm["panoramic_pair_frac"] > opts.nodal_panoramic_frac
    # Rotation-only PAIRS don't make a nodal SHOT: frames up to `overlap` apart look like pure rotation on a walk
    # too (DSCF2977, 2026-11-01: 97 % of pairs, yet NukeX and SfM both solve the translation). What gives a true
    # pan away is SfM's lens: with no baseline it trades focal against distortion into an impossible lens
    # (synthetic pan: focal +98 %, k1 +0.66). So a panoramic shot goes nodal only when SfM's lens is implausible.
    lens_bad = None if rec is None or weak else _implausible_lens(rec, size, opts)
    def has_parallax(d) -> bool:
        return (d is not None and d["path_over_median_depth"] >= opts.parallax_path_depth
                and d["median_tri_angle_deg"] >= opts.parallax_tri_deg)

    parallax = has_parallax(deg_sfm)
    rescue = None
    if lens_bad and not weak and opts.lens_rescue and opts.refine_distortion and opts.camera_move != "nodal":
        # The lens came out impossible. On a walk that is distortion trading against the path (DSCF3002, through
        # glass doors: k1 +0.37); on a pan it is the missing baseline. SfM's own parallax can't tell them apart
        # while its lens is broken (a synthetic pan shows 3.1 deg triangulation, like DSCF3002), so map again with
        # distortion held at zero and judge the parallax of THAT solve: kept only with a plausible lens, as many
        # frames, and real parallax; otherwise the shot goes on to the nodal rule below.
        log(f"solve: SfM's lens is implausible ({lens_bad}); mapping again with distortion fixed")
        fixed = replace(opts, refine_distortion=False)
        rec_r, passes_r = map_and_bridge(fixed, "_rescue")
        rescue = {"lens_was": lens_bad, "passes": passes_r}
        if rec_r is not None:
            rej_r = reject_and_refine(rec_r, opts)
            stage("reject_refine_rescue")
            bad_r = _implausible_lens(rec_r, size, opts)
            enough = len(_registered(rec_r)) >= len(_registered(rec))
            deg_r = degeneracy(db, rec_r)
            moved = has_parallax(deg_r)
            rescue.update(lens_after=bad_r, registered=rec_r.num_reg_images(), parallax=moved,
                          tri_deg=deg_r["median_tri_angle_deg"], path_over_depth=deg_r["path_over_median_depth"],
                          kept=bool(not bad_r and enough and moved))
            if rescue["kept"]:
                rec, rejected, lens_bad, deg_sfm, parallax = rec_r, rej_r, None, deg_r, True
                passes += passes_r
        log(f"solve: lens rescue {'kept' if rescue.get('kept') else 'not kept'} ({rescue.get('lens_after')})")
    go_nodal = {"nodal": True, "free": False}.get(opts.camera_move)
    if go_nodal is None:
        if opts.camera_move != "auto":
            raise SolveError(f"camera_move {opts.camera_move!r}: auto, free or nodal")
        go_nodal = opts.nodal_fallback and (weak or (panoramic and bool(lens_bad)))
    panoramic_kept = panoramic and not go_nodal and not weak
    if go_nodal:
        # no baseline: SfM either can't start or trades focal against distortion freely. A camera turning about
        # its lens is solved as rotations instead, which pins the focal.
        from . import nodal_solve

        why = ("camera_move nodal" if opts.camera_move == "nodal" else
               f"{deg_sfm['panoramic_pair_frac']:.0%} of matched pairs are rotation-only and SfM's lens is "
               f"implausible ({lens_bad})" if panoramic and lens_bad else
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
        stage("nodal")
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
        "mode": mode, "nodal": nodal_info, "camera_move": opts.camera_move, "sfm_lens_check": lens_bad,
        "front_end": opts.front_end, "options": asdict(opts), "image_size": list(size),
        "frames": [first, last], "registered": len(reg), "of": len(names),
        "missing": gaps(reg, first, last), "longest_run": list(run),
        "points3D": rec.num_points3D(), "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "camera": _camera_report(rec, size, opts), "passes": passes, "reject": rejected,
        "degeneracy": deg, "manual_track_matches": n_tracks, "masked": mask_dir is not None,
        "lens_rescue": rescue, "parallax": bool(parallax),
        "seconds": round(time.time() - t0, 1), "stage_seconds": stages,
    }
    if mode == "nodal":            # the model's points sit on their mean rays, so its own error says nothing
        report["mean_reproj_px"] = nodal_info["rms_px"]
    warn = []
    if panoramic_kept and mode == "sfm":
        warn.append(f"{deg_sfm['panoramic_pair_frac']:.0%} of matched pairs look rotation-only, but SfM found "
                    "parallax with a plausible lens, so the camera is solved free. If this was a locked-off pan "
                    "(tripod), set camera_move to nodal and compare the focal.")
    if mode == "nodal":
        warn.append("nodal solve: the shot has no usable baseline, so the camera turns about a fixed point and the "
                    f"points sit on a sphere {nodal_solve.POINT_DISTANCE:g} units out. Fine for a pan; there is no "
                    "depth, so Measure & Scale cannot set metres from these points.")
    if deg["warn"]:
        warn.append("low parallax: " + "; ".join(deg["reasons"]) + ". Focal and distortion are weakly determined; "
                    "check the focal against the lens or lock it from a chart.")
    if mode == "sfm" and lens_bad:
        warn.append(f"SfM's lens is implausible ({lens_bad}) and re-solving it with distortion fixed did not fix it; "
                    "the camera moves (real parallax), so it is kept as SfM solved it. Lock the lens (focal from the "
                    "lens, distortion from a chart) and solve again.")
    if rescue and rescue.get("kept"):
        warn.append(f"SfM's first lens was implausible ({rescue['lens_was']}); re-solved with distortion fixed "
                    "while mapping and freed at the end (lens rescue).")
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
            uv = np.asarray(_project(cam, R @ rec.points3D[p.point3D_id].xyz + t), float)
            if not np.all(np.isfinite(uv)):
                continue
            e = float(np.linalg.norm(uv - np.asarray(p.xy)))
            col = (0, 220, 0) if e < 0.5 else (230, 200, 0) if e < max_px else (230, 0, 0)
            cv2.circle(rgb, (int(round(uv[0])), int(round(uv[1]))), 2, col, -1, cv2.LINE_AA)
        cv2.putText(rgb, f"{frame_of(img.name)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out.append(rgb)
    return out
