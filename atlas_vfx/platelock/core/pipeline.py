"""lock_shot: the whole chain on one shot.

  1. retime each performer           (retime.solve_retime)
  2. align the two solves            (align.anchor_align)
  3. re-shoot the AI background      (reshoot.reshoot_merge, rolling shutter)
  4. warp each performer             (correspondence field -> edge snap to the plate key)
  5. restack by depth                (stack.restack)
  6. QC before and after             (qc.person_report / shot_report)

Everything model-based (segmentation, camera/depth solve, body solve, dense
matching) comes in as data: masks, tracks, depths and a correspondence
callback. That keeps this module testable against the synthetic shot and lets
adapters swap models without touching the maths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import binary_dilation

from . import align, fill, key, qc, reshoot, retime, stack, warp
from .camera import CameraTrack

CorrFn = Callable[[int, int], Optional[tuple]]  # (plate_frame, ai_frame) -> (plate_xy, ai_xy) or None


@dataclass
class PersonInput:
    name: str
    plate_masks: np.ndarray            # (N,H,W) this performer in the plate (segmenter on plate)
    ai_masks: np.ndarray               # (M,H,W) this performer in the AI take
    depth: float | np.ndarray = 1.0    # stacking depth, scalar or per plate frame
    correspondences: CorrFn | None = None


@dataclass
class LockConfig:
    anchor_frame: int = 0
    shutter: float = 0.5               # fraction of a frame (180 deg)
    mb_samples: int = 1                # motion-blur samples for the re-shoot
    neighbours: int = 2                # AI frames either side tried first for bg holes
    max_neighbours: int = 12           # widen the search up to this before inpainting
    inpaint: bool = True               # last-resort fill; marked 0.5 in bg_valid
    fill_radius: float = 1.5
    people_dilate: int = 3             # px removed around AI performers before re-shooting
    retime_smooth: float = 1.0
    corr_smoothing: float = 1.0
    corr_grid: int = 4
    snap: bool = True
    snap_iters: int = 2
    snap_max_shift: float = 8.0
    snap_sigma: float = 2.5
    snap_band: float = 4.0             # px strip around the edge when correspondences exist
    snap_min_px: float = 0.35          # skip the snap when the edge is already this close
    temporal_sigma: float = 0.0        # smooth person maps over time
    frames: Optional[list] = None      # subset of plate frames to process


@dataclass
class PersonResult:
    name: str
    retime: retime.RetimeCurve
    rgb: np.ndarray                    # (N,H,W,3) warped AI performer
    alpha: np.ndarray                  # (N,H,W)
    maps: list                         # per plate frame (map_x, map_y) into ai_frames[i]
    ai_frames: np.ndarray              # (N,) AI frame each plate frame was built from


@dataclass
class LockResult:
    comp: np.ndarray                   # (N,H,W,3)
    bg: np.ndarray                     # (N,H,W,3) re-shot background
    bg_valid: np.ndarray               # (N,H,W)
    bg_maps: list                      # (map_x, map_y) into the AI frame of the same index
    people: dict
    gauge: align.Sim3
    camera_delta: dict
    report_before: dict
    report_after: dict
    frames: list = field(default_factory=list)


def lock_shot(plate: np.ndarray, ai: np.ndarray, ai_depth: np.ndarray,
              plate_track: CameraTrack, ai_track: CameraTrack, people: list[PersonInput],
              scale: float, cfg: LockConfig | None = None, plate_key: np.ndarray | None = None,
              progress: Callable[[str, float], None] | None = None) -> LockResult:
    cfg = cfg or LockConfig()
    N, H, W = plate.shape[:3]
    M = ai.shape[0]
    frames = cfg.frames if cfg.frames is not None else list(range(N))
    say = progress or (lambda *_: None)

    # -- QC before: raw AI against plate, same frame index
    before_people = {}
    for p in people:
        idx = np.clip(np.arange(N), 0, M - 1)
        before_people[p.name] = qc.person_report(p.plate_masks, p.ai_masks[idx])

    # -- 1. retime
    curves = {}
    for p in people:
        curves[p.name] = retime.solve_retime(p.plate_masks, p.ai_masks, smooth_sigma=cfg.retime_smooth)
    say("retime", 1.0)

    # -- 2. align solves
    g = align.anchor_align(ai_track, plate_track, cfg.anchor_frame, scale)
    ai_in_plate = align.transform_track(ai_track, g)
    cam_delta = align.camera_delta(ai_in_plate, plate_track)

    # -- 3. re-shoot background through the plate camera
    union = np.zeros((M, H, W), bool)
    for p in people:
        union |= p.ai_masks > 0.5
    if cfg.people_dilate > 0:
        st = np.ones((3, 3), bool)
        union = np.stack([binary_dilation(u, st, iterations=cfg.people_dilate) for u in union])
    src_valid = (~union).astype(np.float32)

    bg = np.zeros((N, H, W, 3), np.float32)
    bg_valid = np.zeros((N, H, W), np.float32)
    bg_maps = [None] * N
    for n_done, i in enumerate(frames):
        js = sorted(range(max(0, i - cfg.max_neighbours), min(M, i + cfg.max_neighbours + 1)),
                    key=lambda j: abs(j - i)) or [min(i, M - 1)]
        r = reshoot.reshoot_merge([ai[j] for j in js], [ai_depth[j] for j in js], [src_valid[j] for j in js],
                                  ai_track, js, plate_track, i, g, shutter=cfg.shutter,
                                  samples=cfg.mb_samples, fill_radius=cfg.fill_radius,
                                  min_frames=1 + 2 * cfg.neighbours)
        if cfg.inpaint and (r.valid < 0.5).any():
            r.image = fill.pushpull_fill(r.image, r.valid < 0.5)
            r.valid = np.where(r.valid < 0.5, 0.5, r.valid).astype(np.float32)
        bg[i], bg_valid[i], bg_maps[i] = r.image, r.valid, (r.map_x, r.map_y)
        say("background", (n_done + 1) / len(frames))

    # -- 4. people
    results = {}
    for p in people:
        tau = curves[p.name].tau
        jsel = np.clip(np.rint(tau).astype(int), 0, M - 1)
        rgb = np.zeros((N, H, W, 3), np.float32)
        alpha = np.zeros((N, H, W), np.float32)
        maps = [warp.identity_maps(H, W) for _ in range(N)]
        for n_done, i in enumerate(frames):
            j = int(jsel[i])
            m = warp.identity_maps(H, W)
            if p.correspondences is not None:
                c = p.correspondences(i, j)
                if c is not None:
                    m = warp.correspondence_field(c[0], c[1], (H, W), smoothing=cfg.corr_smoothing,
                                                  grid_step=cfg.corr_grid)
            if cfg.snap:
                target = p.plate_masks[i]
                if plate_key is not None:
                    target = np.minimum(target, plate_key[i]) if target.max() > 0 else plate_key[i]
                cur = warp.remap(p.ai_masks[j], *m)
                region = binary_dilation(target > 0.5, iterations=int(cfg.snap_max_shift * 2))
                band = cfg.snap_band if p.correspondences is not None else None
                s = key.edge_snap_iterative(cur, target, iters=cfg.snap_iters, min_px=cfg.snap_min_px,
                                            max_shift=cfg.snap_max_shift, sigma=cfg.snap_sigma,
                                            region=region, band=band)
                m = warp.compose(m, s)
            maps[i] = m
            say(f"person {p.name}", (n_done + 1) / len(frames))
        if cfg.temporal_sigma > 0:
            maps = warp.temporal_smooth(maps, cfg.temporal_sigma)
        for i in frames:
            j = int(jsel[i])
            rgb[i] = warp.remap(ai[j], *maps[i], border="replicate")
            alpha[i] = warp.remap(p.ai_masks[j], *maps[i])
        results[p.name] = PersonResult(p.name, curves[p.name], rgb, alpha, maps, jsel)

    # -- 5. restack
    comp = np.zeros((N, H, W, 3), np.float32)
    for i in frames:
        layers = []
        for p in people:
            d = p.depth if np.ndim(p.depth) == 0 else p.depth[i]
            layers.append(stack.Layer(results[p.name].rgb[i], results[p.name].alpha[i], float(d), p.name))
        comp[i] = stack.restack(bg[i], layers)

    # -- 6. QC after
    after_people = {}
    for p in people:
        after_people[p.name] = qc.person_report(p.plate_masks[frames], results[p.name].alpha[frames],
                                                plate[frames], comp[frames])
    rep_before = qc.shot_report(before_people, cam_delta)
    rep_after = qc.shot_report(after_people)
    return LockResult(comp, bg, bg_valid, bg_maps, results, g, cam_delta, rep_before, rep_after, frames)
