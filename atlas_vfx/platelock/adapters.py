"""Model adapters: the only place model choices live.

The core takes data (masks, tracks, depth, correspondences). Adapters turn a
model's output into that data. Each protocol below is what a real
implementation must provide; file adapters let any tool (ComfyUI nodes,
a script on the render box, Nuke) hand results over as .npz/.json.

Licences matter for paid work: check every model you plug in here. Body
models built on SMPL/SMPL-X are typically research/non-commercial; prefer
ones with a commercial-friendly licence for client shots.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .core.camera import CameraTrack, Intrinsics


@runtime_checkable
class Segmenter(Protocol):
    """e.g. SAM 3.x video segmentation. One mask sequence per performer, same names on plate and AI."""

    def segment(self, frames: np.ndarray, prompts: dict) -> dict[str, np.ndarray]: ...


@runtime_checkable
class VideoSolver(Protocol):
    """Dynamic-scene camera + depth solve that ignores moving people.

    Must return a CameraTrack (OpenCV world->camera) and camera-space z depth
    per frame in the same units as the track. Run the SAME solver on plate
    and AI so their errors are alike.
    """

    def solve(self, frames: np.ndarray, masks_to_ignore: np.ndarray | None = None,
              intrinsics: Intrinsics | None = None) -> tuple[CameraTrack, np.ndarray]: ...


@runtime_checkable
class BodySolver(Protocol):
    """Per-frame body mesh/keypoints projected to 2D, with visibility.

    Returns (N,V,2) pixel positions and (N,V) visibility for one performer.
    Vertex V must be the same body point in every frame and in both takes.
    """

    def solve(self, frames: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class Matcher(Protocol):
    """Dense / semi-dense matcher robust to relighting (RoMa-, MASt3R-style).

    Returns (pts_a, pts_b, confidence) between two images, restricted to masks.
    """

    def match(self, img_a: np.ndarray, img_b: np.ndarray, mask_a: np.ndarray | None,
              mask_b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


CorrFn = Callable[[int, int], "tuple[np.ndarray, np.ndarray] | None"]


def corr_from_body(plate_xy: np.ndarray, plate_vis: np.ndarray, ai_xy: np.ndarray, ai_vis: np.ndarray,
                   min_points: int = 12) -> CorrFn:
    """Correspondence callback from two body solves of the same performer."""
    def fn(i: int, j: int):
        v = (plate_vis[i] > 0.5) & (ai_vis[j] > 0.5)
        if v.sum() < min_points:
            return None
        return plate_xy[i][v], ai_xy[j][v]
    return fn


def corr_from_matcher(matcher: Matcher, plate: np.ndarray, ai: np.ndarray, plate_masks: np.ndarray,
                      ai_masks: np.ndarray, min_conf: float = 0.5) -> CorrFn:
    def fn(i: int, j: int):
        pa, pb, c = matcher.match(plate[i], ai[j], plate_masks[i], ai_masks[j])
        k = c >= min_conf
        if k.sum() < 12:
            return None
        return pa[k], pb[k]
    return fn


def corr_concat(*fns: CorrFn) -> CorrFn:
    """Stack several sources (body solve for coverage + matcher for detail)."""
    def fn(i: int, j: int):
        got = [f(i, j) for f in fns if f is not None]
        got = [g for g in got if g is not None]
        if not got:
            return None
        return np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])
    return fn


# ------------------------------------------------------------ file adapters

def load_masks_npz(path: str) -> dict[str, np.ndarray]:
    """{name: (N,H,W)} from an .npz with one array per performer."""
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def load_body_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """.npz with xy (N,V,2) and vis (N,V)."""
    with np.load(path) as z:
        return z["xy"].astype(np.float64), z["vis"].astype(np.float32)


def load_correspondences_npz(path: str) -> CorrFn:
    """.npz with keys 'p{i}_a{j}_plate' / 'p{i}_a{j}_ai' (N,2) per frame pair."""
    z = dict(np.load(path))

    def fn(i: int, j: int):
        a, b = z.get(f"p{i}_a{j}_plate"), z.get(f"p{i}_a{j}_ai")
        if a is None or b is None:
            return None
        return a, b
    return fn


def static_track(intr: Intrinsics, n: int, readout: float = 0.0) -> CameraTrack:
    """A locked-off camera: identity pose every frame."""
    return CameraTrack(intr, np.repeat(np.eye(3)[None], n, 0), np.zeros((n, 3)), readout)


def flat_depth(n: int, h: int, w: int, distance: float = 10.0) -> np.ndarray:
    """Constant depth for a locked-off shot where the AI background is effectively a card."""
    return np.full((n, h, w), distance, np.float32)


# ------------------------------------------------------- atlas-vfx shot-camera contract

_CV_FROM_NUKE = np.diag([1.0, -1.0, -1.0])


def track_from_shot_camera(cam, readout: float = 0.0, rs_anchor: str = "centre") -> CameraTrack:
    """atlas-vfx shot-camera@0.1 (Nuke axes: Y up, looking down -Z, pixel origin at the corner)
    -> platelock CameraTrack (OpenCV axes, pixel centres on integers).

    Intrinsics come from the first frame; a zoom is reported, not modelled.
    `cam` is a contracts.shot_camera.ShotCamera or a path to one.
    """
    if isinstance(cam, str):
        from ..contracts import shot_camera as sc
        cam = sc.load(cam)
    frames = list(cam.frames)
    K = np.asarray(cam.intrinsics(frames[0]).K, float)
    focals = [float(cam._frame(f)["focal_mm"]) for f in frames]
    if max(focals) - min(focals) > 1e-6:
        import logging
        logging.getLogger("atlas_vfx").warning(
            "platelock: focal length varies across the shot (%.3f-%.3f mm); using frame %d",
            min(focals), max(focals), frames[0])
    p = cam.data["plate"]
    intr = Intrinsics(int(p["width"]), int(p["height"]), K[0, 0], K[1, 1], K[0, 2] - 0.5, K[1, 2] - 0.5)
    Rs, ts = [], []
    for f in frames:
        c2w = np.asarray(cam.cam_to_world(f), float)
        Rc2w = c2w[:3, :3] @ _CV_FROM_NUKE
        C = c2w[:3, 3]
        Rw2c = Rc2w.T
        Rs.append(Rw2c)
        ts.append(-Rw2c @ C)
    return CameraTrack(intr, np.array(Rs), np.array(ts), readout, rs_anchor)
