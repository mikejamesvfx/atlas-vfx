"""Known-answer test bed.

Renders a "plate" (two textured performers on a blue screen, rolling-shutter
camera with shake) and a matching "AI take" with deliberate, known damage:

  * relight (gain + tint) on everything
  * background present (the AI invented it), seen through a drifting,
    global-shutter camera in a different solve gauge (scale/rotation)
  * each performer retimed independently, slightly fatter, offset
  * no rolling shutter anywhere

Every stage can then be scored against the answer.

Performers live in screen space (a moving textured ellipse each). That keeps
the people half honest about what it is: a 2D silhouette/texture lock, which
is exactly what body-solve correspondences reduce to per frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import Sim3
from .camera import CameraTrack, Intrinsics, rot_xyz

BLUE = np.array([0.05, 0.12, 0.62], np.float32)


def bg_texture(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Procedural wall texture in world units (structure at several scales)."""
    r = 0.55 + 0.25 * np.sin(1.7 * X + 0.3 * Y) + 0.12 * np.sin(5.3 * X - 2.1 * Y)
    g = 0.50 + 0.22 * np.sin(1.3 * Y - 0.6 * X) + 0.12 * np.sin(4.1 * Y + 3.7 * X)
    b = 0.45 + 0.18 * np.sin(2.3 * X + 1.9 * Y) + 0.10 * np.cos(6.1 * X * 0.5 + Y)
    chk = ((np.floor(X * 1.5) + np.floor(Y * 1.5)) % 2) * 0.12
    rgb = np.stack([r + chk, g + chk * 0.6, b - chk * 0.4], -1)
    return np.clip(rgb, 0.02, 1.0).astype(np.float32)


@dataclass
class Performer:
    name: str
    base: tuple                 # (x0, y0) px
    radii: tuple                # (rx, ry) px
    colour: tuple               # rgb
    depth: float                # metres from camera (for stacking)
    amp: tuple = (26.0, 7.0)
    freq: tuple = (0.11, 0.23)
    drift: float = 0.8
    lean: float = 0.25
    phase: float = 0.0

    def state(self, s):
        s = np.asarray(s, float)
        cx = self.base[0] + self.amp[0] * np.sin(self.freq[0] * s + self.phase) + self.drift * s
        cy = self.base[1] + self.amp[1] * np.sin(self.freq[1] * s + 1.3 + self.phase)
        th = self.lean * np.sin(0.17 * s + self.phase)
        return cx, cy, th

    def local(self, x, y, s, radii=None, offset=(0.0, 0.0)):
        rx, ry = radii or self.radii
        cx, cy, th = self.state(s)
        dx, dy = x - cx - offset[0], y - cy - offset[1]
        c, sn = np.cos(-th), np.sin(-th)
        a = (c * dx - sn * dy) / rx
        b = (sn * dx + c * dy) / ry
        return a, b

    def shade(self, a, b):
        base = np.asarray(self.colour, np.float32)
        pat = 0.62 + 0.38 * np.sin(7.0 * a) * np.cos(5.0 * b) + 0.12 * np.sin(13.0 * (a + b))
        return np.clip(base[None, :] * pat[:, None], 0, 1)

    def to_pixel(self, a, b, s, radii=None, offset=(0.0, 0.0)):
        rx, ry = radii or self.radii
        cx, cy, th = self.state(s)
        c, sn = np.cos(th), np.sin(th)
        dx, dy = a * rx, b * ry
        return cx + offset[0] + c * dx - sn * dy, cy + offset[1] + sn * dx + c * dy


@dataclass
class AIDamage:
    gain: tuple = (1.12, 1.04, 0.92)       # relight
    camera_yaw_drift: float = 0.0016       # rad / frame
    camera_pitch_drift: float = -0.0007
    camera_trans_drift: tuple = (0.004, 0.0, 0.0)
    gauge_scale: float = 0.5               # AI solve units = metres * scale
    gauge_rot: tuple = (0.1, -0.3, 0.2)
    radii_scale: tuple = ((1.06, 1.03), (1.04, 1.07))
    offsets: tuple = ((2.0, -1.5), (-1.5, 2.0))
    # AI frame j shows performance time h(j): independent per performer
    retime: tuple = ((0.90, 0.6, 0.25, 0.0), (1.05, 0.4, 0.2, -0.5))  # (rate, amp, freq, bias)

    def h(self, k: int, j):
        rate, amp, fr, bias = self.retime[k]
        j = np.asarray(j, float)
        return rate * j + amp * np.sin(fr * j) + bias


@dataclass
class SynthShot:
    plate: np.ndarray            # (N,H,W,3) blue screen plate, rolling shutter
    plate_masks: np.ndarray      # (P,N,H,W) per performer
    plate_bg_gt: np.ndarray      # (N,H,W,3) relit wall through the plate camera (answer for the reshoot)
    plate_track: CameraTrack
    ai: np.ndarray               # (M,H,W,3)
    ai_masks: np.ndarray         # (P,M,H,W)
    ai_bg: np.ndarray            # (M,H,W,3) wall only (what an inpaint would recover)
    ai_depth: np.ndarray         # (M,H,W) in AI solve units
    ai_track: CameraTrack        # in the AI solve gauge, global shutter
    true_gauge: Sim3             # AI gauge world -> plate world (the answer)
    tau_true: np.ndarray         # (P,N) AI frame for each plate frame
    performers: list
    damage: AIDamage
    readout: float
    meta: dict = field(default_factory=dict)


def _render_wall(track: CameraTrack, frame: int, plane_z: float, gain=None):
    H, W = track.intr.height, track.intr.width
    vs, us = np.mgrid[0:H, 0:W].astype(float)
    o, d = track.ray(us, vs, frame)
    lam = (plane_z - o[:, 2]) / d[:, 2]
    X = o + lam[:, None] * d
    rgb = bg_texture(X[:, 0], X[:, 1]).reshape(H, W, 3)
    depth_world = lam.reshape(H, W)  # along-ray param with unit-z ray dirs in cam ≈ cam z
    if gain is not None:
        rgb = np.clip(rgb * np.asarray(gain, np.float32), 0, 1)
    return rgb, depth_world


def _render_people(perfs, H, W, times_of_row, radii=None, offsets=None, gain=None):
    """Composite performers; times_of_row(v) gives performance time per row."""
    vs, us = np.mgrid[0:H, 0:W].astype(float)
    s = times_of_row(vs)
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    masks = []
    for k, p in sorted(enumerate(perfs), key=lambda kp: -kp[1].depth):
        rad = radii[k] if radii is not None else None
        off = offsets[k] if offsets is not None else (0.0, 0.0)
        a, b = p.local(us, vs, s if np.ndim(s) else np.full_like(us, s), rad, off)
        r2 = a * a + b * b
        m = np.clip((1.0 - np.sqrt(r2)) * min(p.radii) * 0.8 + 0.5, 0, 1).astype(np.float32)  # ~1px AA edge
        col = p.shade(a.ravel(), b.ravel()).reshape(H, W, 3)
        if gain is not None:
            col = np.clip(col * np.asarray(gain, np.float32), 0, 1)
        rgb = col * m[..., None] + rgb * (1 - m[..., None])
        alpha = m + alpha * (1 - m)
        masks.append((k, m))
    masks = [m for _, m in sorted(masks)]
    return rgb, alpha, masks


def make_shot(n_plate: int = 24, n_ai: int = 30, width: int = 160, height: int = 96,
              focal: float = 140.0, readout: float = 0.6, plane_z: float = 8.0,
              damage: AIDamage | None = None, seed: int = 0, px_scale: float = 1.0) -> SynthShot:
    """px_scale renders the same shot bigger (frame, focal, performers, offsets all scale)."""
    dmg = damage or AIDamage()
    if px_scale != 1.0:
        from dataclasses import replace
        width, height, focal = int(round(width * px_scale)), int(round(height * px_scale)), focal * px_scale
        dmg = replace(dmg, offsets=tuple((o[0] * px_scale, o[1] * px_scale) for o in dmg.offsets))
    rng = np.random.default_rng(seed)
    intr = Intrinsics.from_fov(width, height, focal)

    # plate camera: handheld-ish pan with shake, true metric world
    def plate_pose(f):
        yaw = 0.006 * f + 0.004 * np.sin(0.9 * f)
        pitch = 0.003 * np.sin(0.7 * f + 0.5)
        roll = 0.002 * np.sin(1.1 * f)
        R = rot_xyz(pitch, yaw, roll)
        C = np.array([0.01 * f, 0.004 * np.sin(0.5 * f), 0.0])
        return R, -R @ C

    Rp, tp = zip(*[plate_pose(f) for f in range(max(n_plate, n_ai) + 2)])
    plate_track = CameraTrack(intr, np.array(Rp)[:n_plate], np.array(tp)[:n_plate], readout, "centre")

    # true AI camera: agrees with the plate at frame 0, then drifts, global shutter
    def ai_pose_true(f):
        R0, t0 = plate_pose(f)
        C0 = -R0.T @ t0
        Rd = rot_xyz(dmg.camera_pitch_drift * f, dmg.camera_yaw_drift * f, 0.0)
        R = Rd @ R0
        C = C0 + np.asarray(dmg.camera_trans_drift) * f
        return R, -R @ C

    Ra, ta = zip(*[ai_pose_true(f) for f in range(n_ai)])
    ai_true = CameraTrack(intr, np.array(Ra), np.array(ta), 0.0)

    # the AI solve reports in its own gauge: X_ai = s*Rg X_true + tg
    Rg = rot_xyz(*dmg.gauge_rot)
    tg = np.array([0.3, -0.2, 0.5])
    to_ai = Sim3(dmg.gauge_scale, Rg, tg)
    from .align import transform_track
    ai_track = transform_track(ai_true, to_ai)
    true_gauge = to_ai.inverse()

    k_ = px_scale
    perfs = [
        Performer("left", (48 * k_, 52 * k_), (13 * k_, 21 * k_), (0.85, 0.55, 0.45), 3.0, amp=(18.0 * k_, 7.0 * k_),
                  drift=0.45 * k_, phase=0.0),
        Performer("right", (112 * k_, 50 * k_), (12 * k_, 22 * k_), (0.40, 0.70, 0.55), 3.6, amp=(14.0 * k_, 6.0 * k_),
                  freq=(0.13, 0.19), drift=-0.35 * k_, phase=1.7),
    ]

    H, W = height, width
    plate = np.zeros((n_plate, H, W, 3), np.float32)
    plate_bg_gt = np.zeros_like(plate)
    plate_masks = np.zeros((len(perfs), n_plate, H, W), np.float32)
    for i in range(n_plate):
        wall, _ = _render_wall(plate_track, i, plane_z, dmg.gain)
        plate_bg_gt[i] = wall
        rgb, alpha, ms = _render_people(perfs, H, W, lambda v: plate_track.row_time(i, v))
        plate[i] = rgb * alpha[..., None] + BLUE * (1 - alpha[..., None])
        for k, m in enumerate(ms):
            plate_masks[k, i] = m
    plate = np.clip(plate + rng.normal(0, 0.004, plate.shape).astype(np.float32), 0, 1)

    ai = np.zeros((n_ai, H, W, 3), np.float32)
    ai_bg = np.zeros_like(ai)
    ai_depth = np.zeros((n_ai, H, W), np.float32)
    ai_masks = np.zeros((len(perfs), n_ai, H, W), np.float32)
    radii = [(p.radii[0] * dmg.radii_scale[k][0], p.radii[1] * dmg.radii_scale[k][1]) for k, p in enumerate(perfs)]
    for j in range(n_ai):
        wall, depth = _render_wall(ai_true, j, plane_z, dmg.gain)
        ai_bg[j] = wall
        ai_depth[j] = depth * dmg.gauge_scale
        # each performer on their own clock; render separately then stack
        layers = []
        for k, p in enumerate(perfs):
            rgb_k, a_k, ms_k = _render_people([p], H, W, lambda v, k=k: np.full(np.shape(v), dmg.h(k, j)),
                                              [radii[k]], [dmg.offsets[k]], dmg.gain)
            layers.append((p.depth, rgb_k, a_k, k))
        comp = wall.copy()
        for _, rgb_k, a_k, k in sorted(layers, key=lambda l: -l[0]):
            comp = rgb_k * a_k[..., None] + comp * (1 - a_k[..., None])
            ai_masks[k, j] = a_k
        ai[j] = comp

    # tau_true: AI frame whose performance time equals the plate frame's (mid-readout) time
    jj = np.linspace(0, n_ai - 1, 4000)
    tau_true = np.stack([np.interp(np.arange(n_plate, dtype=float), dmg.h(k, jj), jj) for k in range(len(perfs))])

    return SynthShot(plate, plate_masks, plate_bg_gt, plate_track, ai, ai_masks, ai_bg, ai_depth,
                     ai_track, true_gauge, tau_true, perfs, dmg, readout,
                     {"plane_z": plane_z, "radii_ai": radii})


def gt_correspondences(shot: SynthShot, k: int, plate_frame: int, ai_frame: int, n: int = 400, seed: int = 0):
    """Stand-in for a body solver / dense matcher: matched points for one performer.

    Returns (plate_xy, ai_xy). Plate positions honour rolling shutter.
    """
    p = shot.performers[k]
    rng = np.random.default_rng(seed + 31 * plate_frame + k)
    r = np.sqrt(rng.uniform(0, 0.92, n))
    th = rng.uniform(0, 2 * np.pi, n)
    a, b = r * np.cos(th), r * np.sin(th)
    s = np.full(n, float(plate_frame))
    for _ in range(4):  # fixed point on the row time
        x, y = p.to_pixel(a, b, s)
        s = shot.plate_track.row_time(plate_frame, y)
    px, py = p.to_pixel(a, b, s)
    sa = shot.damage.h(k, ai_frame)
    ax, ay = p.to_pixel(a, b, np.full(n, sa), shot.meta["radii_ai"][k], shot.damage.offsets[k])
    return np.stack([px, py], 1), np.stack([ax, ay], 1)
