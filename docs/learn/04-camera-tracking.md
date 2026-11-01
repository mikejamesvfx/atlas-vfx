# 04 Camera tracking: Solve Plate and friends, a 3D tracker inside ComfyUI

Built 2026-11-01. Four nodes in `atlas_vfx/camera_side/`: **Solve Plate** (catalogue 4), **Solve Mask** (67),
**Track 2D** (68) and **Solve View** (69). Together they're a camera tracker that runs inside ComfyUI, so a plate can
go from footage to a shot camera without a trip to NukeX and back.

## 1. What it's for

You've got a handheld plate and you want to put things in it: a card, a CG element, an LTX-generated character, a
relit layer. Everything downstream in atlas-vfx (Lock Check, Card Project, Measure & Scale, the LTX clean loop, the
Nuke Builder) needs a camera that matches the plate frame by frame. Before the tracker that camera always came in
from somewhere else (a NukeX `.chan` / JSON, an Alembic, an old pycolmap folder) through Camera Import. Now you can
solve it here.

In artist terms, Solve Plate is CameraTracker's Track + Solve with these settings baked in:

| NukeX CameraTracker | Solve Plate |
|---|---|
| Camera Motion: Free Camera | `camera_move` auto (free SfM, nodal fallback) |
| Camera Motion: Rotation Only | `camera_move` nodal |
| Focal Length: Unknown Constant | `solve_focal` on, `focal_guess_mm` 0 or a zoom mark |
| Lens Distortion: Unknown Lens | `solve_distortion` on, `lens_model` k1 k2 |
| Mask input | `masks` (Solve Mask, or any MASK batch) |
| User tracks | `tracks` (Track 2D) |
| Max Track Error / Delete Rejected | `reject_px` (applied automatically, twice) |
| Solve Error | `mean_reproj_px` in the report |
| Export Camera | the `shot_camera` output, then Camera Export (25) or the Nuke Builder |

The camera comes out **unscaled**: it's in the solve's own units, not metres. Measure & Scale (7) sets the scale and
the ground plane. That's the same as NukeX, where the solve is arbitrary until you set a scene scale.

## 2. The mental model

```
 plate (F-Log, 4K)
   │  Tracking Proxy recipe: luma code values, per-frame stretch, CLAHE, 1024 wide
   ▼
 proxies ──► SIFT features per frame (up to 8192) ──► sequential matching (each frame vs 15 neighbours)
   ▲                                                     │  + Track 2D points as extra matches
   │ Solve Mask: "person. tree. water." ─► no features   ▼
   │                                              two-view geometry per pair (is this pair a pan? planar?)
   │                                                     │
   │                          ┌──────────────────────────┴──────────────────────────┐
   │                          ▼                                                     ▼
   │               incremental SfM (pycolmap)                          nodal solve (rotation only)
   │               poses + 3D points + focal + k1 k2                   rotations + focal + k1 k2
   │               bridge pass for dropped frames                      points on a sphere 100 units out
   │               motion test: drop points > reject_px, re-adjust
   │                          └──────────────────────────┬──────────────────────────┘
   ▼                                                     ▼
 Solve View (3D panel, click to seed) ◄── shot camera (unscaled, lens = brown, longest run of frames) + report
```

The key idea: **a camera tracker never sees "the camera".** It sees thousands of little patches drifting across the
frame and finds the one camera path, lens and set of 3D points that explains all the drifting at once. Parallax
(near things moving more than far things) is what gives it depth and translation. No parallax, no depth: that's the
nodal case.

## 3. How it works

Everything heavy lives in `atlas_vfx/camera_side/plate_solve.py:solve_folder`. The node
(`node_solve_plate.py:AtlasVFXSolvePlate.run`) writes the proxies, converts widgets to `SolveOptions`, calls it, and
turns the result into a shot camera.

### Tracking images

`plate.py:write_tracking_proxies` makes the images the solver sees: luma code values straight off the stream (range
from the stream, so a full-range Fuji plate isn't crushed), a per-frame 0.5 to 99.5 % stretch, then CLAHE (local
contrast) and down to 8 bit, at `proxy_width` (1024 default). On an F-Log plate the luma is already log-ish, which is
what you want: shadows and highlights both keep texture. These proxies are never comped; they exist for features.

### Features and matching

`PycolmapSIFT.extract` runs COLMAP's SIFT on every proxy, one shared camera for the whole clip
(`CameraMode.SINGLE`, since it's one lens), up to `max_features` per frame. A focal guess, if you give one, becomes
the starting camera (`focal_guess_mm / sensor_width_mm * proxy_width` pixels). Masks become COLMAP mask PNGs
(`write_masks`: white in an Atlas mask means moving thing, so no features there), dilated by `mask_dilate_px`.

`PycolmapSIFT.match` is sequential matching: each frame against the next `overlap` frames (15), plus a few
quadratically further ones. Every matched pair gets a **two-view geometry**: COLMAP decides whether the two frames
are related by a general camera move, a plane, or a pure rotation (panoramic). Hold that thought, it matters for
nodal.

Track 2D points go in through `tracks2d.py:add_track_matches`: each tracked position becomes an extra keypoint in its
frame, matched to the same label in the next tracked frame, but only on pairs the matcher already verified.

### What pycolmap actually does

`_map` calls `pycolmap.incremental_mapping`. Incremental SfM in a nutshell:

1. pick a good starting pair with plenty of matches and enough parallax, solve their relative pose and triangulate
   points;
2. register the next frame against the points already in 3D (a PnP solve), triangulate its new points;
3. every so often, **bundle adjust**: nudge every pose, every 3D point, the focal and k1/k2 together so all the
   reprojections line up as well as possible;
4. repeat until no more frames register.

Two repo-specific tweaks in `_mapper_options`:

- the principal point is held at the centre (`refine_principal_point` False), focal and distortion are refined;
- global bundle adjustment runs every 40 % of model growth instead of COLMAP's 10 %, capped at 30 iterations
  (commit 959f92a). On a 336-frame walk that took mapping from 1822 s to 1031 s with the camera unchanged against the
  pixels (0.164 vs 0.163 deg).

The lens is solved **on the raw features**, with the poses. Nothing is undistorted first. The solve's model is
COLMAP RADIAL (f, cx, cy, k1, k2), which the contract stores as `brown` [k1 k2 p1 p2 k3]
(`camera_io.py:from_colmap`).

### Bridging and the motion test

If the mapper drops frames or splits the shot, the **bridge** pass maps again starting from the biggest model, with
looser registration thresholds, so stragglers can attach. What still fails is reported in `missing` and never
interpolated: the camera keeps the **longest contiguous run** of registered frames.

Then `reject_and_refine` is the automatic version of NukeX's Delete Rejected: points whose worst reprojection is over
`reject_px` (2 px at proxy size) are deleted, a global bundle adjust runs, and that repeats (two rounds). A last pass
drops anything still over tolerance without another adjust (a fix after one point behind the camera blew the mean
error up to 1e170 on a real clip). Points on things that moved fail this test because no single 3D position explains
them. In the repo test a textured board sliding through the frame has its points dropped this way, and with a mask it
never gets any.

### Nodal shots: when there is no parallax

A camera turning about its lens (a tripod pan) gives SfM nothing to triangulate. SfM still "solves" it, but with no
baseline it trades focal against distortion freely and invents a lens. On the repo's synthetic pure pan that was focal
+98 % and k1 +0.61.

`nodal_solve.py:solve` is the other path. Every verified match is a ray. Undistort it, rotate it from one frame to the
other, project it back, compare. For a given lens the frame rotations can be fitted exactly (Kabsch, then rotation
averaging over all pairs), so the only thing left to search is the lens itself: 1 to 3 numbers (focal, k1, k2),
started from a sweep and polished with Nelder-Mead. A pure rotation pins the focal well (homography
self-calibration). The points are then placed on their mean rays at 100 units, because a pan has no depth.

Deciding which path a shot takes is the subtle bit, and it changed on the day (commit 792ace7):

- **First rule:** go nodal when most matched pairs look rotation-only. It sent the DSCF2977 beach walk nodal (97 % of
  pairs looked panoramic) and gave focal 18.05 mm and k1 +0.056 against NukeX's 15.52 mm. Wrong. Pairs are frames up
  to 15 apart, and on a walk those look like pure rotation too.
- **Rule now** (`solve_folder`, `_implausible_lens`): nodal only if SfM can't solve the shot at all, **or** the shot is
  panoramic **and** SfM's lens is impossible (|k1| or |k2| over 0.25, focal more than 2x off the guess, or with no
  guess a field of view outside 10 to 120 degrees). Free SfM on DSCF2977 then gave focal 15.80 mm, k1 -0.025.
- `camera_move` forces either path: `free` always SfM, `nodal` always rotation-only.

Speed (from the NOTES and commit 099966e): the nodal solve was 6.4x faster after moving all matches into flat arrays
(a synthetic 24-frame pan went from 46.6 s to 7.3 s). On real clips: DSCF3000, 297 frames, 8 to 9 min free;
DSCF2977, 336 frames, 38 min free SfM before the BA tweak.

### Degeneracy report

`plate_solve.py:degeneracy` writes three numbers into the report: the share of pairs that look panoramic, the median
triangulation angle, and the camera path over the median scene depth. A shot under 2 degrees median angle or 0.05
path-over-depth gets a **low parallax** warning: the focal is weakly determined, check it against the lens or lock it
from a chart (Chart Calibrate, catalogue 49).

### Solve error, and what a good number is

`mean_reproj_px` is the mean distance between where each feature was seen and where the solved camera puts its 3D
point. It's measured **at proxy size** (1024 wide by default), so multiply by 4 for a rough DCI 4K equivalent. NukeX
quotes its Solve Error at plate size.

| Solve | Error | Notes |
|---|---|---|
| Synthetic dolly (repo test) | under 0.8 px required | focal within 1 %, k1 within 0.01, from an 8 % wrong guess |
| DSCF3000 0-296, free lens (NOTES) | 0.43 px at 1024 | 297/297 frames, focal 16.55 to 16.67 mm run to run |
| NukeX on DSCF2977, Mike's retrack (NOTES) | 1.04 px at 4K | focal 15.52 mm |

But the error is **not** the score. It measures how self-consistent the solve is, not whether it's right. A solve can
be wrong and tidy (the nodal pan above has a tiny error and a made-up lens; you'll see it in the exercise). The repo
judges cameras two other ways:

- against another camera of the same shot: **Solve Compare** (6) fits a similarity (Umeyama: scale, rotation,
  translation) between camera centres and reports the centre residual as a % of path and the view-angle difference
  after the fit;
- against the pixels themselves: on DSCF2977, the essential matrix of 164 frame pairs 6 to 12 frames apart, compared
  with each camera's relative rotation. Free SfM: 0.16 deg median. NukeX: 1.00 deg (1.09 over the walk). NukeX
  drifts in the walk; this solve doesn't.

On DSCF3000 against Mike's NukeX camera the view angle agreed to 0.28 / 0.47 / 0.65 deg median (shots S1 / S2 / S3)
with the centre residual at 0.12 % of path. The older pycolmap solve with the focal locked at 16 mm was 6 to 10x
further from NukeX, so work built on it (Measure, depth) is worth redoing.

### Lens and rolling shutter

The solved lens travels in the shot camera's `lens` block and `ShotCamera.project()` applies it, so Lock Check and Card
Project draw on the plate as shot. For pinhole layers there's **clean space** (`clean_space.py:clean_camera`): the same
poses on an undistorted, possibly overscanned frame. Sensor Lens Maps (29) makes the undistort / redistort STMaps, and
`camera_checks.py:lens_needs_clean` refuses to comp pinhole layers onto a lensed plate without them (unless
`allow_raw`).

Solve Plate is a **global-shutter** solve. On a rolling-shutter plate (the X-H2 reads a DCI 4K frame top to bottom in
roughly 27 to 29 ms) a GS solve is bent: each pose absorbs the skew. **RS Solve** (47, `rs_solve.py`) chains after it:
it re-solves poses and points with the readout in the projection (row time from `rs_time.py`, the one time model shared
with the Sensor Pass), undistorts the observations first, and moves the result back onto the input world. On DSCF3000
the free readout came out at 26.6 ms.

### COLMAP to Nuke

COLMAP stores, per image, world-to-camera with **x right, y down, z forward**. Nuke (and USD, and this repo) use
**y up, camera looking down -Z**. `camera_io.py:from_colmap` does:

```
R_c2w_colmap = R_w2c.T
centre       = -R_w2c.T @ t_w2c                # unchanged by the axis flip
R_c2w_nuke   = R_c2w_colmap @ diag(1, -1, -1)  # flip camera y and z
```

then writes focal in mm from the sensor width, the lens as brown, frame numbers as the last number in the image name +
`frame_offset` (1001). Nuke's Euler order is ZXY, meaning `R = Ry @ Rx @ Rz`
(`contracts/shot_camera.py:euler_zxy_to_matrix` and its inverse `matrix_to_euler_zxy`), which is what Camera Export
writes to the `.chan`. The solve's world isn't gravity aligned: up is wherever the first frame put it, until
Measure & Scale.

## 4. The nodes

| Node | Key inputs | Outputs | Notes |
|---|---|---|---|
| **Solve Plate** (4) | `plate_ref`, `sensor_preset` (auto reads the body), `focal_guess_mm`, `solve_focal`, `solve_distortion`, `lens_model`, `proxy_width` 1024, `max_features` 8192, `overlap` 15, `reject_px` 2.0; optional `masks`, `mask_dilate_px` 4, `tracks`, `seed`, `camera_move` auto/free/nodal | `shot_camera`, `path`, `report`, `preview` (points green under 0.5 px, yellow to 2, red beyond) | writes proxies, `database.db`, `sparse/0`, `solve_report.json`, the camera + sidecar under `~/atlas-vfx-work/<plate>/solve/<name>` |
| **Solve Mask** (67) | `classes` ("person. car. bicycle. dog. tree. bush. water. flag."), `window` 16, thresholds, `dilate_px` 6 | `masks`, `preview`, `report` | Grounding DINO boxes on each window's middle frame, SAM 2.1 tracks them both ways, windows unioned. Models via `tools/fetch_solve_models.py` (both Apache-2.0) |
| **Track 2D** (68) | `seeds` JSON `[{"label", "frame", "xy"}]`, `scale` 0.5, `span`, `fb_px` 1.0, `window` 41, `levels` 4 | `tracks` (ATLAS_TRACKS_2D), `preview`, `summary` | pyramidal Lucas-Kanade both ways, kept while the forward-backward error stays under `fb_px` |
| **Solve View** (69) | `shot_camera`, `max_points`, `plate_frames`, `seeds` | `seeds` | three.js panel: points by error, camera path, frustum, per-frame error strip, plate frames with points drawn; **click a plate frame to add a seed** for Track 2D. Display only |

Sockets: cameras are `ATLAS_TRACKED_CAMERA`, the plate is `ATLAS_PLATE_SEQ`, tracks are `ATLAS_TRACKS_2D`.

## 5. Walkthrough: `workflows/solve_plate_s1s3.json`

The shipped graph solves DSCF3000 frames 0 to 296 and checks it against an older camera:

1. **Plate Read** (frames 0 to 296, input space auto, so F-Log is read from the Fuji tag).
2. **Solve Mask** with the default classes.
3. **Solve Plate**: sensor auto (the X-H2 DCI 4K preset, 23.5 x 12.39 mm), focal guess 16 mm, focal and k1 k2 solved,
   1024 proxies, masks in.
4. **Solve View** on the result, 8 plate frames. Scrub, look for red points and frames with a high error.
5. **Track 2D**, empty seeds. Click in Solve View to add seeds, rerun, feed `tracks` back into Solve Plate if a
   stretch is thin on features.
6. **Camera Import** of the old f16 pycolmap folder and **Solve Compare** against it (Preview Any shows the numbers).
7. **Lock Check** on the ground (4 x 4 card) to see if it sticks.
8. **Camera Export** for Nuke.

To use it on your own plate, change Plate Read's path and range, set the sensor (or `custom`), and give a focal guess
if you know the lens. On a tripod shot set `camera_move` to nodal.

## 6. Hands-on: the exercise

```
<ComfyUI venv python> docs/learn/exercises/04_camera_tracking.py
```

It ray traces a little courtyard through a camera with a known path, focal (16.47 mm on 23.5 mm) and barrel lens,
solves it with the repo's code, and grades the answer. No plates, no weights, no GPU, writes only to a temp folder.
About two minutes. Output from a real run:

```
[1] render two plates at 640x360: lens 16.47 mm on a 23.5 mm sensor = 448.5 px, k1 -0.08, k2 0.012
   walk: 24 frames, camera travels 3.0 m while turning 24 deg
   pan:  16 frames, camera stays put and turns 36 deg (a tripod pan)

[2] Solve Plate's engine on the walk, focal guess 1.08 x truth, reject_px 2.0
   mode sfm, 24/24 frames, 10271 points, mean reprojection 0.103 px (at 640 wide), solved in 26.9 s
   motion test: round 1 dropped 94, round 2 dropped 1
   focal 448.54 px vs truth 448.54 (-0.00 %), started +8 % off
   k1 -0.0797 (truth -0.08), k2 +0.0119 (truth 0.012)
   parallax: panoramic pairs 43%, median triangulation angle 4.0 deg, warnings: none

[3] into a shot camera (camera_io.from_colmap) and graded against the true path (Solve Compare)
   shot camera frames 1001-1024, focal 16.470 mm, lens brown [-0.0797, 0.0119, 0.0, 0.0, 0.0], scale.method 'unscaled'
   walk_solve vs truth: 24 frames 1001-1024, scale b/a 0.278152, centre residual median 0.01 % of path,
     view angle median 0.03 deg (p95 0.04), focal 16.47 vs 16.47 mm
   rotation error after a global fit: median 0.0060 deg, max 0.0119 deg
   scale truth/solve 0.278: a solve has no idea how big the world is

[4] COLMAP -> Nuke axes by hand, one frame
   plate_0012.png -> shot frame 1013
   camera centre [0.2609, -0.0813, 0.0586] (solve units, unchanged by the flip)
   Nuke rotate (rot_order ZXY) = (-1.883, -167.576, 179.167) deg
   ok    by-hand conversion matches what from_colmap wrote into the shot camera
   ok    ZXY Euler round trip: R = Ry @ Rx @ Rz gives the same matrix back

[5] the tripod pan, NO focal guess: first camera_move auto, then camera_move nodal
   auto:  panoramic pairs 100%, SfM lens check: k1 +0.262 -> mode nodal
          focal 449.23 px vs truth 448.54 (+0.2 %), k1 -0.0794, 0.182 px, in 36.8 s
   nodal: 16/16 frames, rms 0.191 px, focal 448.83 px vs truth 448.54 (+0.06 %), k1 -0.0795 (truth -0.08)

[6] the checks every camera-taking node runs (camera_side/camera_checks.py)
   walk: nodal=False, lens=True
     -> walk_solve is unscaled (scale.method unscaled: solve units, not metres) ... run Measure & Scale first
   pan: nodal=True, lens=True
     -> pan_solve is a nodal solve (camera turning about one point, no depth), so its Measure widgets mean nothing

total 114.8 s
PASS
```

Things to notice:

- **Step 2:** 43 % of the walk's matched pairs look panoramic, yet it's clearly a translating shot. That's exactly the
  trap the first nodal rule fell into on DSCF2977.
- **Step 3:** the scale between solve and truth is 0.278. Every number is right except how big the world is.
- **Step 4:** the Euler angles look mad (ry -167, rz 179) because the solve's world has its own idea of up. Fine for
  Nuke; Measure & Scale is what makes it sensible.
- **Step 5 sits on a knife edge.** In this run SfM's lens came out with k1 +0.262, just past the 0.25 limit, so auto
  went nodal by itself. In an earlier run of the same scene with 2x supersampled renders SfM's lens came out at
  focal +48 %, k1 +0.177, which is wrong but under the limits, so auto kept SfM, at **0.063 px**, with a
  "100 % rotation-only, if this was a tripod pan set camera_move nodal" warning. A low error with a made-up lens. The
  exercise passes either way, because the lesson is the same: read the warnings, and on a tripod shot say nodal.

Try the knobs at the top of the file: a 30 % wrong focal guess, `REJECT_PX = 0.3`, `WALK_STEP_M = 0.0`.

## 7. Gotchas, and what the NOTES record

**Culling to a pretty number is a trap.** A real NukeX session on DSCF2977 (Fuji X-H2 handheld beach plate, DCI 4K,
filmback 23.5 x 12.39, frames 1001 to 1334) makes the point:

| Step | Max Track Error / Max Error | Error | Tracks |
|---|---|---|---|
| first solve | 2.0 / 4.8 | 1.10 px | ~217 |
| Delete Rejected at Min Length 10 | 1.5 / 2.0 | 0.68 px | ~50, some frames empty |
| re-track + one Solve | auto | 2.35 px | 714 |
| tighten | 3.0 / 8.0 | 1.58 px | ~510 |
| tighten | 2.0 / 5.0 | 1.50 px | ~391 |
| tighten | 1.6 / 4.0 | 1.25 px | ~308 (81 to 184 per frame) |

The 0.68 px "solve" was the worst camera in the table: frames with no tracks have nothing holding them. The final
1.25 px (focal 15.52 mm) is honest, and its leftover error sits on a fast, motion-blurred whip around frame 1085,
where the red tracks are on real rock. Tighten in small steps, watch the per-frame track count, and stop when the
number stops falling. Solve Plate's own cull is deliberately modest: two rounds at `reject_px` 2 px (proxy size),
each followed by a re-adjust, and it never drops whole frames.

**Nuke Indie notes.** In Indie, calling CameraTracker's Update Solve from Python updates the stats but not the camera:
press the button. Python in Indie may create or touch only 10 nodes per session, Indie saves `.nkind` (encrypted, so
nothing here can read Mike's scripts), and the Nuke Builder writes a `build_<shot>.nk` to paste for that reason.

**Other things the NOTES and code record:**

- The camera is **unscaled**. Nodes that want metres call `camera_checks.require_metric` and warn (or refuse in
  strict mode). A **nodal** camera can never be scaled: its points are on a sphere 100 units out, there is no depth.
- **Frame numbering:** the camera keeps the longest run of registered frames, and shot frame = plate index + offset.
  `check_frame_offset` refuses a camera numbered with another offset (every frame would be off by the difference).
- **Run to run:** the multithreaded mapper isn't bit-identical. DSCF3000's focal landed between 16.55 and 16.67 mm.
  Set `seed`, but expect small wobble.
- **A point behind the camera** used to crash the motion test (pycolmap's `img_from_cam` returns None there). Fixed
  (commit 6c4e93c): such observations count as infinitely wrong.
- **Low parallax** shots weakly determine the focal. Believe the warning, give a real focal guess, or lock the lens
  from a chart.
- **Rolling shutter:** don't stack the physical readout on top of a GS solve, that counts the skew twice (deck test,
  2026-10-21). Use RS Solve.
- **Masks:** Solve Mask sees the plate through the shipped OCIO display view, at proxy width; Solve Plate resizes them.
  One mask per plate frame or it refuses.
- **Not built yet:** the feature front end sits behind `FRONT_ENDS` so a learned extractor/matcher can be added
  later; only SIFT exists today. Loop detection needs a vocab tree path and is off by default.

## 8. Check yourself

<details>
<summary>Five questions (click for answers)</summary>

1. **Your walk-and-pan plate reports 90 % panoramic pairs. Is it a nodal shot?**
   Not necessarily. Pairs are frames up to `overlap` apart and look like pure rotation on a walk too. Solve Plate only
   goes nodal if SfM fails or its lens comes out impossible. Check the focal against the lens and the triangulation
   angle.

2. **Solve Plate says 0.4 px, NukeX says 1.1 px on the same shot. Is ours three times better?**
   No. Solve Plate measures at proxy size (1024 wide), NukeX at plate size (4096): 0.4 px at 1024 is about 1.6 px at
   4K. And error measures consistency, not correctness. Compare the cameras (Solve Compare) or check against the pixels.

3. **Write the COLMAP to Nuke rotation conversion.**
   `R_c2w_nuke = R_w2c_colmap.T @ diag(1, -1, -1)`, centre `-R_w2c.T @ t`. Nuke rotate in ZXY means `R = Ry @ Rx @ Rz`.

4. **Why can't Measure & Scale put metres on a tripod pan?**
   A nodal solve has no baseline, so no depth: every point sits on a sphere 100 units out. There's nothing to measure.
   Place things by rotation only, or shoot something with parallax.

5. **You've got water and palm fronds in frame and the solve is noisy. What do you reach for, in order?**
   Solve Mask with "water. tree." (features never land there), then let the motion test drop the rest. If a stretch
   of the shot is thin on features, seed points on rock in Solve View, track them with Track 2D and feed the tracks
   in. Don't just lower `reject_px` until the number looks nice.

</details>
