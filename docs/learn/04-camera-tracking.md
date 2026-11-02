# 04 Camera tracking: Solve Plate and friends, a 3D tracker inside ComfyUI

## In short

- **Four nodes** in `atlas_vfx/camera_side/` (built 2026-11-01): **Solve Plate** (4) tracks and solves the camera,
  **Solve Mask** (67) keeps features off moving things, **Track 2D** (68) adds hand-seeded tracks, **Solve View** (69)
  shows the result in 3D and lets you click seeds.
- **Free SfM by default, nodal only as a fallback.** A shot goes nodal when SfM can't solve it at all, or when it's
  panoramic and SfM's lens is impossible. On a tripod shot, just set `camera_move` to nodal.
- **Solve error is at proxy size** (1024 wide), so multiply by about 4 for DCI 4K. Around 0.1 to 0.5 px at 1024 is
  good, but a low error isn't proof of a right camera: compare cameras or check against the pixels.
- **Mask movers, don't cull to a number.** Solve Mask on "water. tree. person." first, let the automatic motion test
  drop the rest. Deleting tracks until the error looks pretty leaves frames with nothing holding them.
- **COLMAP to Nuke:** `R_c2w_nuke = R_c2w_colmap @ diag(1, -1, -1)`, Euler order ZXY (`R = Ry @ Rx @ Rz`).
- **The camera comes out unscaled** and not gravity aligned. Measure & Scale (7) gives it metres and a ground.

## 1. What it's for

Everything downstream (Lock Check, Card Project, Measure & Scale, the LTX clean loop, the Nuke Builder) needs a
camera that matches the plate frame by frame. Before this, it came in from NukeX, an Alembic or an old pycolmap
folder through Camera Import. Now you can solve it here. In NukeX terms:

| NukeX CameraTracker | Solve Plate |
|---|---|
| Camera Motion: Free Camera / Rotation Only | `camera_move` auto or free / nodal |
| Focal Length: Unknown Constant | `solve_focal` on, `focal_guess_mm` 0 or a lens mark |
| Lens Distortion: Unknown Lens | `solve_distortion` on, `lens_model` k1 k2 |
| Mask input / user tracks | `masks` (Solve Mask) / `tracks` (Track 2D) |
| Max Track Error, Delete Rejected | `reject_px`, applied automatically |
| Solve Error | `mean_reproj_px` in the report |

## 2. The mental model

```
plate -> tracking proxies (luma, stretch, CLAHE, 1024 wide)
      -> SIFT features (masked areas get none) -> sequential matching (+ Track 2D points)
      -> two-view geometry per pair (general move, plane, or pure rotation?)
      -> incremental SfM (pycolmap): poses, 3D points, focal, k1 k2   [or nodal: rotations + lens only]
      -> motion test: drop points over reject_px, re-adjust
      -> shot camera (unscaled, brown lens, longest run of frames) + report -> Solve View
```

A tracker never sees "the camera". It sees thousands of patches drifting and finds the one camera path, lens and set
of 3D points that explains all of it at once. Parallax is what gives it depth and translation. No parallax, no depth:
that's the nodal case.

## 3. How it works

The heavy lifting is `camera_side/plate_solve.py:solve_folder`; the node (`node_solve_plate.py:AtlasVFXSolvePlate.run`)
writes proxies, builds `SolveOptions` and turns the result into a shot camera.

**Tracking images.** `plate.py:write_tracking_proxies` reads luma code values with the range from the stream (so a
full-range Fuji plate isn't crushed), stretches each frame, applies CLAHE and saves 8-bit proxies at `proxy_width`.
F-Log luma is already log-ish, so shadows and highlights both keep texture.

**Features and matching.** `PycolmapSIFT.extract` runs SIFT with one shared camera for the clip, up to
`max_features` per frame. White in an Atlas mask means "moving thing, no features here". `PycolmapSIFT.match` matches
each frame against the next `overlap` (15) frames and gives every pair a two-view geometry: general move, plane or
pure rotation. Track 2D points join through `tracks2d.py:add_track_matches`, only on pairs already verified.

**SfM.** `_map` calls `pycolmap.incremental_mapping`: solve a good starting pair, register frames one by one against
the 3D points, bundle adjust everything (poses, points, focal, k1 k2) as it grows. Two tweaks in `_mapper_options`:
the principal point stays at the centre, and global adjustment runs every 40 % of growth instead of 10 % (commit
959f92a took one 336-frame walk's mapping from 1822 s to 1031 s, same camera). The lens is solved on the raw
features, stored as COLMAP RADIAL and written to the contract as `brown`.

**Bridging and the motion test.** A bridge pass remaps from the biggest model with looser thresholds so dropped frames
can attach. Anything still missing is reported, never interpolated; the camera keeps the longest contiguous run.
`reject_and_refine` is the automatic Delete Rejected: points whose worst reprojection exceeds `reject_px` (2 px at
proxy size) go, the bundle re-adjusts, twice, then a final drop pass. Points on things that moved fail this because no
single 3D position explains them.

**Nodal shots.** A tripod pan gives SfM nothing to triangulate, so it trades focal against distortion and invents a
lens (focal +98 %, k1 +0.61 on the repo's synthetic pan). `nodal_solve.py:solve` treats every match as a ray, fits the
rotations exactly for a given lens, and searches only the lens (focal, k1, k2). Points go on their mean rays at 100
units, because a pan has no depth.

Which path a shot takes changed on the day (commit 792ace7). The first rule ("most pairs look rotation-only") sent the
DSCF2977 beach walk nodal at 18.05 mm, against NukeX's 15.52 mm: wrong, because on a walk pairs 15 frames apart look
like rotation too. Now (`solve_folder`, `_implausible_lens`) it goes nodal only if SfM fails, **or** the shot is
panoramic **and** SfM's lens is impossible: |k1| or |k2| over 0.25, focal more than 2x off the guess, or with no guess
a field of view outside 10 to 120 degrees. Free SfM on DSCF2977 then gave 15.80 mm.

**Degeneracy report.** `plate_solve.py:degeneracy` reports the share of panoramic pairs, the median triangulation
angle and path over scene depth. Under 2 degrees or 0.05 gets a **low parallax** warning: the focal is weakly
determined, so check it against the lens or lock it from a chart (Chart Calibrate, 49).

**Solve error.** `mean_reproj_px` is measured at proxy size; NukeX quotes plate size.

| Solve | Error | Notes |
|---|---|---|
| Synthetic dolly (repo test) | under 0.8 px required | focal within 1 %, k1 within 0.01 |
| DSCF3000 0-296, free lens | 0.43 px at 1024 | 297/297 frames, focal 16.55 to 16.67 mm run to run |
| NukeX on DSCF2977 (NOTES) | 1.04 px at 4K | focal 15.52 mm |

Error measures self-consistency, not correctness. The repo also judges cameras against each other (**Solve Compare**,
6: centre residual as % of path and view angle after a similarity fit) and against the pixels (essential matrices of
frame pairs). On DSCF2977 that pixel check gave free SfM 0.16 deg median against NukeX's 1.00 deg: NukeX drifts in
the walk. On DSCF3000 the solve agreed with Mike's NukeX camera to 0.28 to 0.65 deg median view angle.

**Lens and rolling shutter.** The lens rides in the shot camera, so Lock Check and Card Project draw on the plate as
shot. Pinhole layers use clean space (`clean_space.py:clean_camera`) with Sensor Lens Maps (29) STMaps, and
`camera_checks.py:lens_needs_clean` refuses a lensed plate without them. Solve Plate is global shutter; on the X-H2
(roughly 27 to 29 ms readout) chain **RS Solve** (47, `rs_solve.py`), which puts the readout into the projection using
`rs_time.py`. DSCF3000's free readout came out at 26.6 ms.

**COLMAP to Nuke** (`camera_io.py:from_colmap`). COLMAP is world-to-camera, x right, y down, z forward; Nuke is y up,
looking down -Z:

```
R_c2w_colmap = R_w2c.T
centre       = -R_w2c.T @ t_w2c                # unchanged by the axis flip
R_c2w_nuke   = R_c2w_colmap @ diag(1, -1, -1)
```

Focal goes to mm from the sensor width, frames get `frame_offset` (1001), and Camera Export writes ZXY Eulers
(`contracts/shot_camera.py:matrix_to_euler_zxy`). Up is wherever the first frame put it, until Measure & Scale.

## 4. The nodes

| Node | What you set | Out |
|---|---|---|
| **Solve Plate** (4) | `plate_ref`, `sensor_preset` (auto reads the body), `focal_guess_mm`, `camera_move`; optional `masks`, `tracks` | `shot_camera`, `report`, `preview` (points green under 0.5 px, yellow to 2, red beyond) |
| **Solve Mask** (67) | `classes` ("person. tree. water." and so on) | `masks` (Grounding DINO boxes, SAM 2.1 tracking; fetch with `tools/fetch_solve_models.py`) |
| **Track 2D** (68) | `seeds` JSON `[{"label", "frame", "xy"}]`, `fb_px` | `tracks` (pyramidal Lucas-Kanade both ways, kept while forward-backward error stays under `fb_px`) |
| **Solve View** (69) | `shot_camera`, `plate_frames` | `seeds`: click a plate frame to add one. Display only |

The rest (`proxy_width`, `max_features`, `overlap`, `reject_px`, dilation, windows) has sane defaults; see the
catalogue blocks.

## 5. Walkthrough: `workflows/solve_plate_s1s3.json`

1. **Plate Read** DSCF3000 frames 0 to 296, input space auto (F-Log from the Fuji tag).
2. **Solve Mask** with default classes.
3. **Solve Plate**: sensor auto (X-H2 DCI 4K, 23.5 x 12.39 mm), focal guess 16 mm, masks in.
4. **Solve View**: scrub for red points and high-error frames.
5. **Track 2D** from seeds clicked in Solve View, fed back into Solve Plate if a stretch is thin.
6. **Camera Import** of an older solve and **Solve Compare** against it.
7. **Lock Check** with a 4 x 4 ground card, then **Camera Export** for Nuke.

For your own plate: change Plate Read's path and range, set the sensor, give a focal guess if you know the lens, and on
a tripod shot set `camera_move` to nodal.

## 6. Hands-on: the exercise

```
<ComfyUI venv python> docs/learn/exercises/04_camera_tracking.py
```

It ray traces a courtyard through a known camera (16.47 mm on 23.5 mm, k1 -0.08, k2 0.012), solves a 24-frame walk
and a 16-frame tripod pan with the repo's code, and grades both against the truth. About two minutes, temp folder
only. The lines that teach something:

```
[2] mode sfm, 24/24 frames, 10271 points, mean reprojection 0.103 px (at 640 wide)
    focal 448.54 px vs truth 448.54 (-0.00 %), started +8 % off
    k1 -0.0797 (truth -0.08), k2 +0.0119 (truth 0.012)
    parallax: panoramic pairs 43%, median triangulation angle 4.0 deg, warnings: none
[3] centre residual median 0.01 % of path, view angle median 0.03 deg
    scale truth/solve 0.278: a solve has no idea how big the world is
[4] Nuke rotate (rot_order ZXY) = (-1.883, -167.576, 179.167) deg
    ok    by-hand conversion matches what from_colmap wrote into the shot camera
[5] auto:  panoramic pairs 100%, SfM lens check: k1 +0.262 -> mode nodal
    nodal: 16/16 frames, rms 0.191 px, focal +0.06 %, k1 -0.0795 (truth -0.08)
[6] walk: nodal=False ... unscaled, run Measure & Scale first
    pan:  nodal=True ... its Measure widgets mean nothing
PASS
```

Notice: 43 % of the walk's pairs look panoramic though it clearly translates (the trap the first nodal rule fell into);
the Euler angles look mad because the solve picked its own up; and step 5 sits on a knife edge (see gotchas). Try a 30 %
wrong focal guess, `REJECT_PX = 0.3` or `WALK_STEP_M = 0.0` at the top of the file.

## 7. Gotchas

**Culling to a pretty number is a trap.** A real NukeX session on DSCF2977 (X-H2 handheld beach plate, DCI 4K,
1001 to 1334):

| Step | Max Track Error / Max Error | Error | Tracks |
|---|---|---|---|
| first solve | 2.0 / 4.8 | 1.10 px | ~217 |
| Delete Rejected, Min Length 10 | 1.5 / 2.0 | 0.68 px | ~50, some frames empty |
| re-track, one Solve | auto | 2.35 px | 714 |
| tighten in steps | 3.0/8.0, 2.0/5.0, 1.6/4.0 | 1.58, 1.50, 1.25 px | 510, 391, 308 |

The 0.68 px camera was the worst in the table: empty frames have nothing holding them. The final 1.25 px (focal 15.52
mm, 81 to 184 tracks per frame) is honest; its leftover error sits on a motion-blurred whip around frame 1085 where the
red tracks are on real rock. Tighten in small steps, watch per-frame counts, stop when the number stops falling.

**The nodal rule is borderline on short pans.** In the exercise, SfM's lens on the pan came out at k1 +0.262, just past
the 0.25 limit, so auto went nodal. With 2x supersampled renders it came out at focal +48 %, k1 +0.177: wrong but under
the limits, so auto kept SfM at 0.063 px with only a warning. Read the warnings, and on a tripod shot say nodal.

**Nuke Indie.** Calling CameraTracker's Update Solve from Python updates the stats but not the camera: press the
button. Indie also caps Python at 10 nodes and saves encrypted `.nkind`, which is why the Nuke Builder writes a `.nk` to
paste.

**Also from the NOTES and code:**

- A **nodal** camera can never be scaled (points on a sphere, no depth); `camera_checks.require_metric` warns or
  refuses on unscaled cameras.
- `check_frame_offset` refuses a camera numbered with another offset.
- The mapper isn't bit-identical run to run (DSCF3000 focal 16.55 to 16.67 mm). Set `seed`, expect small wobble.
- Don't stack the physical readout on a GS solve, it counts the skew twice. Use RS Solve.
- One mask per plate frame, or Solve Plate refuses.
- Only SIFT exists today (`FRONT_ENDS` leaves room for a learned matcher); loop detection is off.

## 8. Check yourself

<details>
<summary>Five questions (click for answers)</summary>

1. **Your walk reports 90 % panoramic pairs. Is it nodal?** Not necessarily: pairs up to 15 frames apart look like
   rotation on a walk too. Solve Plate only goes nodal if SfM fails or its lens is impossible. Check the focal and the
   triangulation angle.
2. **Solve Plate says 0.4 px, NukeX 1.1 px. Is ours three times better?** No. 0.4 px at 1024 is about 1.6 px at 4K,
   and error measures consistency, not correctness. Compare the cameras.
3. **Write the COLMAP to Nuke rotation.** `R_c2w_nuke = R_w2c.T @ diag(1, -1, -1)`, centre `-R_w2c.T @ t`; Nuke ZXY
   means `R = Ry @ Rx @ Rz`.
4. **Why can't Measure & Scale put metres on a tripod pan?** No baseline, no depth: the points sit on a sphere 100 units
   out.
5. **Water and palm fronds make the solve noisy. What first?** Solve Mask with "water. tree.", then the motion test. If
   a stretch is thin, seed rock in Solve View and feed Track 2D tracks in. Don't just lower `reject_px`.

</details>
