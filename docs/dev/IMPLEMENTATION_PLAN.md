# Implementation plan: Phase 1, "character on the deck"

Goal: one shot from DSCF3000, S2-S3 (plate frames 129-296 = Nuke 1130-1297), ending in a comp Mike opens in Nuke Indie
17.1 where the test character stands on the clubhouse deck behind tracked foreground grass.

Work the tasks in order. Each one ends with its test passing and a local commit. If a task turns up something that
changes the plan, stop and write it in `docs/dev/NOTES.md`, then tell Mike before going on.

Outputs for the shot live in `C:\Users\miike\atlas-vfx-work\DSCF3000_S2S3\` (create it; never inside the repo).

---

## T0. Scaffold that loads

- `__init__.py` gathers `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` from the three subpackages; a missing
  optional dependency disables only that node and logs why.
- `atlas_vfx/_atlas_camera.py`: locate and import Atlas Camera lazily (see AGENTS.md). Report what it found.
- Junction into V141, restart, confirm the (empty) pack loads with no errors in the V141 log.
- **Test:** `tests/test_import.py` imports `atlas_vfx` without ComfyUI present (pure python parts) and checks the
  camera_side boundary: no module under `camera_side/` imports `layers` or `deliver` (scan the source).

## T1. Contract `shot-camera@0.1`

- `atlas_vfx/contracts/shot_camera_v0_1.schema.json` (draft is there; finish it) + `shot_camera.py` with
  `load()`, `validate()`, `save()`, and helpers `cam_to_world(frame)`, `intrinsics(frame)`, `to_nuke_chan()`.
- **Test:** `tests/test_shot_camera.py`: round-trip save/load; validation rejects a missing frame, wrong units,
  non-orthonormal rotation; `to_nuke_chan` of a synthetic camera matches a hand-built chan to 1e-6.

## T2. Plate Read (catalogue 1) and Tracking Proxy (2)

- Port `prototypes/vfx_probe/plate_proxy.py` and `track_proxy.py` into `camera_side/plate.py`.
- Range from the stream (`ffprobe color_range`); full-range DSCF3000 must decode with `in_range=pc`. Fix the display
  proxy accordingly. Record range, colorspace (default `F-Log2 F-Gamut` for X-H2 F-Log2 files, overridable), body,
  timecode, Nuke frame offset 1001.
- Nodes: `AtlasVFXPlateRead` (path, first/last, ocio config, input space) -> ATLAS_PLATE_SEQ (+ preview IMAGE batch);
  `AtlasVFXTrackingProxy` (ATLAS_PLATE_SEQ, width) -> folder of proxies + IMAGE preview.
- **Test:** `tests/test_plate.py`: a synthetic 10-bit full-range ProRes made by ffmpeg in the test decodes to the
  values written; a legal-range file decodes differently and is labelled so; F-Log2 code 0.391 -> 0.18 ACEScg grey
  through the shipped OCIO config (1e-3).

## T3. Camera Import (5) and Solve Compare (6)

- `camera_side/camera_io.py`: read Nuke `Camera*.json` (from the export snippet) and `.chan` (frame tx ty tz rx ry rz
  [focal]) with rot_order and film back -> shot-camera@0.1. Read a pycolmap model (folder) -> shot-camera@0.1 with the
  axis conversion in AGENTS.md.
- Port `align_nuke.py` into `camera_side/compare.py`: similarity fit, centre residual (% of path), view-angle stats.
- Nodes: `AtlasVFXCameraImport`, `AtlasVFXSolveCompare`.
- **Test:** `tests/test_camera_io.py`: synthetic camera -> chan -> import -> identical within 1e-6 (rotation order
  ZXY checked with a non-trivial rotation); `tests/test_compare.py`: a camera vs itself scaled 7x, rotated and moved
  gives s = 7 and residual 0. **Acceptance on real data:** Nuke Camera1 vs `solve_s1s3_f16` reproduces median centre
  residual about 1 % and view angle about 3 degrees.

## T4. Measure & Scale (7) and Lock Check (8)

- `camera_side/measure.py`: from 2D picks (same physical point in two or more frames) triangulate with the shot camera;
  or pick + fitted plane (port `door_scale3.py`). Several labelled measurements -> metres-per-unit with spread; refuse
  (error) if spread > 5 % unless forced. Up axis: fit from 2+ vertical picks (door jamb, posts); ground plane from
  points in a picked region. Output a scaled, gravity-aligned shot camera (Y up, ground y = 0).
- Picks come in as a small JSON (label, length_m, frames: {frame: [x, y]} in plate pixels). Write
  `docs/PICKING.md`: how Mike picks points in Nuke (a Tracker or the viewer pixel readout) and pastes them.
- Port `card_check.py` -> `AtlasVFXLockCheck`: a grid card on a plane (or at a picked 3D point) projected through
  every frame onto the display proxy -> mp4.
- **Test:** `tests/test_measure.py`: synthetic scene with a known 2.04 m door and 0.18 m step seen from a moving
  camera; recovered scale within 1 %, up axis within 0.5 degrees, ground y within 1 cm.
- **Acceptance on real data:** Mike supplies picks for the door frame, one step riser and the deck rail height; the
  three agree within 5 %; Lock Check video of S2-S3 shows the card stuck to the deck by eye.

## T5. Plate Depth (9) and Set Proxy (10)

- Use ComfyUI core MoGe-3 nodes (as twentyfive does) on display proxies; scale each frame's depth to the shot camera
  by robust median against the solve's points visible in that frame. Store depth as float EXR per frame.
- `layers/set_proxy.py`: from points + depth, fit the deck plane and the wall plane for this shot (RANSAC), output a
  tiny OBJ/USD with both planes trimmed to the visible area. General meshing is not needed in Phase 1.
- **Test:** synthetic: two planes seen by a moving camera -> recovered planes within 1 cm / 0.5 degrees.

## T6. Card Project (14) and Foreground Element (15)

- `layers/card_project.py`: project an RGBA still or sequence onto a card (or the set proxy) and render it through
  the shot camera at plate resolution (numpy/OpenCV homography per frame is enough for planar cards; no GPU renderer).
  Output RGBA EXR sequence, ACEScg, plus the card's 3D corners in the sidecar.
- Foreground grass: generate one tall-grass element on alpha (Z-Image or Qwen-Image, licence register first; key a
  generated plate or ask for a PNG with transparency) and place it on two cards between the camera path and the
  deck at about 1.0 m and 2.0 m from the camera at frame 1130.
- **Test:** synthetic card with a checkerboard projected through a synthetic camera matches a direct render (max
  error < 0.5 px at 1024).

## T7. Character Insert (16)

- `layers/character_insert.py`: the test character from its references (`C:\Users\miike\ComfyUI_V141\ComfyUI\input\howling-yowies\h3\element.png`,
  `element_expressions.png`), MiniMax H3 Ref2VA on the farm (see `prototypes/howling_yowies/h3_gen.py`: farm text encoder on
  render, ref encodes on hub, decode LOCAL), 2.13 m tall, generated on flat grey with the shot's light direction in the
  prompt, standing still, short idle (breathing, head turn). Key the grey to alpha (simple distance key + edge erode;
  note the method in the sidecar). Place the element as a camera-facing card whose bottom edge sits on the deck plane
  at a picked deck point, scaled so the figure is 2.13 m. Render through the shot camera like T6.
- This is a 2.5D insert: good for a camera that mostly walks forward; note its limits in `docs/dev/NOTES.md`.
- **Test:** geometry part only (synthetic): a 2.13 m card on a plane renders with its feet on the plane's image line
  within 1 px across 100 synthetic frames.

## T8. EXR Write (23), Camera Export (25), Nuke Builder (24)

- EXR via OpenImageIO from the vfx_probe venv or the Atlas Camera adapter: per layer RGBA half float, ACEScg,
  colorspace in the header, frames 1130-1297.
- Camera Export: `.chan` in metres (Nuke ZXY) and USD camera via the Atlas Camera exporter if the adapter finds it.
- `deliver/nuke_builder.py` writes `build_DSCF3000_S2S3.py`: a script Mike pastes into the Nuke Script Editor that
  creates Read (ProRes or DPX plate, colorspace F-Log2 F-Gamut via the shipped OCIO config), a Camera from the chan,
  Reads for each layer EXR, the merge tree (plate, the test character, grass over), and a Write. It must not need WriteGeo or any
  NukeX-only node. Keep it idempotent (re-running replaces the nodes it made, found by a name prefix).
- **Test:** `tests/test_nuke_builder.py` checks the generated script compiles (`compile()`), names every file it reads,
  and contains no NukeX-only node classes from a deny list.
- **Acceptance (the slice's done-when):** the builder runs in Nuke Indie 17.1 on ws without edits; grass and the test character hold on
  the plate through the move by eye; the test character's feet stay on the deck within a few px at 4K; every model is in the register.

---

## After the slice

Update `docs/NODE_CATALOGUE.md` statuses, write what was learned into `docs/dev/NOTES.md`, and draft ADR-022 Amendment 3
(evidence) for Mike to review in atlas-nexus. Do not start Phase 2 (clean plate, mattes, relight, look transfer)
without Mike.

---

# Phase 1b: Sensor Pass (rolling shutter + lens)

Added 2026-10-22 from the spec panel (`docs/reviews/SPEC_PANEL_2026-10-22.md`, item 1). The code mostly exists (catalogue
28-31, `camera_side/rs_solve.py`, `camera_side/grid_rs.py`); this section gives it tasks, done-whens and tests.
The margins marked *(proposed)* are the panel's numbers, for Mike to accept or change.

Goal: the plate's own rolling shutter and lens, measured rather than typed, put back onto generated footage as
STMaps, with evidence that it helps on a shot where it should.

Rules for the whole phase:
- One time model: everything RS goes through `camera_side/rs_time.py`. A new RS code path adds itself to
  `tests/test_rs_time.py`.
- Readout default, in this order: chart measurement > RS-aware solve > preset spec. Never the Readout Solve residual
  from a global-shutter camera, and never the full readout through a GS-solved camera (deck test 2026-10-20: -4 %).
- Outputs in `C:\Users\miike\atlas-vfx-work\grid\`, `rs_clips\`, `sensor\`.

## S0. One rolling-shutter time model (done 2026-10-22, 08d2fd7)

- `camera_side/rs_time.py`: row time, pixel convention, anchor, ms/frames, pairing, pose interpolation. sensor/rs.py,
  rs_solve, grid_rs and platelock import it.
- **Test:** `tests/test_rs_time.py`: one fast-pan camera through all four paths agrees to under 1e-6 px; the grid fit
  recovers 30.9000 ms.

## S1. Lens and readout from the chart

- Mike shoots the 96 x 48 in chart (takes A-E: A static lens holds, B pans ISO 2500 4K HQ, C ISO 8000, D 4K HQ off,
  E IBIS on). Run `tools/grid_rs_shoot.py lens` on A, `readout` on B-E with A's lens.
- Note in NOTES: focal (mm on the preset sensor), k1 k2, principal point offset, readout +- sigma per take, RS vs GS
  rms, deg/frame.
- **Test:** `tests/test_grid_rs.py` (exists): known lens + RS rendered through the chart, focal and k1 k2 back,
  readout within 1 %, a locked-off shot reported as not measurable.
- **Acceptance on real data** *(proposed)*:
  - lens rms under 0.5 px over at least 8 still views;
  - readout on B within 5 % of a repeat pan in the same mode (a second B pan, or C if ISO doesn't change it), sigma
    under 1 ms, RS rms at least 30 % below GS rms;
  - D (sub-sampled) reads clearly shorter than B (expected about half), or the note says why not.

## S2. Presets carry readout and focal with a source (done 2026-10-26)

- `camera_side/sensors.py`: each preset gets `readout_ms`, `readout_source`, `readout_status` (unmeasured / solved /
  measured) and a date. Focal per lens and zoom mark the same way (video files carry no focal length; on DSCF3004 the
  focal moved the solved readout 18 -> 23 ms).
- Sensor RS Maps takes its default from the preset and the rule above; the sidecar records which source won.
- Mark Readout Solve (28) deprecated in the catalogue; it stays as a diagnostic.
- **Test:** `tests/test_sensor_presets.py` (new): every preset has the fields; given chart + solve + spec the rule
  picks the chart, given solve + spec it picks the solve, and a GS residual is never picked.

## S3. RS Solve node (done 2026-10-26, synthetic; catalogue 47)

- `AtlasVFXRSSolve` (camera_side, category Atlas VFX/Camera): shot camera + points (from a GS solve) in, RS shot
  camera + readout out, readout free or held, readout profile optional. Wraps `rs_solve.solve(method="lm")`.
  `tools/rs_clip_solve.py` becomes a thin caller of the same code.
- Until shot-camera@0.2 (panel item 7) the readout stays in `extensions.atlas_vfx_rs_solve`.
- **Test:** `tests/test_rs_solve.py` (exists) for the solver; add a node-level run on its synthetic scene that checks
  the output camera validates and the readout is within 1 % of truth.

## S4. Chart node (done 2026-10-28, synthetic; catalogue 49)

- `AtlasVFXChartCalibrate` (camera_side): chart MOV in, lens JSON + readout JSON out (same code as
  `tools/grid_rs_shoot.py`, now `camera_side/chart_calibrate.py` for both). readout_ms wires into Sensor RS Maps'
  `readout_chart_ms`; a locked-off take comes back measurable False with 0 on the wire.
- **Test:** the test_grid_rs renders written to a short synthetic MOV and run through the node (test_grid_rs step 5).

## S5. Prove it on a shot that should show it (fast-pan half passed 2026-10-29 on the solved readout; chart half open)

- Shot: a fast move. A chart take B pan (truth from S1), and DSCF2973 (fast pan, 27.7 ms solved).
- Camera from S3 at the S1 readout. Reproject a rigid surface from a reference frame into the others, RS on vs off,
  against a control with the same resampling (`tools/rs_deck_proof.py` method).
- **Acceptance** *(proposed)*:
  - fast-pan shot: RS on beats RS off by at least 20 % median, and no reference frame gets worse by more than 5 %;
  - S2-S3 deck: RS on is not worse than off (it is +0.9 % today; the shot is too gentle to show more).
- **Fast-pan half, DSCF2973, 2026-10-29: passes both proposed margins** (margins not yet accepted by Mike). No chart
  yet, so the readout is the RS Solve node's free solve (27.73 ms, 48 frames, median 5.31 -> 2.60 px), not S1's.
  `sensor/rs_proof.py` + `tools/rs_s5_proof.py`, run by `tools/run_rs_s5_dscf2973.py`. RS on vs off (RS camera,
  readout on vs off): **+40.1 %** median of 8 refs (1004-1046), worst ref +26.4 %, 213 of 238 pairs better.
  Flipped sign -16.0 % (24 of 238). Against the GS camera the solve started from (the best camera without a
  readout, its own plane, same pixels): +28.0 %, worst ref +10.8 %, 89 of 110 pairs (ref 1046 has no shared
  pixels), so the 40 % isn't just the RS poses looking bad without their readout. S2-S3 deck regression with the
  S3 camera (26.46 ms): +0.9 %, not worse, passes. The RS solve moved this shot's world by 15.6 deg (a
  weak-geometry pan, NOTES 2026-10-29); RS Solve now warns when that happens.
- **Test:** `tests/test_rs_proof.py` (known RS camera over a textured plane: RS on wins, readout 0 is a no-op,
  flipped sign loses).
- Chart half: open, needs S1 (take B pan, at the chart readout).

## S6. RS back on in the LTX clean loop

- Only after S5 passes. `tools/ltx_clean_loop_workflow.py` `USE_RS = True` with the S3 camera and the S1 readout;
  the demo's RS Maps on the preset rule instead of "measured if connected".
- Visible proof needs a rigid element generated in clean space (a box or a sign), where straight verticals make the
  lean measurable to 0.001 px/row; the test character's silhouette was 14x too noisy (2026-10-21).
- **Test:** `tests/test_ltx_sensor_loop.py` (exists) extended with RS maps: remove/apply round trip under 0.001 px
  (0.0001 today, lens only), alpha intact.
- **Acceptance** *(proposed)*: on the rigid element, measured lean matches the maps' predicted skew within 20 %.

## Done-when for Phase 1b

S1, S2, S3 and S5 pass; S6 passes or Mike parks it; catalogue and NOTES updated. Then draft the Sensor Pass part of
ADR-022 Amendment 3. Don't start lens fitting beyond k1 k2, depth bokeh or LoRA work without Mike.
