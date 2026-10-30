# Node catalogue

From ADR-022 Amendment 1 (atlas-nexus). Three parts: the status board (one line per catalogue row), the socket
types, and a contract block per node that loads. Restructured 2026-10-29 from the spec panel
(`docs/reviews/SPEC_PANEL_2026-10-29.md`); the old long status cells became each block's `notes:` line.

`tests/test_node_contracts.py` holds this file to the code: every loaded node has one block, its folder and
category match, the `in:` / `out:` lines and the Types table equal what the code declares, its `test:` files exist
and its `licences:` ids are in `atlas_components.json`. The `in:` / `out:` lines and the Types table are generated:
change a node, run `<V141 python> tests\test_node_contracts.py --update`, commit this file with
`tests/node_contracts.json` and the code. Everything else here is written by hand.

## How to read it

- **Folder**: where the node is now (`atlas_vfx/<folder>/`). camera_side moves into Atlas Camera at the split; this
  column is the source of truth for that, AGENTS.md points here.
- **Status**, one word: built (loads, `NODE_MODULES`), deprecated (loads, not the default path), planned (not
  written), prototype (a script in `prototypes/`), scripts (working tools, no node), exists (elsewhere in the Atlas
  estate).
- **Proof**, how far a built node has been shown to work: synthetic (a test with a known answer), plate script (the
  owner's footage, run as a script), plate V141 (the owner's footage, run as a node in ComfyUI V141), Nuke accepted
  (Mike opened the result in Nuke and accepted it). Anything withdrawn says so.
- **Task**: the plan id that builds or proves it (T2-T8, S1-S6) or the phase.
- **Prototype**: paths under `prototypes/` unless marked *external* (another repo of Mike's, not copied here).
- Every node also needs numpy, opencv-python and opencolorio (`numpy`, `opencv-python`, `opencolorio`) and runs
  inside ComfyUI (`comfyui`, `torch`); the `licences:` lines list what a node adds. `upstream` means the model runs
  in another node and arrives through a socket: the graph that runs it owns the licence check.
- **needs**: `(load)` = a missing package disables the node when ComfyUI starts (logged by `registry.py`, the node
  is missing from the menu); `(run)` = the node loads and fails when queued, with an install message.

## Status board

| # | Node | Folder | Future home | Status | Proof | Task | Prototype |
|---|---|---|---|---|---|---|---|
| 1 | Plate Read | camera_side | Atlas Camera | built | plate V141 | T2 | vfx_probe/plate_proxy.py |
| 2 | Tracking Proxy | camera_side | Atlas Camera | built | plate V141 | T2 | vfx_probe/track_proxy.py |
| 3 | Shot Splitter | camera_side | Atlas Camera | prototype | - | later | vfx_probe/split_analysis.py |
| 4 | Solve Plate | camera_side | Atlas Camera | scripts | plate script | later | vfx_probe/solve_s1s3.py, vfx_probe/resolve_nodist.py |
| 5 | Camera Import | camera_side | Atlas Camera | built | plate V141 | T3 | vfx_probe/align_nuke.py (reader) |
| 6 | Solve Compare | camera_side | Atlas Camera | built | plate V141 | T3 | vfx_probe/align_nuke.py |
| 7 | Measure & Scale | camera_side | Atlas Camera | built | plate V141 (guessed picks) | T4 | vfx_probe/door_scale3.py, ground_check.py, grid_frames.py, zoom_door.py |
| 8 | Lock Check | camera_side | Atlas Camera | built | plate V141 | T4 | vfx_probe/card_check.py |
| 9 | Plate Depth | camera_side | Atlas Camera | built | plate V141 | T5 | |
| 10 | Set Proxy | layers | atlas-vfx | built | plate V141 (guessed scale) | T5 | |
| 11 | Clean Plate | layers | atlas-vfx | planned | - | Phase 2 | |
| 12 | Matte | layers | atlas-vfx | planned | - | Phase 2 | |
| 13 | Layer Stack | layers | atlas-vfx | planned | - | Phase 2 | |
| 14 | Card Project | layers | atlas-vfx | built | plate V141 | T6 | |
| 15 | Foreground Element | layers | atlas-vfx | built | plate V141 | T6 | |
| 16 | Character Insert | layers | atlas-vfx | built | synthetic (plate run withdrawn) | T7 | |
| 17 | Camera-Locked Generate | layers | atlas-vfx | planned | - | Phase 2 | |
| 18 | Light Probe & Relight | layers | atlas-vfx | built | plate V141 (LTX plate) | ahead of Phase 3 | *external* ltx25_hdr/tools/relight_proto.py |
| 19 | Look Transfer | layers | atlas-vfx | planned | - | Phase 2 | |
| 20 | Track Adherence | camera_side | Atlas Camera | exists | - | Phase 2 | |
| 21 | Identity Check | layers | atlas-vfx | planned | - | Phase 2 | |
| 22 | Edge QC | layers | atlas-vfx | planned | - | Phase 2 | |
| 23 | EXR Write | camera_side | Atlas Camera | built | Nuke accepted | T8 | |
| 24 | Nuke Builder | deliver | atlas-vfx | built | Nuke accepted | T8 | howling_yowies/nuke_export_for_claude.py (reverse direction) |
| 25 | Camera Export | camera_side | Atlas Camera | built | Nuke accepted | T8 | vfx_probe/align_nuke.py (chan writer) |
| 26 | Shot Package | deliver | atlas-vfx | planned | - | after the slice | |
| 27 | Farm routing | (farm pack) | ADR-021 | exists | - | - | |
| 28 | Sensor Readout Solve | sensor | atlas-vfx | deprecated | synthetic; biased low on the plate | S2 | |
| 29 | Sensor Lens Maps | sensor | atlas-vfx | built | plate V141 | Sensor pass 1 | |
| 30 | Sensor RS Maps | sensor | atlas-vfx | built | plate V141 | Sensor pass 1, S2 | |
| 31 | Sensor Apply | sensor | atlas-vfx | built | plate V141 | Sensor pass 1 | |
| 32 | Normals | layers | atlas-vfx | built | plate V141 (LTX plate) | with 18 | |
| 33 | Light From Plate | layers | atlas-vfx | built | plate V141 (LTX plate) | with 18 | |
| 34 | Depth Holdout (element) | layers | atlas-vfx | built | plate V141 (LTX plate) | with 18 | |
| 35 | Plate Lock Read EXR | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 36 | Plate Lock Screen Key | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 37 | Plate Lock Camera | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 38 | Plate Lock Person | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 39 | Plate Lock QC | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 40 | Plate Lock | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 41 | Plate Lock Layer | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 42 | Plate Lock Write | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 43 | Plate Lock Cross-View | platelock | atlas-vfx | built | synthetic | Plate Lock | |
| 44 | LTX Prep | layers | atlas-vfx | built | plate V141 | LTX clean loop | |
| 45 | LTX Return | layers | atlas-vfx | built | plate V141 | LTX clean loop | |
| 46 | Comp Preview | deliver | atlas-vfx | built | plate V141 | after T8 | |
| 47 | RS Solve | camera_side | Atlas Camera | built | synthetic; plate script (DSCF2973) | S3 | tools/rs_clip_solve.py (same code) |
| 48 | Physical Sky | layers | atlas-vfx (Prague via Atlas Scene; Hosek-Wilkie fallback here) | built | plate V141 | with 18 | *external* hy_relight/prague make_prague_sky.py, day24_sky.py |
| 49 | Chart Calibrate | camera_side | Atlas Camera | built | synthetic (chart not shot) | S4 | tools/grid_rs_shoot.py (same code) |
| 50 | Chunk Plan | layers | atlas-vfx | built | synthetic | Frame chunks | |
| 51 | Sequence Read Chunk | layers | atlas-vfx | built | synthetic | Frame chunks | |
| 52 | Chunk Stitch | layers | atlas-vfx | built | synthetic | Frame chunks | |

40 rows are built or deprecated, and those are the 40 nodes that load. Categories: `Atlas VFX/Camera`
(camera_side), `Atlas VFX/Layers` (layers), `Atlas VFX/Layers/Plate Lock` (platelock), `Atlas VFX/Layers/Chunks`
(layers), `Atlas VFX/Deliver`
(deliver), `Atlas VFX/Sensor` (sensor).

## Planned and elsewhere

- 3 Shot Splitter: shot boundaries from the proxy; the script exists, no node.
- 4 Solve Plate: pycolmap solves of DSCF3000 in vfx_probe, `camera_side/rs_solve.py` + `tools/rs_clip_solve.py`
  for the RS-aware refit. Phase 1 imports Nuke's camera instead; the RS refit is node 47.
- 11 Clean Plate, 12 Matte, 19 Look Transfer, 22 Edge QC: Phase 2, nothing written.
- 13 Layer Stack: was "T6-T8 (minimal)"; T8 was accepted without it because the Nuke Builder stacks the layer
  folders itself. Phase 2, only if a ComfyUI-side stack is wanted.
- 17 Camera-Locked Generate: generate a layer under the shot camera. No licensed piece chosen. The earlier note named
  MiniMax H3 Fun ControlNet, which is dropped (2026-10-24, licence). Candidates in the register: LTX 2.5 with the
  IC-LoRA in/outpainting (`ltx-2.5-distilled`, `ltx-2.3-ic-lora-in-outpainting`), already used by LTX Prep / Return.
- 20 Track Adherence: Atlas Camera's adherence_score; nothing here yet.
- 21 Identity Check: same performer across shots. Was "by eye in T7", on the H3 character; that subject is gone.
- 26 Shot Package: after the slice.
- 27 Farm routing: the ADR-021 farm pack; the howling-yowies `farm_calls.py` prototype was removed 2026-10-24.

## Types

Custom socket types carry a namespace so they can't collide with another pack (renamed 2026-10-29, spec panel item
1; `tools/rename_socket_types.py` rewrites old graphs). camera_side types start `ATLAS_` because they move into
Atlas Camera; the rest start `ATLAS_VFX_`. Atlas Camera's own `ATLAS_SHOT_CAM` (a shot format: sensor, focal,
resolution) and `ATLAS_PLATE_REF` (one image + colourspace) carry different payloads from ours, so the names stay
apart; `docs/dev/UPSTREAM_TODO.md` has the merge question for the split.

Payloads (all plain JSON-safe dicts unless said):

- `ATLAS_TRACKED_CAMERA`: a shot camera per `atlas_vfx/contracts/shot_camera_v0_1.schema.json` (per-frame pose,
  lens, sensor, plate block, `extensions`). Was `SHOT_CAMERA`.
- `ATLAS_PLATE_SEQ`: a probed movie plate (`camera_side/plate.plate_ref`): path, size, frame range and Nuke range,
  fps, range and colourspace with where each came from, OCIO config, camera body, timecode. Was `PLATE_REF`.
- `ATLAS_VFX_CARD`: an RGBA texture folder + its placement under the shot camera (Foreground Element, Character
  Insert). Was `CARD`.
- `ATLAS_VFX_SENSOR_LENS`: lens model + map folder from Sensor Lens Maps. Was `SENSOR_LENS`.
- `ATLAS_VFX_LTX_PREP`: LTX Prep's record (plate window, crop windows per frame, sizes, Sensor Pass folders) for LTX
  Return and Sensor Apply. Was `LTX_PREP`.
- `ATLAS_VFX_LIGHT_RIG`: sun direction / colour / strength + sky and ground ambient (Light From Plate, Physical
  Sky) for Relight. Was `ATLAS_LIGHT_RIG`.
- `ATLAS_VFX_PLATELOCK_TRACK`, `ATLAS_VFX_PLATELOCK_PEOPLE`, `ATLAS_VFX_PLATELOCK_RESULT`: Plate Lock's camera track
  (RS, readout, anchor), performer chain and lock result (python objects from `platelock/core`, not JSON). Were
  `PLATELOCK_*`.
- `ATLAS_VFX_CHUNK_PLAN`: Chunk Plan's split of a frame range (`layers/chunks.ChunkPlan.to_dict`: first, last,
  max_len, overlap, the chunks' start/end/tag, seams). Chunk Stitch rebuilds the plan from it and checks they agree.
- `MOGE_MODEL`, `MOGE_GEOMETRY`: ComfyUI core's MoGe nodes; not ours, not renamed.

<!-- types:begin (generated) -->
| Type | Made by | Read by |
|---|---|---|
| `ATLAS_PLATE_SEQ` | 1 Plate Read | 2 Tracking Proxy, 5 Camera Import, 8 Lock Check, 9 Plate Depth, 14 Card Project, 24 Nuke Builder, 29 Sensor Lens Maps, 44 LTX Prep, 46 Comp Preview |
| `ATLAS_TRACKED_CAMERA` | 5 Camera Import, 7 Measure & Scale, 47 RS Solve | 6 Solve Compare, 7 Measure & Scale, 8 Lock Check, 9 Plate Depth, 10 Set Proxy, 14 Card Project, 15 Foreground Element, 16 Character Insert, 25 Camera Export, 28 Sensor Readout Solve, 29 Sensor Lens Maps, 30 Sensor RS Maps, 47 RS Solve, 48 Physical Sky |
| `ATLAS_VFX_CARD` | 15 Foreground Element, 16 Character Insert | 14 Card Project |
| `ATLAS_VFX_CHUNK_PLAN` | 50 Chunk Plan | 52 Chunk Stitch |
| `ATLAS_VFX_LIGHT_RIG` | 33 Light From Plate, 48 Physical Sky | 18 Light Probe & Relight |
| `ATLAS_VFX_LTX_PREP` | 44 LTX Prep | 31 Sensor Apply, 45 LTX Return |
| `ATLAS_VFX_PLATELOCK_PEOPLE` | 38 Plate Lock Person | 38 Plate Lock Person, 39 Plate Lock QC, 40 Plate Lock |
| `ATLAS_VFX_PLATELOCK_RESULT` | 40 Plate Lock | 41 Plate Lock Layer, 42 Plate Lock Write |
| `ATLAS_VFX_PLATELOCK_TRACK` | 37 Plate Lock Camera | 40 Plate Lock |
| `ATLAS_VFX_SENSOR_LENS` | 29 Sensor Lens Maps | 30 Sensor RS Maps |
| `MOGE_GEOMETRY` | - | 32 Normals, 34 Depth Holdout (element) |
| `MOGE_MODEL` | - | 9 Plate Depth |
<!-- types:end -->

## Contracts

### 1 Plate Read
`AtlasVFXPlateRead` · camera_side · Atlas VFX/Camera
- in: path STRING, first INT, last INT, ocio_config STRING, input_space STRING, range auto|full|legal, frame_offset INT, preview_width INT, preview_every INT
- out: plate_ref ATLAS_PLATE_SEQ, preview IMAGE, info STRING
- test: tests/test_plate.py
- proof: plate V141 (DSCF3000 S2-S3)
- licences: `ffmpeg`
- needs: ffmpeg/ffprobe on PATH (run)
- writes: nothing (reads the plate, never writes it)
- notes: range from the stream (DSCF3000 is FULL), F-Log vs F-Log2 from the FujiFilm tag when input_space is auto.

### 2 Tracking Proxy
`AtlasVFXTrackingProxy` · camera_side · Atlas VFX/Camera
- in: plate_ref ATLAS_PLATE_SEQ, width INT, out_dir STRING, clip_limit FLOAT, preview_every INT
- out: folder STRING, preview IMAGE
- test: tests/test_plate.py
- proof: plate V141 (DSCF3000 S2-S3)
- licences: `ffmpeg`
- needs: ffmpeg (run)
- writes: log-luma + CLAHE PNG proxies under out_dir

### 5 Camera Import
`AtlasVFXCameraImport` · camera_side · Atlas VFX/Camera
- in: source_path STRING, kind COMBO, plate_ref ATLAS_PLATE_SEQ, sensor_preset auto|fujifilm_xh2_dci4k_24p_dis_off|custom, sensor_width_mm FLOAT, sensor_height_mm FLOAT, m_per_unit FLOAT, rot_order COMBO, chan_focal_mm FLOAT, crop_to_plate BOOLEAN, name STRING, out_dir STRING
- in, optional: offset_first_frame off|to 1|to 1001|to plate start, abc_camera STRING
- out: shot_camera ATLAS_TRACKED_CAMERA, path STRING, summary STRING
- test: tests/test_camera_io.py, tests/test_abc_import.py
- proof: plate V141 (Nuke JSON, .chan, pycolmap; .abc added 2026-10-20)
- licences: `jsonschema`, `pycolmap` (COLMAP kind), `alembic-ogawa-reader` (.abc kind)
- needs: pycolmap for COLMAP models (run)
- writes: the shot camera JSON under out_dir

### 6 Solve Compare
`AtlasVFXSolveCompare` · camera_side · Atlas VFX/Camera
- in: camera_a ATLAS_TRACKED_CAMERA, camera_b ATLAS_TRACKED_CAMERA
- out: report STRING, summary STRING, residual_pct_of_path FLOAT, view_angle_median_deg FLOAT, scale_b_per_a FLOAT
- test: tests/test_compare.py
- proof: plate V141 (reproduces prototypes/vfx_probe/align_nuke.py exactly)
- licences: `jsonschema`
- needs: nothing beyond the base
- writes: nothing

### 7 Measure & Scale
`AtlasVFXMeasureScale` · camera_side · Atlas VFX/Camera
- in: shot_camera ATLAS_TRACKED_CAMERA, picks_path STRING, solve_points STRING, max_spread_pct FLOAT, force BOOLEAN, name STRING, out_dir STRING
- out: shot_camera ATLAS_TRACKED_CAMERA, path STRING, report STRING, summary STRING
- test: tests/test_measure.py
- proof: plate V141, on GUESSED stair picks (the scale is provisional until real picks)
- licences: `jsonschema`, `pycolmap` (solve_points)
- needs: pycolmap when solve_points is set (run)
- writes: the scaled shot camera + report under out_dir

### 8 Lock Check
`AtlasVFXLockCheck` · camera_side · Atlas VFX/Camera
- in: plate_ref ATLAS_PLATE_SEQ, shot_camera ATLAS_TRACKED_CAMERA, centre STRING, orientation ground|vertical, width_m FLOAT, height_m FLOAT, divisions INT, proxy_width INT, preview_every INT, out_path STRING
- in, optional: measure_report STRING
- out: mp4 STRING, preview IMAGE
- test: tests/test_measure.py
- proof: plate V141 (S2-S3)
- licences: `ffmpeg`
- needs: ffmpeg (run)
- writes: a grid-over-plate mp4 at out_path

### 9 Plate Depth
`AtlasVFXPlateDepth` · camera_side · Atlas VFX/Camera
- in: moge_model MOGE_MODEL, plate_ref ATLAS_PLATE_SEQ, shot_camera ATLAS_TRACKED_CAMERA, solve_points STRING, proxy_width INT, resolution_level INT, refine_steps INT, frame_step INT, preview_every INT, out_dir STRING
- out: depth_folder STRING, preview IMAGE, report STRING
- test: tests/test_depth_set_proxy.py
- proof: plate V141 (S2-S3)
- licences: `moge-3-vitl` (upstream, LoadMoGeModel), `openimageio`, `pycolmap` (solve_points)
- needs: ComfyUI core MoGe nodes upstream; OpenImageIO (run)
- writes: per-frame depth EXRs + sidecar under out_dir, scaled to the solve

### 10 Set Proxy
`AtlasVFXSetProxy` · layers · Atlas VFX/Layers
- in: shot_camera ATLAS_TRACKED_CAMERA, depth_folder STRING, solve_points STRING, deck_y FLOAT, deck_band FLOAT, max_tilt_deg FLOAT, inlier_m FLOAT, frame_every INT, pixel_step INT, name STRING, out_dir STRING
- out: obj STRING, usda STRING, report STRING
- test: tests/test_depth_set_proxy.py
- proof: plate V141, on the guessed-scale camera
- licences: `openimageio`, `pycolmap` (solve_points)
- needs: OpenImageIO (run)
- writes: set proxy .obj + .usda + report under out_dir

### 14 Card Project
`AtlasVFXCardProject` · layers · Atlas VFX/Layers
- in: plate_ref ATLAS_PLATE_SEQ, shot_camera ATLAS_TRACKED_CAMERA, card ATLAS_VFX_CARD, scale FLOAT, frame_step INT, preview_every INT, preview_width INT, out_dir STRING
- in, optional: holdout_depth STRING, holdout_margin_m FLOAT, holdout_soft_m FLOAT
- out: exr_folder STRING, preview IMAGE
- test: tests/test_card_project.py
- proof: plate V141 (S2-S3)
- licences: `openimageio`
- needs: OpenImageIO (run)
- writes: an RGBA EXR layer folder (one per plate frame) + sidecar, optional depth holdout

### 15 Foreground Element
`AtlasVFXForegroundElement` · layers · Atlas VFX/Layers
- in: image IMAGE, mask MASK, shot_camera ATLAS_TRACKED_CAMERA, batch_index INT, frame INT, distance_m FLOAT, lateral_m FLOAT, height_m FLOAT, ground_y STRING, yaw_deg FLOAT, erode_px INT, mask_is_background BOOLEAN, ocio_config STRING, name STRING, out_dir STRING
- out: card ATLAS_VFX_CARD, preview IMAGE, report STRING
- test: tests/test_card_project.py
- proof: plate V141 (Z-Image grass + BiRefNet matte)
- licences: `openimageio`, `z-image-turbo` (upstream, in the T6 graph), `birefnet` (upstream)
- needs: OpenImageIO (run)
- writes: the card's ACEScg RGBA texture + placement JSON under out_dir

### 16 Character Insert
`AtlasVFXCharacterInsert` · layers · Atlas VFX/Layers
- in: images IMAGE, shot_camera ATLAS_TRACKED_CAMERA, first_frame INT, last_frame INT, pick_frame INT, pick_u FLOAT, pick_v FLOAT, deck_y FLOAT, height_m FLOAT, key_lo FLOAT, key_hi FLOAT, erode_px INT, mode billboard|fixed, clip_start INT, end_mode hold|pingpong|loop, source_model STRING, ocio_config STRING, name STRING, out_dir STRING
- in, optional: mask MASK, exposure_stops FLOAT
- out: card ATLAS_VFX_CARD, preview IMAGE, report STRING
- test: tests/test_character_insert.py, tests/test_character_time.py
- proof: synthetic. The 2026-10-19 plate run used a MiniMax H3 clip and is withdrawn with H3 (2026-10-24); it needs
  a licensed source to re-prove on the plate.
- licences: `openimageio`; the clip is upstream: source_model names it and the node logs a warning for any model in
  its licence table (MiniMax H3 is kept there only so it still warns)
- needs: OpenImageIO (run)
- writes: an animated card (keyed RGBA frames + placement) under out_dir
- notes: billboard or fixed card standing on the deck at a picked point, scaled to height_m.

### 18 Light Probe & Relight
`AtlasVFXRelight` · layers · Atlas VFX/Layers
- in: element IMAGE, matte MASK, normals IMAGE, light_rig ATLAS_VFX_LIGHT_RIG, encoding srgb|linear, sun_gain FLOAT, ambient_gain FLOAT, studio_flat FLOAT, facing_power FLOAT, fresnel_power FLOAT, edge_rim FLOAT, edge_px FLOAT, rim_max FLOAT, exposure_match BOOLEAN, exposure_bias FLOAT, exposure_smooth INT
- in, optional: background IMAGE, sun_rig ATLAS_VFX_LIGHT_RIG, sun_strength_from sun_gain|physical sky
- out: relit IMAGE, ambient_pass IMAGE, key_pass IMAGE, report STRING
- test: tests/test_relight.py, tests/test_physical_sky.py
- proof: plate V141, a studio actor over an LTX sunset plate (2.5D, normal-based), 2026-10-26
- licences: none beyond the base
- needs: nothing beyond the base
- writes: nothing
- notes: optional sun_rig (Physical Sky) overrides the sun of the plate rig; sun_strength_from picks whose strength.

### 23 EXR Write
`AtlasVFXEXRWrite` · camera_side · Atlas VFX/Camera
- in: images IMAGE, first_frame INT, input_colorspace STRING, colorspace STRING, input_premultiplied BOOLEAN, ocio_config STRING, name STRING, out_dir STRING
- in, optional: mask MASK
- out: pattern STRING
- test: tests/test_nuke_builder.py
- proof: Nuke accepted (T8)
- licences: `openimageio`
- needs: OpenImageIO (run)
- writes: an EXR sequence carrying its colourspace; `exr_io.check_sequence` is what the Nuke Builder verifies

### 24 Nuke Builder
`AtlasVFXNukeBuilder` · deliver · Atlas VFX/Deliver
- in: plate_ref ATLAS_PLATE_SEQ, nuke_camera_json STRING, layer_folders STRING, shot STRING, plate_kind dpx|prores|original, set_proxy_obj STRING, prefix STRING, set_root BOOLEAN, out_dir STRING
- in, optional: x_offset INT
- out: script_path STRING, script STRING
- test: tests/test_nuke_builder.py, tests/test_nuke_text.py
- proof: Nuke accepted (T8, Nuke Indie 17.1 on S2-S3, accepted by Mike)
- licences: `openimageio`, `nuke-indie-17.1` (runs the output)
- needs: OpenImageIO (run)
- writes: a Python script for Nuke's Script Editor, built only through `camera_side/nuke_text.py`

### 25 Camera Export
`AtlasVFXCameraExport` · camera_side · Atlas VFX/Camera
- in: shot_camera ATLAS_TRACKED_CAMERA, units_per_m FLOAT, name STRING, out_dir STRING
- out: chan STRING, usda STRING, nuke_camera_json STRING
- test: tests/test_nuke_builder.py, tests/test_nuke_text.py
- proof: Nuke accepted (T8)
- licences: `jsonschema`
- needs: nothing beyond the base
- writes: .chan + Nuke camera settings JSON + a hand-written .usda under out_dir

### 28 Sensor Readout Solve
`AtlasVFXSensorReadoutSolve` · sensor · Atlas VFX/Sensor
- in: shot_camera ATLAS_TRACKED_CAMERA, solve_folder STRING, points_camera STRING, frame_step INT, refine_pose BOOLEAN, max_readout_ms FLOAT, out_dir STRING
- out: readout_ms FLOAT, report STRING
- test: tests/test_sensor_rs.py, tests/test_sensor_presets.py
- proof: synthetic; on a GS solve of DSCF3000 it reads 23.5 ms against 26.6 ms from the RS-aware solve (biased low)
- licences: `scipy`, `pycolmap`
- needs: scipy (load); pycolmap (run)
- writes: a readout report under out_dir
- notes: deprecated 2026-10-26. DESCRIPTION starts DEPRECATED and it logs a warning when it runs; RS Maps' preset
  rule rejects its value on the record. Stays as a diagnostic.

### 29 Sensor Lens Maps
`AtlasVFXSensorLensMaps` · sensor · Atlas VFX/Sensor
- in: lens_source camera|camera + typed|typed, k1 FLOAT, k2 FLOAT, k3 FLOAT, p1 FLOAT, p2 FLOAT, ca_r FLOAT, ca_b FLOAT, vig_1 FLOAT, vig_2 FLOAT, vig_3 FLOAT, map_scale FLOAT, out_dir STRING
- in, optional: shot_camera ATLAS_TRACKED_CAMERA, plate_ref ATLAS_PLATE_SEQ
- out: sensor_lens ATLAS_VFX_SENSOR_LENS, lens_folder STRING, preview IMAGE, report STRING
- test: tests/test_sensor_lens.py
- proof: plate V141 (DEMO sensor pass, S2-S3; k1 k2 are 0 until the chart shoot measures them)
- licences: `scipy`, `openimageio`
- needs: scipy (load); OpenImageIO (run)
- writes: STMaps (Brown-Conrady + lateral CA) and a vignette map + `.nk` under out_dir

### 30 Sensor RS Maps
`AtlasVFXSensorRSMaps` · sensor · Atlas VFX/Sensor
- in: shot_camera ATLAS_TRACKED_CAMERA, depth_folder STRING, readout_ms FLOAT, readout_source preset rule|measured if connected|typed, direction both|apply RS|remove RS, shutter_angle FLOAT, blur_samples INT, map_scale FLOAT, iters INT, first_frame INT, last_frame INT, out_dir STRING
- in, optional: readout_measured_ms FLOAT, sensor_lens ATLAS_VFX_SENSOR_LENS, readout_chart_ms FLOAT
- out: rs_folder STRING, preview IMAGE, report STRING
- test: tests/test_sensor_rs.py, tests/test_sensor_presets.py, tests/test_sensor_lens.py, tests/test_grid_rs.py
- proof: plate V141 (DSCF3000 S2-S3)
- licences: `scipy`, `openimageio`
- needs: scipy (load); OpenImageIO (run)
- writes: per-frame RS STMaps (apply and/or remove) + `.nk` + sidecar under out_dir
- notes: readout_source 'preset rule' (default) takes chart (readout_chart_ms) > this camera's RS-aware solve >
  the preset's published spec, never a Readout Solve value; typed only with no preset. The sidecar records the mode,
  the winner and what was rejected.

### 31 Sensor Apply
`AtlasVFXSensorApply` · sensor · Atlas VFX/Sensor
- in: images IMAGE, direction apply camera|remove camera, rs_folder STRING, lens_folder STRING, first_frame INT, window STRING, colorspace COMBO, use_vignette BOOLEAN, use_lens BOOLEAN, use_rs BOOLEAN, edge edge|black
- in, optional: ltx_prep ATLAS_VFX_LTX_PREP
- out: images IMAGE, report STRING
- test: tests/test_sensor_lens.py, tests/test_sensor_rs.py
- proof: plate V141 (the S2-S3 LTX renders)
- licences: `scipy`, `openimageio`
- needs: scipy (load); OpenImageIO (run)
- writes: nothing (images out)
- notes: with ltx_prep connected it takes the render's frame range and crop windows from LTX Prep.

### 32 Normals
`AtlasVFXNormals` · layers · Atlas VFX/Layers
- in: moge_geometry MOGE_GEOMETRY, temporal INT, sigma FLOAT
- in, optional: matte MASK
- out: normals IMAGE, report STRING
- test: tests/test_relight.py
- proof: plate V141 (LTX plate, with 18)
- licences: `moge-3-vitl` (upstream, MOGE_GEOMETRY)
- needs: ComfyUI core MoGe nodes upstream
- writes: nothing
- notes: MoGe-3 / MoGe-2 normals, OpenCV to OpenGL, temporal + matte-weighted smoothing.

### 33 Light From Plate
`AtlasVFXLightFromPlate` · layers · Atlas VFX/Layers
- in: background IMAGE, encoding srgb|linear, fov_x FLOAT, sun auto|manual, sun_u FLOAT, sun_v FLOAT, behind FLOAT, smooth INT
- in, optional: sun_colour STRING
- out: light_rig ATLAS_VFX_LIGHT_RIG, preview IMAGE, report STRING
- test: tests/test_relight.py
- proof: plate V141 (LTX plate, with 18)
- licences: none beyond the base
- needs: nothing beyond the base
- writes: nothing
- notes: sun blob + sky / ground ambient from the background.

### 34 Depth Holdout (element)
`AtlasVFXDepthHoldout` · layers · Atlas VFX/Layers
- in: background_geometry MOGE_GEOMETRY, matte MASK, margin FLOAT, soft FLOAT, foot_rows INT, smooth INT, distance_scale FLOAT, blur_px FLOAT
- out: matte MASK, occluder MASK, report STRING
- test: tests/test_holdout.py
- proof: plate V141 (LTX plate, with 18: grass over legs)
- licences: `moge-3-vitl` (upstream, MOGE_GEOMETRY)
- needs: ComfyUI core MoGe nodes upstream
- writes: nothing
- notes: background MoGe depth against the element's feet.

### 35 Plate Lock Read EXR
`AtlasVFXPlateLockReadEXR` · platelock · Atlas VFX/Layers/Plate Lock
- in: path STRING, first INT, last INT, every INT
- out: frames IMAGE, metadata STRING, first_frame INT
- test: tests/test_platelock_nodes.py, tests/test_platelock_lock.py
- proof: synthetic
- licences: `scipy`, `openimageio`, `openexr` (fallback), `imageio` (fallback)
- needs: scipy (load); one of OpenImageIO, OpenEXR, imageio (run; a built-in reader handles uncompressed EXR only)
- writes: nothing
- notes: EXR sequence as linear float, no display transform, plus header metadata (fps, shutter, focal, sensor,
  timecode, gyro/readout).

### 36 Plate Lock Screen Key
`AtlasVFXPlateLockScreenKey` · platelock · Atlas VFX/Layers/Plate Lock
- in: plate IMAGE, screen blue|green, low FLOAT, high FLOAT
- out: matte MASK
- test: tests/test_platelock_nodes.py, tests/test_platelock_lock.py
- proof: synthetic
- licences: `scipy`
- needs: scipy (load)
- writes: nothing
- notes: colour-difference matte off a blue or green screen; a fallback, a Nuke key loaded as MASK is better.

### 37 Plate Lock Camera
`AtlasVFXPlateLockCamera` · platelock · Atlas VFX/Layers/Plate Lock
- in: frames IMAGE, mode locked_off|shot_camera|platelock_json, path STRING, focal_mm FLOAT, sensor_width_mm FLOAT, readout_ms FLOAT, fps FLOAT, rs_anchor centre|top
- out: camera ATLAS_VFX_PLATELOCK_TRACK, info STRING
- test: tests/test_platelock_nodes.py, tests/test_rs_time.py
- proof: synthetic
- licences: `scipy`, `jsonschema`
- needs: scipy (load)
- writes: nothing
- notes: shot camera -> RS camera track, readout + anchor (`adapters.track_from_shot_camera`).

### 38 Plate Lock Person
`AtlasVFXPlateLockPerson` · platelock · Atlas VFX/Layers/Plate Lock
- in: name STRING, plate_masks MASK, ai_masks MASK, depth FLOAT
- in, optional: people ATLAS_VFX_PLATELOCK_PEOPLE, body_plate_npz STRING, body_ai_npz STRING, correspondences_npz STRING
- out: people ATLAS_VFX_PLATELOCK_PEOPLE
- test: tests/test_platelock_nodes.py
- proof: synthetic
- licences: `scipy`; masks and body solves are upstream with no register entry yet (SAM-style video segmentation,
  body-solve or matcher dumps): the graph that makes them owns the licence check, and SMPL-based body models are
  usually non-commercial (`platelock/adapters.py`)
- needs: scipy (load)
- writes: nothing
- notes: one node per performer, chained through people.

### 39 Plate Lock QC
`AtlasVFXPlateLockQC` · platelock · Atlas VFX/Layers/Plate Lock
- in: people ATLAS_VFX_PLATELOCK_PEOPLE
- out: summary STRING, report_json STRING
- test: tests/test_platelock_nodes.py
- proof: synthetic
- licences: `scipy`
- needs: scipy (load)
- writes: nothing (report JSON out)
- notes: raw AI mask against plate mask per performer and frame: overlap, edge distance, motion lag; says which
  stages the shot needs.

### 40 Plate Lock
`AtlasVFXPlateLock` · platelock · Atlas VFX/Layers/Plate Lock
- in: plate IMAGE, ai IMAGE, plate_camera ATLAS_VFX_PLATELOCK_TRACK, ai_camera ATLAS_VFX_PLATELOCK_TRACK, people ATLAS_VFX_PLATELOCK_PEOPLE, scale FLOAT, depth_mode flat|metric|inverse, flat_distance FLOAT, shutter_deg FLOAT, motion_blur_samples INT, neighbours INT, edge_snap BOOLEAN, snap_max_px FLOAT
- in, optional: ai_depth IMAGE, plate_matte MASK
- out: comp IMAGE, background IMAGE, background_valid MASK, result ATLAS_VFX_PLATELOCK_RESULT, report STRING
- test: tests/test_platelock_nodes.py, tests/test_platelock_lock.py, tests/test_platelock_core.py
- proof: synthetic
- licences: `scipy`
- needs: scipy (load)
- writes: nothing
- notes: retime, re-shoot, warp and restack the AI take onto the plate; `platelock/core/` is pure python.

### 41 Plate Lock Layer
`AtlasVFXPlateLockLayer` · platelock · Atlas VFX/Layers/Plate Lock
- in: result ATLAS_VFX_PLATELOCK_RESULT, name STRING
- out: rgb IMAGE, alpha MASK, names STRING
- test: tests/test_platelock_nodes.py
- proof: synthetic
- licences: none beyond the base
- needs: nothing beyond the base
- writes: nothing
- notes: one locked layer with its matte, for the stack.

### 42 Plate Lock Write
`AtlasVFXPlateLockWrite` · platelock · Atlas VFX/Layers/Plate Lock
- in: result ATLAS_VFX_PLATELOCK_RESULT, out_dir STRING, first_frame INT, ai_first_frame INT, ai_path STRING, write_comp BOOLEAN
- out: out_dir STRING
- test: tests/test_platelock_nodes.py, tests/test_nuke_text.py
- proof: synthetic
- licences: `scipy`, `openimageio`, `openexr` (fallback)
- needs: scipy (load); an EXR writer (run)
- writes: re-shot background and per-performer RGBA EXRs, STMaps into the original AI frames, shot.json with retime
  curves and QC, `platelock_lock.nk` (hardened 2026-10-26: `nk_str`, `safe_name`, `resolve_out_dir`)

### 43 Plate Lock Cross-View
`AtlasVFXPlateLockCrossView` · platelock · Atlas VFX/Layers/Plate Lock
- in: plate STRING, ai STRING, plate_ids STRING, out_dir STRING, plate_space STRING, ai_space STRING, proc_width INT, first INT, last INT
- in, optional: ai_ids STRING, matte_channel COMBO, ocio_config STRING, base ai|plate, quality medium|fast|ultrafast, write_comp BOOLEAN
- out: summary STRING, out_dir STRING, preview IMAGE
- test: tests/test_platelock_crossview.py, tests/test_nuke_text.py
- proof: synthetic
- licences: `scipy`, `openimageio`, `openexr` (fallback), `imageio` (fallback)
- needs: scipy (load); an EXR reader/writer (run)
- writes: frame-for-frame dense warp per ID as EXR + OCIO + `.nk` (containment guard 2026-10-20, `nk_str`
  2026-10-26)

### 44 LTX Prep
`AtlasVFXLTXPrep` · layers · Atlas VFX/Layers
- in: plate_ref ATLAS_PLATE_SEQ, layer_folder STRING, first_frame INT, length INT, width INT, height INT, mask_thresh FLOAT, shadow_px INT, dilate_px INT, keyframe_every INT, mode full|crop, crop_margin INT, extra_layers STRING, reference plate|comp, mask_mode layer|none
- in, optional: sensor_rs_folder STRING, sensor_lens_folder STRING
- out: reference IMAGE, mask MASK, reference_half IMAGE, mask_half MASK, start_image IMAGE, ltx_prep ATLAS_VFX_LTX_PREP, keyframes_half IMAGE
- test: tests/test_ltx_integration.py, tests/test_ltx_sensor_loop.py
- proof: plate V141 (LTX clean loop, S2-S3)
- licences: `scipy`, `openimageio`; LTX itself is upstream (`ltx-2.5-distilled`, `ltx-2.3-ic-lora-in-outpainting`,
  `comfyui-ltxvideo`)
- needs: scipy (load); OpenImageIO (run)
- writes: nothing (images + record out)
- notes: plate -> ACEScg -> named SDR view, element alpha as mask, start image + keyframes; crop mode; Sensor Pass
  maps. Reads the plate from ATLAS_PLATE_SEQ itself, so a de-RS'd clean plate can't be fed in yet.

### 45 LTX Return
`AtlasVFXLTXReturn` · layers · Atlas VFX/Layers
- in: images IMAGE, ltx_prep ATLAS_VFX_LTX_PREP, feather_px FLOAT, name STRING, source_model STRING, preview_every INT, out_dir STRING
- out: layer_folder STRING, preview IMAGE
- test: tests/test_ltx_integration.py, tests/test_ltx_sensor_loop.py
- proof: plate V141 (LTX clean loop, S2-S3)
- licences: `scipy`, `openimageio`; the render is upstream (see 44)
- needs: scipy (load); OpenImageIO (run)
- writes: an EXR layer folder + sidecar (inverse view, paste-back with a faded edge)

### 46 Comp Preview
`AtlasVFXCompPreview` · deliver · Atlas VFX/Deliver
- in: plate_ref ATLAS_PLATE_SEQ, layer_folders STRING, width INT, frame_step INT, burn_in BOOLEAN, label STRING
- out: frames IMAGE, fps FLOAT, report STRING
- test: tests/test_comp_preview.py
- proof: plate V141 (the demo graph's "Watch the shot")
- licences: `openimageio`
- needs: OpenImageIO (run)
- writes: nothing (frames out)
- notes: plate + layer EXR folders back to front, premultiplied over, burn-in.

### 47 RS Solve
`AtlasVFXRSSolve` · camera_side · Atlas VFX/Camera
- in: shot_camera ATLAS_TRACKED_CAMERA, solve_folder STRING, points_camera STRING, readout free|held, readout_ms FLOAT, first_frame INT, last_frame INT, max_points INT, max_iter INT, profile_ms STRING, profile_points INT, name STRING, out_dir STRING
- out: shot_camera ATLAS_TRACKED_CAMERA, readout_ms FLOAT, path STRING, report STRING
- test: tests/test_rs_solve.py
- proof: synthetic (free readout 30.887 ms for 30.900 from the preset's 29.2 ms start); plate script on DSCF2973
  through the same code
- licences: `scipy`, `pycolmap`, `jsonschema`
- needs: scipy (load); pycolmap (run)
- writes: the RS shot camera + report under out_dir
- notes: GS shot camera + its COLMAP points in, RS shot camera + readout out, readout free or held (0 = the preset
  spec), optional readout profile. Wraps `rs_solve.solve_shot`, the code `tools/rs_clip_solve.py` calls. The output
  carries `extensions.atlas_vfx_rs_solve`, which RS Maps' preset rule picks up. Warns when it can't keep the input
  world (over 5 deg; DSCF2973 moved 15.6 deg, 2026-10-29).

### 48 Physical Sky
`AtlasVFXPhysicalSky` · layers · Atlas VFX/Layers
- in: latitude FLOAT, longitude FLOAT, altitude_m FLOAT, capture_utc STRING, time_offset_h FLOAT, heading_deg FLOAT, fov_x FLOAT, tilt_deg FLOAT, visibility_km FLOAT, ground_albedo FLOAT, exposure_ev FLOAT, sun_mode metadata|offset|manual, sun_azimuth_offset FLOAT, sun_elevation_offset FLOAT, sun_azimuth FLOAT, sun_elevation FLOAT, width INT, latlong_centre north|camera heading, name STRING, out_dir STRING
- in, optional: shot_camera ATLAS_TRACKED_CAMERA, dataset STRING, ocio_config STRING, display STRING, view STRING, sky_model auto|prague|hosek-wilkie
- out: hdri_acescg IMAGE, light_rig ATLAS_VFX_LIGHT_RIG, preview IMAGE, info STRING, exr_path STRING
- test: tests/test_physical_sky.py, tests/test_sky_hosek.py, tests/test_sun_globe.py
- proof: plate V141 (DSCF2977 check shot, Prague and Hosek-Wilkie through the panel, 2026-10-29); Hosek-Wilkie
  also against the compiled reference C and Prague, and in a clean ComfyUI with nothing of the owner's machine
  (`tools/clean_install_check.py`, ws and Linux CI)
- licences: `hosek-wilkie-sky-model`, `cie-cmf-wyman2013` (fallback, shipped); `atlas-scene`, `prague-sky-model`,
  `prague-sky-model-dataset` (Prague, optional); `three-js` (Sun Globe panel), `openimageio`, `jsonschema`
- needs: nothing for the Hosek-Wilkie sky; Prague needs Atlas Scene through `_atlas_scene.py` and a dataset (run:
  sky_model auto falls back to Hosek-Wilkie and says why, prague raises AtlasSceneMissing with the install steps);
  OpenImageIO (run)
- writes: `<name>.exr` (float32 ACEScg lat-long) + sidecar + info JSON with the atlas_scene commit, under
  ~/atlas-vfx-work/physical_sky/<name>
- notes: GPS + UTC capture time (+ offset hours) -> NOAA sun (metadata / offset / manual) -> the sky. sky_model
  (last widget, default auto): prague = Atlas Scene's Prague (twilight to -4.2 deg, visibility, altitude with the
  full dataset); hosek-wilkie = the analytic fallback in `layers/sky_hosek.py` (sea level, sun 0-90 deg, visibility
  mapped to turbidity by a table fitted to Prague, same radiometric scale; at 20-50 km visibility and sun 20 deg up
  the rig's sun_strength is within about 10% of Prague's and the sky within 25%; near the horizon its sun is too
  bright, and above ~60 km it can't get as clear as Prague: sun_strength 0.55-0.85 of Prague's); auto = Prague when reachable, else
  Hosek-Wilkie. info / sidecar / EXR attribute record the model. Nuke lat-long layout, azimuth grows right, centre
  north or camera heading. Sun under the model's limit (Prague -4.2, Hosek-Wilkie 0): black HDRI + warning, exact
  direction kept. Sun Globe panel (`web/atlas_vfx_sun_globe.js`, three.js r185 vendored): the sun,
  the day's path and the camera frustum on a globe, seen from behind the camera (VIEW NORTH on the button), drag the sun (writes manual), time-of-day scrubber, reset to
  metadata, golden hour, auto-queue toggle (off), sky model switch (writes sky_model), and a flag plus a short
  reason when the picked model won't render this sun (Hosek-Wilkie twilight, anything under -4.2). Drives the native
  widgets only; headless runs are unchanged. Pictures: `docs/images/sun_globe_twilight.png`,
  `docs/images/sky_prague_vs_hosek.png`.

### 49 Chart Calibrate
`AtlasVFXChartCalibrate` · camera_side · Atlas VFX/Camera
- in: video_path STRING, mode lens|readout, sensor_preset fujifilm_xh2_dci4k_24p_dis_off, lens_json STRING, first_frame INT, last_frame INT, every INT, still_px FLOAT, focal_guess_mm FLOAT, name STRING, out_dir STRING
- out: lens_path STRING, focal_mm FLOAT, k1 FLOAT, k2 FLOAT, pp_offset_x_px FLOAT, pp_offset_y_px FLOAT, readout_path STRING, readout_ms FLOAT, sigma_ms FLOAT, rms_rs_px FLOAT, rms_gs_px FLOAT, measurable BOOLEAN, report STRING
- test: tests/test_grid_rs.py
- proof: synthetic; the chart takes aren't shot yet
- licences: `scipy`, `ffmpeg`
- needs: scipy (load); ffmpeg (run)
- writes: `<name>_lens.json` / `<name>_readout.json` + sidecars under atlas-vfx-work/grid
- notes: mode lens (take A: still pairs -> focal on the sensor preset, k1 k2, principal point, rms) or readout
  (takes B-E with A's lens JSON: readout_ms + sigma, RS against GS rms). A locked-off take is reported not
  measurable (measurable False, readout_ms 0, which RS Maps reads as no chart value). readout_ms wires into Sensor
  RS Maps' readout_chart_ms. Wraps `chart_calibrate.py`, the code `tools/grid_rs_shoot.py` calls.

### 50 Chunk Plan
`AtlasVFXChunkPlan` · layers · Atlas VFX/Layers/Chunks
- in: first INT, last INT, max_len INT, overlap INT, chunk INT
- out: chunk_plan ATLAS_VFX_CHUNK_PLAN, start INT, length INT, end INT, tag STRING, chunk_count INT, info STRING
- test: tests/test_chunks.py
- proof: synthetic
- licences: none beyond the base
- needs: nothing
- writes: nothing
- notes: a long range (1001-1400) as equal 8k+1 chunks for LTX 2.x: stride = length - overlap, the last chunk moved
  back to end on `last` (no padded frames, so its seam is longer). 1001-1400 at 121 / 24 is 1001-1121, 1098-1218,
  1195-1315, 1280-1400. start / length / end wire into LTX Prep and the LTX latent / audio lengths. chunk -1 outputs
  every chunk as lists (ComfyUI runs the downstream once per chunk, one queue); for long LTX runs use
  `tools/queue_chunks.py`, which queues one prompt per chunk and fills {tag} / {chunk} in output names.

### 51 Sequence Read Chunk
`AtlasVFXSequenceReadChunk` · layers · Atlas VFX/Layers/Chunks
- in: path STRING, start INT, length INT, numbering frame|sequential, width INT, input_colorspace STRING, colorspace STRING, ocio_config STRING
- out: images IMAGE, first_frame INT, info STRING
- test: tests/test_chunks.py
- proof: synthetic
- licences: `openimageio`
- needs: OpenImageIO (run)
- writes: nothing (reads the sequence, never writes it)
- notes: one chunk of an image sequence (EXR, DPX, PNG, TIFF, JPEG; `####`, `%04d`, `{frame}` or a folder with one
  sequence) as an IMAGE batch. numbering frame (file numbers are shot frames) or sequential (files in order). Any
  missing frame is one error listing them. Colour is passed through as stored unless input_colorspace is set (then a
  named OCIO transform). width resizes; a batch over 8 GB gets a note.

### 52 Chunk Stitch
`AtlasVFXChunkStitch` · layers · Atlas VFX/Layers/Chunks
- in: chunk_plan ATLAS_VFX_CHUNK_PLAN, pattern STRING, numbering frame|sequential, colorspace STRING, name STRING, out_dir STRING
- out: pattern STRING, first_frame INT, info STRING
- test: tests/test_chunks.py
- proof: synthetic
- licences: `openimageio`
- needs: OpenImageIO (run)
- writes: `<name>.####.exr` (RGBA half) from the plan's first frame, `<name>.stitch.json` (sources and weights per
  frame) and a sidecar under atlas-vfx-work/chunks/<name>
- notes: pattern names each chunk's sequence with {chunk} (00) or {tag} (c00). Every chunk is checked before a frame
  is written. Seams crossfade with smoothstep weights that sum to 1 and are 0 outside a chunk (three chunks on a
  frame chain their handovers). Blends the files' own values: scene-linear for EXR, display-encoded for PNG / JPEG.
  Crossfading hides a seam only when the two takes agree; if they ghost, the next step is conditioning each chunk
  on the previous one's tail.
