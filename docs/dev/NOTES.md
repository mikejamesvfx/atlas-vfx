# Notes

Running log for whoever works here. Newest at the top. Keep it short: what was decided, what was learned, what broke.

## 2026-10-29 (Physical Sky: Hosek-Wilkie fallback; branch feat/sky-hosek-fallback)

- Why: the repo is public but Atlas Scene is private, so Physical Sky failed for everyone else. Rule now: Atlas Scene
  owns Prague; atlas-vfx owns one analytic fallback, `layers/sky_hosek.py` (never imports Atlas Scene).
- Model: Hosek-Wilkie 2012 sky + 2013 sun, the authors' BSD-3 v1.4a data from pbrt-v4 at a pinned commit, converted
  by `tools/hosek_wilkie_data.py` (hash-pinned sources, numbers-only parser) to a 191 KB `.npz` + manifest +
  licence notice under `layers/data/`. Colour: CIE 1931 via the Wyman 2013 fit (checked against colour-science's
  table: max error 0.024, integrals within 0.1%), the BT.709 matrix from its chromaticities.
- Units: Atlas Scene's Prague returns radiometric CIE-weighted values (no 683 lm/W); Hosek now matches that scale,
  so exposure_ev means the same in both (median Hosek/Prague 1.03).
- New widget `sky_model` auto | prague | hosek-wilkie, LAST in optional (old GUI graphs fill widgets by position;
  `test_widget_order`). auto = Prague when Atlas Scene + extension + dataset are reachable, else Hosek with the
  reason in notes and the log. info / sidecar / EXR attribute / Sun Globe status name the model. prague with the sun
  below its domain still needs no dataset.
- Prague output is bit-identical to before (two renders hashed before and after).
- Turbidity table: v0 analytic (Koschmieder) replaced by a fit to Prague (`tools/hosek_calibrate.py`) on light
  (sun_strength + E_sky_up). A first fit that also weighed sky colour pulled turbidity away from the light; colour is
  now only reported (dxy about 0.01-0.035). At 20-50 km and sun 20 deg up the rig's sun_strength is within 10% of
  Prague; above ~60 km Hosek can't get as clear (turbidity floor 1, sun_strength 0.55-0.85); near the horizon
  (2-5 deg) its sun is far too bright (x1.8 at 5 deg, x7 at 2 deg). Twilight (-4.2 to 0): black HDRI + a warning
  naming Prague.
- Found on the way: Atlas Scene's render_environment fills the ground from the irradiance its pixels caught, so at
  low widths it misses a sun disc that falls between pixel centres; ours adds the sun analytically (as light_rig
  always did). In UPSTREAM_TODO.
- Not done here: the golden check against the compiled reference C. Building and running downloaded C was blocked
  in this session; `tests/test_sky_hosek.py` SKIPs it until `tools/hosek_wilkie_data.py --golden` is run (MSVC is
  on ws). Not yet run inside V141 (needs a restart for the new widget).

## 2026-10-29 (catalogue spec panel fixes; branch fix/catalogue-panel-2026-10-29)

- Spec panel on NODE_CATALOGUE.md (`docs/reviews/SPEC_PANEL_2026-10-29.md`), all nine items done the same day.
- **Socket types renamed** (breaks graphs saved before it): SHOT_CAMERA -> ATLAS_TRACKED_CAMERA, PLATE_REF ->
  ATLAS_PLATE_SEQ, CARD / SENSOR_LENS / LTX_PREP / PLATELOCK_* / ATLAS_LIGHT_RIG -> ATLAS_VFX_*. Not Atlas Camera's
  ATLAS_SHOT_CAM / ATLAS_PLATE_REF: those carry other payloads (a shot format; one image), checked in its tree.
  The 14 committed workflows are rewritten. Your own saved graphs: `python tools/rename_socket_types.py --dry-run
  <folder>`, then without --dry-run. V141 needs a restart to see the new types.
- `tests/test_node_contracts.py` + `tests/node_contracts.json`: pins every node's sockets, checks every link in
  workflows/, holds the catalogue to the code (folder, category, generated in/out lines and Types table, test,
  licences, needs, proof lines) and the type namespaces. Checked that it catches a renamed output, a wrong folder
  and a stale link type.
- Catalogue restructured: one-word status + a Proof column (synthetic / plate script / plate V141 / Nuke accepted),
  planned rows in their own list, a contract block per node. H3 is out of rows 16, 17, 21, 27: Character Insert's
  plate proof is withdrawn (it was the H3 clip), Camera-Locked Generate has no licensed piece chosen.
- Licence register gaps found and filled: `openexr` and `imageio` (Plate Lock's EXR reader fallbacks, in use
  unregistered), `torch` and `comfyui` (the host and its runtime).
- CLAUDE.md: camera_side is rows 1-9, 20, 23, 25, 47, 49; the catalogue's Folder column is the source of truth.

## 2026-10-29 (Sun Globe for Physical Sky; branch feat/physical-sky)

- `web/atlas_vfx_sun_globe.js`, the first frontend in this repo (root `__init__` now sets `WEB_DIRECTORY = "./web"`).
  A panel on Atlas VFX Physical Sky: a globe in the node's sky frame (right-handed, Y up, north -Z, east +X, azimuth
  clockwise from north) with N E S W, the sun, the capture day's sun path (amber above the horizon, dim under it,
  hourly dots), the camera frustum from the centre (heading / HFOV / tilt, or the shot camera's fov and first-frame
  tilt from the last run) and a little actor in front of it, lit by the sun and throwing its shadow. A line under
  the globe says it in words: sun behind the actor (backlit), beside (lit from screen left / right) or in front
  (front-lit), how far off the lens, in frame or not. Readouts AZ / EL / clock (the capture time's own offset, else
  UTC, plus local solar time), a mode badge, BELOW HORIZON under -4.2 deg.
- Ported from Atlas Camera's Gravity Compass (d1cb096): the native widgets stay the serialized / API contract, the
  panel only drives them (setNative fires the callback), DOM widget with serialize false, pointer capture, shift =
  fine, chained onRemoved disposes the renderer and stops the loop. It hides only the six sun widgets it writes
  (time_offset_h, sun_mode, both offsets, sun_azimuth, sun_elevation); heading, fov and tilt stay ordinary widgets.
  No new Python input: headless graphs and the API are unchanged. Nothing is hidden until the panel has built, so no
  WebGL or no three.js leaves a normal node.
- Drag mapping, decided: dragging the sun writes MANUAL (az / el as dropped, to 0.1 deg). Offset would be "NOAA plus
  a nudge", and a big dragged nudge rides the clock into elevations the node refuses. From offset mode it stays
  offset and rewrites the offsets (manual if the elevation offset would leave -90..90). The scrubber writes
  time_offset_h over the capture's solar day, snapping to 5 minutes of the shown clock (shift: 1); in manual it
  first goes back to metadata. RESET TO METADATA = metadata, time 0, offsets 0. GOLDEN HOUR moves the clock to the
  4 deg sun (evening after noon, morning before). Drag empty space to orbit, double-click resets; VIEW NORTH / VIEW
  CAMERA. Nothing queues while dragging; AUTO-QUEUE (node.properties, default off) queues once when a drag, scrub or
  button ends.
- The node now returns `ui.atlas_sun_globe[0]` (`globe_ui`): the sun used and the NOAA sun before offsets, the
  capture day's path from `physical_sky.sun_path` (10 minute samples; sunrise / sunset bisected at 0 deg, noon by
  ternary search, golden hour at 4 deg, polar day / night), the clock, the camera used, sky_rendered, and the preview
  as a 768 px JPEG. The panel uses the path while lat / lon / capture time match, and textures the upper half of the
  globe with the preview while the widgets still give the sun it rendered (mapping pinned in the test against
  three's own sphere uvs). Without a run it draws from the widgets: the JS carries a line-for-line mirror of
  solar.py's NOAA sun, effective_sun, sun_path, camera_to_sky and project_uv.
- three.js r185 copied from Atlas Camera at e2f8d38 into `web/lib/` with a provenance comment and three.LICENSE; in
  the licence register as three-js (hash taken with LF line ends, since git on ws checks it out CRLF). No CDN, no
  import from the Atlas Camera pack. Both packs load their own copy in the same page; three may log a "multiple
  instances" warning, harmless. test_hygiene now treats `web/` as shipped and scans .js / .html.
- `tests/test_sun_globe.py` (node needed, fails without it): imports the panel as an ES module with a stub `app`,
  runs the JS mirror against the Python on 8 suns, 10 sun-mode cases (errors included), 6 sun paths incl. polar,
  camera basis / projection, the drag round trip and the dome uvs; checks the widget names, unchanged inputs, no
  queue from a pointermove, auto-queue off, the payload shape (DSCF2977 252.355 / 16.121; sunrise / sunset at 0 deg,
  noon = solar.py's highest sun to 0.002 deg, about 79 deg up, north) and a real run() below the horizon.
- Visual check: `tools/sun_globe_review.py` (payload with the V141 python, shots with a python that has Playwright;
  harness `tools/sun_globe_harness.html`). Screenshots in ~/atlas-vfx-work/physical_sky/globe_review/: DSCF2977
  metadata (sun at WSW 16 deg, beside, screen right of a south-facing camera, shadow to the east), a real mouse drag
  (manual 134.0 / 57.1, nothing queued), night (6 h later, -32.3 deg, BELOW HORIZON), view from behind the camera,
  and a dome check from opposite the sun (the preview's sun sits behind the marker; N right when looking west).
  Found and fixed on the way: headless Chromium lost and restored the WebGL context once, and the render-on-change
  loop stayed blank; it now redraws on webglcontextrestored.
- Not yet seen inside V141 (it loads the main checkout; no restart). After the merge: reload the page, drop a
  Physical Sky node, check the panel sits under the widgets and the hidden ones stay hidden in both renderers.

## 2026-10-29 (Phase 1b S5, fast-pan half: DSCF2973 passes on the solved readout)

- RS Solve node on DSCF2973 (`tools/run_rs_solve_dscf2973.py`, GS camera `rs_clips/DSCF2973_f16/DSCF2973_gs`, its
  own COLMAP points): free from the 29.2 ms preset start to **27.73 ms**, 48 frames (1001-1048), 48.4k points,
  415k obs, median 5.31 -> 2.60 px (rms 6.74 -> 4.25), 151 LM iterations in 1509 s (100-iteration cap: 27.73 ms
  too, same everything). Profile 0 ms +112 %, 10 +72 %, 20 +27 %, 26.6 +0.9 %, 30.9 +7.3 %, 40 +101 % (the last two
  hit the cap, upper bounds). Same as the 10-22 tool run. Agrees with DSCF3000's 26.46 ms to 5 %.
- **The 15.6 deg world move is the solve, not a bug.** The camera moved onto the input world by 15.6 deg, 3.06
  units, scale 1.04, poses changed 15.3 deg median (DSCF3000: 0.29 deg). Checked: frame 1001 is held in the solve,
  so the similarity can be read back from it, and in the solve's own gauge the RS poses differ from the GS poses by
  up to 13.6 deg. A fixed 27.73 ms solve (8000 points) converges in 24 iterations to the same 15.5 deg, and a fixed
  0 ms solve moves 0.16 deg, so it's not LM wandering along a gauge with 151 iterations. What changes: the GS solve
  turns 1.78 deg/frame median (16.7 deg net), the RS solve 0.97 deg/frame (10.3 deg net) and explains the rest
  with translation; points move 1.04 units median at about 10 units depth. It's a pan with almost no depth range
  (tracked depths 8-16 units, the plane through them faces the camera), so rotation and sideways translation are
  nearly interchangeable and the readout tips the balance; the point-based similarity back onto the GS world then
  has no good answer. The RS structure is the more consistent one: tracks triangulate at 2.2 px (GS camera 4.2
  px), and 86 % / 68 % of the sand tracks sit on one plane (GS 58 % / 39 %). Units: the GS solve is unscaled.
- So this RS camera does not live in the GS camera's world. Anything placed against the GS camera won't line up
  with it. `rs_solve.world_warnings` (over 5 deg onto the input world or median pose change) now goes into RS
  Solve's warnings and log; `tests/test_rs_solve.py` checks the synthetic run doesn't warn (1.45 deg) and the
  DSCF2973 numbers do. Not changed: the alignment itself (points, so a deck at y = 0 stays put on good shots).
- The proof (`sensor/rs_proof.py`, `tools/rs_s5_proof.py`, runs + table in `tools/run_rs_s5_dscf2973.py`): the
  rs_deck_proof method on any plane. Surface fitted to the shot's own tracks triangulated with the RS camera at each
  row's time; pixels kept where the nearest track is on the plane. Two surfaces because the shot looks at two
  places: the sand before the whip pan (plane from 1001-1013, refs 1004, 1010) and the tool's default plane over
  the whole shot (refs 1016-1046 every 6; it leaves no surface in 1004/1010). 1024 px wide, every other frame of
  the shot a target (pairs with too little surface drop out). The RS on vs off comparison is inside one world
  (same poses, same plane), so the world move can't feed it, but RS off there is the RS poses without their
  readout, so the tool also scores against the GS camera (`--baseline`: its own plane from the same tracks at
  readout 0 with a 6 px ray gate, its control, on the pixels both keep).
- Results, RS on vs off (median over targets) / vs GS camera off:
  1004 +44.7 % / +28.0 %, 1010 +57.6 / +36.7, 1016 +56.9 / +33.1, 1022 +26.4 / +27.3, 1028 +57.7 / +34.5,
  1034 +35.6 / +27.6, 1040 +32.5 / +10.8, 1046 +28.6 / none (no shared pixels). **Median of refs +40.1 %, worst
  +26.4 %**, 213 of 238 pairs better: passes both proposed margins (>= 20 %, no ref below -5 %). Vs the GS camera
  +28.0 %, worst +10.8 %, 89 of 110 pairs: passes too. Flipped sign: -16.0 %, worst -28.3 %, 24 of 238 better.
  The GS camera with the readout pasted on is -70 % against itself without (the double count again). Earlier
  variants this morning (half readout +11.2 %, 40 ms on the second surface +9.7 %, one plane over everything
  +36.8 % with older masking) all point the same way; none of them is the headline.
- Deck regression (`tools/rs_deck_regression.py`, the unchanged rs_deck_proof.py with `DSCF3000_S2S3_rs_free`,
  26.46 ms, maps from the old MoGe depth): +0.9 % median of refs (1160 -4.5, 1170 +0.9, 1180 -1.8, 1190 +14.8,
  1200 +9.0), all pairs +0.3 %, 38 of 74. Not worse: passes, barely, and the per-ref spread is bigger than the
  gain, as before (30.9 ms held camera: +0.6 %).
- Still open: the chart half of S5 (needs S1), and whether the margins stand (Mike). The readout here is solved,
  not measured.
- Review images (not committed), `~/atlas-vfx-work/rs_clips/DSCF2973_s5/proof_v2/review/`:
  `DSCF2973_s5_ref1016_t1024.jpg`, `_ref1034_t1026.jpg`, `_ref1046_t1015.jpg`, `DSCF2973_s5_early_ref1004_t1012.jpg`,
  `_early_ref1010_t1001.jpg`, the `_vs_gs.jpg` ones (GS camera off next to RS off / on, shared pixels) and the
  `_flip` ones. Table: `proof_v2/s5_headline.json`.
  sun by eye (done 2026-10-29, see above; heading stays a typed widget, the globe drags the sun).

## 2026-10-28 (Physical Sky, catalogue 48; branch feat/physical-sky)

- `layers/node_physical_sky.py`, Atlas VFX Physical Sky (Atlas VFX/Layers). GPS + UTC capture time (+ offset hours)
  in; out: a scene-linear, unclipped, float32 ACEScg lat-long (also `<name>.exr`, float, ACEScg in the header, plus
  sidecar), an ATLAS_LIGHT_RIG for Relight, a preview through the named OCIO view, and an info JSON (sun, inputs,
  which atlas_scene was used: path, version, git commit, dirty flag, Prague upstream commit, compiled module path,
  dataset path / size / quick hash). Sun modes metadata / offset / manual, typed heading + fov + tilt or an optional
  shot camera (fov from its intrinsics, every frame's rotation; its first frame looks at `heading_deg`).
- Packaging, not new science: the sky is Atlas Scene's `render_environment` with its Prague evaluator, reached
  through the new optional adapter `atlas_vfx/_atlas_scene.py` (lazy, modelled on `_atlas_camera.py`: already
  imported, ATLAS_SCENE_ROOT, installed, then ~/atlas-scene-DEV/atlas-scene). New import edge layers ->
  `_atlas_scene` (CLAUDE.md + test_import ALLOWED). Atlas Scene is not installed in V141's venv: the adapter puts the
  checkout and `integrations/prague/src` on sys.path and loads the compiled `_native` from the base interpreter's
  site-packages (V141's venv is made from the python312 that pip-installed it, same ABI), or from
  ATLAS_SCENE_PRAGUE_NATIVE. The dataset comes from the widget or ATLAS_PRAGUE_DATASET (set as a user env var on ws;
  a V141 started from an older shell may not see it, the widget always works).
- Sun: our own NOAA in `layers/solar.py` (Atlas Scene's `derive_sun` refuses below -4.2 deg). Same numbers as
  derive_sun to 0 deg on 204 suns; vs the NREL SPA published case +0.0023 deg az, -0.0019 deg el (refraction and
  parallax taken out). DSCF2977: 252.355 az, 16.121 el, as last night.
- The lat-long trap, pinned down: Atlas Scene's Prague boundary maps Atlas (x, y, z) to Prague (x, -z, y) and hands
  the compass azimuth straight in, so in its own lat-long the sun sits at x = (270 - A) / 360 W: azimuth grows to the
  LEFT. That is the mirror we hit last night. Ours is Nuke's layout (centre = -Z, +X a quarter right, +Y top) with a
  right-handed compass (north -Z, east +X), azimuth growing to the right; `_Oriented` hands Atlas Scene's evaluator
  the mirrored directions (one 3x3, det -1), no resampling. latlong_centre north or camera heading. The brightest
  pixel of a 2048 render lands within 1.2 px of the predicted sun for DSCF2977 (north) and a manual 45 / 30 sun
  (camera heading 200), never at the mirrored column. Worth a fix upstream in Atlas Scene (its docs say azimuth is
  clockwise from north); not touched here.
- Light rig: normalised to a sky of 1 (E_sky_up / pi luminance); sun colour max 1 and `sun_strength` = the physical
  key against that sky (7.07 for DSCF2977: 16 deg sun, 40 km visibility). Relight got an optional `sun_rig` input
  (replaces the light rig's sun direction + colour, keeps the plate ambient) and `sun_strength_from` (sun_gain as
  before, or 'physical sky': sun_gain = trim x sun_strength x plate sky luminance). Unconnected it is unchanged
  (test_relight still passes; test_physical_sky checks node output = the old maths).
- Below -4.2 deg: no sky is faked. HDRI black, a logged warning, `sky_rendered: false` in info; the rig keeps the
  exact direction with zero strength. Between -4.2 and 0 Prague renders its twilight sky (not checked by eye yet).
- Things to know: the 2048 render takes about 10-13 s (Prague over 2M directions). Atlas Scene point-samples the
  sun disk (about 3 px wide at 2K), so the HDRI holds 1.06 x the sun's true energy for DSCF2977 (info
  `hdri_sun_energy`; it moves with where the disk falls between pixel centres, and below about 1K the disk can be
  missed). Its ground fill integrates that sampled sun too. The rig's sun is integrated properly (81 x 81 samples
  over the disk), so the rig is right even when the HDRI's disk is a little off. Altitude is ignored with the
  ground-level dataset (noted in info); the full dataset on ws takes it. The rig's colours are Linear Rec.709, like
  Relight's srgb path.
- Licence register: atlas-scene (source), prague-sky-model (runtime, Apache-2.0, Atlas Scene's vendored commit
  2385c91 checked against its THIRD_PARTY.md hashes), prague-sky-model-dataset (external, Apache-2.0 per the authors'
  Zenodo record; ws's ground and full files match its md5s).
- `tests/test_physical_sky.py`: 62 checks, fails (never skips) if Atlas Scene or the dataset is missing.
  `workflows/physical_sky_dscf2977.json` via `tools/physical_sky_workflow.py` (repo copy only; the V141 sidebar copy
  waits for the merge, V141 loads the main checkout). Follow-up: the sun globe widget (JS) for picking heading and
  sun by eye.
## 2026-10-28 (Phase 1b S4: Chart Calibrate node)

- `camera_side/node_chart_calibrate.py`, Atlas VFX Chart Calibrate (catalogue 49, category Atlas VFX/Camera). So
  the chart takes drop straight into a graph once they're shot. mode lens (take A) or readout (takes B-E, with A's
  lens JSON; wire lens_path into lens_json). Outputs lens_path, focal_mm, k1, k2, principal point offset, then
  readout_path, readout_ms, sigma_ms, RS and GS rms, measurable, report. Writes `<name>_lens.json` or
  `<name>_readout.json` plus a `.sidecar.json`, default atlas-vfx-work/grid. Video path through `clean_path`,
  name through `safe_name`, out_dir through `resolve_out_dir` with the take as plate_path (never next to it); the
  lens JSON is read with json only and checked (3x3 K, 5 finite distortion numbers, same plate size as the take).
- Shared code moved to `camera_side/chart_calibrate.py`; `tools/grid_rs_shoot.py` is a thin caller now (same
  command line, plus --preset, --focal-guess-mm, --out). Its result JSON keeps the old keys and adds `k1`, `k2`,
  `measurable` and a sidecar; `_inputs` carries the plate's quick_sha256 instead of a first-MB sha1, and a
  readout JSON's readout_ms is null when the take isn't measurable.
- Decode: the tool used its own ffmpeg pipe (select filter + swscale range conversion to gray16). Both now go
  through the plate reader (`plate.decode_luma16`, which got a `step` argument like `decode_rgb`): range from the
  stream, Y plane straight from ffmpeg, no swscale. No good reason to keep the separate path. The lens step runs
  two passes at `every` (the sampled frames, then the frame after each). The colour space is 'Raw' on purpose:
  only luma codes matter, so a take needn't carry a Fuji recording-mode tag.
- Not measurable: readout is reported only when sigma <= 2 ms and <= 10 % of the value. Otherwise measurable is
  False, the JSON's readout_ms is null (the fit value is kept as readout_ms_fit with the reason) and the node puts
  0 on readout_ms. Sensor RS Maps' `readout_chart_ms` is FLOAT forceInput, same type, and the preset rule skips a
  0 chart value (`sensors.readout_candidates` checks `if chart_ms`), so a locked-off take falls through to the RS
  solve or the spec instead of a fake number. sigma_ms is -1 on the wire when the fit says infinite.
- `tests/test_grid_rs.py` step 5: the step 1-4 renders written as lossless HEVC Rext 4:2:2 10-bit full range (no
  -color_range, NOTES 2026-10-19), lens take with each static view held 2 frames. Node lens: focal 16.4714 mm for
  16.4700 (+0.008 %), k1 -0.0298 (-0.03), k2 +0.0117 (0.012), pp offset +6.07, -4.03 px (+6, -4), rms 0.177 px,
  10 of 10 still views; same as the in-memory calibration to 0.02 px. Readout: 31.111 ms for 30.9 (+0.68 %, sigma
  0.02), 15.415 for 15.3 (+0.75 %), RS rms 0.375 / 0.262 px against GS 4.710 / 2.266. Locked off: not measurable
  (sigma 217 ms, 0.001 deg/frame), 0 on the wire, RS Maps' rule stays on the preset spec; a pan's value wins it as
  chart. Bad name, out_dir outside the roots / in the repo / next to the take, missing take, no or junk lens JSON,
  a pan given as a lens take: all refused. The tool on the same lens take agrees to 1e-13 mm. About 2 minutes on
  top of the old 4 (the renders are reused, no second render pass).
- The module-level fits sit +0.7 % high on both pans already (31.11, 15.42), so the 1 % node gate has some
  headroom but not a lot. It is there in memory too, so it isn't the MOV round trip.
- GUI workflow `chart_calibrate_s2s3` (`tools/chart_calibrate_workflow.py`, saved, not queued: no takes yet):
  lens on GRID_A, readout on GRID_B with the lens JSON wired, readout_ms into Sensor RS Maps on the S3 RS camera
  `DSCF3000_S2S3_rs_free` with the preset rule. Node definitions for Atlas nodes come from the classes (V141 runs
  the main checkout). Fixed on the way: `ui_workflow` treated forceInput FLOATs as widgets, which gave RS Maps two
  stray widget values and a linked socket pretending to be a widget.
## 2026-10-26 (Phase 1b S3: RS Solve node)

- `camera_side/node_rs_solve.py`, Atlas VFX RS Solve (catalogue 47, category Atlas VFX/Camera). GS shot camera in,
  points from its own COLMAP solve (or solve_folder + points_camera, same as Readout Solve), RS shot camera +
  readout_ms out. readout free (start at readout_ms) or held; readout_ms 0 = the sensor preset's spec, and a free
  solve with no preset starts at 0.3 frames. profile_ms (up to 20 readouts, 0-100 ms, numbers only) re-solves at
  fixed readouts and adds the free one. Writes `<name>.shot_camera.json`, `<name>.rs_solve.json`, a sidecar.
  Warns when the camera has lens distortion (the solve is pinhole) or is already an RS solve.
- `rs_solve.solve_shot` + `rs_solve.readout_profile` hold what the node and `tools/rs_clip_solve.py` both run (the
  tool's output files are unchanged; not re-run on footage this session). `measure.points_in_camera` moved out of
  Sensor Readout Solve so camera_side can use it; Readout Solve calls it.
- The node's same-world check on the points is 6 px at plate size (the clip tool's gate, 1.5 px on a 1024 proxy),
  not Measure & Scale's 3 px: a GS solve of an RS plate keeps the skew in its residuals (the synthetic one sits at
  3.1 px).
- `tests/test_rs_solve.py` step 5: the synthetic GS solve written as a COLMAP text model, imported, run through the
  node. Free from 29.2 ms: 30.887 ms for 30.900 (0.04 %), 13 LM iterations; profile 20 ms +412 %, 40 ms +301 %;
  output validates; RS Maps' preset rule takes its readout; held stays at 30.9; bad profile / name refused.
  About 9 s more.
- First V141 run (`tools/run_rs_solve_s2s3.py`, GUI workflow `rs_solve_s2s3`): meant for S2-S3, but it solved all
  297 frames because Camera Import ignored `crop_to_plate` and `name` for kind shot_camera (the file was handed on
  as it was). As a check it is still useful: the node reproduces the script's full-shot solve, 26.60 ms free from
  the 29.2 ms preset start, 18 LM iterations, median 2.59 -> 2.04 px, profile 0 ms +11.3 %, 20 ms +0.9 %, 30.9 ms
  +0.3 %, 40 ms +3.6 % (same as 2026-10-21).
- Re-run after the fix and a V141 restart (2026-10-27), S2-S3 only (1130-1297, 168 frames, 17.7k points, 209k
  obs): **26.46 ms** free, 16 LM iterations in 33 s (151 s with the profile). Median 2.61 px GS -> 2.02 px RS (GS
  poses + the readout on top: 4.99 px, the double count again). Moved onto the input world by 0.29 deg, 53 mm,
  scale 0.9997; poses changed 0.23 deg / 26 mm median. Profile: 0 ms +16.9 %, 10 +7.2 %, 20 +1.2 %, 26.5 0,
  30.9 +0.6 %, 40 +5.7 %, all converged. Agrees with the full shot (26.60 ms) to 0.5 %, and the valley is a little
  deeper on S2-S3 alone. `DSCF3000_S2S3_rs_free.shot_camera.json` is the first saved camera with a solved readout;
  Sensor RS Maps' preset rule picks 26.46 ms from it. The chart (S1) is still the measurement.
- Fixed: Camera Import crops and renames a shot camera JSON like every other kind (`camera_io._from_shot_camera`,
  test in test_camera_io). RS Solve now crops its output to the frames it solved: frames outside kept GS poses under
  a camera that says "apply this readout", so RS Maps would have counted the skew twice there.

## 2026-10-26 (Phase 1b S2: presets carry readout and focal with a source)

- `camera_side/sensors.py`: each preset keeps a list of readout evidence (kind, value, source, date) and the flat
  `readout_ms` / `readout_source` / `readout_status` / `readout_date` are filled from it by the plan's rule at
  import, so they can't drift. Kinds: chart (measured) > rs_solve (solved) > spec (unmeasured); `gs_residual` is
  never picked. Same kind: first listed wins.
- Owner, same day: presets are generic. The Fuji entries are the owner's test kit, so they hold published specs
  only, true for anyone's X-H2 or XF16-55. A solve or chart from one person's footage is evidence about that shot
  and reaches RS Maps with the shot (the camera's RS-solve extension or the chart input), never through the table.
  So X-H2 DCI 4K HQ is 29.2 ms unmeasured (DPReview 30.9 ms x 4076 / 4320); the XF16-55 at 16 mm is 16.0 nominal,
  distortion unmeasured with no k1 k2 written. The test-plate evidence stays here: RS-aware solve 26.6 ms on
  DSCF3000 (2026-10-21, profile valley about 27); focal 16.42 mm (Nuke CameraTracker, Camera1.json 16.417) and
  16.47 mm (pycolmap with focal held at 717.7 px @1024, so not an independent solve). The 2026-10-18 entry below
  says 16.471 mm for the Nuke camera; the file says 16.417, so that was a typo.
- Sensor RS Maps: readout_source 'preset rule' is the new default. Preset from the camera's `sensor.preset`, else
  the plate's body and size. Candidates in order: `readout_chart_ms` (new optional input), the camera's own
  `extensions.atlas_vfx_rs_solve`, the preset's spec. The two saved RS cameras (`DSCF3000_S2S3_rs`,
  `DSCF3000_f16_rs`) were solved with the readout held at 30.9 ms, so the rule picks 30.9 for them (source says
  "readout held"; the poses were fit at that readout, so it is the consistent one). The free 26.6 ms solve is not
  saved as a camera. The GS .abc camera gets the 29.2 ms spec. A connected Readout Solve value is kept only to be rejected
  on the record (log warning + sidecar). A camera with no RS-aware solve gets a warning (the deck test lost 4 %
  with the full readout through a GS camera). 'measured if connected' and 'typed' still work as before (the chart,
  if connected, wins 'measured if connected'). The DEMO graph still says 'measured if connected'; moving it to the
  preset rule is S6.
- Sensor Readout Solve: DESCRIPTION starts DEPRECATED, logs a warning on every run. Kept as a diagnostic.
- `tests/test_sensor_presets.py` (new): fields on every preset and lens mark, the rule (chart > solve > spec, GS
  residual never picked in any position or alone), node runs on the synthetic shot with the sidecar checked.

## 2026-10-26 (spec panel with security pass; Nuke text from data, outputs confined)

- `docs/reviews/SPEC_PANEL_2026-10-26.md`. Three findings, all one shape: nodes built text for Nuke by pasting strings and
  wrote it wherever a widget said. Proven: a `'''` or a newline in the Nuke Builder's `shot`, `prefix`, any path or
  a layer name from a `render.json` ran as Python when the script was pasted (`DATA = json.loads(r'''...''')` and
  f-string comments). The three `.nk` writers did not escape paths or names, and 19 writing nodes took any `out_dir`.
- Fixed the same day (Mike's go-ahead): `camera_side/nuke_text.py` (`py_literal`: DATA as one double-quoted
  literal via `json.dumps(json.dumps(data))`; `nk_str`: `"` `\` `[` `]` `$` `{` `}` and line breaks escaped,
  control characters refused; `nk_name`: `\w` only). `plate.resolve_out_dir` (under `WORK_ROOT` or
  `ATLAS_VFX_OUT_ROOTS`, never the repo or the plate's folder) and `plate.safe_name` (one file-name component,
  refused not repaired) in every writing node, the sensor `work_dir` and both Plate Lock out_dirs. The builder also
  refuses an empty `prefix`: a re-run would have deleted every node in the Nuke script. `WORK_ROOT` now honours
  `ATLAS_VFX_WORK`. New test `tests/test_nuke_text.py` (hostile strings through every writer, node-level refusals).
- Hygiene step, same day: `WORK_ROOT` defaults to `~/atlas-vfx-work` (same folder on ws, no username in the
  code); `_atlas_camera` looks for the sibling pack in `custom_nodes`, then `~`-relative paths; Plate Read and
  Camera Import default paths are blank; every `ocio_config` widget defaults to blank = the shipped config
  (`pl.ocio_path`); path widgets go through `pl.clean_path` (`~`, `%VARS%`). The eleven committed workflows now
  say `~` for the home folder and blank for the config (58 replacements; `tools/ui_workflow.py` does the same on
  save, the V141 sidebar copies keep real paths). Farm addresses and the restart script moved to the untracked
  `docs/MACHINES.local.md` (`*.local.md` ignored). `tests/test_hygiene.py`: no LAN address or token name
  anywhere, no user folder in `atlas_vfx/` or `workflows/`, and a per-file BASELINE for the rest (27 files, 112
  hits: CLAUDE.md, docs, tools, tests, prototypes) that may only fall. Also fixed: CLAUDE.md's junction target
  still said the old `atlas-vfx` repo.
- Step 4, same day: `tests/test_import.py` now scans every `.py` in the six packages (absolute, relative and
  `import_module` strings, resolved per file) against an `ALLOWED` graph that CLAUDE.md states in prose. The two
  edges the panel questioned are kept on purpose and written down: layers -> sensor (LTX Prep/Return take Sensor
  Pass maps) and sensor -> platelock (Readout Solve uses Plate Lock's RS camera so its validation is not rs.py
  checking itself). Nothing else crosses; 91 files, no violation. Catalogue: 12 rows added (35-46: the nine Plate
  Lock nodes, LTX Prep, LTX Return, Comp Preview), node 4 is "scripts only", node 28 "deprecated"; the header lists
  which rows load as nodes and the five ComfyUI categories.
- Step 6, same day: `tests/run_all.py` runs every test script as its own process and prints a table (filter by
  name, `--fail-fast`). `tools/register_hashes.py` finds each weights file named in a register note under the
  model roots (`D:/models`, the V141 `models/`), hashes it and writes a `files` list on the component (name,
  bytes, sha256, subfolder, date): 14 files for the 13 model components, about 70 GB in 4.5 minutes; the two
  sha256 already quoted (MoGe-3, BiRefNet) match. `pyproject.toml` carries lower bounds plus the verified set from
  the V141 venv (scipy was missing from the list); `requirements.txt` is what ComfyUI Manager installs; README
  gains an Install section. The Amendment 3 draft is refreshed to 2026-10-26 (28 tests, the 10-24 owner
  decisions, the relight set, this panel; the Phase 1b margins are item 5 for the owner).
- Still open from the panel: Phase 1b margins (Mike), tool port 8288, shot-camera@0.2, a synthetic-plate example
  workflow, the BASELINE debt (110 hits in 26 files).

## 2026-10-24 (old test character removed; Embedded's Nino is the element)

- Owner: the old test character (a Z-Image still re-rendered by LTX) looked wrong, so it is out completely. Removed:
  workflows ltx_element_baseline_s2s3, ltx_element_integration_s3_{crop,keys}, t8_nuke_builder_s2s3_ltx,
  ltx_night_s2s3_test and both walk-relight graphs; tools ab_element_skew, run_ltx_element, ab_ltx_clean_loop,
  run_ltx_night, ltx_clean_loop_workflow. `ltx_element_clean_loop_s2s3` is the LTX template now (layer and prompt
  blank). The DEMO graphs keep the element branch (Load Video, cut out, Character Insert, Card Project) with no clip
  loaded. Layer and node defaults are "element".
- `actor_spot`: the picked deck point where the test character stands (relabelled 2026-10-29 in the stored picks and cameras too; old files are in atlas-vfx-work/_backup_pick_label_2026-10-29).
  Renaming it means migrating that data.
- Next element: Nino from the Embedded project (own original character), regenerated with open-weight models only.

## 2026-10-24 (MIT, private remote, MiniMax H3 dropped)

- Owner decisions for the Comfy "Open Source Higgsfield" entry (audit: C:\Users\miike\_competition_audit\AUDIT.md):
  MIT licence; private GitHub repo mikejamesvfx/atlas-vfx first, made public later; MiniMax H3 dropped (its licence
  excludes US/EU/UK/KR use).
- This repo (C:\Users\miike\atlas-vfx-oss) is a git filter-repo copy of the old local C:\Users\miike\atlas-vfx: the
  NDA codename scrubbed from every commit and message, `.claude/` dropped from history. 50 commits, dates kept. The old
  repo is untouched as a backup; the V141 junction now points here.
- Removed: workflows element_walk_h3, t7_element_character_insert_s2s3, t7_element_from_saved_clip_s2s3; tools run_t7_element,
  run_element_walk_relight; prototypes h3_gen, farm_calls. Character Insert's source_model default is now blank.
- The H3-derived clip and layer are gone from the workflows (see the entry above).

## 2026-10-22 (one rolling-shutter time model; spec panel steps 1 and 3)

- Spec panel review: `docs/reviews/SPEC_PANEL_2026-10-22.md`. Step 1 done (stale facts in CLAUDE.md and the Amendment 3
  draft). Step 3 done here.
- `camera_side/rs_time.py` is now the only definition of row time, pixel convention, anchor, readout ms/frames,
  neighbour pairing and pose interpolation. sensor/rs.py, rs_solve, grid_rs and platelock core/camera.py import it.
- Drift it found (all small, all silent until now):
  - grid_rs and platelock timed rows from OpenCV pixel centres without the +0.5 shift. A constant time offset of
    0.5/H x readout, so about 0.007 ms. Poses absorb it, so readouts didn't move.
  - platelock divided by H - 1 instead of H: 0.05 % on the readout.
  - platelock's quaternion slerp switched to nlerp below 3.6 deg between frames, which is every frame of a real
    plate: 0.004 px off the geodesic. The threshold is now 1e-12.
  - platelock still clamps outside the track (holds the end pose). The others extrapolate from the inner pair.
    Documented in its docstring, not changed.
- `tests/test_rs_time.py`: conventions, geodesic c2w == w2c and == platelock slerp, then one camera with a fast pan
  through all four paths against a reference built from rs_time alone:
  - rs.py: 1e-11 px.
  - rs_solve: 1e-12 px.
  - platelock: under 1e-6 px.
  - grid_rs: rms 2e-11 px with the readout held, and 30.9000 ms recovered with it free.
  Full suite (25 tests) passes. The Readout Solve synthetic check reads 30.71 ms for 30.9 (was 30.6).
- Panel item 1: `docs/dev/IMPLEMENTATION_PLAN.md` now has a Phase 1b Sensor Pass section (S0-S6, tests named,
  acceptance margins marked proposed for Mike).

## 2026-10-22 (readout on four more X-H2 clips; grid shoot tool)

- New clips in the footage folder: DSCF2973 (beach rocks, fast pan, 48 f), DSCF3001 (lawn/steps, 24 f), DSCF3002
  (deck into the clubhouse, 552 f), DSCF3004 (dark interior to bunkroom, 432 f). All 4096x2160 HEVC 10-bit full range,
  F-Log, 1/800, IBIS on, XF16-55. Mike: 4K quality setting not changed, so all 4K HQ.
- Metadata (exiftool 13.59, full dump incl. unknown Fuji tags): nothing marks 4K HQ vs sub-sampled, every Fuji tag
  is identical across all five clips, and there is no focal length for video (lens range only). What differs:
  2973 f/6.4 ISO 2500, 3000 f/4 ISO 2500, 3001/3002/3004 f/2.8 ISO 8000 (3001 starts the second 3000 ends).
- 30.9 ms is DPReview's figure for full-width 8K (16:9, 4320 rows). DCI 4K HQ reads ~4076 rows: ~29 ms expected
  (scaled, not measured).
- `tools/rs_clip_solve.py CLIP`: proxies -> pycolmap GS (focal prior 717.7 px @1024, refine on unless --fix-focal,
  distortion 0) -> shot camera (longest gap-free run) -> RS solve, readout free -> readout profile.
  Results (atlas-vfx-work/rs_clips/):
  - DSCF2973, focal locked: **27.7 ms**, deep valley (0 ms +112 %, 20 ms +27 %, 30.9 ms +7 %). Same from a 12.5 or
    50 ms start. Free focal fails (pure pan, no baseline: 42 mm, 2 points); the tool now refuses weak GS solves.
  - DSCF3001, focal locked: 16.9 ms, shallow (30.9 ms +3.8 %), same from 12.5 / 29 / 50 ms starts.
  - DSCF3004: solved focal 23.2 mm -> 18.0 ms; locked 16.47 mm fits much better (median 2.45 vs 3.38 px) and gives
    **22.9 ms**, flat 20-27 ms (30.9 ms +3 %). The focal moves the readout a lot on this clip.
  - DSCF3002: GS solve fine (412 frames, focal 16.05 mm solved), RS solve did not converge (free 5.4 ms, profile
    costs jump +376 % / +0 % / +491 % between neighbours). Not a measurement.
  - Some fixed-readout profile points hit the 100-iteration cap (3004 30.9; 2973 30.9 and 40), so their costs are
    upper bounds.
- Synthetic check (test_rs_solve scene, pan scaled 1 / 0.3 / 0.1, noise 0.5 / 3 px, 5 % outliers): the free readout
  stays within 30.9 +-0.8 ms except slow + noisy (0.16 deg/frame, 3 px): 28.3-28.5 ms. So noise and slow motion
  alone don't explain 17 ms; the focal (3004) and the plate's own content are the suspects. Settle with the grid.
- `camera_side/grid_rs.py` + `tools/grid_rs_shoot.py` for Mike's 96 x 48 in chart (32 x 16 squares of 76.2 mm,
  docs/lens distortion grid 2to1 base checker only.pdf): OpenCV SB detection (partial chart ok), calibration
  (pinhole + k1 k2) from still frames, labelling absolute when the whole chart is in view (else carried by
  constant-acceleration homography prediction), then per-frame pose + one readout with the chart as known
  structure, row time from the sensor (distorted) row. `tests/test_grid_rs.py` renders the chart through a known
  lens + RS: focal +0.009 %, k1/k2 within 0.0004, readout 31.11 ms for 30.9 and 15.41 for 15.3 (GS rms 4.7 ->
  0.37 px), locked-off shot reports +-210 ms (not measurable). Passes. Shooting rule: keep the whole chart in
  frame during the pans.

## 2026-10-21 (RS-aware solve converged: analytic Jacobian + LM/Schur)

- `rs_solve.solve(method="lm")` (default): analytic Jacobian (checked vs central differences: points, centres,
  readout 1e-9; rotations 2e-3 from taking the SO(3) right Jacobian as identity) and Levenberg-Marquardt with the
  points eliminated (Schur complement; the camera system is 1783 x 1783 dense). DSCF3000 full solve (44.5k points,
  763k obs): ~12 s per iteration, converged in 17-18 iterations, 3.5 min (was 2 h and stopped at 40 iterations).
- Free readout, all points: **26.6 ms**. Fixed 30.9 ms: median 2.04 px (GS 2.59). Deck flat (14 mm MAD) at
  y = 0, scale +0.8 %, points moved 35 mm median, poses 0.45 deg / 38 mm median.
- Readout profile (`tools/rs_readout_profile.py`, 8000 points, re-solved at each readout): cost above the best
  0 ms +11.3 %, 10 ms +4.8 %, 20 ms +0.9 %, 26.6 ms 0, 30.9 ms +0.3 %, 40 ms +3.6 %. A real valley: the plate
  measures its readout at ~27 ms, and the spec 30.9 ms is within 0.3 % of the best cost. (The old fixed-structure
  Readout Solve's 7 ms was the GS bend, not the sensor.)
- Deck test with the converged camera + 30.9 ms maps (74 frames, refs 1160-1200): vs old camera without RS
  **+2.2 %** (better in 44 of 74), vs old camera + RS +3.9 %, RS on vs off with the new camera +0.9 %. Same as the
  first pass: right direction, small. Per ref still mixed (1160 -6.7 %, 1200 +15.9 % RS on vs off). The deck test
  is probably limited by other things now (MoGe depth from the old camera, low-contrast dark boards, exposure
  drift), not the camera.
- `DSCF3000_S2S3_rs` / `DSCF3000_f16_rs` now hold the converged solve; the 40-iteration first pass is kept as
  `*_rs_fd40.shot_camera.json`.

## 2026-10-21 (RS-aware solve, first pass)

- `camera_side/rs_solve.py`: rolling-shutter bundle adjustment from a GS solve (same model as Hedborg et al.,
  "Rolling Shutter Bundle Adjustment", CVPR 2012, from the Linkoping RS tutorial Mike sent: per-row pose
  interpolated between neighbouring frames, joint with points). rs_anchor centre, like sensor/rs.py. Gauge: first
  frame held + path length; result moved back onto the input world by a robust similarity of the points, so the
  deck stays at y = 0. `tests/test_rs_solve.py` (ground truth): free readout 30.88 ms vs 30.90, pose error
  0.39 -> 0.04 deg, readout pasted on the GS solve is worse (2.4 -> 7.9 px). Passes.
- DSCF3000 f16 solve (297 frames), `tools/rs_solve_dscf3000.py`:
  - free readout (8000 points): **26.0 ms** (started at 12.5 ms; spec 30.9). The old fixed-structure Readout Solve
    said 7 ms. Letting the points move is what recovers it. Hit the 40-iteration cap, still creeping.
  - fixed 30.9 ms (all 44.5k points, 763k obs, 2 h): median reprojection 2.59 px (GS) / 4.72 (GS + 30.9 ms pasted
    on) / **2.16 (RS solve)**. Poses moved 0.12 deg / 17 mm median, scale 0.09 %, points 11 mm. Deck stays flat
    (15 mm MAD before and after) at y = 0 (+-3 mm), tilt 0.15 deg. Also capped at 40 iterations.
  - Cameras: `camera/DSCF3000_f16_rs.shot_camera.json`, `DSCF3000_S2S3_rs.shot_camera.json` (readout in
    extensions.atlas_vfx_rs_solve), points `DSCF3000_f16_rs.points.npz`. RS maps from it at 30.9 ms:
    `sensor/DSCF3000_S2S3_rs_rs` (1150-1215 only).
- Deck reprojection test with the new camera (74 target frames over refs 1160-1200, gradient error vs control):
  new camera + RS 30.9 ms vs old camera no RS **+2.1 %**, vs old camera + RS +2.9 %; RS on vs off with the new
  camera +0.9 % (it was -2.5 % with the old camera). Per ref it is noisy (1160 -9 %, 1190 +9 %, 1200 +15 %).
  So the sign has flipped the right way, but the gain on this gentle shot is small and within the per-ref spread.
  Not yet a reason to switch the LTX loop's RS back on.
- Next if Mike wants it: analytic Jacobian (the 2 h is finite differences, ~25 residual passes per step) so it can
  converge instead of stopping at 40 iterations; then a full free-readout run; re-run MoGe depth alignment against
  the new camera (the maps used the old camera's depth).

## 2026-10-21 (RS defaults follow the deck test)

- `ltx_element_clean_loop_s2s3` is now lens-only (vignette, CA, distortion): RS folder blank (`USE_RS = False` in
  `tools/ltx_clean_loop_workflow.py`). The demo's Sensor RS Maps uses `measured if connected` (the ~7 ms residual
  the Readout Solve gets from the GS solve) instead of typed 30.9 ms. Reason: the deck test below. Go back to
  30.9 ms and RS in the LTX loop only with a camera from a rolling-shutter-aware solve.

## 2026-10-20 night (F-Log re-run, the test character placement, holdout, clean-loop LTX A/B)

- The test character: Mike's placement is centred between the two doors on the deck, foot (-1.32, 0, -9.14) in the .abc world
  (deck y = 0: the doormat triangulates to y 0.007), 2.88 m (35 % over 2.13). His head is out of frame the whole shot
  (camera 0.65 m over deck height at 1130, tilted down): Mike accepted that. Camera exits through the left door
  0.4 m from him, so it is inside his card from about 1275.
- .abc camera == f16_GUESS_stairs camera (sensor agent: scale 1.0000000, 4e-6 deg), so its depth is valid with the
  .abc camera. The test character holdout removing 0 % is correct for this placement: under his alpha the nearest plate depth
  is behind his card (1130: plate >= 4.6 m vs card 5.3 m; the newel post at 2.8 m is inside his card's box but not
  over his figure; 1170: the centre post reads 4.4 m vs card 4.0 m). Through the glass doors MoGe sees the
  interior (7-8 m), not the glass.
- F-Log re-run of the demo + sensor workflow: around the test character the plate is only 0.15 stops darker than the F-Log2 decode
  (this dusk scene sits ~3 stops under mid-grey, where the two curves are close), so the test character exposure 1.5 -> 1.25.
- Clean-loop A/B: `tools/ltx_clean_loop_workflow.py` builds `ltx_element_clean_loop_s2s3` and `ltx_element_baseline_s2s3`
  (same seeds); `tools/ab_ltx_clean_loop.py` compares where LTX repaints plate (inside its mask, outside the test character) with
  the real plate: median error and row skew.
- **A/B result (2026-10-21): no measurable difference on this shot, either way.** Both passes ran cleanly
  (1130-1250, 1920x1024, seeds 42/43, ~37 min each on the 5090; the clean pass's Return applied vignette -> CA +
  distortion -> RS). Two tests:
  - Repainted plate ring (inside LTX's mask, outside the test character): median error 2.32/255 clean vs 2.23/255 baseline; clean
    is a little better early (1130-1175) and a little worse late (1190+, camera close). The ring is thin (dilate 12,
    shadow 24 px) and LTX copies its reference plate almost exactly in both, so this test cannot separate them. The
    row-shift part of the test is noise in this dark plate (phase correlation fails on most frames).
  - the test character's own lean (`tools/ab_element_skew.py`): per-row centre of LTX's the test character silhouette vs his card with and without
    the plate camera. Predicted skew median 0.0034 px/row (max 0.019); LTX's silhouette lean varies by ~0.05 px/row
    frame to frame (pose, arms, edges), 14x the signal. Baseline and clean both 0.049 from the prediction, corr ~0.
  - So: the loop is mechanically right (remove/apply round trip 0.0001, alpha intact) and costs nothing visible, but
    S2-S3 is too gentle to prove the benefit through LTX. To prove it: a shot with a fast pan (skew ~0.03+ px/row),
    or a rigid tracked element (a box, a sign) generated in clean space where straight verticals make the lean
    measurable to 0.001.
- **RS maps vs the real plate, no LTX (`tools/rs_deck_proof.py`): with this camera, the physical readout mostly
  makes things worse.** The deck is an exact plane (y = 0): reproject the deck from a reference frame into frames
  1150-1210 through the camera, plain vs Sensor (remove at ref, plane, apply at target), gradient error on the
  deck boards vs the real plate, against a control with the same resampling and no RS geometry. Median change:
  30.9 ms -4.1 % over 5 refs (1160 -4.1, 1170 -7.4, 1180 -6.7, 1190 +13.8, 1200 0.0); 7.3 ms -1.2 % (-2.0, -1.2,
  -1.9, +3.3, -1.0). Worse as the readout grows, with one exception at 1190 (camera climbing the steps, fastest
  tilt), where 30.9 ms helps by 14 %.
  - Reading: the .abc camera is a global-shutter solve (Nuke CameraTracker) of RS footage. Its poses already absorb
    most of the skew (Readout Solve's ~7 ms residual says the same), so adding the full 30.9 ms on top double counts.
    Where motion is fastest (1190) the GS compromise breaks and real RS shows through.
  - **So: don't apply the full readout through a GS-solved camera.** The Sensor Pass is right as maths but needs
    true global-shutter poses from an RS-aware solve (UPSTREAM_TODO 2026-10-20) before the physical 30.9 ms is the
    right number. Until then, for this shot: RS off (or the Readout Solve residual) on LTX renders; the clean loop's
    lens/vignette are no-ops here anyway (k = 0, vignette 0).
  - Limits: one plane, lower part of frame, 1024-wide maps, gradient error in a dark plate.

## 2026-10-20 (Sensor pass phase 1: rolling shutter + lens as STMaps)
- New package `atlas_vfx/sensor/` (category Atlas VFX/Sensor): Readout Solve, Lens Maps, RS Maps, Apply. Convention:
  plate camera OFF the plate (de-RS, undistort) -> generate / comp clean -> plate camera back ON the render last.
- Row time t(y) = t_frame + (y/H - 0.5) * T on SENSOR (distorted) rows. rs map (output = RS sensor image) is closed
  form because Plate Depth is indexed on the plate's own pixels; the de-RS map is a fixed point (row time and depth
  at the current source estimate, point put on the clean ray at that z in the row's camera). Exact inverses: round
  trip 0.00006 px median on a real 4K frame (1210), < 0.001 px on the synthetic tests.
- Lens: Brown-Conrady (OpenCV convention and param order [k1, k2, p1, p2, k3] for model "brown"), Newton inverse,
  lateral CA as per-channel radial scale about the principal point with G the reference, vignette gain applied to
  linear ACEScg only. camera_io had no distortion model (it only records COLMAP's), so this is the one.
- Sensor Apply composes lens + RS into one map per channel (one filter). Both directions are first class (apply
  camera / remove camera). With LTX_PREP connected it takes the render's frame range and crop windows from LTX
  Prep and cuts the plate-res maps to each window (crop mode) or resizes them (full mode, non-uniform work size).
- No RS readout solver existed (platelock has an RS camera model only), so `sensor/readout_solve.py` sweeps the
  readout with platelock's CameraTrack (independent code from rs.py). Synthetic validation: rs.py makes an RS dot
  clip, the solver gets 30.6 ms back for 30.9 (1 %). On the real plate it says ~7 ms: the global-shutter solve has
  already bent its structure to absorb the skew, so the estimate is biased low and not used (UPSTREAM_TODO: RS BA).
- Plate Depth agrees with the .abc camera: the .abc and DSCF3000_f16_GUESS_stairs are the same camera (similarity
  fit over 1130-1297: scale 1.0000000, rotation 4e-6 deg, centres 5e-7 m), and depth at the doormat matches the .abc
  z within 1.4-2.5 % (1170: 4.10 vs 4.20 m; 1210: 2.59 vs 2.63 m). No rescale needed.
- LTX Prep reads the plate from PLATE_REF itself, so the de-RS'd clean plate (Sensor Apply, remove camera) cannot
  go into it yet: LTX Prep needs an optional clean-plate input (IMAGE batch or EXR folder) for the full loop.
- Speed: 4K maps (rs + ders) 14.5 s/frame in numpy; half-res maps are plenty for 1920 LTX renders.
- Readout for DSCF3000: the MOV's Fuji MVTG maker notes (parsed by hand, no exiftool on ws) say X-H2, 4096x2160,
  24p, but carry no tag I can map to 4K HQ vs sub-sampled. Sensor width in this mode is the full 23.5 mm, which both
  modes use. So the typed 30.9 ms (8K / 6.2K / 4K HQ) is an assumption; 15.3 ms if it was shot sub-sampled. Also
  worth an exiftool check: tag 0x3803 is 0x10, which exiftool's table (from memory) calls F-Log, not F-Log2.
  - Settled 2026-10-20: Mike confirms DSCF3000 was shot 4K HQ, so 30.9 ms is the readout (no longer an assumption).
  - exiftool 13.59 on the MOV: `VideoRecordingMode: F-log` (0x3803), shutter 1/800 (about 11 deg at 24p, so keep
    RS motion blur off), lens XF16-55mm f/2.8 at 16 mm, `PeripheralLighting: On` (in-camera vignette correction:
    keep Sensor vignette near 0), IBIS sensor-shift on (continuous: principal point wanders slightly per frame),
    All-Intra 463 Mbps.
  - Settled 2026-10-20: Mike confirms DSCF3000 is **F-Log**, not F-Log2. exiftool's FujiFilm.pm: 0x3803 0x00 Normal,
    0x10 F-Log, 0x20 HLG, 0x30 F-Log2. Config gains `F-Log F-Gamut` (Fuji data sheet constants, matches
    colour-science log_decoding_FLog to 1e-6 above the toe, ~1e-5 absolute in the toe where OCIO derives the linear
    segment offset for continuity). Plate Read 'auto' now reads the tag from the MOV's MVTG block (moov only, never
    the media) and falls back to the body default with a warning. Read as F-Log2, F-Log mid-grey 0.4593 gives 0.352
    instead of 0.18: everything made before today has the plate about a stop bright in the mids with the wrong
    contrast (the test character's +1.5 stop exposure was matching that). Layers, depth, proxies, LTX passes and comps that came from
    a decoded plate need re-making; camera solves from the tracking proxies are fine (log luma + CLAHE).
- Demo (S2-S3, half-res maps, 620 s for 168 frames): skew up to 0.0126 px/row (27 px top to bottom at 4K) around
  1166-1172 and 1215, round trip 0.00007 px median. On the full-frame LTX render (night_test), the skew measured by
  per-band phase correlation of before vs after matches the maps: slope 1.02 horizontal, 0.87 vertical, peak 1168
  0.0098 measured vs 0.0096 predicted px/row. Reviews in atlas-vfx-work\sensor\ (night LTX, the test character LTX crop through
  the LTX_PREP windows, DEMO comp warp-only). Caveat: night_test and the DEMO comp were made from the RS plate, so
  their plate-derived pixels now carry the skew twice; the clean loop needs LTX Prep to take the de-RS plate.
- workflows/DEMO_sensor_pass_s2s3.json = the demo plus group 8 (Readout Solve, Lens Maps, RS Maps, plate -> clean,
  LTX render -> Sensor Apply). V141 is on port 8188 here (not 8288 as CLAUDE.md says); it was not restarted, so the
  sensor nodes appear after Mike's next restart.

## 2026-10-20 (the test character visible: walk clip, exposure, loop, placement)
- Character Insert: end_mode `loop` (wraps to clip_start) besides hold / pingpong; optional `exposure_stops`
  (linear ACEScg gain on the premultiplied RGB, in the sidecar). tests/test_character_time.py.
- The S2130 clip is a standing pose, so the test character read as static. The demo now uses ELEMENT_walk_h3_s2131 (walks in from the
  left, turns to camera; 107 frames), clip_start 12 (fully in the clip frame), hold, +1.5 stops.
- Placement: the camera is low and walks onto the porch; the top of frame at the deck is under ~2 m, so a 2.13 m the test character
  is never whole anywhere on the deck (moving him to the deck front only cut more of him). On the lawn left of the
  steps (pick 500, 1540 at 1130, y -0.76) he is whole for the first 49 frames, ~1400 px tall, 2 m clear of the camera.
  Known flaw: about 1170-1200 he is drawn over the stair railing and the deck fascia that should hide his legs (flat
  card, no holdout); next fix is a depth holdout (Plate Depth / Set Proxy) in Card Project, Comp Preview and Nuke.
- Comp Preview burn-in: darkened strip, no outline (the outline ghosted at 1280 wide).

## 2026-10-20 (Comp Preview, demo workflow)
- New node Atlas VFX Comp Preview (deliver/comp_preview.py): plate + layer EXR folders, back to front, premultiplied
  over in ACEScg, then the ACES 2.0 SDR view, optional burn-in; IMAGE frames + fps for Create Video / Save Video.
  A preview for ComfyUI, not the delivery. About 1.2 s/frame at 1280 wide on ws (three 4K EXR reads per frame).
  OpenCV 5 putText needs 8-bit: the burn-in draws on an 8-bit copy of the top strip only.
- workflows/DEMO_all_in_one_s2s3.json (screen-share demo): plate, .abc camera, Lock Check, Z-Image grass cards,
  The test character from the saved H3 clip (also saved as a movie), camera export + Nuke builder (build_DSCF3000_S2S3_demo.py,
  so the accepted build stays), and group 7 "Watch the shot" (Comp Preview -> Save Video). Ran on V141 in 851 s
  from cold; layers match the accepted S2-S3 layers to 5e-4. The card the test character stays sharp through the fast move at
  ~1190 where the plate blurs (the LTX integration pass is the fix for that).
- The test character does move (H3 clip mapped frame for frame, silhouette changes by ~14k px over the shot), but it is a standing,
  feet-planted performance; the stills every 24 frames made it look static.
- Restarting V141 from here: the Manager reboot endpoint is not installed; Mike launches
  `Windows_Run_ComfyUI_Default.bat` (stop python, relaunch the .bat).

## 2026-10-20 (Alembic camera import)
- Camera Import takes `.abc` (kind `abc`, auto-detected; optional `abc_camera` when a file has several). Reader is
  `camera_side/abc_io.py`, pure Python on the Ogawa format: object tree, Xform op stacks (translate, rotate,
  rotateX/Y/Z, scale, matrix) with inherits, the Camera `.core` sample. HDF5 Alembic is refused. The venv's
  `alembic` package is SQLAlchemy's migration tool, not the Alembic library; PyAlembic is not installed.
- Frame = sample time x plate fps. Only properties that change set the range: Nuke writes `.ops` on the default
  sampling (0, 1, 2 ... s), which first gave frames 0-14328 instead of 1001-1598.
- Same rules as Nuke JSON: sensor from the preset, focal rescaled if the file's horizontal aperture differs (FOV kept),
  film offset / squeeze / non-identity film back refused (Nuke writes postScale 1 1, which passes). Scale in the parent
  chain is removed from the rotation and reported (`chain_scale_removed`).
- Checked on real exports (files in atlas-vfx-work/ABC, not in git):
  - Mike's Nuke 17.1 export `AVFX_CameraTest_Exp.abc` (our builder's AVFX_Camera, matrix op): against
    `DSCF3000_f16_GUESS_stairs.shot_camera.json`, 297 shared frames, max 9e-7 m, 5e-6 deg, focal 8e-7 mm. Nuke wrote
    1001-1598 (it holds the curve past 1297); crop_to_plate trims it.
  - Blender 5.2 (euler XYZ camera under an animated, rotated empty at scale 1.5): world matrix 1e-6, focal exact.
  - Maya 2026 (rotateOrder zxy, rotatePivot, group scale 1.3, cm, m_per_unit 0.01): 2e-15 m, 1e-6 deg, focal exact.
- tests/test_abc_import.py writes its own .abc files with a minimal Ogawa writer (no binary fixture in git).
- Camera Import `offset_first_frame` (optional, default off): renumber so the camera's first frame is 1, 1001 or
  the plate start, then crop to the plate (works for every kind: abc, chan, Nuke JSON, COLMAP). "Plate start" is the
  plate file's first frame (`frame_offset`, 1001), not the cut: the first version used `nuke_first` (1130 for S2-S3)
  and slid Mike's whole-clip camera +129 frames; caught in the node smoke test. Static cameras are held over the crop.

## 2026-10-19 (Day-to-Night test, Light-A-Video relight, GPU turns)
- Sharing ws with another session (Plate Lock): `tools/gpu_turn.py take/release` holds
  `atlas-vfx-work/GPU_TURN.txt` and waits for an empty V141 queue before any run or restart. My runners use it.
- Day-to-Night (`tools/run_ltx_night.py`, layer `night_test`): LTX-2.5 distilled bf16 + the Day-to-Night IC-LoRA,
  comp (plate + LTX the test character + grass) as the guide, FullHD, 1130-1226, about 9 min. Result: it adds motivated light (the test character's
  lantern lit, a warm light by the door, the doormat picks it up) but barely reads as night: the plate is already
  dusk, and at cfg 1 the distilled model ignores the negative prompt. It also invents detail (door glazing changes,
  the far grass thins). Lock: frame 1130 within 0.5 px, but after that the output lags the plate by about half to one
  frame (night n best matches day n-1 at 1154/1170; flow median ~20 px at 4K at n). The integration pass does NOT
  lag (best match at n on every frame checked; its keyframes pin it). Next for night: a stronger night prompt or the
  dev model at guidance 3-4 (per the card), then lock it back onto the plate with Plate Lock Cross-View (the other
  session's dense per-frame warp, built to ignore relighting) rather than trusting LTX's timing.
- The test character movement clip (`tools/run_element_walk_relight.py gen`): H3 on the farm, the test character walks in from the left on flat grey
  and turns to camera, 107 frames 768x1344, 4.7 min.
- Light-A-Video (AnimateDiff + IC-Light SD1.5, Realistic Vision V6; licences registered; core BiRefNet mask fed in):
  AnimateDiff v1-5-3 holds 32 frames (49 crashed on the positional embedding), so 32 frames every 3rd, 448x784,
  ~5.4 min per setup. Light direction reads well (left key, cast floor shadows, rim) and is consistent over the walk,
  but it softens progressively and loses identity detail (lantern and hessian gone by the end) and replaces the
  backdrop with a light gradient: a lighting reference, not a plate.
- Light transfer (`tools/light_transfer.py`): shade = blur(relit) / blur(source luminance) in linear, multiplied onto
  the full-res H3 source. Keeps every H3 detail and takes the new light; slight halo at the silhouette where the relit
  background differs (fine once matted). Outputs `ELEMENT_walk_lighttransfer_{night_porch,sunset_left}.mp4` in V141
  output/atlas-vfx. Worth a node (catalogue 18 Relight) if Mike likes it.

## 2026-10-19 (LTX-2.5 integration pass on the test character)
- Goal: re-render the T7 the test character card into the plate so it takes the plate's light, blur and grain, without touching the
  lock. LTX-2.5 distilled (bf16) + the in/outpainting IC-LoRA (a 2.3 file; LTX-2.5 runs 2.3 LoRAs), following
  Lightricks' LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled recipe: stage 1 at 960x512, 8 steps; stage 2 at 1920x1024,
  2 steps; Laplacian blend each stage. Frames 1130-1250 (121). About 22-23 min per run on the 5090.
- New nodes: LTX Prep (plate -> ACEScg -> named ACES 2.0 SDR view, the test character's projected alpha as the mask with 24 px of
  shadow room below and 12 px dilation, start image + keyframes = the 2.5D comp) and LTX Return (inverse view ->
  ACEScg, upscale to 4K, alpha = the feathered mask, premultiplied EXR layer `layers/element_ltx_keys`). Outside the mask
  alpha is 0, so the plate and the lock are untouched (test: bit-exact; an unchanged pass returns within 0.19 %).
  View round trip on the real plate: median 0.0006 %, nothing over 5 %.
- What failed first: start image only. The test character held for about 2 s, then LTX painted the green fill (LTXVInpaintPreprocess
  puts green under the mask) into his silhouette: a green furry figure from ~1138 on, int8 and bf16 alike. A 49-frame
  bf16 test looked fine only because it was short; I briefly blamed the int8-convrot transformer, wrongly.
- Fix: the 2.5D comp as LTXVAddGuide keyframes every 24 frames (strength 0.8). Greenness inside the test character is <= 0 on every
  sampled frame (was +0.13 to +0.26 without keyframes).
- Result (`layers/LTX_element_compare_keys.jpg`): the test character takes the dusk grade and the plate's softness; through the fast
  blur around 1190 he blurs with the plate where the 2.5D card stays sharp. Feet stay where the card put them.
  Costume (lantern, ribbon, hessian) holds. Cost: fine detail is softer (generated at 1920 wide, upscaled 2.1x).
  No obvious contact shadow yet.
- Licence register: LTX-2.5 distilled, Gemma 4 + LTX projection text encoder, LTX VAEs, latent upscaler, the 2.3
  in/outpainting IC-LoRA, ComfyUI-LTXVideo (its LICENSE is the LTX-2 Community License, not Apache). Client work: paid
  licence or legal check first.
- Nuke A/B: `nuke/build_DSCF3000_S2S3_ltx.py` builds the same tree with the LTX the test character, prefix AVFXL_, offset 1200 px in
  the DAG, root untouched, so it sits beside the AVFX_ tree. Builder gained `x_offset`.
- Crop mode (`LTX_MODE=crop`, layer `element_ltx_crop`): LTX Prep cuts a 1216x1920 portrait window that follows the test character
  (from his projected alpha box + 96 px margin), 1:1 with the plate while he fits (50 of 121 frames, 1130-~1185),
  growing smoothly to 1.125x (full plate height) as he gets close; LTX Return pastes it back with the edge faded where
  the window is inside the plate. Same keyframes. 20.5 min. Clearly sharper than the full-frame pass (face and feet
  at 1130, hessian at 1154 and 1230, feet at 1210 planted with some contact darkening where full-frame smeared
  them); no green anywhere (<= 0.02); no visible window seam. The Nuke A/B script now uses this layer.
- Next if it's worth it: a stronger shadow prompt or a darker
  shadow strip in the start image; try int8 with keyframes (faster); Day-to-Night on the whole comp.

## 2026-10-19 (slice accepted)
- Mike ran the builder in Nuke Indie 17.1 after the OCIO fix: "that tracked great". Grass and the test character hold through the
  move. Phase 1 "character on the deck" done-when is met (scale still guessed from the stairs: tape one riser).
- ADR-022 Amendment 3 drafted for Mike in `docs/dev/ADR-022_amendment3_DRAFT.local.md` (untracked since 2026-10-29: kept out of the public repo; atlas-nexus not edited). Phase 2
  waits for Mike, per the plan.

## 2026-10-19 (T8)
- Mike ran the build in Nuke 17.1: it worked, except F-Log2 F-Gamut was not in Nuke's colorspace list. Our added
  colour space had no `categories`; Nuke's Read/Write menus only list spaces tagged `file-io` (all the built-in
  input spaces are). Added `categories: [file-io]` to it in the shipped config, and test_plate now checks every
  active input space carries the tag. Nuke reads a config once, so restart Nuke (or reload the config) and re-run
  the build script.
- Nuke Builder writes `atlas-vfx-work/DSCF3000/nuke/build_DSCF3000_S2S3.py`. Run it in the Script Editor with
  `exec(open(r'C:/Users/miike/atlas-vfx-work/DSCF3000/nuke/build_DSCF3000_S2S3.py').read())`. It makes AVFX_Plate
  (DPX 1130-1297, F-Log2 F-Gamut), AVFX_Layer_element / grass_far / grass_near (ACEScg, premultiplied, black outside
  their range) merged over back to front, AVFX_Write_comp (EXR ACEScg into atlas-vfx-work/DSCF3000/comp, not
  rendered), AVFX_Camera (keys from the chan, ZXY, 23.5 x 12.393 mm) and AVFX_SetProxy (ReadGeo of the set OBJ).
  It sets the root to the shipped OCIO config (ACEScg working), 1130-1297, 24 fps, 4096x2160. Re-running deletes
  only AVFX_* nodes. Every knob is set on its own and anything Nuke refuses is listed at the end instead of
  stopping the build; a missing file stops it before any node is made.
- Not run in Nuke yet: Indie can't run headless, so the acceptance check (runs without edits, layers hold by eye) is
  Mike's. The test runs the script twice against a stand-in `nuke` module (idempotency, merge wiring, camera keys,
  Camera3 -> Camera2 fallback). The knob names most likely to differ in 17.1 are the root OCIO ones
  (`colorManagement`, `OCIO_config`, `customOCIOConfigPath`, `workingSpaceLUT`); if they're refused, set the config
  by hand once.
- The camera is parsed from the chan by the script and keyed directly, not `import_chan`, so rot order and units are
  ours. The chan covers the whole camera (1001-1297); only 1130-1297 matters for the shot.
- Camera Export also writes `.usda` by hand (no pxr in V141; Atlas Camera's USDExporter takes its own solve types,
  logged in UPSTREAM_TODO). Transform samples match cam_to_world to 5e-9 in the test.
- EXR Write node: IMAGE (+ mask) in any named OCIO space -> premultiplied ACEScg RGBA half with colorInteropID.
  `exr_io.check_sequence` checks the layer sequences (complete, RGBA half, ACEScg, plate size) before the builder
  writes the script; all three layers pass.
- Plate in Nuke is the DPX conversion (frame numbers match, full-range codes). It went through swscale, so it has the
  ~1-code error noted for T2; the layers were rendered against the numpy decode. Not visible in a comp; noted.

## 2026-10-19 (T7)
- The test character from MiniMax H3 Ref2VA on the farm (text encoder on render, ref encodes on hub, decode on ws), 768x1344
  portrait, 175 frames (H3 lengths are 5 + 17k; 175 covers the 168-frame shot), 20 steps, seed 2130, refs element.png +
  element_expressions.png at "max". 11.5 min. Locked-off camera, full body, feet planted the whole clip (feet row spread
  0 px over 175 frames), head turn mid-clip. Licence: register entry was already there; the node logs the H3
  warning and writes it into the card and sidecar (`source_model` input). Own test only, never on client work.
- Changed from the plan: the matte. The plan's "simple distance key" failed on the real clip. H3's "flat grey" is a
  vignetted backdrop with a floor, a soft seam and floor shadows, and the test character's hessian is grey-brown. A smooth backdrop
  plate (degree-4 polynomial, refitted on backdrop pixels) fixed the vignette but not the floor. The node now takes
  an optional MASK; the run feeds core BiRefNet per frame. The distance key stays as the fallback (and is tested),
  the backdrop plate is still used for edge decontamination. Two fixes the test drove in the distance key: alpha
  normalised by the local foreground distance (a plain clip saturated early and left grey in every edge), and a
  snap to 1 inside.
- Placement: the guessed `element_spot` (frame 1210, 2048/1800) sits on the camera path, 0.12 m from it at the closest:
  the camera walks through him. Moved him to the deck right of the door, 0.6 m off the wall: pick at 1130, plate px
  (2202, 1079) = (-0.74, 0, -9.48), closest approach 1.22 m. A 2.13 m figure never fits in frame whole here: the
  camera is 0.65 m above the deck at 1130 and the wall is 5.7 m away, so at 1130 we see 0-2.05 m of him, then the
  camera closes to his legs and torso, feet in frame 1130-1226, out of frame right by about 1236.
- Billboard card (turns to face the camera every frame, feet line fixed on the deck). Synthetic test: 100 frames, foot
  columns on the deck line within 0.18 px at 4096 (plan: 1 px). Card Project now takes animated cards
  (`corners_by_frame`, `texture_pattern` + `texture_by_frame`; `card_at()`), clip ping-pongs past its end.
- Real result (`layers/T7_element_comp_check.jpg`): he stands on the boards, about door height, and holds through the
  move by eye. 2.5D limits: one fixed view of him (no parallax inside the figure, the view never changes as the camera
  swings about 30 deg round him), no contact shadow (BiRefNet drops the floor shadow; add one in comp), he is sharp
  where the plate has motion blur and defocus around 1190, and the grade is the generator's, not the dusk plate's.
- `tools/run_t7_element.py` can reuse the saved clip (`T7_CLIP=atlas-vfx/T7_element_h3_s2130_00001_.mp4`, copied into V141
  input) so matte or placement changes don't regenerate. GUI workflows: `t7_element_character_insert_s2s3` (generate)
  and `t7_element_from_saved_clip_s2s3`.

## 2026-10-19 (T6)
- Licence register first: Z-Image-Turbo (Apache-2.0), its qwen_3_4b text encoder and the FLUX ae VAE (Apache-2.0,
  Comfy-Org/z_image_turbo repack), BiRefNet (MIT, core LoadBackgroundRemovalModel / RemoveBackground), all checked
  on Hugging Face 2026-10-19.
- Card Project (`layers/card_project.py`): per-pixel ray/plane intersection, not a homography, so it stays right when
  the camera walks up to and past a close card. Synthetic: 683k pixels within 0.0002 px of the forward projection at
  1024 (plan: < 0.5 px). Bug found by the test: (u, v) by separate projections onto U and V is only right for
  rectangles; now solved with the Gram matrix.
- Foreground Element: Z-Image-Turbo on a plain grey backdrop (template settings: 8 steps, cfg 1, res_multistep,
  AuraFlow shift 3; seed 1130, batch 2 -> one clump per card), BiRefNet foreground mask, erode 1 px, edge
  decontamination against the backdrop median, OCIO 'sRGB Encoded Rec.709 (sRGB)' -> ACEScg, premultiplied RGBA half
  EXR (colorInteropID lin_ap1_scene). The finest outer blades are lost by BiRefNet; seed heads and blades hold.
- Placement lesson: the camera is ~1.37 m above the lawn and the vertical FOV is 41 deg, so 1 m ahead the frame bottom
  is ~0.97 m up. A 0.9 m clump there was entirely below frame (first render showed only seed tips). Final: near clump
  1.35 m tall, 1.0 m ahead, 0.6 m left; far clump 1.4 m tall, 2.0 m ahead, 1.3 m right beside the stair stringer;
  both on the lawn at y -0.76. Visible 1130 to ~1160 with correct parallax (near leaves faster).
- Output: `atlas-vfx-work/DSCF3000/layers/DSCF3000_f16_GUESS_stairs_grass_{near,far}/grass_*.%04d.exr`, 1130-1297,
  4096x2160; textures + card JSON in `elements/`. Check image `layers/T6_grass_comp_check.jpg`. Both renders ~9 min.
- For the comp (T8 / Mike): the grass is brighter and sharper than the dusky, soft plate foreground and has no motion
  blur (the near clump moves fast around 1138). Grade, defocus and vector blur in Nuke; or add blur here later.

## 2026-10-19 (guessed picks, Mike: "just guess")
- Picks file: `atlas-vfx-work/DSCF3000/picks/DSCF3000_S2S3_picks_GUESS.json`. Camera: `DSCF3000_f16_GUESS_stairs`
  (0.821 m per solve unit, Y up, deck y = 0). Everything downstream of T4 now uses it.
- First try (clicked corners tracked with LK, `tools/track_picks.py`): step riser / deck rail / verandah post disagreed
  by 36 % and the verticals by 22 deg. The camera walks straight at the house (triangulation depth is weak near the
  frame centre) and LK slides along white-on-white edges (5-10 px residuals). Not usable.
- What worked: the stairs. Solve-point heights in the stair region cluster at 5 levels (pad, 3 treads, deck) one
  riser apart (0.22 u). New measurement kind `plane_distance` (pad plane to deck plane, mean normal, regions kept close),
  guessed 4 x 0.18 m = 0.72 m. Deck plane near the steps vs near the door agree within 1.2 % (0.821 vs 0.831 m/u);
  measuring along one normal instead of the mean had them 8 % apart.
- Sanity: camera ~1.37 m above the pad while walking on the lawn; 0.6-0.9 m above the deck climbing the steps,
  ~1.4 m at the door. Deck at y +0.005 m, tilt 0.5 deg. Lock Check card at actor_spot holds on the deck 1130-1220.
- Checks that do NOT agree: deck rail 1.32 m and verandah post 3.29 m against guesses 1.0 / 2.4. Their picks are the
  unreliable tracked ones, so they are reported only. If a riser is taped and is not 0.18, scale moves in proportion.
- The prototype's "door" (frame 0, assumed 2.08 m) gave 1.078 m/u: 31 % bigger than the stairs say. It was not the
  door (it is ~1.6 m at 0.821 m/u). MoGe-3's own metric scale (1.56 x in the 1.078 world) implies ~0.69 m/u, between.
- Set Proxy: with max_tilt 10 deg the wall drifted onto a 6.4 deg leaning mix of surfaces; default is now 3 deg
  (cameras here are gravity aligned). Wall 7.0 x 2.3 m, tilt 1.6 deg, rms 1.7 cm; deck 6.6 x 5.8 m, rms 1.4 cm.

## 2026-10-19 (T5)
- Plate Depth (`camera_side/plate_depth.py` + node): MoGe-3 (ComfyUI core `LoadMoGeModel`, model in the register)
  on the display proxy at 1024, `fov_x` = the shot camera's horizontal FOV, then each frame scaled by a 3-MAD-trimmed
  median of solve-point depth / MoGe depth. EXR per frame: Z (z depth, not 1/z, 0 = invalid) + A, float32 zip,
  `colorInteropID data` (OIIO 3.2 writes the new OpenEXR standard attribute) + `atlas_vfx:*` attributes.
- On S2-S3 with the provisional camera: 168 frames in ~4.5 min on the 5090, median 985 solve points per frame, point
  ratios agree within 2.4 % (p90 13.7 %), no frame needed interpolation. The scale moves 1.7 % frame to frame
  (median): fine for plane fitting, would flicker if depth were used per frame. Smooth it before T6+ use depth that way.
- Solve points now follow the camera through Measure & Scale (`measure.points_for_camera`: import m_per_unit, then
  the measure transform, then a reprojection check). Shared by Measure & Scale, Plate Depth and Set Proxy.
- Set Proxy (`layers/set_proxy.py` + node): constrained RANSAC, deck = horizontal near deck_y, wall = vertical facing
  the camera path; each trimmed by the other (the floor through the doorway is the deck's level but not the deck) and
  outlined by a density test (10 cm cells) so stray coplanar points do not stretch the hull. OBJ + USDA (hand-written,
  no pxr in V141) + JSON with plane equations.
- Provisional result (deck_y 1.41, lawn-ground camera): deck 8.3 x 7.4 m at y 1.15, tilt 2.5 deg against the
  provisional up, rms 1.5 cm; wall 8.6 x 3.4 m, rms 1.7 cm. Drawn on 1135-1255 the deck sits on the boards and the
  wall's base meets the deck along the door base. The 2.5 deg tilt should drop with Mike's verticals.
- Synthetic test: depth scale within 0.034 %, deck 0.002 deg / 0.0 mm, wall 0.039 deg / 0.3 mm with 5 % junk.
- Bug found on the way: the node used the camera's whole range (S1 lawn included) for solve points and the camera
  path; it now uses only the depth's frames. Deck went 12.0 -> 8.3 m wide.

## 2026-10-19 (T4, code done, waiting on Mike's picks)
- Measure & Scale (`camera_side/measure.py`): picks JSON (docs/PICKING.md, Nuke bottom-left pixels by default).
  Ends picked in 2+ frames are triangulated (works with the Nuke camera too); ends picked in one frame hit a plane
  fitted to solve points in a region (pycolmap camera only). Scale = median, refused over 5 % spread unless forced.
  Up = mean of the verticals; ground from a region (frames pooled) or picked points. Output: metres, Y up, ground
  y = 0, first camera at x = z = 0 looking -Z; `<name>.measure.json` holds every number and labelled points.
- Synthetic test: scale 0.08 % off, up 0.10 deg, ground within 1 cm, camera height within 0.5 mm.
- First real run taught two guards: (1) ground from one frame's bottom strip is a thin band of points, and RANSAC
  happily fitted a near-vertical plane through it (74-81 deg off the door). Regions can now pool frames, line-like
  sets are refused, and a ground more than 5 deg off the verticals is refused unless forced. (2) ComfyUI cached
  Measure & Scale when the picks file changed but its path did not: IS_CHANGED now hashes the picks file, and Camera
  Import keys on its source's size/mtime.
- Provisional real run (prototype door picks, frame 0 only, door ASSUMED 2.08 m, lawn pooled over 1001-1101) on
  the f16 camera: jamb 1.929 u (prototype 1.920), 1.078 m/u, lawn 3.1 deg off the door vertical (lawn slopes),
  walking camera 1.57-1.61 m above the lawn. Lock Check card at the threshold sits flat on the deck at 1135; it
  goes behind the camera at ~1195 as the camera passes it. That suggests the frame-0 "door jamb" is the post/opening
  at the top of the steps rather than the entrance door. Mike's picks replace all of this.
- For S2-S3, use the deck as ground (the test character stands on it). Heights against the sloping lawn drift (3.5 m at 1001).
- Lock Check (`camera_side/lock_check.py`): grid card (ground or vertical) + measured points, near-clipped, on the
  display proxy -> mp4 + sidecar in `atlas-vfx-work/DSCF3000/lock_check/`.

## 2026-10-19 (T3)
- Camera Import (`camera_side/camera_io.py`): Nuke Camera JSON, .chan (any rot_order) and pycolmap models ->
  shot-camera@0.1. Sensor always from the preset (`camera_side/sensors.py`); Nuke's vaperture is recorded and
  ignored; a haperture that is not the sensor width rescales focal to keep Nuke's horizontal FOV. Lens shift
  (win_translate) and non-SRT/RST xform orders are refused rather than guessed. COLMAP: largest model wins, image
  number = plate index, intrinsics scaled proxy -> plate, gaps in the frames refused (import a clean range).
- Contract fix found here: `intrinsics()` took fy from sensor height_mm. 12.39 is a rounded measurement (square
  pixels need 12.3926), which skewed fy by 0.02 % and moved points ~1 px at the edge of a 4K frame. fy is now
  fx * pixel_aspect, like Nuke; sensor height is only checked. Schema description updated.
- Solve Compare reproduces align_nuke.py to every printed digit: Camera1 vs solve_s1s3_f16 over 1001-1297 =
  0.9710 % of path, 2.9385 deg view angle, scale 0.104875.
- On the Phase 1 frames only (S2-S3, 1130-1297, fitted on those frames) the two agree much better: 0.09 % of path,
  0.29 deg median view angle (p95 0.38). The 3 deg is mostly S1 (the lawn). Either camera is fine for the slice.
- Imported S2-S3 cameras are in `atlas-vfx-work/DSCF3000/camera/` (shot camera + sidecar each). The pycolmap one
  is in raw solve axes (world Y points down); Measure & Scale (T4) does the gravity alignment and metres.
- pycolmap was also missing from the V141 venv; installed pycolmap==4.2.1 (numpy already satisfied). V141 now
  matches what CLAUDE.md says it has.

## 2026-10-19 (T2)
- Plate Read + Tracking Proxy built (`camera_side/plate.py`, two node modules) and run in V141 on S2-S3: 34 s for
  168 frames, proxies in `atlas-vfx-work/DSCF3000/track_0129-0296_w1024/` with `track_proxy.json`. GUI workflow
  `workflows/t2_plate_read_s2s3.json` (also in the V141 Workflows sidebar under atlas-vfx).
- swscale is not accurate enough for 10-bit full-range Y'CbCr->RGB (about 1 code off, tints neutrals). ffmpeg now
  only decodes native planes; range (H.273), matrix and chroma siting are applied in numpy. Synthetic codes come
  back exact, BT.709 colour patches within code rounding.
- Plan said "synthetic full-range ProRes": ProRes cannot carry a full-range flag (ffmpeg writes tv regardless; it is
  video range by spec). The full-range test file is lossless HEVC Rext 4:2:2 10-bit, the X-H2's own format; ProRes
  is used for the legal-range case, like Mike's conversion. Also: passing -color_range to ffmpeg when making test
  files silently inserts a range conversion (900 -> 903), so the test sets the range as HEVC metadata only.
- DSCF3000 stream tags say bt709 transfer/primaries; ignored. Input space comes from the body (X-H2 -> F-Log2
  F-Gamut). Chroma siting is `unspecified`, so HEVC's default (co-sited left) is used.
- Fixed display proxy vs the prototype (frame 200): shadows in the deck and doorway are back (mean 24.8 vs 19.7).
  Tracking proxies differ from the prototype's by 13/255 on average, all global tone from the old clipping; the
  structure is the same. Not re-solved yet.
- OpenColorIO was missing from the V141 venv (CLAUDE.md said it was there); installed opencolorio==2.6.0.

## 2026-10-19 (T1)
- shot-camera@0.1 finished: strict schema (unknown keys rejected, free `extensions` object for producers), plus rule
  checks in `shot_camera.validate()`: every frame once, orthonormal det +1 rotations, mm sanity on sensor/focal,
  sensor aspect vs pixel aspect within 1 %. Added `source.kind = synthetic` and `plate.pixel_aspect`.
- `to_nuke_chan` matches Nuke's own export layout (tab separated, 6 decimals, frame tx ty tz rx ry rz focal) and
  unwraps angles frame to frame. Mike's Camera1_metres.chan (528 frames) round trips exactly.
- For T3: Mike's Camera1.json has `vaperture 15.7` (Nuke's default), real is 12.39. Camera Import must take the
  sensor height from the preset, not the Nuke camera, or validate() will reject it (pixel aspect 0.79).

## 2026-10-19 (T0)
- Root `__init__.py` puts the repo on sys.path and calls `atlas_vfx.registry.load_nodes()`. Each subpackage lists
  its node modules in `NODE_MODULES`, one node per module. An ImportError disables that module's node; a class can
  also declare `REQUIRES = ("OpenImageIO",)`. Both get logged and the rest load.
- `_atlas_camera.py` looks in this order: already imported (V141 loads atlas-camera first, alphabetical), then
  `ATLAS_CAMERA_ROOT`, the installed pack, the dev tree. Nothing is imported until asked for. `report()` goes in logs.
- Junction `custom_nodes\atlas-vfx` -> repo is in place on V141. Log line:
  `atlas-vfx 0.0.1: 0 nodes loaded, 0 disabled; Atlas Camera: imported 0.8.1 at ...\custom_nodes\atlas-camera`.

## 2026-10-18 (repo created)
- One repo for now, split later: camera_side/ goes to Atlas Camera, the rest stays (ADR-022 Amendment 2).
- Evidence so far, all on DSCF3000 S1-S3: Nuke CameraTracker 16.471 mm and pycolmap (focal pinned 717.7 px @1024,
  k=0) agree to ~1 % of path length and ~3 deg median view direction. Scale ~10.33 m per Nuke unit from the frame-0
  door at an ASSUMED 2.08 m. Mike still to measure the door, a step riser and the deck rail.
- Lock check (card on the wall plane) held across S1-S3 by eye.
- Phase 1 slice: "character on the deck", S2-S3, Nuke frames 1130-1297. The test character is 2.13 m.
