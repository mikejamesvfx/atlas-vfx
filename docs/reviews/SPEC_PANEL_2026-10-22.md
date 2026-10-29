# Spec panel review, 2026-10-22

What was reviewed: CLAUDE.md, docs/dev/IMPLEMENTATION_PLAN.md, NODE_CATALOGUE.md, NOTES.md (2026-10-18 to 2026-10-22),
UPSTREAM_TODO.md, ADR-022 + the Amendment 3 draft, and the code as of commit 46ba655 (46 commits, 86 modules,
24 script tests). Mode: critique. Panel: Wiegers (requirements), Adzic (examples), Fowler (architecture),
Nygard (failure modes), Crispin (testing). Nothing in the plan or the ADR was edited; this file only proposes.

## Verdict

Phase 1 is done and well evidenced. Everything since (Sensor Pass, RS-aware solve, grid tool, Plate Lock, LTX
loop) was built from NOTES entries and chat, not from a spec. The code is ahead of the documents, and the
documents now say things that aren't true any more. The biggest gap is the Sensor Pass: four rolling-shutter
code paths, no done-when, and its key input (the readout per camera mode) is still unmeasured.

| Score (0-10) | Phase 1 slice | Work since the slice |
|---|---|---|
| Clarity | 8 | 5 |
| Completeness | 8 | 4 |
| Testability | 9 | 6 |
| Consistency | 7 | 4 |

## Critical

**1. The plan stops at Phase 1. (Wiegers)**
IMPLEMENTATION_PLAN.md ends "Do not start Phase 2 without Mike". Since then: catalogue 28-31 (Sensor Pass),
rs_solve, grid_rs, Plate Lock (9 nodes), LTX Prep/Return, Comp Preview, Alembic import. None of these has a task,
a done-when or a named acceptance test in the plan. The sensor-pass design lives in memory and NOTES.
Recommendation: add a "Phase 1b: Sensor Pass" section in the T-task format, with done-when items like these:
- Readout per mode measured from the chart: within 5 % of the repeat take, sigma < 1 ms.
- The deck test on S2-S3, or a fast-pan shot, improves by a set margin with RS on.
- The LTX clean loop round trip stays under 0.001 px.

**2. Four rolling-shutter models, no shared definition. (Fowler, Nygard)**
- `sensor/rs.py` (maps).
- `sensor/readout_solve.py`, through `platelock/core/camera.py`.
- `camera_side/rs_solve.py` (bundle adjustment).
- `camera_side/grid_rs.py` (chart fit).

All say "t = f + (y/H - 0.5)·r on sensor rows", but each writes it again. If one changes, for example the anchor
or how the interpolation is done, the maps stop matching the solve that made them, and no test notices.
Recommendation:
- Put one `row_time(y, H, readout, anchor)` and one pose-interpolation function in `camera_side`.
- Have all four import them.
- Add a cross-test: a camera from rs_solve, sent through the rs.py maps, reprojects its own points within 0.05 px.

**3. Readout Solve (node 28) is known to be wrong, and it's still the demo default. (Nygard)**
- NOTES 2026-10-20/21: it reads ~7 ms on DSCF3000 because the GS solve has already bent its structure. rs_solve
  replaced it.
- Even so, `DEMO_sensor_pass_s2s3` RS Maps uses "measured if connected", which takes that ~7 ms.
- rs_solve and grid_rs are tools only, not nodes.

Recommendation:
- Mark node 28 deprecated in the catalogue.
- Plan a node for rs_solve: shot camera + points in, RS shot camera + readout out.
- Plan a node for the grid: chart MOV in, lens + readout JSON out.
- Write the readout default as a rule: measured from the chart > rs_solve > preset spec. Never the GS residual.

## Major

**4. CLAUDE.md and the Amendment 3 draft contradict the evidence. (Wiegers)**
- CLAUDE.md "Shape of the repo" lists three subpackages. The registry loads five (`sensor`, `platelock` too).
- CLAUDE.md says "4K HQ, so rolling-shutter readout 30.9 ms". NOTES 2026-10-22: 30.9 ms is the 8K full-width
  figure. DCI 4K should be about 29 ms (scaled). Measured: 26.6 ms on DSCF3000, 27.7 ms on DSCF2973.
- CLAUDE.md gives V141 on port 8288. NOTES and memory: 8188.
- CLAUDE.md says "Lens ... distortion about 0". That's a solve assumption (fixed at 0), not a measurement. The
  grid shoot will tell.
- The Amendment 3 draft still says "X-H2 -> F-Log2 F-Gamut" in its Plate Read evidence. Fixed 2026-10-20: F-Log.
  If Mike pastes it as written, the ADR records the wrong curve.

Recommendation: fix these five lines, and add a short Amendment 3 note: F-Log correction, Sensor Pass, RS solve.

**5. The split boundary is checked for one folder out of five. (Fowler)**
- `test_import.py` only checks that camera_side doesn't import layers/deliver.
- `sensor/readout_solve.py` imports `platelock.adapters`.
- `platelock/crossview_job.py` imports `camera_side.exr_io`.
- The platelock docstring says `core/camera.py` and `core/align.py` move to Atlas Camera at the split.
- sensor's future home ("atlas-vfx" in the catalogue) depends on platelock and camera_side.

Nothing stops a cycle forming before Mike says "split".
Recommendation: write the allowed import graph down once. Proposal:
- contracts: imports nothing.
- camera_side: contracts only.
- platelock/core and sensor: camera_side + contracts.
- layers: all of the above.
- deliver: everything.

Then make test_import.py check every package against that graph.

**6. Sensor presets have no readout with a source. (Wiegers, Adzic)**
The readout is typed in a node (30.9). `sensors.py` holds sizes only. Proposal: each preset gets a
`readout_ms` with its source and date, and RS Maps defaults from the preset. Example:

    fujifilm_xh2_dci4k_24p_dis_off: readout_ms 29.0, source "scaled from DPReview 8K 30.9 (4076/4320 rows)",
    status "unmeasured" -> after take B: readout_ms <value>, source "grid_rs DSCFxxxx take B", status "measured"

Do the same for focal per lens/zoom mark: video has no focal length in the metadata, and on DSCF3004 the wrong
focal moved the readout 18 to 23 ms.

**7. shot-camera@0.1 keeps the readout in `extensions`. (Fowler)**
- `extensions.atlas_vfx_rs_solve` holds the readout.
- RS Maps, the Nuke builder and Atlas Camera all need it.
- Distortion param order is also undefined in the schema (UPSTREAM_TODO).

Recommendation: draft shot-camera@0.2 with these fields:
- `sensor.readout_ms` and `sensor.readout_source`.
- `lens.distortion {model: brown, params: [k1, k2, p1, p2, k3], units: plate px}`.
- A `poses_are: global_shutter | rs_corrected` flag, so a GS-solved camera can't be given the full readout by
  mistake (the deck-test failure of 2026-10-20).

**8. The scale is still a guess. (Crispin)**
Everything metric hangs on 0.18 m risers. ADR Amendment 3 open item 1 ("tape one riser") is still open. Not a
spec defect, but it's the cheapest way to cut the risk: one tape measure next to the chart shoot.

## Minor

**9. Tools have no tests and fixed paths. (Crispin)**
- 27 scripts in `tools/`, several of which make the numbers quoted in NOTES (rs_clip_solve, rs_deck_proof,
  rs_readout_profile).
- Paths are hard-coded (`OUT = ...atlas-vfx-work\grid`). Fine for one machine.
- No smoke test, so a refactor of rs_solve can quietly break them.

Recommendation: for each tool that feeds a number into NOTES, add one tiny synthetic run (as test_grid_rs does for
grid_rs) or an `--selftest` flag.

**10. grid_rs edge cases with no test. (Crispin, Adzic)**
Given/When/Then cases to add:
- Given a take where the chart leaves frame for > 3 frames, when indexing resumes on a partial chart, then
  labels match truth (or the frames are dropped, not mislabelled).
- Given IBIS on (take E), then the readout is still within 5 %, or the rms says the model doesn't fit.
- Given a 1-frame flash (exposure change) mid-pan, then detection skips it without breaking the sequence.

**11. Catalogue is incomplete.**
- Missing: Plate Lock (9 nodes), LTX Prep/Return, Comp Preview, Holdout.
- Node 4 "Solve Plate" still says "proven". It's still scripts (rs_clip_solve is the closest).

Recommendation: add rows, plus a "superseded" status.

## What the panel liked

- Every claim in NOTES has a number and a path. Failures are written down, including wrong turns (F-Log2, the
  door scale, the 7 ms readout).
- Ground-truth synthetic tests for every solver (rs_solve 30.88 vs 30.90, grid_rs 31.11 vs 30.9, locked-off
  readout reported as not measurable).
- The deck test was used as a falsifier and changed the defaults. That's the right reflex.

## Suggested order (none started)

1. Fix the stale lines (item 4). Ten minutes, and stops wrong facts spreading.
2. Grid shoot, then preset readout + focal with source (6), then make the readout default a rule (3).
3. One row_time / interpolation module + cross-test (2).
4. Phase 1b Sensor Pass section in the plan with done-when (1), shot-camera@0.2 draft (7).
5. Import graph test (5), catalogue rows (11), tool smoke tests (9, 10).
