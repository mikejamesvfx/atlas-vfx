# Spec panel review, 2026-10-29 (NODE_CATALOGUE.md)

What was reviewed: `docs/NODE_CATALOGUE.md` at 03ef465 (main, after PR #3), checked against the code it describes:
`atlas_vfx/registry.py` and every subpackage's node modules, the socket types each node declares, `tests/`, the
14 committed workflows, `prototypes/`, `atlas_components.json`, CLAUDE.md, the plan, NOTES, and the socket types
Atlas Camera declares in its dev tree (read only). Mode: critique, all focuses.
Panel: Wiegers (requirements), Adzic (examples), Fowler and Newman (architecture, the split), Nygard (failure
modes), Crispin (testing), Hightower (release). Nothing in the code was changed; this file only proposes.

## Verdict

The catalogue's headline is right: 37 nodes load, the rows it names (1, 2, 5-10, 14-16, 18, 23-25, 28-49) are
exactly the 37 `AtlasVFX*` keys in the code, and the categories match (Camera 11, Layers 11, Plate Lock 9, Sensor
4, Deliver 2). It works as a status board. It does not work as a spec: it has no inputs, outputs or socket types,
no test per node, and no licence link, so the things most likely to break at the split or at publish are written
nowhere. Two rows still lean on MiniMax H3, which was dropped five days ago. One architecture issue blocks a clean
split and should be decided before more camera-side nodes land.

| Score (0-10) | Now |
|---|---|
| Clarity | 6 (status cells carry the spec; row 48 is ~900 characters) |
| Completeness | 4 (no contracts, no tests, no licences) |
| Testability | 4 (nothing a script can check except the count) |
| Consistency | 6 (count and categories right; CLAUDE.md, rows 13, 16, 17, 48 drift) |

---

## Status (same day, branch fix/catalogue-panel-2026-10-29)

All nine items fixed; see NOTES 2026-10-29. Item 1 went with neither (a) nor (b): Atlas Camera's two types carry
different payloads (`ATLAS_SHOT_CAM` is a shot format, `ATLAS_PLATE_REF` one image), so ours became
`ATLAS_TRACKED_CAMERA` / `ATLAS_PLATE_SEQ` and the rest `ATLAS_VFX_*`; the merge question is in UPSTREAM_TODO.
Item 8 also found two runtimes in use without a register entry (`openexr`, `imageio`), now registered.

## Findings, most severe first

### 1. CRITICAL (architecture, the split): camera-side socket types don't match Atlas Camera's

Newman: camera_side is meant to move into Atlas Camera "as a folder move". Its nodes speak `SHOT_CAMERA` (3
outputs) and `PLATE_REF`; Atlas Camera already has `ATLAS_SHOT_CAM` and `ATLAS_PLATE_REF`. After the move, Atlas
Camera ships two types for one idea, its own nodes can't link to the moved ones, and renaming either side breaks
every saved graph that uses it (all 14 workflows here link through these sockets: `DEMO_all_in_one_s2s3.json` alone
has 24 atlas-vfx nodes).

Fowler: the unprefixed names are also a namespace risk on their own. `CARD`, `SHOT_CAMERA`, `PLATE_REF`,
`SENSOR_LENS`, `LTX_PREP` are global strings in ComfyUI; any other pack that picks the same name collides silently.
The only prefixed atlas-vfx type is `ATLAS_LIGHT_RIG`.

- Recommendation: decide now, while it is cheap, one of (a) camera_side adopts Atlas Camera's names
  (`ATLAS_SHOT_CAM`, `ATLAS_PLATE_REF`) if the payloads agree, or adapters convert at the boundary, or (b) Atlas
  Camera adopts these at the split. Either way prefix the rest (`ATLAS_VFX_CARD`, ...). Write the choice into the
  catalogue's new type table (item 3) and `docs/dev/UPSTREAM_TODO.md`, which today says nothing about types.
- Tell Mike: this touches committed workflows; `tools/ui_workflow.py` can rewrite link types in one pass.

### 2. CRITICAL (compliance): rows 16 and 17 still cite MiniMax H3

Wiegers: row 16 (Character Insert) gives its prototype as `howling_yowies/h3_gen.py`, deleted 2026-10-24, and its
status "ran on S2-S3" was the H3 run. Row 17 (Camera-Locked Generate) lists "H3 Fun ControlNet" as the existing
pieces of a planned node. CLAUDE.md: no workflow, tool or demo may use H3 or its outputs. A planned node whose
only named basis is a banned model is a requirement nobody can meet. Row 27's `howling_yowies/farm_calls.py` was
deleted the same day.

- Recommendation: row 16 status "built (T7; the H3 proof run is withdrawn, needs a licensed source to re-prove)",
  prototype blank. Row 17 pieces: name the licensed candidates (LTX 2.5 IC-LoRA is in the register) or "none
  chosen". Row 27 prototype blank.

### 3. MAJOR (requirements): the catalogue has no contracts

Adzic: the wiring that matters is buried in prose, if anywhere. Examples from the rows: Chart Calibrate's
`readout_ms` into RS Maps' `readout_chart_ms`; Physical Sky's `ATLAS_LIGHT_RIG` into Relight's `sun_rig`; RS
Solve's `extensions.atlas_vfx_rs_solve` read by RS Maps' preset rule. A reader can't tell what plugs into what
without opening the code, and 49 of the outputs are `STRING` (mostly folder or file paths), so the type system
doesn't help either.

- Recommendation: keep the table as the status board, trimmed to one line per cell, and add two sections:
  - **Types**: each custom socket type, its payload (a dict schema or the contract file it follows), who makes it,
    who reads it, and its future home.
  - **Contracts**: one short block per built node. Example for row 49:

    ```
    49 Chart Calibrate (AtlasVFXChartCalibrate, camera_side, Atlas VFX/Camera)
    in:    video_path STRING (read only), mode lens|readout, sensor_preset, lens_json STRING (readout mode),
           first/last/every INT, still_px, focal_guess_mm FLOAT, name, out_dir STRING
    out:   lens_path STRING, focal_mm, k1, k2, pp_offset_x_px, pp_offset_y_px FLOAT, ... readout_ms FLOAT
    files: <name>_lens.json | <name>_readout.json + sidecars, under atlas-vfx-work/grid
    feeds: readout_ms -> Sensor RS Maps.readout_chart_ms (0 = no chart value)
    test:  tests/test_grid_rs.py (synthetic takes, check_node)
    proof: synthetic only; real chart takes not shot
    licences: opencv-python, numpy, scipy
    ```

  Generate the in/out lines from `INPUT_TYPES` / `RETURN_NAMES` with a small tool so they can't drift; write the
  rest by hand.

### 4. MAJOR (testing): no node-level contract is pinned

Crispin: `test_import.py` proves 37 nodes load and the import graph holds. Nothing proves a socket kept its name
or type, which is what a saved graph depends on. 15 node wrappers are never named in any test (Plate Read, Camera
Import, Solve Compare, Measure & Scale, Lock Check, Plate Depth, Set Proxy, Card Project, Foreground Element,
Character Insert, LTX Prep, LTX Return, Plate Lock Cross-View, Nuke Builder, Comp Preview). Their core modules are
tested; the node layer (defaults, widget ranges, return order) is not. CLAUDE.md says each task names its test; the
catalogue has no column for it.

- Recommendation: `tests/test_node_contracts.py` dumps every node's `INPUT_TYPES()`, `RETURN_TYPES`,
  `RETURN_NAMES` and `CATEGORY` to a committed `tests/node_contracts.json` and fails on any difference, printing the
  diff (update the file on purpose, in the same commit as the change). Second check in the same script: every link
  in `workflows/*.json` names a socket that exists with the same type. That turns item 1's rename into a visible,
  testable change.
- Add a `test` line per node (item 3) and let the contracts test fail when a built row names no test.

### 5. MAJOR (consistency): CLAUDE.md's camera_side list is out of date

CLAUDE.md gives camera_side catalogue rows "1-9, 20, 23, 25". Rows 47 (RS Solve) and 49 (Chart Calibrate) are
camera_side in the code and the catalogue. CLAUDE.md is the file that defines what moves at the split.

- Recommendation: "1-9, 20, 23, 25, 47, 49". Better: CLAUDE.md points at the catalogue's Folder column as the
  source of truth and stops listing numbers, and the contracts test (item 4) checks each node's module folder
  against its row.

### 6. MINOR (consistency): stale rows

- Row 13 Layer Stack: "new", task "T6-T8 (minimal)". T6-T8 are done and T8 was accepted without it; the plan never
  mentions it. Either "Phase 2" or "dropped: the Nuke Builder stacks the layers".
- Row 48 Physical Sky: "branch feat/physical-sky"; it is on main since PR #3. "Not yet run inside V141" is still
  true (NOTES line 52); keep that part.
- Row 21 Identity Check: "by eye in T7" was a check of the H3 character; with item 2 it has no subject.
- Row 18's prototype `ltx25_hdr/tools/relight_proto.py` and row 48's `hy_relight/prague make_prague_sky.py` are
  outside this repo; the column header says `prototypes/`. Mark external paths as such.

### 7. MINOR (clarity): the status vocabulary isn't used as defined

Wiegers: the header defines built / proven / prototype / scripts only / exists / deprecated / new. No row says
"proven"; "built" covers everything from "accepted by Mike in Nuke" (24) to "synthetic only" (47, 49) to "on
GUESSED stair picks" (7). The "Phase 1 task" column mixes task ids, phases and "with 18".

- Recommendation: Status is one word from the list, plus a separate **Proof** column with a fixed scale: synthetic
  / owner plate (script) / owner plate (V141) / accepted in Nuke. Rename "Phase 1 task" to "Task" and use plan ids
  only.

### 8. MINOR (compliance, release): no licence link per node

Hightower: before publish, someone has to answer "what does this node pull in". The register has the answers
(moge-3-vitl, z-image-turbo, birefnet, atlas-scene, prague-sky-model, three-js, ...) but no row points at them.
Nodes that take model output through a socket (Plate Lock Person's SAM masks, body-solve npz) depend on models the
register can't see; the catalogue should say so, so the workflow that makes the masks carries the licence check.

- Recommendation: a `licences:` line per contract block (item 3) with register ids; "upstream, via socket" where
  the model runs outside this pack. The hygiene test can check each id exists in `atlas_components.json`.

### 9. MINOR (failure modes): optional dependencies are invisible

Nygard: `registry.py` supports `REQUIRES` and disables a node whose imports fail; no node uses `REQUIRES` today and
the catalogue doesn't say which nodes need what. Physical Sky loads without Atlas Scene and raises
`AtlasSceneMissing` at run time; the OIIO-only and MoGe-dependent nodes aren't marked. On a fresh install the user
learns at queue time.

- Recommendation: a `needs:` line per contract block (OpenImageIO, PyOpenColorIO, pycolmap, Atlas Scene, Atlas
  Camera, MoGe core nodes), saying "disabled at load" or "fails at run" for each.

---

## Proposed order

1. Item 2 (H3 rows): text only, today.
2. Items 5 and 6: text only.
3. Item 1: Mike decides the type names; nothing else in this list should rename sockets before that.
4. Item 4: the contracts test, which then guards the rename.
5. Items 3, 7, 8, 9: restructure the catalogue, generated in/out lines first.

## Evidence

- Node keys: 37 `"AtlasVFX*":` mappings under `atlas_vfx/`; categories by `CATEGORY =` per module.
- Socket types: `RETURN_TYPES` across `atlas_vfx/` (STRING 49, IMAGE 28, MASK 7, FLOAT 6, SHOT_CAMERA 3, CARD 2,
  ATLAS_LIGHT_RIG 2, SENSOR_LENS, PLATE_REF, PLATELOCK_TRACK, PLATELOCK_RESULT, PLATELOCK_PEOPLE, LTX_PREP, INT 1
  each). Atlas Camera: ATLAS_SHOT_CAM, ATLAS_PLATE_REF, ATLAS_CAMERA, ATLAS_SOLVE among others.
- H3: `atlas_components.json` id `minimax-h3-ref2va` (DROPPED 2026-10-24); NOTES 2026-10-24 lists `h3_gen` and
  `farm_calls` as removed.
- Prototypes present: everything in rows 1-8's column exists under `prototypes/vfx_probe/` and
  `prototypes/howling_yowies/nuke_export_for_claude.py`.
