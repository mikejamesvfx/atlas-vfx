# atlas-vfx: working rules for people and agents

Layered, camera-locked generation over tracked real plates, delivered to Nuke, as ComfyUI nodes. Read this, then
`docs/dev/IMPLEMENTATION_PLAN.md` (what to build next and how to prove it) and `docs/NODE_CATALOGUE.md` (every node, its
sockets, tests, licences and proof). Write plain prose in docs and messages.

## Shape of the repo

```
__init__.py                 ComfyUI entry point: NODE_CLASS_MAPPINGS gathered by atlas_vfx/registry.py from five
                            subpackages (camera_side, layers, platelock, deliver, sensor)
atlas_vfx/camera_side/      nodes that will move into Atlas Camera (the catalogue's Folder column says which). No
                            imports from layers/ or deliver/, ever.
atlas_vfx/layers/           layer nodes. May import camera_side, contracts, sensor and `_atlas_scene`.
atlas_vfx/deliver/          Nuke builder, comp preview. May import everything.
atlas_vfx/sensor/           Sensor Pass: rolling shutter + lens as STMaps. May import camera_side, contracts, platelock.
atlas_vfx/platelock/        Plate Lock: lock an AI take back onto its plate. core/ is pure python. May import
                            camera_side and contracts.
atlas_vfx/contracts/        shot_camera_v0_1.schema.json + loader/validator. Pure python, no ComfyUI imports.
atlas_vfx/_atlas_camera.py  the ONLY place that imports Atlas Camera (lazy adapter).
atlas_vfx/_atlas_scene.py   the ONLY place that imports Atlas Scene (optional, lazy; Physical Sky's sky render).
web/                        ComfyUI frontend (Sun Globe panel, vendored three.js r185). Display only: panels drive
                            native widgets, every node runs the same without them.
config/                     the shipped OCIO config (ACES 2.0 studio + F-Log2 and F-Log F-Gamut input spaces)
prototypes/                 the scripts that first proved each step. Port them, don't import them.
tests/                      plain script tests (see Testing)
atlas_components.json       licence register: every model, runtime and source this repo uses
```

The import graph, enforced by `tests/test_import.py` over every `.py` in the six packages (absolute, relative and
`import_module` strings alike): contracts imports nothing of atlas_vfx; camera_side: contracts and `_atlas_camera`;
platelock: camera_side, contracts; sensor: those plus platelock; layers: camera_side, contracts, sensor and
`_atlas_scene` (optional: every node loads without Atlas Scene, only the sky render needs it); deliver: every package
and `_atlas_camera` (not `_atlas_scene`). `registry` is the root's alone. A new edge is a decision: change this
paragraph and the test's `ALLOWED` in the same commit.

## Rules

- **Widgets, and the files they name, are untrusted; outputs are confined.** ComfyUI's API has no auth, so any
  string a node receives may be hostile. Every output path goes through `plate.resolve_out_dir` (must resolve under
  the work root, `~/atlas-vfx-work` or `ATLAS_VFX_WORK`, or a folder in `ATLAS_VFX_OUT_ROOTS`; never the repo, never
  the plate's folder) and every widget that becomes a file name through `plate.safe_name`. Anything written for Nuke
  to run (Python, every `.nk`) is built with `camera_side/nuke_text.py` (`py_literal`, `nk_str`, `nk_name`), never by
  pasting strings into source. Files a node reads are data, never code: no pickle, exec or `allow_pickle`.
  Subprocesses take argument lists, never a shell. Panels in `web/` set text with `textContent`; `innerHTML` only
  for fixed markup or numbers. Tests that write set `ATLAS_VFX_OUT_ROOTS` to their temp folder.
- **Never modify or move a plate.** Plates and their conversions are read-only inputs; outputs go to the work root.
- **Nothing of one machine in the shipped tree.** `atlas_vfx/`, `workflows/`, `config/`, `web/` carry no user
  folder or username: path widgets go through `plate.clean_path` (`~` and `%VARS%` expand), committed workflows write
  the home folder as `~` and leave the OCIO config blank (= the shipped config). `tests/test_hygiene.py` checks it,
  and fails on any LAN address.
- **Production data stays out of git**: no plates, frames, renders, EXRs, model weights or Nuke scripts
  (`.gitignore` covers them). Tests build synthetic plates.
- **Licence register first.** Before any new model, library or weights file is used, add it to
  `atlas_components.json` with its licence and where you verified it. Non-commercial licences are for local tests
  only and the node must log a warning. MiniMax H3 is not used here (its licence excludes US/EU/UK/KR use).
- **OCIO always explicit.** Read plates with the range from the stream. Working space ACEScg. Every display-referred
  hop is a named OCIO transform. EXRs carry their colourspace.

## Conventions

- Node class names `AtlasVFX<Thing>`, display names "Atlas VFX <Thing>", categories `Atlas VFX/Camera`,
  `Atlas VFX/Layers`, `Atlas VFX/Layers/Plate Lock`, `Atlas VFX/Sensor`, `Atlas VFX/Deliver`.
- Custom socket types are namespaced: camera_side's start `ATLAS_` (`ATLAS_TRACKED_CAMERA`, `ATLAS_PLATE_SEQ`),
  everything else `ATLAS_VFX_`. Never reuse an Atlas Camera type name for a different payload (`ATLAS_SHOT_CAM`,
  `ATLAS_PLATE_REF` are taken). A socket change breaks saved graphs: run `tests/test_node_contracts.py --update`,
  commit `tests/node_contracts.json` and the catalogue with the code, and give old graphs a mapping in
  `tools/rename_socket_types.py`.
- Camera maths in metres, right-handed, Y up, camera looking down -Z (Nuke/USD). COLMAP is x right, y down, z forward:
  `R_c2w_nuke = R_c2w_colmap @ diag(1, -1, -1)`. Nuke Euler order is ZXY: `R = Ry @ Rx @ Rz`.
- Every node that writes files also writes a small JSON sidecar (inputs, versions, hashes) so a shot can be rebuilt.
- Save a GUI (non-API) workflow for every graph you build (`tools/ui_workflow.py`).

## Testing

Plain scripts in `tests/` run with the ComfyUI venv's python (`python tests/test_<name>.py`, or `tests/run_all.py`),
exit non-zero on failure, no pytest. Each node's catalogue block names its tests (`tests/test_node_contracts.py`
fails otherwise). Synthetic tests build a plate from a known camera and points, so solves and scales are checked
against ground truth.
