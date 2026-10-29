# atlas-vfx

VFX nodes for real plates, for ComfyUI (category `Atlas VFX/...`) and plain Python. MIT licence (see LICENSE). Early: the nodes work on the author's test plates; expect changes.

- `atlas_vfx/camera_side/` plate I/O, tracking, solve and import, scale, lock check, depth, camera export. Moves to Atlas Camera later.
- `atlas_vfx/layers/` set proxy, cards, foreground elements, character insert, relight, physical sky, LTX prep/return.
- `atlas_vfx/sensor/` Sensor Pass: rolling shutter and lens as STMaps.
- `atlas_vfx/platelock/` Plate Lock: lock an AI take back onto its plate.
- `atlas_vfx/deliver/` Nuke builder, comp preview.
- `atlas_vfx/contracts/` shot-camera@0.1 and friends.
- `prototypes/` the scripts that proved it on DSCF3000. Reference only.
- `docs/` the node list (NODE_CATALOGUE.md) and how to make picks (PICKING.md). `docs/dev/` is the working record: the
  plan, notes, upstream asks. `docs/reviews/` holds the spec panel reviews.

Start with AGENTS.md (rules, layout, import graph) and SECURITY.md (threat model, reporting).

## Install

- Clone (or junction) this repo into `<ComfyUI>/custom_nodes/atlas-vfx` and restart ComfyUI. The nodes appear under
  `Atlas VFX/Camera`, `Atlas VFX/Layers` (with `Plate Lock`), `Atlas VFX/Sensor` and `Atlas VFX/Deliver`.
- `requirements.txt` is what ComfyUI Manager installs (numpy, scipy, opencv-python, opencolorio, jsonschema;
  bounds in `pyproject.toml`). Two optional extras, by hand into the ComfyUI venv: `pycolmap==4.2.1` for COLMAP
  import, `OpenImageIO>=3.0` for EXR read/write. Nodes load without them and say what to install when queued;
  `docs/NODE_CATALOGUE.md` lists what each node needs.
- `ffmpeg` and `ffprobe` on PATH (plates are decoded as native planes; the colour maths is in numpy).
- Tests: `python tests/run_all.py` with the ComfyUI venv's python runs every `tests/test_*.py` as a script
  (about 7 minutes; `python tests/run_all.py plate camera` filters by name). No pytest.
- Point the nodes at your own plate with the `path` widget of Plate Read (`~` expands) and your camera with
  Camera Import (Nuke JSON or .chan, Alembic, a COLMAP model, or a shot-camera JSON).

## Where things go

- Nodes write under `~/atlas-vfx-work` (`ATLAS_VFX_WORK` moves it; `ATLAS_VFX_OUT_ROOTS`, `os.pathsep`-separated,
  allows more folders). An `out_dir` anywhere else, inside the repo, or next to the plate is refused: ComfyUI's API
  has no auth, so widgets are not trusted.
- Path widgets expand `~` and `%VARS%`. A blank `ocio_config` means the shipped `config/atlas_aces2.0_studio_flog2.ocio`.
- `workflows/*.json` document the DSCF3000 runs with the home folder written as `~`; they run unchanged on a
  machine with the same layout under its own home folder, else re-point the plate and camera widgets.
- `tests/test_hygiene.py` keeps the shipped tree free of user folders and LAN addresses; machine notes stay in
  untracked `*.local.md` files.
