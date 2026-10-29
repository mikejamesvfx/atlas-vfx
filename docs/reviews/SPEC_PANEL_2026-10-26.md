# Spec panel review, 2026-10-26 (security focus)

What was reviewed: CLAUDE.md, docs/dev/IMPLEMENTATION_PLAN.md (Phase 1 and 1b), NODE_CATALOGUE.md, NOTES.md,
UPSTREAM_TODO.md, the Amendment 3 draft, the 2026-10-22 panel, atlas_components.json, pyproject.toml, .gitignore,
the eleven committed workflows, the git history, and the code at b48e185 (54 commits, 183 tracked files, 34 nodes in
five subpackages, 26 script tests). Mode: critique. Panel: Wiegers (requirements), Adzic (examples), Fowler
(architecture), Nygard (failure modes, trust boundaries), Crispin (testing), Hightower (operations, release).

Security was reviewed over the whole tree, not only the last commit. Two independent reads of the code were made
with the same threat model (one by the panel, one by a separate reviewer told nothing of the panel's findings);
both found the Nuke Builder injection, and the second read added the `.nk` and output-path findings. The injection
was proven with a script kept outside the repo. All 26 tests were run. Nothing in the repo was changed; this file
only proposes.

## Threat model the panel used

- ComfyUI's HTTP API has no authentication. ws listens on 127.0.0.1:8188; the farm hub and render box are on the
  LAN behind the farm-remote token (addresses in the untracked `docs/MACHINES.local.md`). Anyone who can POST a
  prompt can run any node here with any string in any widget: paths, names, prefixes, JSON file paths, newlines
  and quotes included.
- Nodes read files that may come from other people or machines (render.json sidecars, camera JSON, .chan, .abc,
  pycolmap models, picks JSON, OCIO configs, MOVs) and write wherever `out_dir` points.
- Three nodes write files Nuke executes: the Nuke Builder writes Python that Mike runs by hand
  (`exec(open(...).read())`); Sensor RS Maps, Sensor Lens Maps, Plate Lock Write and Plate Lock Cross-View write
  `.nk` scripts he opens or inserts. Those are the places where a node's output crosses into another process with
  Mike's user rights. Everything else stays inside ComfyUI, which can already run arbitrary custom-node code for
  anyone who reaches its API.
- The repo is private today and will be made public later (owner decision 2026-10-24), so anything personal in
  the tree becomes public then.

## Verdict

The code is careful where it counts for a VFX tool: subprocesses take argument lists (no shell), nothing is
unpickled, no node talks to the network, the plate is never written, and the licence register is ahead of the
code. The git history is clean of secrets and binaries, and all 26 tests pass. Three things need fixing before
the repo is public, all the same shape: nodes build text that Nuke will execute (Python and `.nk`) by pasting
strings into it, and they write that text wherever a widget says. One is proven, the other two are plain from the
code. Separately, the tree carries 119 personal paths plus two LAN addresses in 44 files, with the work root
hard-wired to Mike's home folder.

On the spec side, the Phase 1b section the last panel asked for is in the plan and S0 is done, but S1 to S6 have
not moved in four days and the acceptance margins are still marked "proposed". The catalogue still lacks 12 nodes
that load today (the nine Plate Lock nodes, LTX Prep, LTX Return, Comp Preview). Nothing in CLAUDE.md or the plan
says who may call these nodes, where outputs may go, or what the generated Nuke scripts are trusted to do; the
security notes below are the draft for that.

| Score (0-10) | Phase 1 slice | Phase 1b and the rest | Security posture |
|---|---|---|---|
| Clarity | 8 | 6 | 5 (no written trust boundary) |
| Completeness | 8 | 5 | 5 |
| Testability | 9 | 6 | 4 (no test guards the generated scripts or the tree hygiene) |
| Consistency | 7 | 5 | 5 (one node guards the repo dir, nineteen don't; three script writers, three escaping styles) |

---

## Security findings

### Vuln 1: Code injection via the generated Nuke Python script: `atlas_vfx/deliver/nuke_builder.py:226-230`

* **Fixed 2026-10-26** (`nuke_text.py_literal`, fixed header, `safe_name` on shot / layer names, `\w+` prefix;
  `tests/test_nuke_text.py`). Line numbers below are from b48e185.
* Severity: Medium (High once the repo is public and other people's shots feed it)
* Category: `code_injection`
* Confidence: 9/10 (proven by running it)
* Description: `script_text()` builds the script by string formatting. Two of the interpolations let data become
  code:
  1. Line 230: `DATA = json.loads(r'''<json.dumps(data)>''')`. `json.dumps` escapes backslashes and double quotes
     but never single quotes, so a `'''` anywhere in any string that reaches `build_data` ends the raw-string
     literal and whatever follows is parsed and run as Python.
  2. Lines 226-227: `# atlas-vfx Nuke build for {data['shot']}` and `# replaces only nodes named {data['prefix']}*`
     are f-strings inside comments. A newline in `shot` or `prefix` ends the comment, and the next line is code.
     This one needs no JSON balancing at all.
  Strings that reach these: the `shot` and `prefix` widgets, every path (plate, camera chan, OCIO config, set
  proxy, out_dir), the camera JSON's `rot_order`, and the layer name and pattern, which `find_layer` (lines
  36-41) reads from `*.render.json` in whatever folder `layer_folders` names. The same `shot` string is also the
  file name (`build_{shot}.py`, node_nuke_builder.py:63), so `..\` in it writes the script outside `out_dir`.
  `tests/test_nuke_builder.py:98` runs the generated text through `exec`, so the test process is exposed too.
* Exploit scenario: a workflow posted to the ComfyUI API, or a `render.json` dropped into a layer folder, carries
  `"card": "S\"}''') if 0 else 0 ; __import__('os').system('...') ; _ = (r'''"`. The first statement still parses
  as valid JSON (`{"shot": "S"}`), so `DATA` is set without an error, the injected statement runs, and the rest of
  the body fails later. Proof: a script that generated a build with that string in `shot` and executed only the
  head of the script (json as the only global) set an `INJECTED` variable from `os.getpid()`. Mike runs Nuke with
  his own rights on ws.
* Recommendation: never place data inside a Python literal by string concatenation. Write
  `DATA = json.loads({json.dumps(json.dumps(data))})`: the outer `json.dumps` makes a double-quoted Python string
  literal with every quote, backslash and control character escaped, which `json.loads` then decodes. Or write the
  JSON to `build_<shot>.json` next to the script and have the script `json.load` it. Drop `shot` and `prefix` from
  the comment lines (or `repr()` them). Then validate `shot`, `prefix` and layer names against
  `re.fullmatch(r"[\w.\- ]+", ...)` and add the Given/When/Then cases under item 3 to `tests/test_nuke_builder.py`.

### Vuln 2: Unescaped paths and names in generated `.nk` scripts: `atlas_vfx/sensor/maps_io.py:87`, `atlas_vfx/platelock/core/deliver.py:124-125,134-135`, `atlas_vfx/platelock/crossview_job.py:205-206`

* **Fixed 2026-10-26** (`nuke_text.nk_str` / `nk_name` in all three writers, `safe_name` on performer names;
  `tests/test_nuke_text.py`).
* Severity: Medium
* Category: `code_injection` (Nuke script format)
* Confidence: 8/10
* Description: the three `.nk` writers put `out_dir`, `ai_path` and performer names straight into
  `file "{path}"`, `name {name}` and `label "{label}"` with no escaping. A double quote followed by a newline
  closes the knob and the node block; the rest of the value becomes further Nuke script. Nuke scripts can carry
  knobs that run Python when a node is created (`onCreate`) or when the script loads, so this is code execution
  when Mike opens or File > Inserts the generated `.nk`. The values come from the `out_dir` and `ai_path` widgets
  of Sensor RS Maps, Sensor Lens Maps, Plate Lock Write and Plate Lock Cross-View, and from the Person node's
  `name`.
* Exploit scenario: `ai_path` on Plate Lock Write is set to
  `x.####.exr"\n}\nNoOp {\n onCreate "python: import os; os.system(...)"\n name a\n}\nRead {\n file "y`. The
  node writes `platelock_lock.nk` containing a NoOp whose `onCreate` runs when the script is loaded in Nuke.
* Recommendation: one `nk_str(s)` helper that escapes `\`, `"` and newlines (Nuke uses backslash escapes in quoted
  knob values) and is used for every quoted value; validate node names and performer names against `[\w]+`;
  refuse paths with control characters. Add a test that generates each `.nk` with a hostile path and checks the
  node count by parsing `^\w+ \{` lines.

### Vuln 3: Output paths and names are not confined to the work root: `atlas_vfx/deliver/node_nuke_builder.py:55,63` and 18 other writing nodes

* **Fixed 2026-10-26** (`plate.resolve_out_dir` and `plate.safe_name` in every writing node, the sensor `work_dir`
  and both Plate Lock out_dirs; `tests/test_nuke_text.py`).
* Severity: Medium (Low on its own, Medium with Vuln 1 and 2 as the payload)
* Category: `path_traversal` / arbitrary file write
* Confidence: 8/10 (the write happens; reaching the API is the precondition)
* Description: every writing node takes `out_dir` (or `out_path`) as given (`.strip().strip('"')`) and a `name`
  that goes straight into the file name; `mkdir(parents=True)` creates the folder and the writers overwrite. The
  only guard anywhere refuses paths inside the repo (`platelock/node_write.py:40`, `crossview_job.py:93`) and
  nothing else. Files written include `.py`, `.nk`, `.json`, `.obj`, `.usda`, `.exr`, `.png`, `.mp4`.
* Exploit scenario: a workflow sets the Nuke Builder's `out_dir` to `custom_nodes/<some pack>/` and `shot` to a
  module name plus the Vuln 1 payload. The attacker-authored `.py` lands in a folder ComfyUI imports on its next
  restart: code execution in ComfyUI without anyone opening Nuke. For the data-only nodes it is "overwrite any file
  the ComfyUI user can write" with image or JSON content.
* Recommendation: one `resolve_out_dir(user_value, default)` in `camera_side/plate.py`: resolve, require the
  result to be under `WORK_ROOT` (or an allow-listed second root), refuse the repo and the plate's folder (as
  `write_tracking_proxies` already does for proxies). Require `name`, `shot` and `prefix` to be a single path
  component (`re.fullmatch(r"[\w.\- ]+", ...)`). Every node calls it; the two Plate Lock guards become calls to it.

### Hygiene: personal paths, LAN addresses and a home-folder work root in the public tree

* Severity: Low today, Medium at the public flip
* Category: `information_disclosure` / portability
* Description: `miike` appears 119 times in 44 tracked files: 11 workflows (absolute `C:\Users\miike\...` plate,
  config and output paths in every widget), `atlas_vfx/camera_side/plate.py:30` (`WORK_ROOT`),
  `atlas_vfx/platelock/node_write.py:12` and `node_crossview.py:14` (`DEFAULT_OUT`), `node_plate_read.py:9` and
  `node_camera_import.py:38` (default widget paths), `atlas_vfx/_atlas_camera.py:20-21` (installed and dev
  roots), `tools/ui_workflow.py:14` (`sys.path.insert(0, r"C:\Users\miike\.claude\skills\comfyui")`), most
  `tools/*.py`, four tests, CLAUDE.md (ten, including the two farm addresses and the name of the farm token's env
  var) and three docs. The token value itself is nowhere in the tree or the history (pattern scan over
  `git log -p`); only its name was.
* Why it matters: on any other machine the default `WORK_ROOT` makes `C:\Users\miike\atlas-vfx-work\...` on that
  user's drive (`mkdir(parents=True)` creates it), the eleven workflows cannot run without editing every path
  widget, and the public repo documents the owner's username, home layout, LAN topology and the name of the farm
  token. None of this is a vulnerability on its own; it is the surface someone starts from.
* Recommendation: one setting, `ATLAS_VFX_WORK` (env var, default `<ComfyUI>/output/atlas-vfx` when ComfyUI is
  present, else `~/atlas-vfx-work`), read in `plate.py` and used by the platelock defaults. Keep the Atlas Camera
  roots but behind `ATLAS_CAMERA_ROOT` first (already the case) and drop the `Desktop\AtlasCamera_Claude` dev path
  at the split. Move the farm addresses out of CLAUDE.md into an untracked `docs/MACHINES.local.md` (or `.claude/`,
  already ignored). Re-save the workflows with placeholder paths, or move them to `examples/` with a note. Add
  `tests/test_hygiene.py`: fail if any tracked text file contains a user folder, a LAN address or the owner's username (allow-list
  CLAUDE.md until the machine block moves).
* **Fixed 2026-10-26** (step 2 of the order below): `WORK_ROOT` defaults to `~/atlas-vfx-work`; the farm block
  moved to the untracked `docs/MACHINES.local.md`; `atlas_vfx/` and `workflows/` carry no user folder (path
  widgets expand `~`, workflows say `~`, OCIO blank = shipped config); `tests/test_hygiene.py` fails on any LAN
  address or token name, on a user folder in a shipped file, and on a personal-path count that grows anywhere else.
  The remaining debt (CLAUDE.md, docs, tools, tests, prototypes) is the test's BASELINE table, which can only fall.

### Defence in depth (not vulnerabilities; the panel's recommendations)

1. **No written trust boundary.** CLAUDE.md should carry four lines: the API is unauthenticated so widgets and the
   files they name are untrusted; nodes write only under the work root; anything a node writes for Nuke (Python,
   `.nk`) is code and must be built from escaped data, never by pasting strings; files a node reads are data,
   never code (no pickle, no exec, no `allow_pickle`).
2. **Supply chain.** `atlas_components.json` records a sha256 for 2 of about 15 weight files (MoGe-3, BiRefNet).
   The LTX, Z-Image, Qwen3, FLUX ae, IC-Light, AnimateDiff and Realistic Vision entries give sizes only. Record
   the sha256 for each (one `certutil -hashfile` per file) so a swapped file is detectable. `pyproject.toml` pins
   nothing but pycolmap; pin minimums that match the V141 venv (numpy, opencv, jsonschema) so a fresh install does
   not silently take a different major.
3. **`tools/*.py` still point at port 8288** (`comfy_run.py:11`, `gpu_turn.py:18`, `ui_workflow.py:11`); CLAUDE.md
   and NOTES say 8188. Not a security issue, but a tool that targets the wrong ComfyUI by habit is how a plate
   ends up decoded on the wrong box.
4. **`ATLAS_CAMERA_ROOT` adds a directory to `sys.path`** (`_atlas_camera.py:82`). Environment variables are
   trusted, so this is fine; note it in the trust-boundary lines so nobody later reads the root from a widget.

### Checked and clean

- `subprocess` calls (plate.py ffprobe/ffmpeg, lock_check.py ffmpeg, tools): argument lists, no `shell=True`;
  the plate path is gated by `Path.is_file()` and stored resolved, so protocol prefixes and option-looking names
  cannot reach ffmpeg.
- No `pickle`, `torch.load`, `yaml.load`, `eval` or `exec` in `atlas_vfx/`. `np.load` in `platelock/adapters.py`
  uses the default `allow_pickle=False`, so an object array in a hostile `.npz` raises instead of running.
- `abc_io.py` (hand-written Ogawa reader) parses with `struct` and bounds checks and raises `AbcError` on bad
  data; `camera_io.py` and `shot_camera.py` parse numbers and JSON only, with a local schema and no remote `$ref`.
- `importlib` and `sys.path` only ever load fixed module names from fixed roots or `ATLAS_CAMERA_ROOT`; no widget
  string reaches an import name.
- No network calls in `atlas_vfx/`; `tools/` talk to 127.0.0.1 only.
- The plate is never written: Plate Read reads in place, Tracking Proxy refuses to write next to it.
- Git: no binaries, no `.claude/`, no secrets in any commit; `.gitignore` covers plates, renders, weights and
  Nuke scripts. The last commit (relight.py sun halo) is pure numpy with no security surface.
- MiniMax H3 is gone from workflows, tools and prototypes; only the licence warning string remains in
  `node_character_insert.py:23`.

---

## Spec critique

### Critical

**1. Phase 1b has not moved and its margins are still "proposed". (Wiegers)**
The plan's S1-S6 were written 2026-10-22. Since then: S0 done, nothing else. `sensors.py` still holds sizes only
(S2 not started), there is no `AtlasVFXRSSolve` or `AtlasVFXChartCalibrate` (S3, S4), no `test_sensor_presets.py`,
and the chart has not been shot (S1 blocks S5 and S6). Every acceptance number in S1, S5 and S6 is still marked
*(proposed)*. Recommendation: Mike accepts or edits the margins in one sitting and the word "proposed" comes out;
then S2 (presets with a readout source) can be built before the shoot, since it needs no footage.

**2. Three nodes write code for Nuke by pasting strings, each its own way. (Fowler, Nygard)**
Vulns 1-3 above. The plan's T8 line "contains no NukeX-only node classes from a deny list" is the only property
the builder test checks about the generated code, and nothing in the plan or catalogue says the `.nk` writers
exist. Recommendation: one `deliver/nuke_text.py` (or in camera_side if the sensor nodes move) with
`py_literal(data)` and `nk_str(s)`, used by all three writers, and two stated properties in the plan: "every
string in a generated script round-trips as data" and "a generated script has exactly the nodes the writer
intended".

**3. No examples for the hostile inputs. (Adzic)**
Given/When/Then cases the tests should carry:
- Given a layer folder whose render.json names the card `a''' ; import os ; _ = r'''`, when the Nuke Builder
  writes the script, then `json.loads(DATA)` in the script returns that exact name as data and the script has the
  same number of top-level statements as a clean build (parse with `ast`).
- Given `shot` containing a newline, when the Nuke Builder runs, then it refuses.
- Given `ai_path` containing `"` and a newline, when Plate Lock Write writes `platelock_lock.nk`, then the script
  has exactly the Read and TimeWarp nodes the writer intended, and the path round-trips through Nuke's quoting.
- Given `out_dir` outside the work root or inside the repo, when any writing node runs, then it refuses with the
  CLAUDE.md rule in the message (today only Plate Lock Write and Cross-View refuse, and only the repo).
- Given a tracked text file containing `C:\Users\`, when `test_hygiene.py` runs, then it fails and names the file.

### Major

**4. The catalogue is 12 rows short and one row is wrong. (Wiegers)**
34 nodes load (`NODE_MODULES`): camera_side 9, layers 10, platelock 9, deliver 2, sensor 4. The catalogue has
34 rows but a different set: Plate Lock's nine nodes, LTX Prep, LTX Return and Comp Preview have no row, while
rows 3, 4, 11-13, 17, 19-22, 26 and 27 are not nodes. Node 4 "Solve Plate" still says "proven"; it is still
scripts (`tools/rs_clip_solve.py`). The 2026-10-22 panel asked for this (item 11); rows 32-34 were added, the rest
was not. Recommendation: add the 12 rows under their real categories (Plate Lock has its own category string),
mark 4 "scripts only", 28 "deprecated".

**5. The import graph is still enforced for one package out of five. (Fowler)**
Same as the last panel's item 5, still open, and the tree has moved against it:
- `layers/ltx_integration.py:201` imports `atlas_vfx.sensor.jobs`; CLAUDE.md says layers "may import camera_side
  and contracts".
- `sensor/readout_solve.py:21` imports `platelock.adapters`; the catalogue says sensor's future home is atlas-vfx
  but platelock's `core/camera.py` is marked for Atlas Camera.
- `platelock/core/camera.py:24` imports `camera_side.rs_time` (allowed, and the right direction).
`test_import.py` checks only camera_side. Recommendation: write the allowed graph into CLAUDE.md (contracts: none;
camera_side: contracts; platelock/core and sensor: camera_side + contracts; layers: those plus sensor if that is
wanted; deliver: everything) and make `boundary_violations` run over every package against it.

**6. The work root is a constant in one user's home folder. (Hightower)**
`plate.WORK_ROOT = C:\Users\miike\atlas-vfx-work` is imported by 15 node modules as the default output. README says
"install and run steps to come"; there is no way to run this pack on another machine without editing source.
Recommendation: the `ATLAS_VFX_WORK` setting under Hygiene above, plus a README install section (junction or
clone into custom_nodes, the two optional extras, where outputs land, how to point at your own plate).

**7. Eleven shipped workflows cannot run anywhere but ws. (Adzic, Hightower)**
Every path widget is absolute. They are the only executable examples of the nodes. Either make them examples with
a tiny synthetic plate the user can generate (`tests/test_plate.py` already builds one with ffmpeg), or say in
the README that they document the DSCF3000 runs and are not meant to run elsewhere.

### Minor

**8. Tools target port 8288.** See defence item 3.

**9. shot-camera@0.2 not drafted** (last panel's item 7). The readout still lives in `extensions`, the Brown
param order is still only in `sensor/lens.py` and UPSTREAM_TODO, and there is no `poses_are` flag. S3 depends on
this ("until shot-camera@0.2 the readout stays in extensions"), so it is on the Phase 1b path.

**10. No test runner and no CI.** 26 scripts, run one at a time by hand; no `.github`, no `run_tests.py`. The
NOTES claim ("full suite passes", 2026-10-22) could only be checked with 26 commands (done below: it holds). A
ten-line `tests/run_all.py` that runs each script and prints a table would do; CI can wait until the repo is
public (the tests need ffmpeg and, for some, torch, so a plain GitHub runner would only cover the pure-python
half).

**11. ADR Amendment 3 draft is still "Proposed" and dated 2026-10-19.** It says "ten script tests passing";
there are 26. Its "Later evidence" section stops at 2026-10-22; the MIT/private/H3-dropped decisions of
2026-10-24 and the relight nodes of 2026-10-26 are only in NOTES.

## Follow-up on the 2026-10-22 panel

| Item | Status 2026-10-26 |
|---|---|
| 1 Plan stops at Phase 1 | Done: Phase 1b S0-S6 in the plan. Margins still proposed (item 1 above) |
| 2 Four RS models | Done: `camera_side/rs_time.py`, `tests/test_rs_time.py` |
| 3 Readout Solve still the demo default | Rule written into the plan; node 28 not yet marked deprecated; RS Maps default unchanged (`measured if connected`) |
| 4 CLAUDE.md / Amendment 3 stale lines | Done in CLAUDE.md and the draft; `tools/*.py` still say 8288 |
| 5 Import graph for one of five packages | Open, and layers now imports sensor |
| 6 Presets with readout + source | Open (`sensors.py` unchanged) |
| 7 shot-camera@0.2 | Open |
| 8 Scale is a guess | Open (riser not taped) |
| 9 Tools have no tests, fixed paths | Open |
| 10 grid_rs edge cases | Open |
| 11 Catalogue incomplete | Partial: 32-34 added; 12 loaded nodes still missing; node 4 still "proven" |

## What the panel liked

- The licence register is ahead of the code and the H3 decision was carried through the tree the same day.
- Every subprocess is an argument list, every file the builder reads is checked for existence before a node is
  made, and the plate is treated as read-only in code, not just in a rule.
- The history scrub was done properly: no codename, no `.claude/`, no secrets, no binaries, 54 commits intact.
- Plate Lock Write's repo guard is the right reflex; it only needs to anchor on the work root and spread.
- Synthetic ground-truth tests for every solver, and the deck test used as a falsifier, both still hold, and
  the whole suite passes today.

## Suggested order

1. **Done 2026-10-26.** `py_literal` / `nk_str` helpers, the three writers on them, `resolve_out_dir` in every
   writing node, the item 3 tests except the hygiene one (it waits on step 2), and the trust-boundary lines in
   CLAUDE.md (step 3). Vulns 1-3, item 2, defence 1.
2. **Done 2026-10-26.** `ATLAS_VFX_WORK`, `~` in widgets and workflows, hygiene test with a falling baseline, farm
   block out of CLAUDE.md, README "Where things go". Hygiene, items 6, 7 (the workflows now run from any home
   folder that has the same layout; a synthetic-plate example is still open).
3. Done with step 1 (CLAUDE.md rule "Widgets, and the files they name, are untrusted").
4. **Done 2026-10-26.** Catalogue rows 35-46, node 4 "scripts only", node 28 "deprecated"; `test_import.py` scans
   all six packages against the graph now written in CLAUDE.md (layers -> sensor and sensor -> platelock kept,
   with their reasons). Items 4, 5.
5. Mike: accept or edit the Phase 1b margins; tape the riser; shoot the chart. Then S2, shot-camera@0.2, S3.
6. **Done 2026-10-26** (all but CI, which waits for the public flip). `tools/register_hashes.py` wrote sha256 and
   size for the 14 weights files into the register; `pyproject.toml` has lower bounds and the verified set,
   `requirements.txt` for Manager, README Install; `tests/run_all.py`; Amendment 3 draft refreshed. Defence 2,
   items 6 (README), 10, 11.

## Test run

2026-10-26, V141 python, every `tests/test_*.py` as a script: 26 of 26 pass, about 7 minutes wall time.
Slowest: test_grid_rs 227 s, test_platelock_lock 49 s, test_rs_solve 28 s; sixteen finish in 3 s or less.
After the Vuln 1-3 fixes, same day: 27 of 27 pass (new `tests/test_nuke_text.py`; four tests gained an
`ATLAS_VFX_OUT_ROOTS` line so nodes may write under the temp folder).
After steps 2, 4 and 6, same day, through `tests/run_all.py`: 28 of 28 pass in 356 s (new `tests/test_hygiene.py`;
`test_import.py` now covers the import graph of all six packages). Slowest: test_grid_rs 193 s,
test_platelock_lock 44 s, test_rs_solve 28 s.
