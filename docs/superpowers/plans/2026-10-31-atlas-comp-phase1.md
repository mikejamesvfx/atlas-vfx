# Atlas Comp Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A compositor's screen at `http://127.0.0.1:8188/atlas` that opens a shot, stacks layers over the plate,
shows them through the OCIO view with A/B and f-stops, and runs the Matte, Relight and Sky recipes as Preview or
Render with live previews.

**Architecture:**
- A new server package `atlas_vfx/comp/`:
  - shot store on disk;
  - recipe filling;
  - frame compositing;
  - aiohttp routes on ComfyUI's PromptServer.
- A static browser app in `app/`, plain ES modules.
- Recipes are JSON (an API graph, plus fields mapped to graph inputs), generated from the existing graph builders.
- The browser queues filled graphs on ComfyUI's own `/prompt` with its `client_id`, so live previews and progress
  arrive on ComfyUI's websocket.

**Tech Stack:**
- Python 3.12 in V141: aiohttp (ComfyUI's), numpy, OpenCV, OCIO, OIIO.
- Browser: plain ES modules, no build.
- Tests: plain scripts in `tests/` (no pytest), run with the V141 python.

**Spec:** `docs/superpowers/specs/2026-10-31-atlas-comp-design.md`

## Global Constraints

- **No new dependencies.**
- **Browser files go in `app/`, never `web/`.** ComfyUI loads every `.js` under `web/` into its graph editor.
- **Imports:** `atlas_vfx/comp/` may import every atlas_vfx package and `_atlas_camera`; nothing imports it except
  the root `__init__.py`. `tests/test_import.py` `ALLOWED`, `AGENTS.md` and `CLAUDE.md` change in the same commit.
- **Every HTTP input is untrusted:**
  - Names go through `plate.safe_name`.
  - Shot folders go through `plate.resolve_out_dir`, under `WORK_ROOT / "comp"`.
  - Sequence sources must resolve under `WORK_ROOT`. The plate is any readable file named at shot creation.
  - Recipe params are validated against the recipe's fields.
  - No pickle, exec or eval.
- **Disk is the truth:** `WORK_ROOT/comp/<shot>/shot.json`, and renders under
  `WORK_ROOT/comp/<shot>/layers/<layer>/<vNNN>/`.
- **Colour:** composite in linear ACEScg; display through a named OCIO view (default
  `sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)`).
- **Text:** casual plain prose in docs and UI text, no em-dashes.
- **Commits:** end with the two attribution lines (Co-Authored-By / Claude-Session).

## Review Focus

- **A shot or layer name with `..`, slashes or spaces:** saved under a safe name, never outside `WORK_ROOT/comp`.
  Test in Task 1.
- **A frame outside a layer's range (or a missing version):** that layer is skipped and the rest composite; no 500.
  Test in Task 3.
- **Two renders of one layer in a row:** each gets its own version folder (v001, v002), and neither overwrites the
  other. Test in Task 2.
- **A recipe param of the wrong type or outside its choices** (`"direction": "../x"`, `"seed": "abc"`): a 400 with
  the field named, and nothing queued. Test in Task 2 (fill) and Task 4 (route).
- **The app loaded with no shots yet:** an empty state that says how to make one, not a blank page. Checked in
  Task 5's live check.

---

### Task 1: Shot store and the import-graph change

**Files:**
- Create: `atlas_vfx/comp/__init__.py`, `atlas_vfx/comp/shot.py`
- Modify: `tests/test_import.py` (`PACKAGES` + `ALLOWED`), `AGENTS.md`, `CLAUDE.md` (import graph paragraph)
- Test: `tests/test_comp.py`

**Interfaces:**
- Produces, in `atlas_vfx.comp.shot`:
  - `COMP = "comp"`, and `root() -> Path`, which is `pl.WORK_ROOT / COMP` read at call time.
  - `create_shot(name: str, plate: str, first: int, last: int, frame_offset: int = 1001, ocio_config: str = "") -> dict`.
    It writes `shot.json` as `{"name", "plate", "first", "last", "frame_offset", "ocio_config", "layers": []}` and
    raises `ValueError` if the shot exists.
  - `list_shots() -> list[str]`
  - `load_shot(name: str) -> dict`: each layer gets `"versions": [...]`.
  - `save_shot(shot: dict) -> None`
  - `add_layer(name: str, layer: str, kind: str, source: str = "", recipe: str = "", params: dict | None = None) -> dict`.
    - `kind` is `"sequence"` (source is a `####` pattern under WORK_ROOT) or `"recipe"`.
    - Defaults: `blend="over"`, `opacity=1.0`, `visible=True`, `version=None`.
  - `update_layer(name: str, layer: str, **changes) -> dict`.
    - Allowed keys: `visible`, `opacity` (0-1), `blend` (`over|plus|multiply|screen`), `version`, `params`, `order`.
    - Anything else raises `ValueError`.
  - `versions(name: str, layer: str) -> list[str]`, and `next_version(name: str, layer: str) -> str` (`"v001"`...).
    `next_version` creates the folder, so two calls in a row give v001 then v002.
  - `layer_pattern(shot: dict, layer: dict) -> str | None`: the `####` pattern of the chosen version (or the
    sequence source); `None` when nothing is rendered.

- [ ] **Step 1: Write the failing tests in `tests/test_comp.py`** (plain script style of `tests/test_artist_nodes.py`;
  set `ATLAS_VFX_WORK` / `pl.WORK_ROOT` and `ATLAS_VFX_OUT_ROOTS` to a temp folder):
  - `test_shot_roundtrip`: `create_shot("S2S3", plate, 1130, 1226)`, then `list_shots() == ["S2S3"]`,
    `load_shot("S2S3")["first"] == 1130`, and a second create raises `ValueError`.
  - `test_names_are_safe`: `create_shot("../evil", ...)` either raises `ValueError` or lands inside `root()`.
    `add_layer(..., kind="sequence", source=str(tmp_outside/"x.####.exr"))` raises `ValueError` ("outside").
  - `test_layers`: add a sequence layer, then `update_layer(opacity=0.5, blend="screen")`, which is saved;
    `update_layer(blend="dodge")` and `update_layer(colour="x")` raise.
  - `test_versions`: `next_version` twice gives `"v001"`, `"v002"`. `layer_pattern` returns `None` before any EXR,
    and the v002 pattern after writing `layer.1130.exr` into `v002` and `update_layer(version="v002")`.
- [ ] **Step 2: Run `<V141 python> tests/test_comp.py`.** Expected: an ImportError on `atlas_vfx.comp`.
- [ ] **Step 3: Implement `atlas_vfx/comp/shot.py`.** JSON read and write only; names through `pl.safe_name`; the
  shot folder through `pl.resolve_out_dir("", root() / name)`. Sequence sources are checked by resolving them and
  testing `pl.WORK_ROOT.resolve()` is a parent. Version folders are matched by `^v\d{3}$`.
- [ ] **Step 4: Extend the import graph.**
  - Add `"comp"` to `PACKAGES`, and `"comp": {"contracts", "camera_side", "platelock", "sensor", "layers",
    "deliver", "_atlas_camera"}` to `ALLOWED` in `tests/test_import.py`.
  - Add a matching sentence to `AGENTS.md`'s import-graph paragraph and to `CLAUDE.md`'s.
- [ ] **Step 5: Run `tests/test_comp.py` and `tests/test_import.py`.** Expected: all ok.
- [ ] **Step 6: Commit** ("Atlas Comp: shot store on disk (comp package, import graph)").

### Task 2: Recipes (JSON graph + fields, validated fill)

**Files:**
- Create: `atlas_vfx/comp/recipes.py`, `tools/comp_recipes.py`, `recipes/matte.json`, `recipes/relight.json`,
  `recipes/sky.json`
- Test: `tests/test_comp.py` (add)

**Interfaces:**
- **Recipe JSON:**
  - Top level: `{"name", "label", "description", "graph": {API graph}, "fields": [...], "context": {...},
    "outputs": [[node, input], ...], "preview": {"set": [[node, input, value], ...], "drop": [node, ...]}}`.
  - Each field: `{"key", "label", "type": "choice|text|int|float|bool|layer", "default", "choices"?, "min"?,
    "max"?, "targets": [[node, input], ...]}`.
  - `context`: keys from `{"plate", "first_frame", "length", "plate_first", "plate_last", "element_pattern",
    "element_folder"}` mapped to `[[node, input], ...]`.
  - `outputs`: the `out_dir` / `name` inputs the version folder is written into.
- **Consumes:** `shot.load_shot`, `shot.next_version`, `shot.layer_pattern` (Task 1).
- **Produces, in `atlas_vfx.comp.recipes`:**
  - `list_recipes() -> list[dict]`: each recipe without its graph.
  - `load_recipe(name: str) -> dict`: raises `ValueError` for an unknown name; a name never becomes a path except
    through the known list.
  - `validate(recipe: dict, params: dict) -> dict`: the defaults merged in; unknown key, wrong type, value not in
    choices, or out of bounds raises `ValueError(f"{key}: ...")`.
  - `fill(recipe: dict, params: dict, context: dict, out_dir: str, mode: str = "render") -> dict`:
    - returns a deep-copied API graph;
    - preview applies `preview.set`, then removes `preview.drop` and every node whose inputs link to a removed node,
      repeated until stable.
  - `context_for(shot: dict, layer: dict) -> dict`, from the shot fields:
    - `length` = Shot Range's 8k+1-shorter count;
    - `plate_first` = `first - frame_offset`;
    - `element_pattern` = the pattern of the layer named by the `element` param, if any.
- **Recipes** (`tools/comp_recipes.py` writes them from `tools/alpha_matte_workflow.graph`,
  `tools/ltx_relight_workflow.graph_4k` and `tools/physical_sky_workflow.graph`):
  - **matte:**
    - Fields: `element` (layer) → Sequence Read Chunk path; `still_frames`; `seed`.
    - The graph's LoadImage is replaced by `AtlasVFXSequenceReadChunk` reading the element pattern as sRGB through
      OCIO, `input_colorspace` `ACEScg`.
    - Outputs: EXR Write `out_dir` and `name`.
    - Preview: length 9, and `drop` the full VAE decode branch (nodes 53, 60, 65, 66, 67), keeping the Tiny VAE
      branch.
  - **relight:**
    - Fields: `element` (layer) → Layer Picker is replaced by the context `element_folder` / `element_pattern`;
      `direction`, `height`, `look`, `seed`, `mode`.
    - Outputs: LTX Return `out_dir` and Relight Transfer Files `out_dir`.
    - Preview: length 17, and `drop` Relight Transfer Files, Notes, Viewer and the full decode. A Tiny VAE `VAEDecode`
      on the cropped latent feeds a PreviewImage.
  - **sky:** fields `latitude`, `longitude`, `capture_utc`, `heading_deg`, `fov_x`, `width`. Outputs: `out_dir`.
    Preview: `width` 512.

- [ ] **Step 1: Write the failing tests:**
  - `test_recipes_listed`: names are `{"matte", "relight", "sky"}`, and every target `[node, input]` exists in its
    graph.
  - `test_validate`: relight `{"direction": "../x"}` raises with "direction"; `{"seed": "abc"}` raises with "seed";
    `{"zzz": 1}` raises; `{}` returns the defaults.
  - `test_fill_render_vs_preview`:
    - In render, the filled graph's EXR/Return `out_dir` equals the given version folder.
    - In preview, the graph has no `AtlasVFXRelightTransferFiles` node, and every remaining `[node, slot]` link
      points at a node that still exists.
  - `test_two_renders_two_versions`: `next_version` twice, then two fills; the `out_dir`s differ (v001, v002).
- [ ] **Step 2: Run `tests/test_comp.py`.** Expected: fails on the missing module.
- [ ] **Step 3: Implement `recipes.py`.** Add `tools/comp_recipes.py` (imports the graph builders, rewires as
  above, and writes `recipes/*.json` through `ui_workflow.portable`), then run it to write the three JSONs.
- [ ] **Step 4: Run the tests and `tests/test_hygiene.py`.** Expected: ok, with no personal paths in `recipes/`.
- [ ] **Step 5: Commit** ("Atlas Comp: recipes (matte, relight, sky) as validated JSON graphs").

### Task 3: Frame compositing for the viewer

**Files:**
- Create: `atlas_vfx/comp/frames.py`
- Test: `tests/test_comp.py` (add)

**Interfaces:**
- **Consumes:** `shot.load_shot`, `shot.layer_pattern`; `layers.node_viewer.read_frame`, `render`, `views`;
  `layers.card_project.over`; `camera_side.plate` (`plate_ref` from the shot's plate via
  `node_plate_read.AtlasVFXPlateRead`, `decode_rgb`, `to_working`).
- **Produces, in `atlas_vfx.comp.frames`:**
  - `BLENDS = ("over", "plus", "multiply", "screen")`
  - `composite(shot: dict, frame: int, width: int = 1280, only: list[str] | None = None) -> tuple[np.ndarray, list[str]]`:
    - returns linear ACEScg RGBA at `width` (the plate at the bottom when present) and the names of the layers used;
    - a layer is skipped (and left out of the list) when invisible, unrendered, or missing that frame;
    - blends are premultiplied: over `b*(1-a)+f`, plus `b+f`, multiply `b*f` over the layer alpha, screen
      `1-(1-b)(1-f)`; `opacity` scales the layer's RGBA.
  - `display_jpeg(shot: dict, frame: int, width=1280, exposure=0.0, view: str = DEFAULT_VIEW, a: list[str] | None = None, b: list[str] | None = None, wipe=0.5, quality=88) -> bytes`:
    - no `b`: composite `a` (or everything) through the view;
    - with `b`: wipe at `wipe`, using the Viewer's `render`.
  - Plate frames are cached (LRU of 8, keyed by frame and width) because a MOV decode is slow.

- [ ] **Step 1: Write the failing tests** (synthetic EXR layers written with `exr_io.write_rgba` in a temp work
  root, shot plate `""`):
  - `test_blends`: a red half-alpha layer over a grey layer gives `0.5*red + 0.5*grey` for over; plus adds; screen
    and multiply against hand values at one pixel; opacity 0.5 halves the layer.
  - `test_skips_missing`: layer B has only frame 1130. `composite(..., 1131)` uses only A, and the used list says so
    with no exception.
  - `test_jpeg`: `display_jpeg` starts with `b"\xff\xd8"` and decodes (cv2) to width 1280. With `b` and wipe 0.25,
    column 300 comes from A and column 400 from B.
- [ ] **Step 2: Run.** Expected: fails on the missing module.
- [ ] **Step 3: Implement `frames.py`.** Resize each layer to `width` with INTER_AREA. A shot with an empty `plate`
  has no plate layer.
- [ ] **Step 4: Run `tests/test_comp.py`.** Expected: all ok.
- [ ] **Step 5: Commit** ("Atlas Comp: server-side frame compositing through the OCIO view").

### Task 4: Routes on ComfyUI's server

**Files:**
- Create: `atlas_vfx/comp/server.py`
- Modify: `__init__.py` (call `atlas_vfx.comp.server.register()` in a try/except that logs and never stops the
  nodes loading)
- Test: `tests/test_comp_server.py`

**Interfaces:**
- **Consumes:** Tasks 1-3.
- **Produces:**
  - `build_routes() -> aiohttp.web.RouteTableDef` and `register() -> str`:
    - `register` adds the routes to `server.PromptServer.instance.app.router`;
    - it returns `"not inside ComfyUI"` outside ComfyUI.
- **Routes:**

  | Route | Response |
  |---|---|
  | `GET /atlas` | `app/index.html` |
  | `GET /atlas/app/{name}` | files directly in `app/` with suffix `.js .css .html .svg` only; anything else 404 |
  | `GET /atlas/api/shots` | `{"shots": [...]}` |
  | `POST /atlas/api/shots` | json `{name, plate, first, last}` → the shot |
  | `GET /atlas/api/shots/{shot}` | the shot with versions |
  | `POST /atlas/api/shots/{shot}/layers` | json `{layer, kind, source?, recipe?, params?}` |
  | `PATCH /atlas/api/shots/{shot}/layers/{layer}` | json changes |
  | `GET /atlas/api/recipes` | `{"recipes": [...]}` |
  | `POST /atlas/api/shots/{shot}/layers/{layer}/graph` | json `{mode}` → `{"prompt": graph, "version": "v00N"}` (render takes `next_version`; preview writes to `<layer>/preview`) |
  | `GET /atlas/api/shots/{shot}/frame?f=&w=&exposure=&view=&a=&b=&wipe=` | `image/jpeg` |
  | `GET /atlas/api/views` | the OCIO views list |

  - Every `ValueError` becomes HTTP 400 `{"error": str}`; anything else becomes a 500 with a logged traceback.

- [ ] **Step 1: Write the failing tests** with `aiohttp.test_utils.TestClient` / `TestServer` over an
  `aiohttp.web.Application` holding `build_routes()` (temp work root):
  - create a shot, then list, get, add a layer and patch it;
  - `GET /atlas/app/../atlas_vfx/__init__.py` and `/atlas/app/x.py` give 404;
  - a graph POST with a bad param gives 400 naming the field;
  - a render graph twice gives v001 then v002;
  - a frame request gives `image/jpeg`.
- [ ] **Step 2: Run `tests/test_comp_server.py`.** Expected: fails on the missing module.
- [ ] **Step 3: Implement `server.py` and the `__init__.py` hook.** Heavy work (the frame composite) goes through
  `loop.run_in_executor` so ComfyUI's event loop never blocks.
- [ ] **Step 4: Run the new test plus `tests/test_import.py` and `tests/test_node_contracts.py`.** Expected: ok.
- [ ] **Step 5: Commit** ("Atlas Comp: routes on ComfyUI's server (/atlas)").

### Task 5: The browser app

**Files:**
- Create: `app/index.html`, `app/style.css`, `app/api.js` (fetch wrappers + ComfyUI websocket),
  `app/main.js` (state + panes)
- Test: `tests/test_comp_app.py`

**Interfaces:**
- **Consumes:** the Task 4 routes; ComfyUI's `POST /prompt` (`{"prompt", "client_id"}`), `ws /ws?clientId=`
  (`progress`, `executing`, `execution_error` text messages; binary preview frames, an 8-byte header then a JPEG),
  `GET /history/{id}`.
- **Layout** (dark, compositor-like, for a 1920-wide screen):
  - Top bar: shot picker, a "New shot" form, and the view picker.
  - Left: the layer stack. Each row has eye (visible), name, blend select, opacity slider and a version select;
    "+ Layer" adds an existing EXR sequence or a recipe.
  - Centre: the viewer `<img>` fed by `/frame`, with A/B buttons (pick two layers or versions), a wipe slider, an
    exposure slider (stops) and a live-preview overlay showing the latest binary preview while a job runs.
  - Right: the inspector, drawing the selected recipe layer's fields:
    - choice → select, text → textarea, int/float → number, bool → checkbox, layer → a select of the shot's layers;
    - then the **Preview** and **Render** buttons, and job status.
  - Bottom: the timeline, with a frame scrubber (Nuke frames, first-last), the current frame, and one bar per layer
    showing which frames have EXRs.
- **Behaviour:**
  - Queueing = `POST graph` then `POST /prompt` with the page's `clientId`.
  - On `executing` with `node: null`, reload the shot and refresh the viewer.
  - Scrubbing debounces `/frame` requests (one in flight).
  - Empty state, with no shots: a short "Make your first shot: plate file, first and last frame" form in the centre.

- [ ] **Step 1: Write `tests/test_comp_app.py`:**
  - the four files exist;
  - no `http(s)://` URLs to other hosts in them;
  - no `eval(` and no `innerHTML =` with interpolated server strings (use `textContent`);
  - every `fetch("/atlas/api/...")` path in `api.js` matches a route in `build_routes()`;
  - `index.html` loads `main.js` as `type="module"`.
- [ ] **Step 2: Run it.** Expected: fails (no files).
- [ ] **Step 3: Implement the four files.**
- [ ] **Step 4: Run `tests/test_comp_app.py`.** Expected: ok.
- [ ] **Step 5: Live check on V141.**
  - Restart V141 without `--enable-triton-backend`.
  - Open `/atlas` with the `/browse` skill:
    - the empty state shows;
    - create shot DSCF3000 1130-1226;
    - add `showcase_abc_showcase_element` as a sequence layer;
    - scrub, wipe and change exposure;
    - add a Matte recipe layer on the element and run Preview, watching the live preview appear;
    - run the Sky recipe as Render, and check the version shows in the stack.
  - Screenshots go to the scratchpad.
- [ ] **Step 6: Commit** ("Atlas Comp: the browser app at /atlas").

### Task 6: Docs and full run

**Files:**
- Modify: `README.md` (an "Atlas Comp" section: open `/atlas`, what Phase 1 does), `CHANGELOG.md`,
  `docs/dev/NOTES.md` (the live check results), `.gitignore` if needed
- Test: `tests/run_all.py`

- [ ] **Step 1: Write the docs in plain prose.** Name the routes and the shot folder layout.
- [ ] **Step 2: Run `<V141 python> tests/run_all.py`.** Expected: all pass (39 of 39).
- [ ] **Step 3: Commit** ("Atlas Comp Phase 1: docs").
