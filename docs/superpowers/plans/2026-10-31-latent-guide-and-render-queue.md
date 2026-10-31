# Latent Guide and Render Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode reference footage once to a float32 latent and guide LTX IC-LoRAs from it (the priority), then let
Atlas Comp send recipe jobs to the home farm and Comfy Cloud from a queue tab, with preflight and a result check.

**Architecture:**
- **Two new layer nodes:**
  - Latent Pre-render: VAE in float32, writes `.latent` safetensors plus a sidecar.
  - Guide From Latent: replays `LTXAddVideoICLoRAGuide`'s keyframe and attention steps with the cached latent.
- **Comp side:**
  - a target registry (local config, untracked);
  - a preflight against each target;
  - a queue that splits each job into pre (local), generate (target) and post (local);
  - a Cloud adapter on Comfy's API.

**Tech Stack:**
- Python 3.12 (V141), torch, safetensors (in ComfyUI), aiohttp.
- ComfyUI-LTXVideo's `nodes_lt` and `iclora_attention`, imported lazily.
- Plain-JS app. Tests are plain scripts.

**Spec:** `docs/superpowers/specs/2026-10-31-render-queue-design.md`

## Global Constraints

- **No new dependencies.** The licence register comes first for anything new.
- **Latent files** are safetensors, loaded with `comfy.utils.load_torch_file(..., safe_load=True)`; never pickle.
- **Untrusted input:** every widget and HTTP input; output paths through `plate.resolve_out_dir` / `safe_name`.
- **Nothing machine-specific in tracked files:**
  - no tokens or LAN addresses;
  - targets come from `atlas_comp_targets.local.json` beside the work root, ignored by git;
  - the farm token from its environment variable only (its name lives in `docs/MACHINES.local.md`).
- **Guide From Latent:** frame_idx 0 and downscale 1 only; anything else is refused with a reason.
- **Imports:** layers may import ComfyUI-LTXVideo modules only inside node `run`, so atlas-vfx loads without the pack.
- **Commits** end with the two attribution lines.

## Review Focus

- **A cached latent made with another VAE, size or frame count** than the job's latent: refused, naming the
  mismatch (sidecar check), never a silent wrong guide.
- **The same frames encoded twice:** the second call reuses the file (hash match), with no second encode.
- **A target that goes offline mid-queue:** that job fails with the reason, and the queue moves on.
- **A Cloud job:** nothing is uploaded or spent before the artist confirms the estimate.
- **A target whose ComfyUI lacks the guide node:** preflight refuses before any upload.

---

### Task 1: Latent Pre-render node (priority)

**Files:**
- Create: `atlas_vfx/layers/node_latent_prerender.py`
- Test: `tests/test_latent_guide.py`

**Interfaces:**
- **Produces** `AtlasVFXLatentPrerender` (catalogue 65, Atlas VFX/Artist):
  - **inputs:** `images` IMAGE (display-referred reference, LTX work size), `vae` VAE (wrap the input with
    `LTXVVAEForceFloat32` upstream), `name` STRING, `out_dir` STRING;
  - **outputs:** `latent` LATENT, `path` STRING, `info` STRING;
  - **writes:** `<out>/<name>.latent` (safetensors `{"latent_tensor": fp32}`) and `<name>.json`, holding the frames'
    sha256, frame count, size, the VAE's `downscale_index_formula`, its dtype, and the file's sha256;
  - **reuse:** if the sidecar's frame hash matches, it loads the file instead of encoding (`info` says
    `"cached"`).
- **Pure helpers:** `frames_hash(images: np.ndarray) -> str`, `keep_frames(n: int, t: int) -> int` (8n+1 trim as
  LTX does).

- [ ] **Step 1:** Write `test_prerender_writes_and_reuses`. Use a fake VAE (an object with
  `encode(pixels)` returning a deterministic `(1,128,T,H/32,W/32)` tensor and `downscale_index_formula=(8,32,32)`).
  - The first call writes `.latent` and `.json`; the second call does not call `encode` (counter) and returns an
    equal tensor.
  - The latent dtype is float32, and frames beyond 8k+1 are trimmed (10 frames give 9).
  - A name with `..` is refused.
- [ ] **Step 2:** Run `<V141 python> tests/test_latent_guide.py`; expect an ImportError.
- [ ] **Step 3:** Implement `node_latent_prerender.py`: cast the encode output `.float()`, write with
  `safetensors.torch.save_file`, read with `comfy.utils.load_torch_file(safe_load=True)`.
- [ ] **Step 4:** Run; expect ok.
- [ ] **Step 5:** Commit ("Latent Pre-render: encode a reference once at float32, reuse it").

### Task 2: Guide From Latent node (priority)

**Files:**
- Create: `atlas_vfx/layers/node_guide_from_latent.py`
- Test: `tests/test_latent_guide.py` (add); live check on V141.

**Interfaces:**
- **Consumes:** the Task 1 file format.
- **Produces** `AtlasVFXGuideFromLatent` (catalogue 66):
  - **inputs:** `positive`, `negative` CONDITIONING; `latent` LATENT (the empty video latent); `guide` LATENT
    (from Pre-render, or `LoadLatent`); `strength` FLOAT; `latent_downscale_factor` FLOAT (wired from the IC-LoRA
    loader, must be 1);
  - **outputs:** `positive`, `negative`, `latent`, matching `LTXAddVideoICLoRAGuide`'s outputs;
  - **body:** mirrors `iclora.LTXAddVideoICLoRAGuide.execute` for the whole-clip path with frame_idx 0
    (`causal_fix` true): `resolve_frame_idx`, then `nodes_lt.LTXVAddGuide.append_keyframe(..., causal_fix=True)`,
    then `append_guide_attention_entry` on both conditionings, with
    `iclora_tokens_added = F*H*W` of the guide.
  - **refuses:** a guide whose H×W does not equal the target latent's, or a downscale ≠ 1.

- [ ] **Step 1:** Write `test_guide_matches_image_guide`. With ComfyUI's root on `sys.path` and the fake VAE from
  Task 1:
  - run `LTXAddVideoICLoRAGuide.execute(image)` and ours with `encode(image)`;
  - assert equal conditioning keyframe tensors (`keyframe_idxs`, the appended latent, the noise_mask) and equal
    `guide_attention_entries`.
  - Also: a mismatched H×W raises, and downscale 2 raises.
- [ ] **Step 2:** Run; expect an ImportError.
- [ ] **Step 3:** Implement. Import `nodes_lt` from `comfy_extras.nodes_lt`, or the pack's module as `iclora.py`
  does: read its import line and use the same one.
- [ ] **Step 4:** Run; expect ok.
- [ ] **Step 5:** **Live:** run the relight recipe on DSCF3000 1130 (17 frames, seed 42) on V141 twice, once with
  the image guide and once with Pre-render then Guide From Latent. Compare the decoded frames.
  - **Expected:** max abs difference under 1e-3 (identical apart from encode precision: the image path's encode is
    float32 too through `LTXVVAEForceFloat32`).
  - Record the encode time saved in NOTES.
- [ ] **Step 6:** Catalogue rows and blocks 65-66, contracts `--update`, CHANGELOG. Commit ("Guide From Latent:
  IC-LoRA guide from a cached float32 latent").

### Task 3: Relight and matte recipes use the cached latent

**Files:**
- Modify: `tools/comp_recipes.py` (relight, matte), regenerate `recipes/*.json`
- Test: `tests/test_comp.py` (add)

- [ ] **Step 1:** Write a test: the filled relight graph has Pre-render (out_dir in the layer's `cache/`) feeding
  Guide From Latent, and no `LTXAddVideoICLoRAGuide`. Its links are intact, and preview still drops the 4K transfer.
- [ ] **Step 2:** Run; fail.
- [ ] **Step 3:** Rewire both recipes. Pre-render's `out_dir` is set from a new context key `cache_dir` (=
  `layers/<layer>/cache`, a folder never counted as a version).
- [ ] **Step 4:** Run `tests/test_comp.py`; ok.
- [ ] **Step 5:** Commit.

### Task 4: Targets and preflight

**Files:**
- Create: `atlas_vfx/comp/targets.py`
- Test: `tests/test_comp_targets.py`

**Interfaces:**
- `load_targets() -> list[dict]` reads `pl.WORK_ROOT / "atlas_comp_targets.local.json"`. Each target is
  `{"name", "kind": "comfyui"|"comfy_cloud", "url"?}`; the default is ws only. URLs must be `http(s)` and are never
  logged with tokens.
- `preflight(target, graph) -> dict` returns `{"ok": bool, "missing_nodes": [...], "missing_models": [...],
  "notes": [...]}`.
  - comfyui: one `/object_info` GET, then a check that every class_type exists, and that every loader string input
    is in that loader's option list.
  - comfy_cloud: through an injected client: node search, account models.

- [ ] **Step 1:** Tests with a fake HTTP fetch:
  - a graph using a node the fake target lacks gives `ok` False, naming it;
  - a model missing from `UNETLoader` options is named;
  - the targets file absent gives `[ws]`;
  - a URL with `file://` is refused.
- [ ] **Steps 2-4:** Fail, implement, pass.
- [ ] **Step 5:** Commit.

### Task 5: The queue (ws and farm)

**Files:**
- Create: `atlas_vfx/comp/queue.py`
- Modify: `atlas_vfx/comp/server.py` (routes)
- Tests: `tests/test_comp_queue.py`, `tests/test_comp_server.py`

**Interfaces:**
- `Queue.submit(shot, layer, mode, target) -> job_id`.
- Stages:
  - **pre (local):** a graph cut to Prep, Light Ball, Pre-render, written to `cache/`;
  - **generate:** POST `/prompt` to the target with the cached latent uploaded via `/upload/image` or the farm's
    shared root, then poll `/history`;
  - **post (local):** download outputs via `/view`, write the version, check the frame count.
- Routes:
  - `GET /atlas/api/queue`
  - `POST /atlas/api/queue` (`{shot, layer, mode, target}`)
  - `POST /atlas/api/queue/{id}/cancel`
- One worker thread per target.

- [ ] **Step 1:** Tests with a fake target server (aiohttp test server implementing `/object_info`, `/prompt`,
  `/history`, `/view`):
  - a job runs to done and writes the version;
  - a target missing a node is refused at preflight with no upload;
  - a result short of frames fails the job and leaves the version unpicked;
  - a target that dies mid-job fails it and the queue continues.
- [ ] **Steps 2-4:** Fail, implement, pass.
- [ ] **Step 5:** **Live:** queue a relight Preview to the render box (3090), using the target URL from Mike's local
  config.
- [ ] **Step 6:** Commit.

### Task 6: Queue tab in the app

**Files:**
- Modify: `app/index.html`, `app/main.js`, `app/api.js`, `app/style.css`
- Test: `tests/test_comp_app.py`

- [ ] **Step 1:** Static checks:
  - a Render-to target picker beside Render;
  - a Queue panel listing jobs with target, stage, frames and cancel;
  - every new fetch path is a real route.
- [ ] **Steps 2-4:** Fail, implement, pass.
- [ ] **Step 5:** Live check with `/browse`.
- [ ] **Step 6:** Commit.

### Task 7: Comfy Cloud target

**Files:**
- Create: `atlas_vfx/comp/cloud.py`
- Test: `tests/test_comp_cloud.py`

- **Adapter:** upload the reference frames as a video or image batch, submit the generate stage (standard and LTX
  nodes only, image guide, since atlas nodes are absent there), poll, fetch outputs.
- **Estimate first:** nothing is uploaded until the artist confirms.
- **Credentials:** the Comfy account's API key from the environment (`COMFY_CLOUD_API_KEY`), never stored.
- **The Relight LoRA import** is Mike's action on his account. Until then, Cloud runs the matte recipe only.
- **Tests:** a fake Cloud client. Nothing is uploaded before confirm; a missing LoRA is refused at preflight; the
  frame count is checked.
- **Live:** one Alpha-Gen matte on Cloud, with its credits shown first.
- **Commit.**

### Task 8: Docs and full run

- README (queue, targets file, Cloud opt-in), CHANGELOG, NOTES with the farm and Cloud live results.
- Licence register: Comfy Cloud as a service entry.
- `tests/run_all.py` green. Commit.
