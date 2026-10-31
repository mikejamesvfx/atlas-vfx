# Changelog

Versions follow `pyproject.toml` (and `atlas_vfx.__version__`). Before 1.0, a minor version can break saved graphs;
when it does, the entry says so and names the tool that rewrites them.

## Unreleased

- **Frame chunks** for shots longer than one LTX run (catalogue 50-52, `Atlas VFX/Layers/Chunks`). Chunk Plan
  splits a range such as 1001-1400 into equal 8k+1 chunks with a set overlap (the last one moved back to end on the
  last frame, nothing padded); Sequence Read Chunk reads one chunk of an EXR/DPX/PNG sequence; Chunk Stitch joins
  the rendered chunks into one sequence numbered from the first frame, crossfading the overlaps.
  `tools/queue_chunks.py` queues one prompt per chunk from an API workflow. New socket type `ATLAS_VFX_CHUNK_PLAN`.
- **Relight Transfer** (catalogue 53, `Atlas VFX/Layers`): an LTX relight IC-LoRA runs at its own size (2.3 Relight
  at 1280x704, 2.5 Day-to-Night / Golden-Hour), and this node moves only the change of light onto the full-res
  element, keeping every element pixel. mode guided measures the gain at the LoRA's size and brings it up with a
  matte-weighted guided filter (element luminance, plus MoGe depth / normals); mode sh fits the light as spherical
  harmonics on MoGe-3 normals and re-shades at their sharpness. Starter: `workflows/relight_transfer_element.json`
  (`tools/relight_transfer_workflow.py`). The three LoRAs are in the licence register. `subsample` runs the fast
  guided filter (default 2: 2.6x faster at 4K, within 0.01 stop of exact).
- **Relight Transfer smudges**: `lift_limit` (default 3 stops) hands the near-black parts of the plate that the take
  lifts further than that to the take's own pixels (soft but clean, instead of the plate's noise lifted 30x), and an
  optional `holdout` mask drops the light of anything the relight invented and fills it from around it.
- **Mirror for LTX relights**: LTX Prep `mirror` flips what LTX sees left-right and LTX Return flips the take back;
  Relight Transfer `relit_mirrored` does the same for a raw take. On DSCF3000 the LTX-2.3 Relight LoRA kept the sun
  on frame right whatever the light ball said; mirrored, with the ball on the right, the sun lands on the left.
  Starter: `workflows/ltx_relight_mirror.json` (`tools/ltx_relight_workflow.py`): Plate Read, LTX Prep (comp
  reference, mirror on, 1280x704, 97 frames), the light ball pasted top-right, the LTX-2.3 Relight IC-LoRA on the
  LTX-2.5 distilled base, LTX Return. Its layer is the `relit` input of the Relight Transfer starter. Put the ball
  PNG (`the right__low.png`) in ComfyUI's input folder.

- **Relight Transfer Files** (catalogue 54, `Atlas VFX/Layers`): Relight Transfer for 4K and long shots. It takes
  the element, matte (blank = the element's alpha), relit take and holdout as sequence paths, reads them one frame
  at a time, runs MoGe-3 per frame inside the node (normals cached on disk, removed at the end) and writes the EXRs
  and gain EXRs as it goes. Same maths as 53. 97 frames of DSCF3000 at 4K in one prompt peaked at 13.4 GB of RAM;
  the IMAGE graph ran out of 64 GB and had to be cut into chunks, which popped up to a stop at the joins. Starter:
  `workflows/relight_transfer_files.json`.

- **Artist nodes** (catalogue 55-61, `Atlas VFX/Artist`), for a Nuke / AE / Photoshop user: Shot Range (frames
  once, in Nuke numbers; LTX 8k+1, Plate Read's index, RAM), Sun Direction and Light Ball (the sun in words; prompt,
  ball and the mirror trick done for you; the balls ship in `assets/relight_balls`, MIT), Layer Picker (rendered
  layers from a list), Grade (Nuke's Grade maths), Viewer (A/B wipe, f-stops, OCIO view, zoom to alpha, gain false
  colour) and Notes (reports as readable lines, warnings first). No new dependencies.
- **Blueprints**: `subgraphs/relight_element_4k.json` and `subgraphs/ltx_clean_loop.json` show up in ComfyUI's node
  library as one box each, with only the settings an artist changes (`tools/blueprints.py`). The LTX relight
  starters now use the artist nodes.

- **LTX Alpha-Gen mattes** (catalogue 63-64): Alpha Prep / Alpha Return wrap the LTX-2.5 Alpha-Gen IC-LoRA (a matte
  from any picture or clip, no green screen, soft edges kept). A single still works: it is held for 9 frames and the
  mattes are averaged before the clamp. Starter `workflows/alpha_matte.json`, blueprint "Atlas VFX Alpha Matte".
- **Tiny VAE** (catalogue 62): taeltx2_3 (madebyollin/taehv, MIT, safetensors) for a quick-look decode of any LTX
  latent in a second. Loading the pack also gives LTX samplers live previews through it when ComfyUI's live preview
  method is TAESD (`ATLAS_VFX_LTX_LIVE_PREVIEW=0` turns that off; anything unexpected falls back to ComfyUI's preview).

## 0.1.0 (2026-10-30), first public beta

The first tagged release of the public repo. 37 nodes in five families (Camera, Layers, Plate Lock, Sensor,
Deliver); what each is proven on is in the README's Status section and `docs/NODE_CATALOGUE.md`.

Since the public repo started (one commit, 2026-10-29):

- **Physical Sky renders anywhere.** A Hosek-Wilkie sky ships in `atlas_vfx/layers/sky_hosek.py` (BSD-3 data,
  checked against the compiled reference C), fitted to Prague so the two agree on the light at 20-50 km visibility.
  New last widget `sky_model`: auto (Prague when Atlas Scene is reachable, else Hosek-Wilkie), prague,
  hosek-wilkie. Old graphs load unchanged and get auto.
- **Sun Globe panel**: sky model switch, a flag and a short reason when the picked model won't render the sun
  (Hosek-Wilkie twilight, anything under -4.2 deg), opens behind the camera looking where it looks, no clipped text
  on a narrow node.
- **OpenImageIO is now required** (`requirements.txt`, `pyproject.toml`): most of what the nodes write is EXR, and
  a ComfyUI Manager install without it could not run the starter. pycolmap stays optional.
- **Chart Calibrate drops a bad chart detection** instead of averaging it in. OpenCV's exhaustive chessboard
  search is not deterministic and now and then returns a bogus patch; on CI one such view moved the lens focal 7 %.
  A view far worse than the rest is dropped, the lens solved again, and the node warns with the frame.
- **Starter workflow** `workflows/starter_physical_sky.json`: no plate, no other packs.
- **CI** (`.github/workflows/tests.yml`, Linux, no GPU): every test script on Python 3.11 and 3.12, and
  `tools/clean_install_check.py`, which loads the pack into a fresh ComfyUI v0.43.0 with an empty home folder and
  renders the starter.
- **Docs**: node map, Prague vs Hosek-Wilkie and Sun Globe images in `docs/images/`; README status and first-run
  sections.

Breaking for graphs saved in the private history: the custom socket types were namespaced before the public commit
(`ATLAS_TRACKED_CAMERA`, `ATLAS_PLATE_SEQ`, `ATLAS_VFX_*`). `tools/rename_socket_types.py` rewrites old graphs.
