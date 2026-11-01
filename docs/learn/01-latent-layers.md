# 01 Latent layers: getting layers in and out of LTX without losing the lock

This guide covers the part of atlas-vfx that hands your comp layers to LTX-2.5 and takes them back: LTX Prep,
LTX Return, Latent Pre-render, Guide From Latent, the Chunk nodes and the Tiny VAE. Read it with
`docs/NODE_CATALOGUE.md` open (blocks 44, 45, 50 to 52, 62, 65, 66). The hands-on part is
`docs/learn/exercises/01_latent_layers.py`, which runs in a couple of seconds on synthetic data.

## 1. What it's for

You've tracked a plate, put a generated element on a card in 3D (Card Project), and it sits in the shot. It's locked,
but it looks pasted on: wrong softness, no grain, no contact shadow, lit by whatever the generator imagined. LTX
can fix that, because it's a video model that has seen a lot of real footage. The catch is that video models don't
care about your camera solve. Hand LTX a whole frame and it will happily repaint the plate, nudge the horizon and
drift the element a few pixels a second.

So the system has one job: let LTX paint light, blur, grain and shadow onto a layer **while the camera lock stays
ours**. LTX only ever paints inside a mask built from the layer's own projected alpha. Everything outside that mask
comes back as alpha 0, so the plate (and the track under it) is bit-for-bit untouched.

The same plumbing runs the relight passes (the Relight IC-LoRA, Day-to-Night), the Alpha-Gen matte, and shots longer
than one LTX run.

"Latent encode" here means the moments a picture is turned into LTX's latent (the VAE encode): the reference video,
the start image, the keyframes, and the stage-2 upscale. atlas-vfx decides what goes into each, in which colour
space, at what size, and (with Latent Pre-render) lets you encode the reference once and reuse it.

## 2. The mental model

```
 plate (as shot)                                                  delivered
     |                                                                ^
     v  input space -> ACEScg                                         |  over the plate in Nuke
 [optional: Sensor Pass "remove camera" -> clean space]    [optional: "apply camera" back on]
     |                                                                ^
     +--- comp layers over it (Card Project EXRs, premult ACEScg) ---+|
     |                                                               ||
     v  named OCIO view: ACES 2.0 SDR (display-referred 0-1)         ||
 LTX PREP --> reference video, mask, start image, keyframes, record   |
     |                                                                |
     v  VAE encode (once, or cached by Latent Pre-render)             |
   LTX sampler (IC-LoRA guide + keyframe guides + start image)        |
     |                                                                |
     v  VAE decode, Laplacian blend keeps the reference outside mask  |
 LTX RETURN --> exact inverse of the SAME view -> ACEScg, plate size, |
                alpha = feathered mask, premultiplied EXR layer ------+
```

Three rules hold the whole thing together:

1. **LTX sees ordinary video.** It was trained on display pictures, so it gets the plate through a named display view,
   not scene-linear numbers.
2. **What goes out comes back through the exact inverse.** Return reads which view Prep used from Prep's record and
   inverts that one, not whatever today's default happens to be.
3. **The mask is the contract.** Inside it, LTX's pixels. Outside it, alpha 0. The lock never depends on the model
   behaving.

## 3. How it works in the code

### The layer going in

A "layer" is a Card Project output folder: one premultiplied ACEScg RGBA EXR per plate frame plus a `*.render.json`
naming the card, its frame pattern, its frame range, its lens, and (since the lens work) a `versions` map with a
`plate` version (redistorted to sit on the plate) and a `clean` version (pinhole). LTX Return writes the same kind of
folder, so a returned take can be the input of another pass. Layer Picker (catalogue 58) lists these folders so you
pick instead of typing paths.

### LTX Prep (`atlas_vfx/layers/node_ltx_prep.py`, helpers in `layers/ltx_integration.py`)

For each frame of the run, `AtlasVFXLTXPrep.run`:

1. decodes the plate (`plate.decode_rgb`) and converts it to ACEScg (`plate.to_working`), using the range and log
   curve the Plate Read node recorded;
2. if you gave Sensor Pass folders, takes the plate's camera **off** (`ltx_integration.remove_camera`: de-rolling
   shutter, undistort, devignette, all in ACEScg), so LTX sees clean pinhole frames that match the card layers;
3. comps the layers over it, back to front (`card_project.over`), reading the `clean` version when it's working in
   clean space and the `plate` version otherwise;
4. pushes the result through the named view with `ltx_integration.to_view` (forward processor from
   `view_processors`, display `sRGB - Display`, view `ACES 2.0 - SDR 100 nits (Rec.709)` by default) and resizes to
   the work size;
5. builds the mask with `ltx_integration.make_mask`: layer alpha above `mask_thresh` (0.05), smeared **downwards**
   by `shadow_px` (24) so LTX has room to paint a contact shadow under feet, then dilated by `dilate_px` (12);
6. on frame 0 runs `roundtrip_report`: ACEScg to view to inverse on the real plate, and records the error.

It returns seven things:

| output | what it is | where it goes in the graph |
|---|---|---|
| `reference` / `reference_half` | the plate (or plate plus layers, `reference = comp`) in the view, at work size and half | the IC-LoRA guide, through `LTXVInpaintPreprocess` (which greens the masked area) |
| `mask` / `mask_half` | the inpaint mask at both sizes | `LTXVInpaintPreprocess` and the Laplacian blends |
| `start_image` | frame 0 with every layer comped in | `LTXVImgToVideoInplace` |
| `keyframes_half` | the comp every `keyframe_every` (24) frames plus the last, half size | one `LTXVAddGuide` each |
| `ltx_prep` | the record (`ATLAS_VFX_LTX_PREP`): plate ref, frames, sizes, view, round-trip report, crop windows, mirror flag, sensor folders, warnings, and the masks kept in memory | LTX Return |

The length must be 8k+1 (Prep raises otherwise) because LTX's VAE squeezes 8 frames into one latent frame plus the
first one on its own. Work sizes step in 64 px; the latent is 1/32 of the picture in each direction and has 128
channels, so 97 frames at 1280x704 is a `(1, 128, 13, 22, 40)` latent.

Two modes matter for detail. `mode full` scales the whole plate to the work size. `mode crop` cuts a window that
follows the element (`crop_windows`): it keeps the work aspect, is never smaller than the work size (so the element
is at native 1:1 when it fits), grows only as much as the element plus `crop_margin` needs, and is smoothed over
plus or minus 8 frames because a jittery crop reads to LTX as camera shake.

### What gets encoded, and when

In the shipped clean-loop graph (`subgraphs/ltx_clean_loop.json`, the same nodes as
`workflows/ltx_element_clean_loop_s2s3.json`) LTX runs Lightricks' two-stage inpaint recipe. These are the encodes,
in order, with the strengths the graph ships with:

| stage | picture | encoded by | strength |
|---|---|---|---|
| 1 (960x512, 8 steps) | `start_image`, scaled to half | `LTXVImgToVideoInplace` | 0.7 |
| 1 | `reference_half` greened under `mask_half` | `LTXAddVideoICLoRAGuideAdvanced` (the in/outpainting IC-LoRA guide) | 1.0 |
| 1 | `keyframes_half` frames 24, 48, 72, 96, 120 | five chained `LTXVAddGuide` nodes | 0.8 each |
| 2 (1920x1024, 2 steps) | stage 1's decode (Laplacian-blended, upscaled) | `VAEEncodeTiled` | n/a, it is the latent being refined |
| 2 | `start_image` at full size | `LTXVImgToVideoInplace` | 1.0 |

Stage 1 samples with `euler_ancestral` over the sigmas `1.0 ... 0.0` (seed 43); `LTXVCropGuides` then strips the
guide frames back off the latent before decode. Stage 2 refines with `euler` over `0.725, 0.4219, 0.0` (seed 42),
using stage 1's cropped conditioning. After each stage `LTXVLaplacianPyramidBlend` blends the decode with the
reference by the mask, so outside the mask the pixels LTX hands to Return are the reference's own.

The keyframes are there for a reason (section 7): with the start image alone, the element faded into LTX's green
inpaint fill after about two seconds.

### Latent Pre-render and Guide From Latent (65, 66)

The IC-LoRA guide node encodes the reference every run. `AtlasVFXLatentPrerender.run`
(`layers/node_latent_prerender.py`) does that encode once and keeps it:

- trims the clip to the 8k+1 frames LTX would keep (`keep_frames`);
- hashes the frames at 16 bits a channel (`frames_hash`), so identical pictures give an identical hash;
- if `<name>.latent` and `<name>.json` exist, the frame hash matches and the file's own sha256 matches, it loads the
  file (`info` says `"cached": true`) and never calls the VAE;
- otherwise encodes, casts to float32 and writes plain safetensors (`latent_tensor` plus ComfyUI's
  `latent_format_version_0` marker so Load Latent reads it back unscaled), plus a sidecar with the hash, frame count,
  size, the VAE's `downscale_index_formula` and dtype. Default folder: `<work>/latent_cache`.

Wire the VAE through `LTXVVAEForceFloat32` first so the encode itself is float32, as
`workflows/relight_tracked_dscf3000_s2s3.json` does.

`AtlasVFXGuideFromLatent.run` (`layers/node_guide_from_latent.py`) then does what LTX's "Add Video IC-LoRA Guide"
does after its encode, by calling LTX's own code from the loaded ComfyUI-LTXVideo pack: `resolve_frame_idx`,
`LTXVAddGuide.append_keyframe(..., causal_fix=True)`, and `append_guide_attention_entry` on both conditionings with
`F*H*W` guide tokens. Same code, same numbers. NOTES (2026-10-31) has the proof: the relight preview on DSCF3000 1130
(17 frames, seed 42) run with the image guide and with the cached latent gave LTX Return EXRs with a max abs
difference of 0.0. The cache file was `(1, 128, 3, 22, 40)`.

Why bother if it's not about speed? NOTES is honest: the encode is a small part of a run. The value is that every
machine (ws, the farm, Comfy Cloud) starts from the same numbers, and the thing that travels is a few MB of latent,
not the frames. The render queue's "pre" stage runs Pre-render locally for exactly that reason.

### LTX Return (`layers/node_ltx_return.py`)

`AtlasVFXLTXReturn.run` takes LTX's decoded frames and the Prep record and, per frame:

1. resizes back to the work size if needed and flips the take back if Prep mirrored it;
2. gets the processors from the record (`ltx_integration.prep_processors`), so it inverts the exact view Prep applied;
3. `return_frame` (full mode) or `return_window` (crop mode): inverse view to ACEScg, upscale to plate size, alpha =
   the work-size mask upscaled and feathered (`feather_px`, 3). In crop mode the window is pasted back with its edge
   ramped wherever the window sits inside the plate, so there's no seam;
4. if Prep worked in clean space, writes that frame to `clean/` and then puts the plate camera back on
   (`apply_camera`: vignette, then CA and distortion, then rolling shutter) for the main frame;
5. writes `<name>.<frame>.exr`, RGBA half, premultiplied ACEScg, plus `<name>.render.json` (with `versions`, the
   round-trip report, coverage, window scales) and a sidecar.

Outside the mask alpha is 0, so in Nuke the plate shows through exactly. `tests/test_ltx_integration.py` checks this
with an identity "LTX" (it hands back its own reference): outside the mask the comp equals the plate bit for bit,
and an unchanged pass comes back within a few tenths of a percent inside it.

### Phase D: elements come in through the inverse of LTX's view

Generated elements (Z-Image, an LTX take, any model) arrive as display pictures. `layers/element_colour.py` decides
how they become ACEScg before they go on a card. The old way treated them as an sRGB texture
(`sRGB Encoded Rec.709 (sRGB)` to ACEScg). That looks right as a texture, but LTX Prep then tone-maps the card
again on the way out, so the element reached LTX darker and flatter than it was made. NOTES (flow review
2026-11-01) measured 0.12 darker in the mids. The default now is `inverse display view (ACES 2.0 SDR, as LTX)`: what
the element looked like on screen is exactly what LTX sees. Foreground Element and Relight Transfer both have this
`source_colour` switch. The exercise reproduces the 0.12.

### Chunks for long shots (50, 51, 52, `layers/chunks.py`)

LTX takes 8k+1 frames a run, 121 is the usual cap. `chunks.plan(first, last, max_len, overlap)` makes equal chunks
with `stride = length - overlap`, and slides the last chunk back so it ends exactly on `last` (no padded frames, its
seam is just longer). 1001-1400 at 121/24 gives 1001-1121, 1098-1218, 1195-1315, 1280-1400. Every chunk has the same
length, so one graph serves them all: wire Chunk Plan's `start`, `length` and `tag` into LTX Prep, the empty latent
lengths and the output names, then either set `chunk` to -1 (ComfyUI runs the downstream once per chunk) or use
`tools/queue_chunks.py` (one prompt per chunk, fills `{tag}` / `{chunk}` in names). Chunk Stitch joins the results
with smoothstep weights (`chunks.weights`) that sum to 1, in the files' own values (scene-linear for EXR).

### Tiny VAE and live previews (62, `layers/ltx_live_preview.py`)

Tiny VAE loads `taeltx2_3` (MIT) from `models/vae_approx` for a quick look decode: seconds, blurry, right in framing,
motion and light. Loading the pack also wraps ComfyUI's previewer so LTX samplers preview through it when the live
preview method is TAESD; anything unexpected falls back to ComfyUI's own preview, and `ATLAS_VFX_LTX_LIVE_PREVIEW=0`
turns it off.

### How the layers stay camera-locked

It's worth spelling out, because it's the reason this exists instead of "just run LTX on the plate":

- **Placement never comes from LTX.** The card is placed in 3D by Card Project from the tracked camera. LTX Prep
  only ever sees that card already projected into each frame, and its mask is that projected alpha. NOTES
  (2026-10-19): through the fast blur around frame 1190 the element blurs with the plate, and its feet stay where
  the card put them.
- **Outside the mask is not LTX's.** The Laplacian blends hand the reference back outside the mask, and Return sets
  alpha 0 there anyway. Two guarantees, one in the graph and one in our code.
- **Resizes are undone exactly.** The work size is a non-uniform scale of about 1 % from 4K; Return scales back with
  the plate size from the record. Crop windows are pasted back where they were cut.
- **The plate's camera is handled outside LTX.** With Sensor Pass folders, Prep removes the lens and rolling shutter
  so LTX works on pinhole frames that match the pinhole cards, and Return puts them back. On S2-S3 the A/B showed no
  measurable difference (the shot is too gentle), and the RS part should stay off on shots solved by a
  global-shutter tracker (section 7), but the loop is mechanically right: remove then apply round trips to 0.0001.

## 4. The nodes

| # | node | key inputs | key outputs |
|---|---|---|---|
| 44 | Atlas VFX LTX Prep | `plate_ref`, `layer_folder`, `first_frame`, `length` (8k+1), `width` / `height`, `mask_thresh`, `shadow_px`, `dilate_px`, `keyframe_every`, `mode` full\|crop, `reference` plate\|comp, `mask_mode` layer\|none; optional Sensor folders, `mirror`, `allow_raw_lens` | `reference`, `mask`, their `_half`s, `start_image`, `keyframes_half`, `ltx_prep` |
| 45 | Atlas VFX LTX Return | `images` (LTX decode), `ltx_prep`, `feather_px`, `name`, `source_model` | `layer_folder` (EXR layer), `preview` |
| 65 | Atlas VFX Latent Pre-render | `images` (the reference at work size), `vae`, `name`, `out_dir` | `latent`, `path`, `info` |
| 66 | Atlas VFX Guide From Latent | `positive`, `negative`, `vae`, `latent` (empty video latent), `guide` (Pre-render's latent), `strength`, `latent_downscale_factor` (wire from the IC-LoRA loader, must be 1) | `positive`, `negative`, `latent` |
| 50 | Atlas VFX Chunk Plan | `first`, `last`, `max_len`, `overlap`, `chunk` (-1 = all) | `chunk_plan`, `start`, `length`, `end`, `tag` |
| 51 | Atlas VFX Sequence Read Chunk | `path`, `start`, `length`, `numbering`, colour widgets | `images`, `first_frame` |
| 52 | Atlas VFX Chunk Stitch | `chunk_plan`, `pattern` with `{tag}` or `{chunk}`, `numbering`, `name` | `pattern`, `first_frame` |
| 62 | Atlas VFX Tiny VAE | `tae_name` | `vae` |
| 58 | Atlas VFX Layer Picker | `layer` (folders under the work root) | `folder`, `pattern`, `first`, `last` |

Socket types: `ATLAS_VFX_LTX_PREP` goes from Prep to Return (and Sensor Apply), `ATLAS_VFX_CHUNK_PLAN` from Plan to
Stitch. Guide From Latent finds ComfyUI-LTXVideo at run time, so atlas-vfx loads fine without the pack.

## 5. Walkthrough: the two shipped graphs

**Integration pass, `workflows/ltx_element_clean_loop_s2s3.json`** (the clean-loop subgraph is the same nodes). Open
it and follow the wires from LTX Prep (node 100):

1. `reference_half` and `mask_half` go into `LTXVInpaintPreprocess` (110), which feeds the IC-LoRA guide (115). The
   IC-LoRA loader (102, the in/outpainting LoRA) wires its `latent_downscale_factor` into the guide.
2. `start_image` goes through `ImageScale` and `LTXVPreprocess` into `LTXVImgToVideoInplace` (114, 0.7) on an empty
   960x512x121 latent (111).
3. `keyframes_half` is split by `ImageFromBatch` nodes into the five `LTXVAddGuide`s (171 to 175, frames 24 to 120,
   0.8), chained after the IC-LoRA guide.
4. Stage 1 samples (122), crops the guides (124), decodes (125) and blends with the greened reference by
   `mask_half` (126).
5. Stage 2 upscales that, encodes it (131), adds the full-size start image (132, 1.0), samples two steps (138),
   decodes (140) and blends with the full reference by `mask` (142).
6. LTX Return (150) gets the blend and Prep's record and writes `layers/element_ltx`.

**Relight pass, `workflows/relight_tracked_dscf3000_s2s3.json`.** Here Prep runs with `reference comp` and
`mask_mode none` (the whole comp gets relit, Return delivers a full replacement plate with alpha 1). Sun Direction
(from Physical Sky's light rig) decides the light and the `mirror` switch; Light Ball draws the ball onto Prep's
reference; that picture goes into **Latent Pre-render** (60) with the VAE wrapped by `LTXVVAEForceFloat32` (15);
**Guide From Latent** (43) adds it to an empty 1280x704x97 latent with the Relight LoRA's `latent_downscale_factor`.
After sampling, `LTXVCropGuides`, decode, and LTX Return writes `comp_relit`. Compare with
`workflows/ltx_relight_mirror.json`, which is the same pass with LTX's own `LTXAddVideoICLoRAGuide` in place of the
two Atlas nodes: that pair is exactly what the bit-identical check compared.

For long shots start from `workflows/chunked_sequence_starter.json` (Chunk Plan at 1001-1400, 121/24, `chunk` -1,
feeding Sequence Read Chunk) and `workflows/chunked_sequence_stitch.json`.

## 6. Hands-on

From the repo root, with the python of the ComfyUI venv that runs atlas-vfx (it has torch, safetensors,
PyOpenColorIO and OpenCV):

```
python docs/learn/exercises/01_latent_layers.py
```

No plate, no weights, no GPU. A fake VAE with the LTX layout stands in for the real one, and "LTX" is an identity
that hands back its reference. It writes only into a fresh temp folder. This is what it printed on the 5090 box
(about 2 s):

```
1. LTX's frame rule: 8k+1 frames in, (k+1) latent frames out
     10 frames given ->    9 kept ->   2 latent frames
     97 frames given ->   97 kept ->  13 latent frames
    128 frames given ->  121 kept ->  16 latent frames
   ok   17 frames at 1280x704 -> latent (1, 128, 3, 22, 40)

2. Latent Pre-render: encode once, float32 on disk, reuse on a hash match
   wrote reference.latent + reference.json, latent (1, 128, 3, 2, 3)
   sidecar: frames 17, size [96, 64], layout [8, 32, 32], dtype float32, frames_sha256 d1d9a889499e...
   ok   stored as float32 even though this VAE returned half
   second run: cached=True, 4.6 ms, encodes so far 1
   ok   one changed pixel: the hash differs, so it encodes again
   ok   a name with .. is refused (widgets are untrusted)

3. Into LTX and back: ACEScg -> 'ACES 2.0 - SDR 100 nits (Rec.709)' view -> inverse
   dusk plate: median error 0.0009 %, p99 0.0025 %, share over 5 % 0.00 %, max luminance 0.33
   neutral   20.0 -> view 0.9803 -> back    19.99
   neutral  500.0 -> view 1.0000 -> back   128.00
   warm (6.0, 2.4, 0.8) -> view [1.     0.8358 0.75  ] -> back [4.53 2.36 1.07]
   warning LTX Prep would log: 14.8 % of the plate does not survive the SDR view round trip (max luminance 3.29);
   highlights inside the mask come back clipped

4. A generated element (display picture) into ACEScg: inverse view vs sRGB texture
   display 0.50 -> ACEScg 0.3690 (inverse view) vs 0.2140 (sRGB texture)
   ...and what LTX then sees: 0.5000 vs 0.3844
   ok   sRGB texture: tone-mapped twice, so it reaches LTX darker in the mids  (mean 0.124 darker)

5. The inpaint mask: alpha > thresh, smeared DOWN for a contact shadow, then dilated
   element rows 100-149; with shadow_px 20 the mask covers rows 100-169

6. LTX Return with an identity 'LTX': outside the mask the plate is bit-exact
   alpha 0 on 94.8 % of pixels; inside, the round trip differs by 0.23 %
   ok   plate untouched outside the mask (bit-exact)

7. Shots longer than one run: Chunk Plan and Chunk Stitch weights
     c00  1001-1121
     c01  1098-1218  seam 24
     c02  1195-1315  seam 24
     c03  1280-1400  seam 36
   frame 1098: c00 0.995, c01 0.005
   frame 1109: c00 0.530, c01 0.470
   frame 1121: c00 0.005, c01 0.995

all checks passed in 1.9 s
```

(Trimmed: the script prints every check.) The numbers to notice:

- **Part 2:** the second run takes milliseconds and the encode count stays at 1. One changed pixel re-encodes.
- **Part 3:** a dusk plate survives the SDR view almost perfectly. A neutral highlight even comes back in float up to
  about 100 (500 caps at 128), but a saturated warm highlight clips the red channel at about 2.0 and comes back as
  4.5 instead of 6. That's the round-trip warning you'll see in Prep's log on bright, colourful plates.
- **Part 4:** the 0.12 darkening that phase D fixed, reproduced.
- **Part 6:** the lock guarantee. Outside the mask, comp equals plate exactly.

Things to try (they're in the script as comments): multiply the identity "LTX" output by 1.2 inside the mask and
check the outside is still bit-exact; set `shadow_px` and `dilate_px` to 0; plan 1001-1400 at 97/16.

## 7. Gotchas, and what NOTES says went wrong

- **Green figures without keyframes.** The first integration pass used the start image only. The element held for
  about 2 s, then LTX painted `LTXVInpaintPreprocess`'s green fill into the silhouette (from about frame 1138, int8
  and bf16 alike; a 49-frame test looked fine only because it was short). The fix was the 2.5D comp as
  `LTXVAddGuide` keyframes every 24 frames at 0.8: greenness inside the element went from +0.13 to +0.26 down to
  0 or below on every sampled frame. Don't remove the keyframes to save time.
- **Small elements come back soft in `mode full`.** At 1920 wide the element is upscaled about 2.1x on return. On the
  relight starter it covered about 3 % of the frame and its face came back soft and drawn over. `mode crop` keeps it
  1:1 when it fits (50 of 121 frames on the S2-S3 test) and was clearly sharper, with no visible window seam.
- **The Relight LoRA kept the sun on frame right** on DSCF3000 whatever the light ball or prompt said. That's what
  `mirror` is for: Prep flips everything, you light from the right, Return flips the take back so the sun lands on
  the left. Add the light ball after Prep, on the mirrored frame. Sensor Apply refuses mirrored frames.
- **SDR clips.** LTX sees display-referred 0-1. Saturated highlights inside the mask come back clipped (exercise
  part 3). `mask_mode none` replaces the whole plate, so then they're lost everywhere; Prep's warning says so.
- **8k+1, and multiples of 32.** Prep raises on a length that isn't 8k+1; Pre-render quietly trims to 8k+1, as LTX's
  guide does.
- **Guide From Latent covers one case.** Whole clip from frame 0, `latent_downscale_factor` 1 (what the Relight and
  Alpha-Gen LoRAs declare). A guide whose latent height and width differ from the job's is refused ("pre-render the
  reference at the job's work size"). For anything else use LTX's image guide.
- **Pre-render's cache key is the frames, not the VAE.** Reuse checks the frame hash and the file hash. The sidecar
  records the VAE layout and dtype but doesn't compare them, so if you switch VAE on the same frames, change `name`
  or clear `latent_cache` first.
- **Don't push the full readout through a global-shutter solve.** The deck test (2026-10-21) showed the NukeX-style
  camera already absorbs most of the rolling-shutter skew; adding the physical 30.9 ms on top double counts. Use the
  RS-aware solve's camera, or keep RS off on LTX renders for those shots.
- **Triton breaks the LTX-2.5 VAE decode.** Starting ComfyUI with `--enable-triton-backend` makes the decode fail
  with `Triton Error [CUDA]: invalid argument`. Start without it for LTX work.
- **Crossfades hide seams only when takes agree.** Chunk Stitch blends; it can't reconcile two chunks that drew the
  element differently. Conditioning each chunk on the previous chunk's tail is the next step and isn't built yet.

## 8. Check yourself

<details>
<summary>Five questions (answers inside)</summary>

1. **Why does LTX Prep push the plate through the ACES 2.0 SDR view instead of handing LTX ACEScg?**
   LTX was trained on ordinary display video, so it expects display-referred 0-1. Return applies the exact inverse
   of the view Prep recorded to get back to ACEScg.

2. **You run a 121-frame integration pass and the element turns into a green furry shape after a couple of
   seconds. What's missing?**
   The keyframe guides: `keyframes_half` into `LTXVAddGuide` every 24 frames at about 0.8. The start image alone
   doesn't hold the element against the inpaint fill.

3. **What makes it impossible for an LTX pass to shift the camera track?**
   LTX only paints inside the projected-alpha mask; the Laplacian blend keeps the reference outside it, and Return
   writes alpha 0 there, so the plate under the layer is bit-exact. Placement comes from Card Project's 3D card, not
   from the model.

4. **Latent Pre-render returned `"cached": true` but you'd just swapped to a different VAE. Is the guide right?**
   No. The cache key is the frames (and the file), not the VAE. Clear the cache or change `name`, and wire the VAE
   through `LTXVVAEForceFloat32`.

5. **Chunk Plan, 1001-1400, max 121, overlap 24: why is the last seam 36 frames?**
   The last chunk is slid back so it ends exactly on 1400 rather than padding frames, so it starts at 1280 and
   overlaps the previous chunk (ending 1315) by 36.

</details>
