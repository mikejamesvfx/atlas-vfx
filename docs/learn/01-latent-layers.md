# 01 Latent layers: getting layers in and out of LTX without losing the lock

## In short

- **What it does:** hands your comp layers to LTX-2.5 so it can paint light, blur, grain and contact shadow onto
  them, then brings the take back as a premultiplied ACEScg EXR layer for Nuke.
- **The rule that keeps the lock:** LTX only paints inside a mask built from the layer's own projected alpha.
  Outside it, Return writes alpha 0, so the plate and its track come back bit-exact. Placement always comes from
  Card Project's 3D card, never from the model.
- **Colour:** LTX sees display video through a named OCIO view (ACES 2.0 SDR), and Return inverts that exact view.
  Saturated highlights clip in that trip; neutral greys survive.
- **Keyframes are not optional:** without `LTXVAddGuide` keyframes every 24 frames the element turns into green
  inpaint fill after about two seconds.
- **Numbers:** lengths are 8k+1 frames, sizes step in 64 px, the latent is 1/32 size with 128 channels (97 frames at
  1280x704 is `(1, 128, 13, 22, 40)`). Long shots run as equal chunks, 121 frames with 24 overlap.
- **Main gotcha:** Latent Pre-render's cache is keyed on the frames, not the VAE. Swap VAE on the same frames and it
  silently reuses the old latent.

Nodes covered: LTX Prep, LTX Return, Latent Pre-render, Guide From Latent, Chunk Plan / Read / Stitch, Tiny VAE
(`docs/NODE_CATALOGUE.md` blocks 44, 45, 50 to 52, 62, 65, 66). Exercise: `docs/learn/exercises/01_latent_layers.py`.

## 1. What it's for

You've tracked a plate and put a generated element on a card in 3D (Card Project). It's locked, but it looks pasted
on: wrong softness, no grain, no contact shadow. LTX can fix that, but a video model doesn't care about your camera
solve. Hand it a whole frame and it will repaint the plate and drift the element a few pixels a second.

So the system has one job: let LTX paint the look **while the camera lock stays ours**. The same plumbing runs the
relight passes (Relight IC-LoRA, Day-to-Night), the Alpha-Gen matte and shots longer than one LTX run.

"Latent encode" means the moments a picture becomes LTX's latent (the VAE encode): the reference video, the start
image, the keyframes and the stage-2 upscale. atlas-vfx decides what goes into each, in which colour space, at what
size, and (with Latent Pre-render) lets you encode the reference once and reuse it.

## 2. The mental model

```
 plate -> ACEScg -> [optional Sensor Pass: remove camera] -> comp layers over it
     -> named OCIO view (ACES 2.0 SDR, display 0-1)
     -> LTX PREP: reference, mask, start image, keyframes, record
     -> VAE encode (or Latent Pre-render's cached latent) -> LTX sampler
     -> VAE decode, Laplacian blend keeps the reference outside the mask
     -> LTX RETURN: inverse of the SAME view -> ACEScg, plate size, alpha = feathered mask
     -> [optional: apply camera back on] -> premult EXR layer over the plate in Nuke
```

Three rules hold it together:

1. **LTX sees ordinary video**, through a named display view, not scene-linear numbers.
2. **What goes out comes back through the exact inverse.** Return reads the view from Prep's record.
3. **The mask is the contract.** Inside, LTX's pixels; outside, alpha 0.

## 3. How it works in the code

**The layer going in.** A layer is a Card Project folder: one premultiplied ACEScg RGBA EXR per frame plus a
`*.render.json` with a `versions` map (`plate`, redistorted onto the plate, and `clean`, pinhole). LTX Return writes
the same kind of folder, so a returned take can feed another pass. Layer Picker (58) lists them.

**LTX Prep** (`layers/node_ltx_prep.py`, helpers in `layers/ltx_integration.py`). Per frame, `AtlasVFXLTXPrep.run`:

1. decodes the plate to ACEScg (`plate.decode_rgb`, `plate.to_working`) with Plate Read's recorded range and curve;
2. optionally takes the plate camera off (`ltx_integration.remove_camera`: rolling shutter, distortion, vignette) so
   LTX sees pinhole frames that match the cards;
3. comps the layers over, back to front (`card_project.over`), using the `clean` or `plate` version to match;
4. pushes the result through the view (`ltx_integration.to_view`, default display `sRGB - Display`, view
   `ACES 2.0 - SDR 100 nits (Rec.709)`) and resizes to the work size;
5. builds the mask (`make_mask`): alpha above `mask_thresh` 0.05, smeared **down** by `shadow_px` 24 for a contact
   shadow, dilated by `dilate_px` 12;
6. on frame 0 runs `roundtrip_report` (view and inverse on the real plate) and records the error.

It outputs `reference` / `reference_half`, `mask` / `mask_half`, `start_image` (frame 0 with every layer), and
`keyframes_half` (the comp every `keyframe_every` 24 frames plus the last), plus the `ltx_prep` record for Return.

`mode full` scales the whole plate to the work size. `mode crop` cuts a window that follows the element, never
smaller than the work size (so the element is 1:1 when it fits), smoothed over plus or minus 8 frames because a
jittery crop reads to LTX as camera shake.

**What gets encoded, and when.** The shipped clean-loop graph (`subgraphs/ltx_clean_loop.json`, same nodes as
`workflows/ltx_element_clean_loop_s2s3.json`) runs Lightricks' two-stage inpaint recipe:

| stage | picture | encoded by | strength |
|---|---|---|---|
| 1 (960x512, 8 steps) | `start_image`, half | `LTXVImgToVideoInplace` | 0.7 |
| 1 | `reference_half` greened under `mask_half` | `LTXAddVideoICLoRAGuideAdvanced` | 1.0 |
| 1 | `keyframes_half` at 24, 48, 72, 96, 120 | five `LTXVAddGuide` | 0.8 each |
| 2 (1920x1024, 2 steps) | stage 1's blended, upscaled decode | `VAEEncodeTiled` | the latent being refined |
| 2 | `start_image`, full | `LTXVImgToVideoInplace` | 1.0 |

`LTXVCropGuides` strips the guide frames before decode, and after each stage `LTXVLaplacianPyramidBlend` puts the
reference back outside the mask.

**Latent Pre-render and Guide From Latent (65, 66).** `AtlasVFXLatentPrerender.run` (`layers/node_latent_prerender.py`)
trims to 8k+1, hashes the frames at 16 bits a channel, and if `<name>.latent` and its sidecar exist with a matching
frame hash and file sha256, loads it without calling the VAE (`"cached": true`). Otherwise it encodes, stores float32
safetensors and a sidecar. Wire the VAE through `LTXVVAEForceFloat32` first. `AtlasVFXGuideFromLatent.run` then calls
LTX's own guide code (`LTXVAddGuide.append_keyframe`, `append_guide_attention_entry`) on the cached latent. NOTES
(2026-10-31): on DSCF3000 the image guide and the cached latent gave LTX Return EXRs with a max difference of 0.0.
The point isn't speed, the encode is a small part of a run: it's that every machine (ws, the farm, Comfy Cloud)
starts from the same few MB of latent.

**LTX Return** (`layers/node_ltx_return.py`). Per frame: undo the resize and any mirror, get the inverse processors
from the record (`prep_processors`), `return_frame` or `return_window` (inverse view to ACEScg, plate size, alpha =
feathered mask, crop windows pasted back with ramped edges), optionally put the plate camera back on
(`apply_camera`), then write RGBA half premultiplied ACEScg EXRs plus `render.json` and a sidecar.

**Phase D, elements through the inverse view.** Generated elements arrive as display pictures.
`layers/element_colour.py` used to treat them as an sRGB texture, which Prep then tone-mapped again, so they reached
LTX 0.12 darker in the mids (flow review 2026-11-01). The default `source_colour` is now `inverse display view
(ACES 2.0 SDR, as LTX)`: what you saw on screen is what LTX sees.

**Chunks** (50 to 52, `layers/chunks.py`). `chunks.plan` makes equal chunks with `stride = length - overlap` and
slides the last one back to end exactly on the last frame. 1001-1400 at 121/24 gives 1001-1121, 1098-1218,
1195-1315, 1280-1400. One graph serves every chunk (set `chunk` to -1, or use `tools/queue_chunks.py`), and Chunk
Stitch blends with smoothstep weights that sum to 1, in scene-linear for EXR.

**Tiny VAE** (62) loads `taeltx2_3` (MIT) for quick blurry previews, including live sampler previews
(`ATLAS_VFX_LTX_LIVE_PREVIEW=0` turns that off).

**Why the layers stay locked:** placement comes from Card Project's tracked 3D card; outside the mask is the
reference twice over (graph blend and Return's alpha 0); resizes and crops are undone from the record; and the lens
and rolling shutter are handled outside LTX (remove then apply round trips to 0.0001). NOTES (2026-10-19): through
the fast blur around frame 1190 the element blurs with the plate and its feet stay where the card put them.

## 4. The nodes

The widgets you actually set (full lists in the catalogue):

| # | node | set these | gives you |
|---|---|---|---|
| 44 | LTX Prep | `plate_ref`, `layer_folder`, `first_frame`, `length` (8k+1), `width` / `height`, `mode` full or crop, `reference` plate or comp, `mask_mode` layer or none, `mirror` | reference, mask, start image, keyframes, `ltx_prep` |
| 45 | LTX Return | `images`, `ltx_prep`, `name` | EXR layer folder, preview |
| 65 | Latent Pre-render | `images` (reference at work size), `vae`, `name` | `latent`, `path`, `info` |
| 66 | Guide From Latent | conditionings, empty video latent, `guide`, `strength`, `latent_downscale_factor` (from the IC-LoRA loader, must be 1) | conditionings, latent |
| 50 / 52 | Chunk Plan / Stitch | `first`, `last`, `max_len`, `overlap`, `chunk`; Stitch's `pattern` with `{tag}` | per-chunk start/length/tag; stitched pattern |

## 5. Walkthrough: the shipped graphs

**Integration pass, `workflows/ltx_element_clean_loop_s2s3.json`.** From LTX Prep (node 100): `reference_half` and
`mask_half` into `LTXVInpaintPreprocess` (110) and the IC-LoRA guide (115); `start_image` into
`LTXVImgToVideoInplace` (114) on an empty 960x512x121 latent; `keyframes_half` split into five `LTXVAddGuide`s (171
to 175). Stage 1 samples, crops guides, decodes and blends (126); stage 2 upscales, encodes (131), refines two steps
and blends with the full reference (142). LTX Return (150) writes `layers/element_ltx`.

**Relight pass, `workflows/relight_tracked_dscf3000_s2s3.json`.** Prep runs `reference comp`, `mask_mode none` (a full
replacement plate). Sun Direction picks the light and `mirror`, Light Ball draws onto Prep's reference, which goes
through **Latent Pre-render** (60) and **Guide From Latent** (43). `workflows/ltx_relight_mirror.json` is the same
pass with LTX's own image guide: that pair is what the bit-identical check compared.

Long shots: `workflows/chunked_sequence_starter.json` and `chunked_sequence_stitch.json`.

## 6. Hands-on

```
python docs/learn/exercises/01_latent_layers.py
```

No plate, weights or GPU: a fake VAE with LTX's layout and an identity "LTX" that hands back its reference. About
2 s. The lines worth reading:

```
   ok   17 frames at 1280x704 -> latent (1, 128, 3, 22, 40)
   second run: cached=True, 4.6 ms, encodes so far 1
   ok   one changed pixel: the hash differs, so it encodes again
   dusk plate: median error 0.0009 %, p99 0.0025 %, share over 5 % 0.00 %, max luminance 0.33
   neutral   20.0 -> view 0.9803 -> back    19.99
   warm (6.0, 2.4, 0.8) -> view [1.     0.8358 0.75  ] -> back [4.53 2.36 1.07]
   ok   sRGB texture: tone-mapped twice, so it reaches LTX darker in the mids  (mean 0.124 darker)
   element rows 100-149; with shadow_px 20 the mask covers rows 100-169
   ok   plate untouched outside the mask (bit-exact)
     c03  1280-1400  seam 36
   frame 1109: c00 0.530, c01 0.470
all checks passed in 1.9 s
```

Notice: the cache reuses in milliseconds and one changed pixel re-encodes; a neutral highlight survives the SDR trip
but a saturated warm one clips (that's Prep's round-trip warning); the 0.12 phase D darkening is reproduced; outside
the mask, comp equals plate exactly. The script's comments suggest things to try (scale the identity "LTX" inside the
mask, zero `shadow_px`, plan at 97/16).

## 7. Gotchas, and what NOTES says went wrong

- **Green figures without keyframes.** Start image alone held the element about 2 s, then green fill crept in (from
  about frame 1138). Keyframes every 24 at 0.8 took greenness from +0.13 to +0.26 down to 0 or below.
- **Small elements go soft in `mode full`** (about 2.1x upscale on return). `mode crop` keeps them 1:1 when they fit.
- **The Relight LoRA kept the sun on frame right** on DSCF3000. Use `mirror`: light from the right, Return flips it
  back. Add the light ball after Prep. Sensor Apply refuses mirrored frames.
- **SDR clips saturated highlights** inside the mask, and everywhere with `mask_mode none`.
- **8k+1 and multiples of 32.** Prep raises on a bad length; Pre-render quietly trims.
- **Guide From Latent covers one case:** whole clip from frame 0, downscale factor 1, latent size matching the job.
- **Pre-render's cache key is the frames, not the VAE.** Switch VAE on the same frames and it reuses the old latent:
  change `name` or clear `latent_cache` first.
- **Don't push the full rolling-shutter readout through a global-shutter solve** (deck test 2026-10-21: it double
  counts). Use the RS-aware solve's camera or keep RS off.
- **`--enable-triton-backend` breaks the LTX-2.5 VAE decode.** Start ComfyUI without it.
- **Chunk Stitch only hides seams when takes agree.** Conditioning a chunk on the previous tail isn't built yet.

## 8. Check yourself

<details>
<summary>Five questions (answers inside)</summary>

1. **Why does Prep give LTX the SDR view and not ACEScg?** LTX was trained on display video. Return inverts the exact
   view Prep recorded.
2. **The element goes green after two seconds. What's missing?** The `LTXVAddGuide` keyframes every 24 frames at
   about 0.8.
3. **What stops an LTX pass shifting the track?** It only paints inside the projected-alpha mask; outside, the blend
   and Return's alpha 0 keep the plate bit-exact. Placement comes from the 3D card.
4. **Pre-render said cached, but you'd swapped VAE. Is it right?** No. The key is the frames, not the VAE. Clear the
   cache or change `name`.
5. **1001-1400 at 121/24: why is the last seam 36?** The last chunk slides back to end on 1400, so it starts at 1280
   and overlaps the one ending 1315 by 36.

</details>
