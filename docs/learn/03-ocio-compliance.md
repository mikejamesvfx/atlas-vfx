# 03 OCIO compliance: one colour pipeline from the camera to Nuke

## 1. What it's for

Every pixel in atlas-vfx crosses a lot of borders. It starts as Fuji F-Log code values in an HEVC file, becomes scene
linear for the comp, gets shown to a video model as ordinary display video, comes back from that model as display
video again, lands in an EXR, and finally gets read by Nuke. Each border is a place where colour quietly goes wrong:
a stop of exposure here, a flattened contrast there, an element that's a touch dimmer every time it goes round.

"OCIO compliant" in this repo means one boring thing: **at every one of those borders, the conversion is a named
transform from one OCIO config, and the name is written down where the pixels land.** No hand-rolled gamma, no
"looks about right" LUT, no guessing what an EXR is. If you can read the sidecar or the EXR header, you know exactly
what the numbers mean and how they got there.

The rules, as AGENTS.md puts them:

| Rule | What it means in practice |
|---|---|
| Read plates with the range from the stream | Full vs legal range comes from ffprobe's `color_range`, not from an assumption |
| Working space ACEScg | All comp maths (overs, grades, relight gains) happens in linear AP1 |
| Every display-referred hop is a named OCIO transform | Going to or from "video" always means a display/view of the config, forward or inverse |
| EXRs carry their colourspace | Every EXR header names its space, twice, and the Nuke Builder refuses ones that don't match |

## 2. The mental model

```
 camera file (F-Log codes, 10-bit, full range)
      |  Plate Read: range from the stream, F-Log vs F-Log2 from the Fuji tag
      v
 normalised code values 0-1  --[ 'F-Log F-Gamut' -> ACEScg ]-->  ACEScg (scene linear, the working space)
                                                                   |            ^
                         Card Project, Grade, Relight, Comp ...    |            |
                                                                   v            |
                         [ ACEScg -> sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709) ]   (forward view)
                                                                   |            ^
                                                         LTX / any video model  |
                                                                   |            |
                         [ same display/view, INVERSE ] -----------+------------+
      |
      v
 EXR layers: RGBA half, premultiplied ACEScg, colorInteropID lin_ap1_scene + atlas_vfx:ocio_colorspace "ACEScg"
      |
      v
 Nuke Builder: root set to OCIO + the shipped config, plate Read in F-Log F-Gamut, layer Reads in ACEScg, Write ACEScg
```

The one idea worth holding on to: **a video model is a display-referred island.** It sees display pictures and hands
back display pictures. The repo walks pixels onto that island through the ACES 2.0 SDR view and walks them off through
the exact inverse of the same view, so a pixel that the model didn't touch comes back where it started.

## 3. How it works

### The config

`config/atlas_aces2.0_studio_flog2.ocio` is OCIO's built-in ACES 2.0 studio config
(`studio-config-v4.0.0_aces-v2.0_ocio-v2.5`) plus two colour spaces it doesn't ship:

| Space | Curve | Gamut | 18 % grey code |
|---|---|---|---|
| `F-Log F-Gamut` | Fuji F-Log (the original) | F-Gamut (BT.2020 primaries, D65), CAT02 to AP0 | 0.4593 |
| `F-Log2 F-Gamut` | Fuji F-Log2 | same | 0.391 |

Both are a `LogCameraTransform` (data sheet constants, checked against colour-science's `log_decoding_FLog` and
`log_decoding_FLog2`) followed by a matrix to ACES2065-1. `prototypes/vfx_probe/build_ocio_flog2.py` is the script
that first built the F-Log2 one; it prints the curve error against colour-science (NOTES records 1e-6 above the toe,
about 1e-5 absolute in the toe for F-Log).

The roles that matter: `scene_linear` is **ACEScg** (the working space), `compositing_log` and `color_timing` are
**ACEScct**. The default display/view is `sRGB - Display` / `ACES 2.0 - SDR 100 nits (Rec.709)`, and the code pins
the same pair in `camera_side/plate.py` (`DEFAULT_DISPLAY`, `DEFAULT_VIEW`, `WORKING_SPACE`).

Every `ocio_config` widget in the pack defaults to blank, which means "the shipped config" (`plate.ocio_path`). That's
also why committed workflows leave it blank: they run on anyone's machine.

### Plate in: range and curve from the file

`camera_side/plate.py:plate_ref` probes the movie and decides two things, and records where each answer came from:

- **Range.** `color_range=pc` or `jpeg` means full, `tv` or `mpeg` means legal, RGB pixel formats are full, and a
  missing flag falls back to legal with a warning. `range_override` (`full` / `legal`) exists only for files with a
  broken flag; the override is logged and written into the record as `override (stream says ...)`.
- **Input space.** `input_space` `auto` reads the FujiFilm maker note tag 0x3803 (VideoRecordingMode) out of the MOV's
  `moov` atom, never the media: 0x10 is F-Log, 0x30 is F-Log2 (`FUJI_RECORDING_MODE`, `FUJI_MODE_SPACE`). HLG or
  Normal footage is refused. No tag at all falls back to the camera body default (`F-Log2 F-Gamut` for an X-H2) with
  a warning.

Decoding (`plate.decode_rgb`) is done in numpy, not by ffmpeg's swscale. ffmpeg only hands over the native Y'CbCr
planes; the H.273 maths (`_norm_luma`, `_norm_chroma`), the matrix and the chroma siting are applied here. Why: swscale's
10-bit full-range conversion was about a code off and tinted neutrals (measured 2026-10-19: a Y=300 neutral came back
as R 299.1, G 301.4, B 298.9). The decode gives normalised code values in 0-1, which is exactly what an OCIO log input
space expects.

Then it's one call: `plate.to_working(img, ref)` runs `ref["colorspace"] -> ACEScg` through the config, and
`plate.to_display(img, ref)` runs the plate's space through the named display/view for previews.

### The display island: LTX Prep and LTX Return

`layers/ltx_integration.py:view_processors` builds a pair of OCIO processors from one `DisplayViewTransform`: forward
(ACEScg to the SDR view) and inverse (the same transform with `TRANSFORM_DIR_INVERSE`). LTX Prep comps the layer over
the plate in ACEScg and shows LTX the forward view. It also records the display and view it used in its
`ATLAS_VFX_LTX_PREP` record, and `prep_processors` makes LTX Return invert **that** recorded view, so changing the
defaults later can't break an old pass.

Return then upscales, inverts the view back to ACEScg, and writes a premultiplied ACEScg RGBA EXR layer whose alpha is
the feathered mask. Outside the mask alpha is 0, so the plate is untouched.

The inverse view isn't magic, so Prep measures what it costs on the plate itself (`roundtrip_report`: ACEScg to view
to inverse, relative luminance error, the share of pixels off by more than 5 %). If more than 0.5 % of the plate is
off (`ROUNDTRIP_CLIP_SHARE`), `roundtrip_warnings` says so, and louder when `mask_mode none` replaces the whole plate.
The exercise shows where the losses really are: in float, neutral greys survive the ACES 2.0 SDR view up to about 64
in ACEScg, values past display 1.0 clip near 128, and bright saturated colours (a sunset sky, a red light) come back
desaturated and much darker because the view compresses gamut. Storing the display frames as 8-bit costs a bit more
on top: shadows below about 0.002 round to zero and the top end drifts a tenth of a stop or so.

### Generated elements: the inverse view, not "sRGB texture"

`layers/element_colour.py` is the fix from the 2026-11-01 flow review (phase D). Generated pictures (an element, a
character) used to come in as a flat texture: `sRGB Encoded Rec.709 (sRGB) -> ACEScg`. That's fine for a texture,
but LTX Prep then tone-maps it **again** on the way out, so elements came back dimmer and flatter than they were made
(NOTES: about 0.12 darker in the mids after LTX). Now the default `source_colour` is
`inverse display view (ACES 2.0 SDR, as LTX)`: what the element looked like on screen is what it looks like after the
round trip. The old path is still there as `sRGB texture` if you want it. Foreground Element, Character Insert and
Relight Transfer all expose the choice.

### EXRs that say what they are

`camera_side/exr_io.py:write_rgba` writes RGBA half, zip, premultiplied by default, and stamps the colourspace twice:
`oiio:ColorSpace` (which OIIO 3.2 turns into the OpenEXR standard `colorInteropID`, `lin_ap1_scene` for ACEScg) and
`atlas_vfx:ocio_colorspace` in plain words, plus `atlas_vfx:premultiplied`. `check_sequence` reads headers without
pixels and refuses a sequence with a missing frame, the wrong channels, a non-half format, the wrong size or the wrong
colourspace. The Nuke Builder runs it on every layer before it writes a single line.

Why bother when the config has file rules? Look at them: the shipped config's rule for `*.exr` says **ACES2065-1**.
An untagged ACEScg EXR read by anything trusting the rules would be read as AP0. The header is the truth, not the
extension.

### Nuke out

`deliver/nuke_builder.py` writes a Python script (and a `.nk` paste file for Nuke Indie) that, with `set_root` on,
switches the root to OCIO with `customOCIOConfigPath` = the shipped config and `workingSpaceLUT` = ACEScg, reads the
plate in its own space (e.g. `F-Log F-Gamut`), reads every layer as ACEScg, merges them over back to front, and sets
up an ACEScg EXR Write. STMap lens reads are set to `Raw` with `raw true`, because UV maps are data, not colour.
Every value goes through `camera_side/nuke_text.py`, never string pasting.

### The speed path: ocio_fast

An OCIO CPU processor runs on one thread, and the ACES 2.0 output transform costs about 3 s per 4K frame that way.
`camera_side/ocio_fast.py` wraps the processor (`fast(proc)` returns a `Banded`), cuts the frame into row bands of
about 130k pixels and runs them on a thread pool. PyOpenColorIO releases the GIL inside `applyRGB`, so this scales.
It's not a LUT or a GPU approximation: OCIO computes every pixel, and `tests/test_ocio_fast.py` asserts the output is
bit-identical to the single-thread processor on a 4K HDR frame with negatives and highlights. The docstring records
0.22 s per 4K frame on 24 threads (about 13x). Arrays under 65k pixels, float64, non-contiguous arrays, or
`ATLAS_VFX_OCIO_THREADS=1` take the plain path. Every processor helper in the pack (`plate._cpu`,
`ltx_integration.view_processors`, `element_colour`) hands out the banded wrapper.

### Elsewhere

Plate Lock (`platelock/colour.py`) matches frames in ACEScct (the `compositing_log` role) and refuses colour space
names that aren't in the config. The Grade node (catalogue 59) is Nuke's Grade maths on linear images. Look Transfer
(catalogue 19, the log-space grade for inserted studio elements) is **planned for Phase 2, nothing written yet**.

## 4. The nodes

| # | Node | Colour sockets and widgets | What it guarantees |
|---|---|---|---|
| 1 | Plate Read | `ocio_config` (blank = shipped), `input_space` (auto), `range` auto/full/legal | `plate_ref` records `range`, `range_source`, `colorspace`, `colorspace_source`, `working_space` |
| 23 | EXR Write | `input_colorspace`, `colorspace` (ACEScg), `input_premultiplied` | RGBA half EXR with `colorInteropID` + `atlas_vfx:ocio_colorspace`, checked after writing |
| 15, 16 | Foreground Element, Character Insert | `source_colour`: inverse display view / sRGB texture | display picture into premultiplied ACEScg the way LTX will see it |
| 44 | LTX Prep | uses the plate's config; `reference plate/comp`, `mask_mode` | plate in ACEScg, shown through the named SDR view; the view is recorded |
| 45 | LTX Return | `ltx_prep` record | inverts the recorded view, writes ACEScg premultiplied, sidecar holds the round-trip report |
| 46 | Comp Preview | uses the plate's config | over in ACEScg, then the SDR view, for watching only |
| 60 | Viewer | `view`, `exposure`, `gamma`, `ocio_config` | any display/view of the shipped config, exposure and gamma for the view only |
| 24 | Nuke Builder | `set_root`, `lens_delivery` | root OCIO, plate in its space, layers ACEScg, sequences checked first |

## 5. Walkthrough with shipped workflows

1. **`workflows/t2_plate_read_s2s3.json`.** One Plate Read: frames 129-296, `ocio_config` blank, `input_space` auto,
   `range` auto, frame offset 1001. Run it and read the `info` output: you should see `range: full` with
   `range_source: stream color_range=pc`, and `colorspace: F-Log F-Gamut` with a source that names tag 0x3803. If
   it says "default for FUJIFILM X-H2", the file has no recording-mode tag and you're on the F-Log2 guess: check it.
   The preview batch is already through the named view.
2. **`workflows/ltx_element_clean_loop_s2s3.json`.** Plate Read feeds LTX Prep (work size 1920x1024, mask 0.05,
   24 px shadow room, 12 px dilate), LTX renders, and LTX Return (feather 3.0) writes `element_ltx`. Open the Return
   sidecar: it holds the colour round-trip report for the plate, and the view Prep used.
3. **`workflows/t8_camera_export_nuke_builder_s2s3.json`.** Plate Read, Camera Import, Measure & Scale, Camera Export
   and the Nuke Builder, over layer folders made earlier (by Card Project, LTX Return or EXR Write). The builder runs
   `check_sequence` on each layer first. Paste the script (or the `.nk`) into Nuke and check the root's colour
   management, the plate Read's colourspace and the layer Reads.

## 6. Hands-on

Run from the repo root with the ComfyUI venv python (it needs PyOpenColorIO and OpenImageIO):

```
python docs/learn/exercises/03_ocio_compliance.py
```

No plate, no model, no GPU; it writes three tiny EXRs into a temp folder and deletes them. This is the output on the
owner's workstation (1.4 s):

```
2. 18 % grey: F-Log vs F-Log2 (data sheet grey codes 0.4593 and 0.391)
   F-Log  code 0.4593 read as F-Log   -> 0.1800 ACEScg
   F-Log2 code 0.391  read as F-Log2  -> 0.1800 ACEScg
   F-Log  code 0.4593 read as F-Log2  -> 0.3522 ACEScg  (+0.97 stops)
      code 0.25: F-Log   0.0316  as F-Log2   0.0393  (+0.31 stops)
      code 0.35: F-Log   0.0779  as F-Log2   0.1188  (+0.61 stops)
      code 0.55: F-Log   0.3441  as F-Log2   0.8410  (+1.29 stops)
      code 0.70: F-Log   0.9666  as F-Log2   3.4741  (+1.85 stops)

3. The same 10-bit codes decoded as full range and as legal range (plate._norm_luma, H.273)
   code   full     legal    F-Log lin(full)  F-Log lin(legal)  error
     64  0.0626  +0.0000        -0.0035         -0.0106      (toe)
    470  0.4594  +0.4635         0.1802          0.1855      +0.04 st
    940  0.9189  +1.0000         4.2275          7.2813      +0.78 st
   1019  0.9961  +1.0902         7.0931         13.3142      +0.91 st

4. ACEScg -> ACES 2.0 SDR view -> inverse (what a video model sees, and what comes back)
   ACEScg    display   back (float)  rel err   back (8-bit)  stops off
     0.002   0.0003        0.002     0.00%       0.000    lost (rounds to 0)
     0.180   0.3492        0.180     0.00%       0.180    -0.00
    16.000   0.9747       15.995     0.03%      17.059    +0.09
   100.000   0.9990      100.213     0.21%     128.000    +0.36
   500.000   1.0000      128.000    74.40%     128.000    -1.97
   saturated colours (ACEScg in -> back)
   [ 8.00  0.20  0.05] -> [ 1.74  0.48  0.27]
   [ 6.00  1.20  0.15] -> [ 2.47  1.05  0.43]
   dusk plate      median 1.14e-05, share off >5 % 0.000%, warning: no
   same + hot sky  median 1.20e-05, share off >5 % 5.729%, warning: yes

5. A generated (display) picture into ACEScg, then out through LTX Prep's view
   inverse display view (ACES 2.0 SDR, as LTX)   max err 0.0000, mean drift in the mids +0.000
   sRGB texture                                  max err 0.2527, mean drift in the mids -0.117

6. EXRs carry their colourspace
   the config's own file rule for '*.exr': ('ACES2065-1', 0)
   header: format half, ... colorInteropID 'lin_ap1_scene', atlas_vfx:ocio_colorspace 'ACEScg', premultiplied 1
   refused: colorspace 'ACES2065-1' / 'lin_ap0_scene'

7. ocio_fast: same processor, row bands on a thread pool
   2K frame through the ACES 2.0 SDR view: 746 ms single thread, 57 ms banded on 22 threads
   ok   banded output is bit-identical (no LUT, no approximation)

PASS  (1.4 s)
```

Things to notice:

- Step 2: the wrong curve isn't a flat exposure offset. It's +0.3 stops in the shadows and nearly +2 stops up top,
  so the contrast is wrong too. You can't fix it with a gain.
- Step 3: mid-grey barely moves between full and legal (+0.04 stops), but the top end moves most of a stop. A "range
  bug" on a log plate mostly hits highlights.
- Step 4: in float, the ACES 2.0 SDR view is far more invertible than you'd guess for neutrals. The real losses are
  past display 1.0, in bright saturated colours, and in 8-bit storage. That's why Prep measures the plate rather than
  trusting a rule of thumb.
- Step 5: the `-0.117` mid drift is the "elements came back dimmer" bug, reproduced in a few lines.

The script has "try changing" comments: swap the view to `Un-tone-mapped`, store the display frames at 10 bits
instead of 8, or set `ATLAS_VFX_OCIO_THREADS=1`.

## 7. Gotchas, and the mistakes NOTES records

- **The F-Log / F-Log2 mix-up (fixed 2026-10-20).** DSCF3000 was decoded as F-Log2 (the X-H2 body default) until the
  maker-note tag was read: tag 0x3803 = 0x10 is F-Log. F-Log's mid-grey 0.4593 read as F-Log2 gives 0.352 instead
  of 0.18, so everything decoded before then had the plate about a stop bright in the mids with the wrong contrast. A
  character's +1.5 stop exposure had been matching that. Layers, depth, proxies, LTX passes and comps from a decoded
  plate had to be remade; camera solves from the tracking proxies (log luma + CLAHE) were fine. On the dusk plate the
  re-run moved the comp only 0.15 stops near the character, because that scene sits about 3 stops under grey, where
  the two curves are close.
- **The `pc` mistake.** The first prototype scripts forced `in_range=tv`, but DSCF3000 is full range (`pc`), so every
  code was stretched. Now the range comes from the stream, and the override is recorded.
- **Making test files.** Passing `-color_range` to ffmpeg silently inserts a range conversion (measured 900 to 903).
  ProRes can't carry a full-range flag at all, so `tests/test_plate.py` uses lossless HEVC Rext 4:2:2 10-bit (the
  X-H2's own format) for full range and ProRes only for legal.
- **Stream colour tags lie.** DSCF3000's stream says bt709 transfer and primaries. They're ignored; the input space
  comes from the Fuji tag or the body.
- **Nuke's Read menu hides spaces without `file-io`.** The first F-Log2 space had no categories, so Nuke didn't list
  it (T8, 2026-10-19). Both Fuji spaces now carry `categories: [file-io]`, and `test_plate.py` checks every active
  input space does. Nuke reads a config once: restart it after changing the config.
- **Open: the config in Nuke Indie.** NOTES (2026-11-01) records that the shipped config (profile 2.5) didn't load in
  the owner's Nuke Indie, with F-Log missing from the colourspace list, and that's parked. Why isn't diagnosed in the
  repo; an older OCIO in Nuke than the config's 2.5 profile would be the first thing to check.
- **The DPX plate in Nuke.** The DPX conversion went through swscale, so it carries the ~1-code error; the layers were
  rendered against the numpy decode. Not visible in a comp, but it's there.
- **HDR back from the island.** The inverse view can't bring back what the model never saw. `tools/hdr_spike.py`
  compares an SDR-to-HDR IC-LoRA against the plate's own ACEScg as ground truth, next to the plain inverse view.
  It's a spike: nothing is built on it until the numbers say so.

## 8. Check yourself

<details>
<summary>Five questions (answers inside)</summary>

1. **A Fuji MOV has no recording-mode tag. What does Plate Read do?**
   Falls back to the camera body default (F-Log2 F-Gamut for an X-H2), logs a warning, and writes "default for ...
   (file has no recording-mode tag)" into `colorspace_source`. Set `input_space` by hand if you know better.

2. **Why can't you fix an F-Log plate decoded as F-Log2 with a Grade?**
   The error grows with brightness (about +0.3 stops in the shadows to nearly +2 at the top), so it's a different
   curve, not an exposure offset. Re-decode with the right input space.

3. **An element looks right in the generator but comes back dimmer from LTX. Which setting is the suspect?**
   `source_colour` on Foreground Element or Character Insert set to `sRGB texture`. That path tone-maps the element
   twice; the inverse display view doesn't.

4. **Why does LTX Return invert the view stored in the LTX Prep record instead of the current default?**
   So it always undoes exactly what Prep applied, even if the defaults change between the two runs.

5. **The shipped config's file rule says EXRs are ACES2065-1. Why is that not a problem for this pack's layers?**
   Every EXR the pack writes names its space in the header (`colorInteropID lin_ap1_scene` and
   `atlas_vfx:ocio_colorspace ACEScg`), `check_sequence` refuses anything that doesn't match, and the Nuke Builder
   sets each Read's colourspace explicitly.

</details>
