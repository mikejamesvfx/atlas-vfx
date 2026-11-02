# 03 OCIO compliance: one colour pipeline from the camera to Nuke

## In short

- **Four rules** (from AGENTS.md): read plates with the range from the stream; work in ACEScg; every display-referred
  hop is a named OCIO transform; every EXR names its colourspace in the header.
- **F-Log is not F-Log2.** F-Log mid-grey read as F-Log2 comes out about a stop bright (+0.97), and the error grows
  with brightness, so a Grade can't fix it. Plate Read takes the curve from the Fuji maker-note tag.
- **Range comes from the file** (`color_range`), never from an assumption. A wrong range barely moves mid-grey on a log
  plate but costs most of a stop up top.
- **A video model is a display-referred island.** LTX sees the ACES 2.0 SDR view and its output goes back through the
  exact inverse of that same view.
- **Generated elements come in through the inverse view too**, not as an sRGB texture, or they come back about 0.12
  darker in the mids.
- **EXRs carry their colourspace** (`colorInteropID` plus `atlas_vfx:ocio_colorspace`), and the Nuke Builder refuses
  any layer that doesn't match.

## 1. What it's for

A pixel in atlas-vfx crosses a lot of borders: Fuji F-Log codes in an HEVC file, scene linear for the comp, display
video for LTX, display video back again, an EXR, then Nuke. Each border is a chance to lose a stop or flatten the
contrast without anyone noticing.

"OCIO compliant" here means one boring thing: **at every border the conversion is a named transform from one OCIO
config, and the name is written down where the pixels land.** No hand-rolled gamma, no "looks about right" LUT, no
guessing what an EXR is.

## 2. The mental model

```
 camera file (F-Log codes, 10-bit, full range)
      |  Plate Read: range from the stream, F-Log vs F-Log2 from the Fuji tag
      v
 code values 0-1  --[ F-Log F-Gamut -> ACEScg ]-->  ACEScg (working space: Card Project, Grade, Relight, comp)
                                                       |                    ^
                       [ forward: sRGB - Display / ACES 2.0 - SDR 100 nits ]  [ same view, INVERSE ]
                                                       |                    |
                                                       +----> LTX ----------+
      v
 EXR layers: RGBA half, premultiplied ACEScg, tagged in the header
      v
 Nuke Builder: root OCIO + shipped config, plate Read in its log space, layers ACEScg
```

The idea to hold on to: a pixel the model didn't touch comes back where it started, because the trip off the island
is the exact inverse of the trip on.

## 3. How it works

### The config

`config/atlas_aces2.0_studio_flog2.ocio` is OCIO's ACES 2.0 studio config (`studio-config-v4.0.0_aces-v2.0_ocio-v2.5`)
plus two Fuji input spaces:

| Space | Gamut | 18 % grey code |
|---|---|---|
| `F-Log F-Gamut` | F-Gamut (BT.2020 primaries, D65), CAT02 to AP0 | 0.4593 |
| `F-Log2 F-Gamut` | same | 0.391 |

Each is a `LogCameraTransform` with data sheet constants (checked against colour-science) and a matrix to ACES2065-1.
`prototypes/vfx_probe/build_ocio_flog2.py` first built them. Roles: `scene_linear` is ACEScg, `compositing_log` and
`color_timing` are ACEScct. `camera_side/plate.py` pins the defaults (`WORKING_SPACE`, `DEFAULT_DISPLAY`,
`DEFAULT_VIEW`). A blank `ocio_config` widget means the shipped config (`plate.ocio_path`), which is why committed
workflows leave it blank.

### Plate in

`camera_side/plate.py:plate_ref` decides two things and records where each answer came from:

- **Range.** `pc`/`jpeg` is full, `tv`/`mpeg` is legal, RGB formats are full, a missing flag falls back to legal with
  a warning. `range_override` is for broken flags only, and gets logged as `override (stream says ...)`.
- **Input space.** `auto` reads Fuji tag 0x3803 (VideoRecordingMode) from the MOV's `moov` atom: 0x10 is F-Log,
  0x30 is F-Log2. HLG or Normal is refused. No tag falls back to the body default (F-Log2 for an X-H2) with a warning.

`plate.decode_rgb` does the H.273 maths in numpy rather than ffmpeg's swscale, which was about a code off and tinted
neutrals (a Y=300 neutral came back R 299.1, G 301.4, B 298.9). Then `plate.to_working` runs the plate's space to
ACEScg, and `plate.to_display` makes previews through the named view.

### The display island: LTX Prep and LTX Return

`layers/ltx_integration.py:view_processors` builds forward and inverse processors from one `DisplayViewTransform`.
Prep comps in ACEScg, shows LTX the forward view and records the display/view it used; `prep_processors` makes Return
invert **that** recorded view, so changing defaults later can't break an old pass. Return writes premultiplied ACEScg
with the feathered mask as alpha, so the plate outside the mask is untouched.

Prep also measures what the round trip costs on the plate (`roundtrip_report`) and warns past 0.5 % of pixels off by
more than 5 % (`ROUNDTRIP_CLIP_SHARE`). In float, neutral greys survive the SDR view up to about 64 ACEScg and clip
near 128; the real losses are bright saturated colours (gamut compression) and 8-bit storage. The report checks
luminance only, so a hue shift at the same brightness won't trip it.

### Generated elements

`layers/element_colour.py` is the 2026-11-01 flow review fix (phase D). Elements used to come in as
`sRGB Encoded Rec.709 (sRGB) -> ACEScg`, then Prep tone-mapped them again on the way out, so they came back dimmer
and flatter. The default `source_colour` is now `inverse display view (ACES 2.0 SDR, as LTX)`; `sRGB texture` is still
there. Foreground Element, Character Insert and Relight Transfer expose the choice.

### EXRs that say what they are

`camera_side/exr_io.py:write_rgba` writes RGBA half, zip, premultiplied, and stamps `oiio:ColorSpace` (OIIO 3.2 turns
it into `colorInteropID`, `lin_ap1_scene` for ACEScg), `atlas_vfx:ocio_colorspace` and `atlas_vfx:premultiplied`.
`check_sequence` reads headers only and refuses missing frames, wrong channels, non-half, wrong size or wrong
colourspace. It matters: the shipped config's file rule for `*.exr` says **ACES2065-1**, so an untagged ACEScg EXR
would be read as AP0 by anything trusting the rules.

### Nuke out

`deliver/nuke_builder.py` writes a Python script (and a `.nk` paste file for Indie) that sets the root to OCIO with the
shipped config and ACEScg working space, reads the plate in its own space, reads layers as ACEScg and writes ACEScg
EXRs. STMaps read as `Raw`, because UVs are data. Every value goes through `camera_side/nuke_text.py`.

### The speed path: ocio_fast

A CPU processor runs on one thread, about 3 s per 4K frame through the ACES 2.0 view. `camera_side/ocio_fast.py`
(`fast(proc)`) splits the frame into row bands on a thread pool (PyOpenColorIO releases the GIL), about 13x faster on
24 threads. OCIO still computes every pixel: `tests/test_ocio_fast.py` asserts bit-identical output. Small or float64
arrays, or `ATLAS_VFX_OCIO_THREADS=1`, take the plain path.

### Elsewhere

Plate Lock (`platelock/colour.py`) matches in ACEScct and refuses names not in the config. Look Transfer (catalogue
19, the log-space grade for studio elements) is **planned, not built**.

## 4. The nodes

Sockets and widgets are in `docs/NODE_CATALOGUE.md`; here's what each guarantees for colour.

| # | Node | Guarantee |
|---|---|---|
| 1 | Plate Read | records `range`, `colorspace` and where each came from |
| 15, 16 | Foreground Element, Character Insert | display picture into ACEScg the way LTX will see it |
| 23 | EXR Write | tagged RGBA half EXR, checked after writing |
| 44, 45 | LTX Prep, LTX Return | named SDR view out, its recorded inverse back, round-trip report in the sidecar |
| 46, 60 | Comp Preview, Viewer | any display/view of the config, for watching only |
| 24 | Nuke Builder | root OCIO, plate in its space, layers ACEScg, sequences checked first |

## 5. Walkthrough with shipped workflows

1. **`workflows/t2_plate_read_s2s3.json`.** Run it and read `info`: expect `range: full` from
   `stream color_range=pc` and `colorspace: F-Log F-Gamut` sourced from tag 0x3803. If it says "default for FUJIFILM
   X-H2", there's no tag and you're on the F-Log2 guess.
2. **`workflows/ltx_element_clean_loop_s2s3.json`.** Plate Read, LTX Prep, LTX, LTX Return. Open Return's sidecar for
   the round-trip report and the view Prep used.
3. **`workflows/t8_camera_export_nuke_builder_s2s3.json`.** The builder checks every layer, then writes the script.
   Paste it into Nuke and check the root colour management and each Read's colourspace.

## 6. Hands-on

```
python docs/learn/exercises/03_ocio_compliance.py
```

Needs PyOpenColorIO and OpenImageIO (both in the ComfyUI venv), no plate or GPU, about 1.5 s. The lines worth
staring at:

```
   F-Log  code 0.4593 read as F-Log2  -> 0.3522 ACEScg  (+0.97 stops)
      code 0.25: F-Log   0.0316  as F-Log2   0.0393  (+0.31 stops)
      code 0.70: F-Log   0.9666  as F-Log2   3.4741  (+1.85 stops)
    470  0.4594  +0.4635         0.1802          0.1855      +0.04 st
    940  0.9189  +1.0000         4.2275          7.2813      +0.78 st
     0.180   0.3492        0.180     0.00%       0.180    -0.00
   500.000   1.0000      128.000    74.40%     128.000    -1.97
   [ 8.00  0.20  0.05] -> [ 1.74  0.48  0.27]
   inverse display view (ACES 2.0 SDR, as LTX)   max err 0.0000, mean drift in the mids +0.000
   sRGB texture                                  max err 0.2527, mean drift in the mids -0.117
   refused: colorspace 'ACES2065-1' / 'lin_ap0_scene'
   ok   banded output is bit-identical (no LUT, no approximation)
```

In order: the wrong curve is +0.3 stops in the shadows and nearly +2 up top (contrast, not exposure); a range bug
barely moves grey (code 470) but hits highlights (940); neutrals round-trip cleanly while a hot saturated red comes
back much darker; the -0.117 mid drift is the "dimmer elements" bug; a mislabelled AP0 frame is refused; banding is
exact. Try the "try changing" comments: the `Un-tone-mapped` view, 10-bit display storage, one OCIO thread.

## 7. Gotchas, and the mistakes NOTES records

- **F-Log decoded as F-Log2 (fixed 2026-10-20).** DSCF3000 went through as the X-H2's F-Log2 default until the tag was
  read. Everything decoded before then had the mids about a stop bright with the wrong contrast, and had to be remade;
  solves from the tracking proxies were fine.
- **The `pc` mistake.** Early prototypes forced legal range on a full-range plate. Range now comes from the stream.
- **Making test files.** ffmpeg's `-color_range` silently converts (900 became 903), and ProRes can't carry a
  full-range flag, so `tests/test_plate.py` uses lossless HEVC for full range.
- **Stream colour tags lie.** DSCF3000 claims bt709; the input space comes from the Fuji tag or the body instead.
- **Nuke hides spaces without `file-io`.** Both Fuji spaces carry `categories: [file-io]` and the tests check it.
  Nuke reads a config once, so restart after changing it.
- **Open: the config in Nuke Indie.** NOTES (2026-11-01) says the shipped profile-2.5 config didn't load in the owner's
  Nuke Indie, with F-Log missing from the list, while an earlier T8 note says Nuke 17.1 accepted the build. Not
  diagnosed; an older OCIO inside Nuke than the config's profile is the first thing to check.
- **The DPX plate** went through swscale, so it carries the one-code error. Invisible in a comp, but there.
- **HDR back from the island.** The inverse view can't restore what the model never saw. `tools/hdr_spike.py` tests an
  SDR-to-HDR IC-LoRA against the plate's own ACEScg; nothing is built on it yet.

## 8. Check yourself

<details>
<summary>Five questions (answers inside)</summary>

1. **A Fuji MOV has no recording-mode tag. What does Plate Read do?**
   Uses the body default (F-Log2 for an X-H2), warns, and says so in `colorspace_source`. Set `input_space` by hand
   if you know better.

2. **Why can't a Grade fix an F-Log plate decoded as F-Log2?**
   The error grows with brightness (+0.3 to nearly +2 stops), so it's the wrong curve, not an offset. Re-decode.

3. **An element comes back from LTX dimmer than it was made. What's the suspect?**
   `source_colour` set to `sRGB texture`, which tone-maps it twice. Use the inverse display view.

4. **Why does LTX Return invert the view recorded by LTX Prep, not the current default?**
   So it always undoes exactly what Prep applied, even if defaults change between runs.

5. **The config's file rule says EXRs are ACES2065-1. Why doesn't that hurt the pack's layers?**
   Every EXR names its space in the header, `check_sequence` refuses mismatches, and the Nuke Builder sets each
   Read's colourspace explicitly.

</details>
