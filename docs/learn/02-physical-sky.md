# 02 Physical Sky: a real sun and sky from where and when you shot

Hands-on: `docs/learn/exercises/02_physical_sky.py` (under a second, no plate, no GPU, no Atlas Scene).

## In short

- **Physical Sky** (catalogue 48) turns a plate's GPS and capture time into a scene-linear ACEScg sky HDRI, a light
  rig Relight can use, and a preview. It's a clear-sky model: no clouds, no set bounce.
- **Two sky models.** Prague belongs to Atlas Scene and is reached only through `_atlas_scene.py` (optional, renders
  twilight). Hosek-Wilkie ships with the repo (`layers/sky_hosek.py`, sun above 0 deg only). `auto` picks Prague when
  it can and says why in `info`.
- **GPS plus time give the sun** through NOAA's algorithm in `layers/solar.py`. The time must carry its offset
  (`Z` or `+11:00`): an hour of doubt is 15 deg of sun.
- **Pixels are physical radiance**, unclipped float32, in ACEScg via a named OCIO hop. A sun disc reads in the
  millions. The light rig is normalised so the sky on an up-facing surface is 1 and `sun_strength` is the key against it.
- **A tracked camera doesn't know north.** You type `heading_deg` for its first frame, and the sun direction is only
  as good as that number.
- **Main gotchas:** time zones, small HDRIs that miss the sun disc (use the rig for the key), and Hosek's sun
  being far too bright within a few degrees of the horizon.

## 1. What it's for

The camera wrote GPS and a timestamp, which is enough to know exactly where the sun was. Physical Sky turns that
into an **HDRI** (float32 ACEScg lat-long EXR plus sidecar), a **light rig** (`ATLAS_VFX_LIGHT_RIG`: sun direction
per frame in camera space, its colour and strength against the sky) and an OCIO **preview** with the sun ringed and
the camera's field of view marked.

Think of it as the sun-and-sky half of a lighting reference, made from metadata instead of a chrome ball. What it
gives you that a chrome ball often can't is the sun's exact direction and its strength relative to the sky, at any
hour (`time_offset_h`). Sun Direction (56) and Light Ball (57) then turn that rig into words and a ball for the LTX
relight.

## 2. The mental model

```
GPS + capture time (with offset) [+ time_offset_h]
        │  layers/solar.py (NOAA, geometric)
        ▼
sun azimuth / elevation ── sun_mode: metadata | offset | manual
        │
   ┌────┴──────────────────────┐
   ▼                           ▼
sky model                    camera (typed, or a tracked shot camera)
prague | hosek-wilkie          │ sun_in_camera(): per frame, x right, y up, looking down -Z
   ▼                           ▼
Linear Rec.709 radiance ──► light_rig (sky = 1, sun_strength, sun_dir) ──► Relight / Sun Direction
   ▼ OCIO "Linear Rec.709 (sRGB)" -> "ACEScg"
hdri_acescg EXR + preview
```

Keep two questions apart: **where is the sun?** (astronomy, answered in-repo) and **what does the sky look like?**
(a sky model, of which there are two).

## 3. How it works

### 3.1 Where the sun is: `layers/solar.py`

`solar.sun_at(latitude, longitude, when)` is NOAA's compact solar position algorithm in pure python. It returns
compass azimuth (clockwise from true north) and elevation, geometric (no refraction), like Prague. It matches NREL's
published SPA case to about 0.002 deg and Atlas Scene's `derive_sun` exactly (`tests/test_physical_sky.py`). The repo
carries its own because `derive_sun` refuses suns below -4.2 deg, and a relight still needs a sun under the horizon.

`solar.parse_utc` refuses a bare time on purpose. `physical_sky.effective_sun` adds the modes: `metadata` (NOAA plus
`time_offset_h`), `offset` (plus az/el nudges) and `manual` (typed az/el). `physical_sky.sun_path` samples the day
for sunrise, sunset, solar noon and golden hour (the sun at 4 deg) for the Sun Globe.

### 3.2 Prague or Hosek-Wilkie, and who owns what

| | Prague | Hosek-Wilkie |
|---|---|---|
| owned by | Atlas Scene, reached only via `_atlas_scene.py` | atlas-vfx, `layers/sky_hosek.py`, never imports Atlas Scene |
| needs | Atlas Scene, its Prague extension, a `.dat` dataset (ground level, ~103 MB, is enough) | nothing extra (coefficients ship hash-checked) |
| sun range | down to -4.2 deg (twilight) | 0 to 90 deg |
| haze | `visibility_km` directly | `visibility_km` mapped to turbidity 1 to 10, fitted to Prague |

The fallback exists because the repo is public and Atlas Scene isn't (NOTES 2026-10-29). The rule: nobody copies
Atlas Scene's sky code. How close it is (`tools/hosek_calibrate.py`): at 20 to 50 km with a 20 deg sun,
`sun_strength` within about 10% of Prague and the sky within about 25%; the absolute scale matches (median ratio
1.03), so `exposure_ev` means the same in both. It can't get as clear as Prague above about 60 km, and near the
horizon its sun is far too bright (x1.8 at 5 deg, x7 at 2 deg).

![Prague and Hosek-Wilkie side by side](../images/sky_prague_vs_hosek.png)

### 3.3 Frames (the bit that bit on 2026-10-28)

All in `layers/physical_sky.py`:

- **Sky frame:** right-handed, Y up, north -Z, east +X. `azel_to_dir` gives `(sin A cos e, sin e, -cos A cos e)`.
- **Lat-long:** Nuke's layout. The centre column is north (or the camera heading, via `latlong_centre`) and
  **azimuth grows to the right**. `sun_pixel` gives where the sun lands; the EXR header records the convention.
- **The mirror:** Atlas Scene's own lat-long runs azimuth to the left. `latlong_to_atlas` (a det -1 3x3) is applied
  to directions in `class _Oriented` before the evaluator sees them, so there's no resampling. The Hosek evaluator
  goes through the same wrapper. A test pins the brightest pixel to the predicted sun, never the mirrored column.
- **Camera:** OpenGL space, which Relight expects. `sun_in_camera` puts the sun there.

### 3.4 How the tracked camera orients the sky

`heading_deg` is the compass bearing the camera faces **on its first frame**. With a `shot_camera` connected,
`sun_in_camera` turns the solved world about Y so frame one faces that bearing (`world_yaw_deg` in info), then
applies each frame's rotation. The solve's tilt and fov win over the typed ones. On the DSCF3000 tracked relight the
heading is still a 180 placeholder (NOTES 2026-11-01), so treat that shot as illustrative.

### 3.5 What a pixel value means

Both models return unclipped Linear Rec.709 radiance in radiometric CIE-weighted units (multiply Y by
`sky_hosek.LM_PER_W` = 683 for cd/m^2). In the exercise a 16 deg evening zenith reads about 2 and the sun disc over a
million, which is why everything stays float32 (half overflows at 65504).

The hop to working space is named: `physical_sky.to_working` runs `"Linear Rec.709 (sRGB)" -> "ACEScg"` with the
shipped config, and the EXR says `ACEScg` in its header. The preview is a separate display hop, auto-exposed;
never grade the HDRI through it. `exposure_ev` scales the HDRI only.

The rig integrates the sun disc properly (`_sun_irradiance`), while the HDRI point-samples a disc about 3 px wide at
2K. So the HDRI's share of sun energy wobbles (`hdri_sun_energy` in info) and below about 1K the disc can vanish.
Use the rig for the key.

### 3.6 The light rig and relight

`physical_sky.light_rig` normalises so the sky on an up-facing surface has luminance 1. The keys that matter:
`sky`, `ground`, `sun_colour` (max channel 1), `sun_strength` (the key a sun-facing surface gets against a sky of 1),
`sun_dir` per frame and `sun_elevation_deg`. For DSCF2977 Prague gave `sun_strength` 7.07; the exercise gets 7.32
from Hosek.

- **Relight (18)**: its optional `sun_rig` input swaps in the physical sun's direction and colour and keeps the
  plate's own ambient (`relight.override_sun`). With `sun_strength_from` = physical sky,
  `relight.physical_sun_gain` sets `sun_gain = trim x sun_strength x plate sky luminance`. Unconnected, Relight is
  bit-for-bit unchanged.
- **Sun Direction (56)**: `node_sun_direction.from_light_rig` snaps the camera-space sun to one of 8 sectors and
  low/mid/high for the LTX Relight IC-LoRA, and applies the mirror trick (the LoRA only lit from frame right on
  DSCF3000, so left-lit shots are mirrored in LTX Prep and flipped back in Return).

### 3.7 The Sun Globe (display only)

`web/atlas_vfx_sun_globe.js` draws a three.js globe under the node: compass, sun, the day's path, the camera frustum
and a little lit actor. It only drives native widgets (`time_offset_h`, `sun_mode`, offsets, manual az/el), so an
API run is identical without it. Dragging the sun switches to manual; the scrubber sets `time_offset_h`; there are
buttons for reset, golden hour and view north or camera. A JS mirror of `solar.py` draws before the first run, and
`tests/test_sun_globe.py` checks the two agree.

![The Sun Globe at twilight](../images/sun_globe_twilight.png)

## 4. The nodes

Full sockets are in `docs/NODE_CATALOGUE.md` blocks 48, 18, 56 and 57. What a learner must set on **Physical Sky**:

| widget | what to put in it |
|---|---|
| `latitude`, `longitude` | GPS, degrees north / east |
| `capture_utc` | ISO 8601 **with offset** |
| `heading_deg` | compass bearing of the first frame |
| `visibility_km` | 20 hazy to 131.8 very clear |
| `sun_mode` | metadata, offset or manual |
| `width` | lat-long width, 2K or more if you want the disc |
| optional `shot_camera` | from Solve Plate, Camera Import or Measure & Scale |
| optional `sky_model` | auto, prague, hosek-wilkie (last widget on purpose: saved graphs fill by position) |

Outputs: `hdri_acescg`, `light_rig`, `preview`, `info` (JSON), `exr_path`. Blank `out_dir` writes under
`~/atlas-vfx-work/physical_sky/<name>`.

## 5. Walkthrough

**`workflows/starter_physical_sky.json`**: one node into a Preview, the DSCF2977 check shot, no plate, no Atlas Scene.

1. Queue it. Without Atlas Scene, info says Prague is unavailable and Hosek-Wilkie rendered. That's expected.
2. The preview shows the sun ringed at about WSW, 16 deg up, outside the yellow field-of-view lines: off frame right.
3. In `info`, find `sun.azimuth_deg` 252.355, `elevation_deg` 16.121, `relative_to_heading_deg` (about +72) and
   `light_rig.sun_strength`.
4. Drag the sun on the globe (mode flips to manual), hit RESET, then GOLDEN HOUR, and queue again. Scrub past sunset:
   Hosek refuses, the HDRI goes black, and the rig keeps the exact direction.
5. Load the EXR in Nuke as ACEScg on a SphericalTransform; north is the centre column.

**`workflows/relight_tracked_dscf3000_s2s3.json`** shows the real wiring: tracker, then Physical Sky with the tracked
camera and `latlong_centre` = camera heading, then Sun Direction and Light Ball into the LTX relight. It needs the
owner's plate, so read it for the wiring.

## 6. Hands-on

```
python docs/learn/exercises/02_physical_sky.py
```

Ten steps, synthetic inputs, writes nothing. The lines that teach the most (full run on ws: PASS in about 0.5 s):

```
   ok   a bare time is refused  (capture time '2025-12-25T18:39:25' has no offset; ...)
   Sydney summer solstice: solar noon 2025-12-21T01:53:09Z UTC, sun az 360.00, el 79.57 (simple astronomy says 79.57)
   DSCF2977 (Burning Palms, 6:39 pm AEDT): az 252.355, el 16.121
   ok   north is the centre column, east a quarter to the RIGHT (azimuth grows right)
   ok   72 deg off the lens with a 69.5 deg lens: the sun is off frame right
   brightest pixel (205, 210) lum 1471059.0; predicted sun pixel (205.3, 209.6)
   sky luminance: 5 deg from the sun 24.276, 90 deg round 7.297, zenith 2.125, low horizon opposite the sun 19.392
   visibility  20.0 km -> turbidity 5.00: sun_strength 4.14, zenith blue/red 2.58
   visibility 100.0 km -> turbidity 1.00: sun_strength 8.52, zenith blue/red 3.93
   ok   sun_strength: the key is several times the sky (NOTES: Prague gave 7.07)  (7.317)
   OCIO 'Linear Rec.709 (sRGB)' -> 'ACEScg': red [0.6131, 0.0702, 0.0206], white [1.0, 1.0, 1.0]
   +108 deg from the camera's back, 16 deg up -> 'the right', 'low'
   ok   Hosek-Wilkie refuses a twilight sun (-2 deg); Prague (Atlas Scene) goes down to -4.2
```

Notice: solstice noon elevation is the napkin sum `90 - |lat - decl|`; the brightest pixel lands within a pixel of
the prediction; haze weakens the sun against its sky and greys the zenith, and that ratio is what a relight cares
about. Try the script's suggestions: a northern city (the noon sun swings south), heading 250 (the sun goes dead
ahead, backlit), width 256 (watch the HDRI's sun energy wander while the rig holds).

## 7. Gotchas

- **Time zones.** Bare times are refused. Check the sun against the plate's shadows; fix a wrong clock with offset
  mode or the scrubber.
- **The mirrored sky.** Render Prague outside this node and expect the mirror; inside, `_Oriented` handles it.
- **Heading is yours.** Get it from a map, a landmark, a compass on set or the shadows.
- **Nothing is faked below the model.** Hosek stops at 0 deg, Prague at -4.2: black HDRI, a warning, and a rig with
  direction but zero strength. Prague's twilight render hadn't been checked by eye as of 2026-10-28.
- **Hosek near the horizon** is 1.8x to 7x too bright: prefer Prague for a golden-hour key, or trim `sun_gain`.
- **Small HDRIs lose the disc**, and **half float overflows** on it. Keep float32, take the key from the rig.
- **Altitude** only counts with Prague's full dataset.
- **Prague tests skip** on a machine without Atlas Scene unless `ATLAS_VFX_REQUIRE_PRAGUE=1`.

## 8. Check yourself

1. Metadata says `2025-12-25T18:39:25` with no zone. Why is it refused, and what would you do?
2. A Prague lat-long rendered directly through Atlas Scene has the sun on the wrong side. Why?
3. The rig says `sun_strength` 7. What does that mean, and how does Relight's physical sky mode use it?
4. You connect a tracked pan as `shot_camera`. What must `heading_deg` be, and what happens over the frames?
5. A dusk shot 2 deg below the horizon gives a black HDRI. Bug? What would render it?

<details>
<summary>Answers</summary>

1. It could be local or UTC, and an hour is 15 deg of sun. Add the camera's zone (`+11:00` for AEDT, or convert to
   `Z`), then check against the shadows and correct with `time_offset_h` or offset mode.
2. Atlas Scene's lat-long runs azimuth to the left. Physical Sky corrects it with the det -1 `latlong_to_atlas` in
   `_Oriented`.
3. A sun-facing surface gets 7 times the sky on an up-facing surface. Physical sky mode sets
   `sun_gain = trim x sun_strength x plate sky luminance` and keeps the plate's ambient.
4. The bearing of the first frame. The world is turned so frame one faces it, then each frame's rotation applies, so
   the sun moves in camera space as the camera pans.
5. Not a bug: Hosek covers 0 to 90 deg and never fakes outside it. The rig keeps the direction. Prague renders down
   to -4.2 deg.

</details>
