# Picking points for Measure & Scale

Measure & Scale turns a few known lengths in the plate into metres, finds up, and puts the ground at y = 0. You
pick the ends of each length in Nuke and hand over a small JSON. This page is how.

## What to pick for DSCF3000 S2-S3

| label | what | vertical | length |
|---|---|---|---|
| `door_frame` | door frame, threshold (a) to underside of the head (b), one side of the jamb | yes | tape it |
| `step_riser` | one step riser, bottom edge (a) to nosing (b) | yes | tape it |
| `deck_rail` | deck surface (a) to top of the rail (b), at a post | yes | tape it |
| `actor_spot` (point) | where the test character stands on the deck | - | - |

For vertical lengths, **a is the bottom and b is the top**. That is how up is found, so it matters. The three
lengths should agree on the scale within 5 %, or the node refuses (it tells you which one is off).

Ground: for this shot use the **deck** as the ground (the test character stands on it), so y = 0 is the deck surface. Either give a
region of deck in one frame (needs the pycolmap camera, which has 3D points) or pick 1+ ground points (any camera).

## Plane distance (no point picks): what the guessed run used

`"kind": "plane_distance"` measures the perpendicular distance between two planes fitted to the solve's points in two
regions, e.g. the step pad and the deck at the top of the steps = 4 risers. Hundreds of points per plane make it far
steadier than clicked points when the camera walks straight at the subject (triangulation is weak near the middle of
the frame) or the corners are white-on-white (trackers slide along edges). With `"vertical": true`, a is the LOWER
plane and b the UPPER one, and the mean normal gives up. Keep the two regions close together: fitted planes are never
exactly parallel, so measuring between far-apart patches adds (angle x offset).

```json
{"label": "pad_to_deck_4_risers", "kind": "plane_distance", "length_m": 0.72, "vertical": true,
 "a": {"frames": [1105, 1110, 1115, 1120, 1125, 1130, 1135], "region": [1600, 1880, 3800, 2160]},
 "b": {"frames": [1150, 1154, 1158, 1162, 1166, 1170], "region": [1120, 1000, 3440, 1280]}}
```

(regions here in `top_left` pixels.) If you tape one riser, put 4 x that in `length_m` and re-run: nothing else changes.

## Two ways to pick a point

A point is picked in **one or more frames**. Picked in 2+ frames (spread apart, at least 15-20 frames, the further the
better) it is triangulated through the shot camera: this works with the Nuke camera or the pycolmap one. Picked in one
frame it needs a `plane` (a region of the solve's points it lies on, like the wall beside the door), which only works
with the pycolmap camera.

### A. Viewer readout (no scripting)

1. Viewer at full res, proxy off, the plate Read (DPX or ProRes) in, format 4096x2160.
2. Go to the frame, hover the point, read `x y` from the bottom-left of the viewer (Nuke's pixel coordinates:
   origin bottom-left, y up). Type them into the JSON under that frame number (Nuke frame, 1130-1297 for S2-S3).

### B. Trackers (more frames, less typing)

One Tracker per point end, named `PICK_<label>_a`, `PICK_<label>_b`, or `PICK_<label>` for a single point. Use the
first track in each. Either track it over a range (every keyed frame is used) or set a few keys by hand on frames
far apart. Then paste this in the Script Editor (edit the top block first):

```python
# ---- edit ----
OUT = r"C:\Users\miike\atlas-vfx-work\DSCF3000\picks\DSCF3000_S2S3_picks.json"
MEASUREMENTS = {             # label: (length in metres, vertical?)
    "door_frame": (2.04, True),
    "step_riser": (0.18, True),
    "deck_rail": (0.90, True),
}
POINTS = ["actor_spot"]
GROUND = {"points": ["actor_spot"]}      # or {"frame": 1160, "region": [x0, y0, x1, y1]} in Nuke px (pycolmap camera)
# ---- end ----
import json, os, nuke
NCOL, CX, CY = 31, 2, 3                # Tracker4 'tracks' table: 31 columns, track_x = 2, track_y = 3 (first track)

def keyed(name):
    n = nuke.toNode(name)
    if n is None:
        raise RuntimeError("no node " + name)
    k = n["tracks"]
    ax = k.animation(CX)
    if ax is None:
        raise RuntimeError(name + ": first track has no keys")
    frames = sorted({int(round(key.x)) for key in ax.keys()})
    return {str(f): [k.getValueAt(f, CX), k.getValueAt(f, CY)] for f in frames}

doc = {"origin": "nuke_bottom_left", "measurements": [], "points": [], "ground": GROUND}
for label, (length, vertical) in MEASUREMENTS.items():
    doc["measurements"].append({"label": label, "length_m": length, "vertical": vertical,
                                "a": keyed("PICK_%s_a" % label), "b": keyed("PICK_%s_b" % label)})
for label in POINTS:
    doc["points"].append({"label": label, "frames": keyed("PICK_" + label)})
if not os.path.isdir(os.path.dirname(OUT)):
    os.makedirs(os.path.dirname(OUT))
with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=1)
print("wrote", OUT)
```

The Tracker table indices (31 columns, x = 2, y = 3) are the usual Tracker4 layout but I have not run this on Indie
17.1. If it errors, paste the error back, or use method A.

## The JSON

```json
{
  "origin": "nuke_bottom_left",
  "measurements": [
    {"label": "door_frame", "length_m": 2.04, "vertical": true,
     "a": {"1240": [2210.5, 642.0], "1270": [2318.0, 590.5]},
     "b": {"1240": [2208.0, 1460.0], "1270": [2316.5, 1502.0]}},
    {"label": "wall_board", "length_m": 1.2, "vertical": false,
     "a": {"1001": [2240.0, 1300.0]}, "b": {"1001": [2780.0, 1300.0]},
     "plane": {"frame": 1001, "region": [2240, 1200, 2800, 1880]}}
  ],
  "points": [
    {"label": "actor_spot", "frames": {"1250": [1980.0, 610.0], "1290": [1904.0, 380.0]}}
  ],
  "ground": {"points": ["actor_spot"]}
}
```

- `origin`: `nuke_bottom_left` (viewer readout and Trackers) or `top_left` (image tools, OpenCV).
- Frames are Nuke frames (plate index + 1001). Pixels are full plate resolution (4096x2160).
- `plane.region` and `ground.region` are `[x0, y0, x1, y1]` boxes in the same pixel convention, on the frame given.
- Optional per measurement: `"use_for_scale": false` keeps a length as a check only.

## What comes out

`<name>.shot_camera.json` (metres, Y up, ground y = 0, first camera at x = z = 0 looking down -Z), `<name>.measure.json`
(every length as measured, the scale spread, up and ground methods, labelled points in metres) and a sidecar, in
`atlas-vfx-work\DSCF3000\camera\`. Then run Lock Check on it: the card should stay stuck to the deck.
