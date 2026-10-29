# Plate Lock

Takes the best AI take of a bluescreen shot and puts it back on the plate, frame for frame: each performer's
timing, the camera move, the rolling shutter skew and the silhouette edge all come from the plate, while the light
and the invented background come from the AI.

The rule it's built around is geometry from the plate, light from the model. The AI take is solved into the same
parts as the plate (camera, depth, per-performer motion), the two solves are put in one space anchored on frame 0
(where the AI started from the plate), and the AI is re-driven by the difference.

## What each stage does

1. **Retime, per performer.** DTW on mask motion (position, size, shape), then a joint solve of time and constant
   offset, since on smooth motion an offset and a lag look alike. Open-ended, so a take that runs long is fine.
2. **Align solves.** Sim3 that makes the AI camera equal the plate camera at the anchor frame, with a scale you give
   it (performer height ratio, or a known object).
3. **Re-shoot the background.** The plate is blue behind them, so there's nothing to warp onto. Each AI background
   pixel is lifted to 3D with its depth and photographed through the plate camera row by row (rolling shutter),
   optionally integrated over the shutter. Holes behind the performers fill from neighbouring AI frames, then a
   push-pull fill; `background_valid` is 1 for real AI pixels, 0.5 for filled.
4. **Warp each performer.** A smooth field from correspondences (body solve vertices, matcher points, or nothing),
   then an edge snap to the plate's screen key, banded to the edge when correspondences already hold the interior.
5. **Restack** by depth over the re-shot background.
6. **QC** before and after: overlap, edge distance in pixels, motion lag, and which stages the shot needed.

Rolling shutter only needs modelling for the background. For people, warping onto each plate frame copies the skew.

## Nodes (Atlas VFX/Layers/Plate Lock)

| Node | In | Out |
|---|---|---|
| Read EXR | sequence path | linear frames, header metadata JSON, first frame number |
| Screen Key | plate | matte (fallback; a Nuke pull is better) |
| Camera | frames + locked_off / shot_camera / platelock_json, readout ms | ATLAS_VFX_PLATELOCK_TRACK |
| Person | name, SAM masks on plate and AI, depth, optional body/matcher .npz | ATLAS_VFX_PLATELOCK_PEOPLE (chain one per person) |
| QC | people | summary, report JSON |
| Plate Lock | plate, AI, both cameras, people, scale, depth mode, shutter | comp, background, background_valid, result, report |
| Layer | result, name | one performer's RGB + alpha |
| Write | result, out dir | EXRs, STMaps, shot.json, platelock_lock.nk |

SAM masks and depth come from the nodes you already use. Depth must be metric (MoGe metric) or inverse metric, not
a normalised preview. `flat` treats the AI background as a card, which is right for a locked-off shot.

## Delivery

Everything stays a map on the original AI frames, so comp keeps control:

```
bg/bg.####.exr                re-shot background RGBA (A: 1 real, 0.5 filled)
bg/bg_stmap.####.exr          STMap into the AI frame of the same number
<person>/<person>.####.exr    warped performer RGBA, header says which AI frame
<person>/<person>_stmap.####.exr
comp/comp.####.exr            review comp
shot.json                     retime curves, gauge, camera delta, QC, sidecar (inputs, versions, core hash)
platelock_lock.nk             Reads + a TimeWarp per performer (step lookup = AI frame per plate frame)
```

STMaps follow Nuke's convention (s = (x+0.5)/W, t = 1-(y+0.5)/H in the AI frame). Wire TimeWarp into an STMap
node's src and the STMap Read into its stmap input, uv = rgb.

## Conventions

`core/` uses OpenCV camera axes (x right, y down, z forward) with pixel centres on integers. The shot-camera
adapter converts from the repo's Nuke axes (`R_c2w_cv = R_c2w_nuke @ diag(1, -1, -1)`) and the corner pixel origin.
Readout is in frames internally; the Camera node takes milliseconds and fps. `rs_anchor` says which row a solved
pose represents (most solvers land on mid-readout, `centre`).

## Known limits

- Frames are processed in numpy at plate resolution. Fine for review sizes; full 6K is slow, a GPU path is next.
- Intrinsics are taken from the first frame; a zoom is reported, not modelled.
- Where the AI changed a shape (a sleeve, the hair mass), no warp gets there. QC lists those frames as regen
  candidates: put them back through LTX with the plate's poses and camera as control, then lock again.
- Model adapters (SAM, camera/depth solver, body solver, matcher) are protocols in `adapters.py`; check each
  model's licence in `atlas_components.json` before a paid job. SMPL-based body models are usually non-commercial.

## Tests

```
<V141 python> tests\test_platelock_core.py    every stage against a known answer
<V141 python> tests\test_platelock_lock.py    synthetic shot end to end, delivery round trip
<V141 python> tests\test_platelock_nodes.py   registry + the node graph on torch tensors
```

The synthetic shot (`core/synth.py`) is a rolling-shutter plate of two performers on blue and an AI take with known
damage: relit, retimed per performer, fatter and offset, a drifting global-shutter camera in another solve gauge.
