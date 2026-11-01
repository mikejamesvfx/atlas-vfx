# Learning atlas-vfx

Four guides for VFX artists and TDs who know Nuke, tracking, comp and colour, and a bit of ComfyUI, but haven't read
this code yet. Each one explains a part of the pipeline in artist terms, then shows where it lives in the code, which
nodes you wire, which shipped workflow to open, and what the NOTES say went wrong along the way. Each ends with five
check-yourself questions.

Every guide comes with an exercise in `exercises/`. They're plain Python, build their own synthetic data (no plates,
no model weights, no network, no GPU), print what you should be looking at step by step, and exit non-zero if a check
fails. Run them from the repo root with the ComfyUI venv's python:

```
python docs/learn/exercises/01_latent_layers.py
```

| Guide | What it covers | Exercise | Runtime |
|---|---|---|---|
| [01 Latent layers](01-latent-layers.md) | Getting comp layers into LTX and back without losing the camera lock: LTX Prep and Return, Latent Pre-render, Guide From Latent, chunks for long shots, Tiny VAE previews, elements through the inverse of LTX's view | `01_latent_layers.py` (fake VAE with LTX's shape, identity "LTX") | a few seconds |
| [02 Physical Sky](02-physical-sky.md) | A real sun and clear sky from the plate's GPS and capture time: solar position, Prague (Atlas Scene) vs the Hosek-Wilkie fallback, sky and camera frames, ACEScg radiometry, the light rig that drives Relight, the Sun Globe panel | `02_physical_sky.py` (Sydney solstice, Hosek render, turbidity sweep) | under a second |
| [03 OCIO compliance](03-ocio-compliance.md) | One colour pipeline from F-Log to Nuke: the shipped ACES 2.0 config, range from the stream, F-Log vs F-Log2, the display island around LTX, EXRs that carry their colourspace, the ocio_fast path | `03_ocio_compliance.py` (uses the shipped config with PyOpenColorIO) | a couple of seconds |
| [04 Camera tracking](04-camera-tracking.md) | The tracker inside ComfyUI: Solve Plate, Solve Mask, Track 2D, Solve View, nodal shots, solve error, lens and rolling shutter, COLMAP to Nuke, and a worked comparison against NukeX's CameraTracker | `04_camera_tracking.py` (ray-traced courtyard with a known camera, solved and checked against ground truth) | about two minutes |

A suggested order if you're new: 04 (you need a camera before anything locks), 03 (colour touches every node), 02
(the sun needs the camera), then 01 (LTX sits on top of all three).

The guides describe the code as of 2026-11-01. `docs/NODE_CATALOGUE.md` stays the source of truth for sockets and
widgets; if a guide and the catalogue disagree, trust the catalogue and tell us.
