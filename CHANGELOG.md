# Changelog

Versions follow `pyproject.toml` (and `atlas_vfx.__version__`). Before 1.0, a minor version can break saved graphs;
when it does, the entry says so and names the tool that rewrites them.

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
