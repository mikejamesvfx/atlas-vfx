# Security

## Threat model

ComfyUI's HTTP API has no authentication. Anyone who can reach it can queue any node with any widget value, so this
pack treats every widget, and every file a widget names, as untrusted:

- Outputs are written only under the work root (`~/atlas-vfx-work`, moved by `ATLAS_VFX_WORK`) or a folder listed in
  `ATLAS_VFX_OUT_ROOTS`, never into this repo or next to a plate. File names from widgets are sanitised.
- Scripts written for Nuke to run (the Nuke Builder's Python, every `.nk`) are built with escaping helpers
  (`atlas_vfx/camera_side/nuke_text.py`), never by pasting strings. Review a generated script before you run it.
- Files the nodes read (camera JSON, .chan, .abc, COLMAP models, npz, EXR, movies, OCIO configs) are parsed as data:
  no pickle, no exec. External tools (ffmpeg, ffprobe, git) run with argument lists, never a shell.
- Shipped data files (the Hosek-Wilkie sky coefficients) are read from fixed paths inside the package with
  `allow_pickle=False` and checked against their sha256 manifest; no widget names them. Physical Sky's Prague
  `dataset` widget names a file that Atlas Scene's compiled Prague loader parses, so treat it like any other
  input path.
- Nodes read any input path you give them, like most ComfyUI file nodes. Keep ComfyUI on 127.0.0.1, or behind
  authentication, if other people or machines can reach it.

`tests/test_nuke_text.py` holds the escaping cases and `tests/test_hygiene.py` keeps user folders and LAN addresses
out of the shipped tree.

## Reporting

Please report a vulnerability privately through GitHub's "Report a vulnerability" (Security tab of this repo), not
in a public issue. Include the node, the input that triggers it and what it does.
