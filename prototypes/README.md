# Prototypes

Working scripts from 2026-10-17/18 that proved each step on DSCF3000. They are reference, not library code: port the
logic into `atlas_vfx/`, with tests, and leave these as they are. Paths inside them point at Mike's machine
(`...\_pick\vfx_probe\`), which is where their outputs still live.

| Script | What it proved | Becomes |
|---|---|---|
| vfx_probe/build_ocio_flog2.py | ACES 2.0 config + F-Log2 F-Gamut input space (18 % grey -> 0.18) | config/ (already copied) |
| vfx_probe/plate_proxy.py | display proxies + per-frame stats. BUG: assumes legal range; DSCF3000 is full range | T2 Plate Read |
| vfx_probe/track_proxy.py | log-luma + CLAHE proxies; made S2-S3 solvable (297/297 vs 184/297) | T2 Tracking Proxy |
| vfx_probe/split_analysis.py | exposure/change curve, shot split suggestions | Shot Splitter (later) |
| vfx_probe/solve_s1s3.py | pycolmap solve from proxies, per-shot registration report | Solve Plate (later) |
| vfx_probe/resolve_nodist.py | re-map with focal/distortion pinned (FOCAL, FIX_FOCAL env); the 16.47 mm solve | Solve Plate (later) |
| vfx_probe/door_scale3.py | known-length scale from picks + fitted wall plane (frame 0 clean, 40/60 not) | T4 Measure & Scale |
| vfx_probe/ground_check.py | lawn plane + camera height sanity check | T4 Measure & Scale |
| vfx_probe/grid_frames.py, zoom_door.py | gridded / zoomed frames for picking pixels by eye | T4 picking docs |
| vfx_probe/card_check.py | grid card on a plane projected through every frame | T4 Lock Check |
| vfx_probe/align_nuke.py | Nuke Camera1 vs pycolmap: 1 % centre residual, 3 deg view angle; chan in metres | T3 Camera Import + Compare, T8 Camera Export |
| howling_yowies/h3_gen.py | MiniMax H3 Ref2VA on the farm (TE render, refs hub, decode local), CUT syntax | T7 Character Insert |
| howling_yowies/farm_calls.py | where each farm call ran | debugging T7 |
| howling_yowies/nuke_export_for_claude.py | getting data OUT of Nuke Indie via the Script Editor | the model for T8's builder (the reverse) |
