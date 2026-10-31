"""Atlas Comp: the server side of the generative compositing app (docs/superpowers/specs/2026-10-31-atlas-comp-design.md).
The shot store on disk (shot.py), recipes (recipes.py), viewer frames (frames.py) and the /atlas routes on ComfyUI's
server (server.py). It may import every atlas_vfx package; only the root __init__ imports it."""
