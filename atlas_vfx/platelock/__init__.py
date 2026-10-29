"""Plate lock: put the best AI take back on its plate, frame for frame (retime, re-shoot, warp, restack).

core/ is pure python (numpy, scipy, OpenCV): no ComfyUI or torch, so it runs in the tests and outside Comfy.
At the split, core/camera.py and core/align.py (rolling-shutter cameras, solve alignment) are the candidates
to move into Atlas Camera; the rest stays here.
"""

# One node per module, gathered by atlas_vfx.registry.
NODE_MODULES: tuple = ("node_read_exr", "node_screen_key", "node_camera", "node_person", "node_qc",
                       "node_lock", "node_layer", "node_write", "node_crossview")
