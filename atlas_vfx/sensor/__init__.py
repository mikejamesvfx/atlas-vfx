"""Sensor pass: put the plate's rolling shutter and lens back onto generated footage (and take them off the plate),
deterministically, as STMaps. Stays in atlas-vfx (it may import camera_side, contracts and platelock).

rs.py (rolling shutter maps), lens.py (Brown-Conrady, lateral CA, vignette), apply.py (apply / remove, crop windows),
readout_solve.py (readout from tracks), maps_io.py (EXR + Nuke Reads), jobs.py (what the nodes run).
"""

NODE_MODULES: tuple = ("node_readout_solve", "node_lens_maps", "node_rs_maps", "node_apply")
