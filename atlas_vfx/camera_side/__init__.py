"""Camera-side nodes (plate I/O, tracking, solve/import, scale, lock check, depth, camera export).

Moves into Atlas Camera later. Rule: no imports from atlas_vfx.layers or atlas_vfx.deliver.
"""

# Node modules in this folder (one node per module). The root registry imports them.
NODE_MODULES: tuple = ("node_plate_read", "node_tracking_proxy", "node_camera_import", "node_solve_compare",
                       "node_measure_scale", "node_lock_check", "node_plate_depth", "node_exr_write", "node_camera_export", "node_rs_solve",
                       "node_chart_calibrate")
