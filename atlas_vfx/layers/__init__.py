"""Layer nodes (set proxy, card project, foreground elements, character insert, layer stack). Stays in atlas-vfx."""

# Node modules in this folder (one node per module). The root registry imports them.
NODE_MODULES: tuple = ("node_set_proxy", "node_foreground_element", "node_card_project", "node_character_insert", "node_ltx_prep", "node_ltx_return",
                       "node_normals", "node_light_from_plate", "node_relight", "node_depth_holdout",
                       "node_physical_sky")
