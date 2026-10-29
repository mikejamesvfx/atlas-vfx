# camera_side

**Moves into Atlas Camera later** (ADR-022 Amendment 2). Write it so it can be lifted out as a folder:

- no imports from `atlas_vfx.layers` or `atlas_vfx.deliver` (tests/test_import.py checks this)
- Atlas Camera code is used through `atlas_vfx._atlas_camera`, read-only
- outputs follow `contract.shot-camera@0.1`

Nodes: Plate Read, Tracking Proxy, Shot Splitter, Solve Plate, Camera Import, Solve Compare, Measure & Scale,
Lock Check, Plate Depth, Track Adherence, EXR Write, Camera Export. See docs/NODE_CATALOGUE.md.
