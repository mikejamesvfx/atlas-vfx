"""Sun Direction from a light rig (catalogue 56 + 48): the real sun, placed relative to the camera, in the Relight
LoRA's words.

  - typed camera looking north: sun east = the right, south (behind the camera) = the front, north (in view) =
    behind, west = the left (and mirrored, the trick); elevation picks low / mid / high.
  - a tracked camera panning 90 deg right under a fixed sun: the sun swings round in camera space, the node reads
    it at the chosen frame and reports the spread.
  - the node: a rig replaces the typed direction and height; no rig keeps them.
Run: <V141 python> tests/test_sun_rig.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import node_sun_direction as sd  # noqa: E402
from atlas_vfx.layers import physical_sky as ps  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def rig_for(az, el, heading=0.0, shot_camera=None):
    dirs, _meta = ps.sun_in_camera(az, el, heading, 0.0, shot_camera)
    return ps.light_rig(dirs, np.ones(3), np.ones(3), 0.2, el, 60.0, 16 / 9)


def main() -> int:
    cases = [(90, 20, "the right", "low"), (180, 45, "the front", "mid"), (0, 70, "behind", "high"),
             (270, 15, "the left", "low"), (45, 30, "the back-right", "low")]
    for az, el, want_dir, want_h in cases:
        r = sd.from_light_rig(rig_for(az, el))
        check((r["direction"], r["height"]) == (want_dir, want_h),
              f"camera north, sun az {az} el {el}: {r['direction']}, {r['height']} (want {want_dir}, {want_h})")
    s = sd.sun(*[sd.from_light_rig(rig_for(270, 15))[k] for k in ("direction", "height")], "golden")
    check(s["mirror"] and "right" in s["direction"], "a real sun on the left is rendered mirrored (the trick)")

    # a tracked camera panning 90 deg to the right (yaw -90 about +Y) under a sun due east
    mats = []
    for i in range(9):
        m = np.eye(4)
        a = math.radians(-90.0 * i / 8)
        m[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        mats.append(m)
    plate = {"path": "synthetic", "width": 1920, "height": 1080, "colorspace": "ACEScg", "range": "full"}
    cam = sc.build(name="pan", frame_start=1001, fps=24.0, plate=plate, sensor={"width_mm": 23.5,
                   "height_mm": 23.5 * 1080 / 1920}, cam_to_world=mats, focal_mm=16.0, source={"kind": "synthetic"})
    rig = rig_for(90, 20, heading=0.0, shot_camera=cam.data)
    first, last = sd.from_light_rig(rig, "first"), sd.from_light_rig(rig, "last")
    check(first["direction"] == "the right" and last["direction"] == "behind",
          f"panning onto the sun: first frame {first['direction']}, last {last['direction']} "
          f"(spread {first['directions_over_shot']})")

    node = sd.AtlasVFXSunDirection()
    out = node.run("the left", "high", "golden", "auto", light_rig=rig_for(180, 20))
    m, prompt, sun = out["result"]
    check(sun["asked"] == "the front" and sun["height"] == "low" and "from_light_rig" in sun and not m,
          f"node: the rig replaces the typed direction ({sun['asked']}, {sun['height']})")
    out = node.run("the left", "high", "golden", "auto")
    check(out["result"][2]["asked"] == "the left" and out["result"][0], "node: no rig keeps the typed sun")
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
