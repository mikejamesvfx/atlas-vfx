"""camera_side/camera_checks.py: the checks every camera-taking node runs (catalogue 4 and the nodes it feeds).

  - require_metric: silent for a scaled camera, warns for an unscaled solve, says "nodal" for a nodal one, raises
    when strict.
  - check_frame_offset: a camera numbered with another plate offset is refused (every frame would be off).
  - lens_needs_clean: a lensed camera without a lens folder is refused unless allow_raw; pinhole passes.
  - wired: Lock Check refuses the auto card on a nodal camera; Camera Export labels units honestly; LTX Prep's
    round-trip warning fires on a clipped plate and names mask_mode none.
Run: <V141 python> tests/test_camera_checks.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import camera_checks as cc  # noqa: E402
from atlas_vfx.camera_side import camera_export as ce  # noqa: E402
from atlas_vfx.camera_side import node_lock_check as nlc  # noqa: E402
from atlas_vfx.contracts import shot_camera as sc  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (cc.CameraCheckError, ValueError) as e:
        return str(e)
    return None


def camera(n=6, moving=True, scale="known_lengths", lens=None):
    M = []
    for i in range(n):
        m = np.eye(4)
        m[:3, 3] = [0.1 * i if moving else 0.0, 1.6, 0.0]
        M.append(m)
    plate = {"path": "synthetic", "width": 1024, "height": 540, "colorspace": "ACEScg", "range": "full"}
    cam = sc.build(name="chk", frame_start=1001, fps=24.0, plate=plate, sensor={"width_mm": 23.5,
                   "height_mm": 23.5 * 540 / 1024}, cam_to_world=M, focal_mm=16.0, source={"kind": "synthetic"},
                   scale={"method": scale, "m_per_source_unit": 1.0}, plate_frame_offset=1001, lens=lens)
    return cam


def main() -> int:
    metric, unscaled, nodal = camera(), camera(scale="unscaled"), camera(moving=False, scale="unscaled")
    check(cc.require_metric(metric, "T") is None, "scaled camera: no metric warning")
    w = cc.require_metric(unscaled, "T")
    check(w is not None and "Measure & Scale" in w, f"unscaled camera warns ({w})")
    w = cc.require_metric(nodal, "T")
    check(w is not None and "nodal" in w and cc.is_nodal(nodal), "nodal camera (centres identical) says nodal")
    check(raises(cc.require_metric, unscaled, "T", strict=True) is not None, "strict raises")

    ref = {"frame_offset": 1001}
    check(cc.check_frame_offset(metric, ref, "T") is None, "same frame offset passes")
    msg = raises(cc.check_frame_offset, metric, {"frame_offset": 1000}, "T")
    check(msg is not None and "+-1 off" not in msg and "-1 off" in msg, f"other offset refused ({msg})")

    lensed = camera(lens={"distortion_model": "brown", "distortion_params": [-0.08, 0.01, 0, 0, 0]})
    check(cc.has_lens(lensed) and not cc.has_lens(metric), "has_lens")
    check(cc.lens_needs_clean(metric, False, "T") is None, "pinhole camera needs no lens folder")
    check(cc.lens_needs_clean(lensed, True, "T") is None, "lensed camera with a lens folder passes")
    check(raises(cc.lens_needs_clean, lensed, False, "T") is not None, "lensed camera with no lens folder refused")
    w = cc.lens_needs_clean(lensed, False, "T", allow_raw=True)
    check(w is not None and "allow_raw" in w, "allow_raw turns the refusal into a warning")

    # wired nodes
    msg = raises(nlc._centre, nodal, "auto", {})
    check(msg is not None and "nodal" in msg, "Lock Check refuses the auto card on a nodal camera")
    check(nlc._centre(nodal, "0,1,-5", {})[1] == "explicit", "Lock Check takes an explicit centre on a nodal camera")
    check(ce.nuke_settings(metric)["units"] == "metres", "Camera Export: scaled camera labelled metres")
    check(ce.nuke_settings(unscaled)["units"].startswith("solve units"), "Camera Export: unscaled labelled solve units")
    check(ce.nuke_settings(nodal)["units"].startswith("nodal"), "Camera Export: nodal labelled nodal")
    s = ce.nuke_settings(lensed)
    check(s["lens"]["distortion_model"] == "brown" and s["scale_method"] == "known_lengths",
          "Camera Export carries the lens and scale method into the Nuke camera JSON")

    rt_ok = {"share_over_5pct": 0.0, "max_luminance": 0.8}
    rt_hot = {"share_over_5pct": 0.04, "max_luminance": 6.0}
    check(li.roundtrip_warnings(rt_ok, "layer") == [], "round trip: clean plate, no warning")
    w = li.roundtrip_warnings(rt_hot, "layer")
    check(len(w) == 1 and "clipped" in w[0] and "WHOLE" not in w[0], "round trip: clipped plate warns")
    check("WHOLE" in li.roundtrip_warnings(rt_hot, "none")[0], "round trip: mask_mode none says the whole plate")

    d = copy.deepcopy(metric.data)
    d["extensions"] = {"atlas_vfx_solve_plate": {"degeneracy": {"nodal": True}}}
    check(cc.is_nodal(sc.ShotCamera(d)), "is_nodal reads the solve's own nodal flag")
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
