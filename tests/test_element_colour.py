"""layers/element_colour.py: generated (display) pictures into ACEScg the way LTX Prep sees them (flow review
2026-11-01: elements came in as flat sRGB textures, then LTX Prep tone-mapped them again on the way out).

  1. inverse display view -> LTX Prep's view gives the generated picture back (the element survives the round
     trip as it was made); the old sRGB texture path does not (it comes back darker in the mids).
  2. Foreground Element's make_texture follows source_colour; the old texture path is still there on request.
  3. Relight Transfer with the inverse-view encoding: a relit take identical to the element is a no-op.
Run: <V141 python> tests/test_element_colour.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.layers import element_colour as ec  # noqa: E402
from atlas_vfx.layers import foreground_element as fe  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    rng = np.random.default_rng(1)
    pic = rng.uniform(0.05, 0.9, (32, 48, 3)).astype(np.float32)          # a generated display picture
    fwd, _inv = li.view_processors("")                                     # what LTX Prep shows LTX

    lin = ec.to_acescg(pic, "", ec.VIEW_INVERSE)
    back = li.to_view(lin, fwd)
    err = float(np.abs(back - pic).max())
    check(err < 2e-3, f"inverse view -> LTX Prep view gives the picture back (max {err:.5f})")
    old = li.to_view(ec.to_acescg(pic, "", ec.SRGB_TEXTURE), fwd)
    mid = (pic > 0.3) & (pic < 0.7)
    drift = float(np.mean(old[mid] - pic[mid]))
    check(drift < -0.02, f"the old sRGB texture path comes back darker through the view (mean {drift:+.3f})")
    try:
        ec.to_acescg_processor("", "nope")
        check(False, "an unknown source colour is refused")
    except ValueError:
        check(True, "an unknown source colour is refused")

    mask = np.zeros((32, 48), np.float32)
    mask[4:28, 6:42] = 1.0
    a, ra = fe.make_texture(pic, mask, "", erode_px=0, source=ec.VIEW_INVERSE)
    b, rb = fe.make_texture(pic, mask, "", erode_px=0)
    check("inverse" in ra["colour"] and "sRGB Encoded" in rb["colour"] and not np.allclose(a, b),
          f"make_texture follows source_colour ({ra['colour'][:30]}... vs {rb['colour']})")

    import torch

    from atlas_vfx.layers.node_relight_transfer import AtlasVFXRelightTransfer

    node = AtlasVFXRelightTransfer()
    spec = node.INPUT_TYPES()
    check(ec.VIEW_INVERSE in spec["required"]["encoding"][0], "Relight Transfer offers the inverse-view encoding")
    el = torch.from_numpy(np.repeat(pic[None], 2, 0))
    m = torch.from_numpy(np.repeat(mask[None], 2, 0))
    kw = {k: (v[1].get("default") if isinstance(v[1], dict) else None) for k, v in spec["required"].items()
          if k not in ("element", "matte", "relit", "encoding")}
    kw.update({"mode": "global"} if "global" in spec["required"]["mode"][0] else {})
    out = node.run(element=el, matte=m, relit=el, encoding=ec.VIEW_INVERSE, **kw)[0].numpy()
    err = float(np.abs(out - el.numpy()).max())
    check(err < 5e-3, f"Relight Transfer (inverse view): relit = element is a no-op (max {err:.4f})")
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
