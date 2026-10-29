"""Batch cross-view lock from the command line (same job as the Plate Lock Cross-View node).

Example (V141 python, from the repo root):
  python tools\\platelock_crossview.py ^
      --plate D:\\shot\\plate\\plate.####.exr --ai D:\\shot\\ai\\ai.####.exr ^
      --id hero=D:\\shot\\mattes\\hero.####.exr --id friend=D:\\shot\\mattes\\friend.####.exr ^
      --plate-space ACEScg --ai-space ACEScg --proc-width 2048 ^
      --out C:\\Users\\miike\\atlas-vfx-work\\platelock\\shot010
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.platelock import crossview_job  # noqa: E402
from atlas_vfx.platelock.core.crossview import CVConfig  # noqa: E402
from atlas_vfx.platelock.node_crossview import summary_text  # noqa: E402


def _ids(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plate", required=True)
    ap.add_argument("--ai", required=True)
    ap.add_argument("--id", action="append", required=True, help="name=plate matte sequence (repeat)")
    ap.add_argument("--ai-id", action="append", help="name=AI matte sequence (repeat, optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plate-space", default="ACEScg")
    ap.add_argument("--ai-space", default="ACEScg")
    ap.add_argument("--ocio", default=None)
    ap.add_argument("--proc-width", type=int, default=2048)
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    ap.add_argument("--matte-channel", default="A")
    ap.add_argument("--base", choices=["ai", "plate"], default="ai")
    ap.add_argument("--quality", choices=["medium", "fast", "ultrafast"], default="medium")
    a = ap.parse_args(argv)
    side = crossview_job.run(a.plate, a.ai, _ids(a.id), a.out, ai_ids=_ids(a.ai_id) or None,
                             plate_space=a.plate_space, ai_space=a.ai_space, ocio_config=a.ocio,
                             first=a.first, last=a.last, matte_channel=a.matte_channel,
                             cfg=CVConfig(proc_width=a.proc_width, preset=a.quality), base=a.base,
                             progress=lambda n, t: print(f"\r{n}/{t}", end="", flush=True))
    print()
    print(summary_text(side))


if __name__ == "__main__":
    main()
