"""Rename atlas-vfx socket types in saved ComfyUI workflows (GUI or API JSON).

The 2026-10-29 rename (spec panel item 1) gave every custom type a namespace, so it can't collide with another pack
or with Atlas Camera's own types (ATLAS_SHOT_CAM is a shot format, ATLAS_PLATE_REF a single image: different
payloads from ours). Camera-side types take ATLAS_ (they move into Atlas Camera at the split), the rest ATLAS_VFX_.

Only exact quoted strings are swapped ("SHOT_CAMERA" -> "ATLAS_TRACKED_CAMERA"), which is all a workflow holds for a
type: node input/output "type" fields and the type at the end of each link. A widget value equal to an old name would
also change; none of the committed workflows has one.

  python tools/rename_socket_types.py FILE_OR_FOLDER [...]          rewrite in place, print what changed
  python tools/rename_socket_types.py --dry-run FILE_OR_FOLDER [...]

Folders are searched for *.json (not recursive). Run it on any graph saved before the rename, e.g. your own copies in
ComfyUI's user/default/workflows. Files with no old names are left untouched."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RENAMES = {
    "SHOT_CAMERA": "ATLAS_TRACKED_CAMERA",
    "PLATE_REF": "ATLAS_PLATE_SEQ",
    "CARD": "ATLAS_VFX_CARD",
    "SENSOR_LENS": "ATLAS_VFX_SENSOR_LENS",
    "LTX_PREP": "ATLAS_VFX_LTX_PREP",
    "PLATELOCK_TRACK": "ATLAS_VFX_PLATELOCK_TRACK",
    "PLATELOCK_RESULT": "ATLAS_VFX_PLATELOCK_RESULT",
    "PLATELOCK_PEOPLE": "ATLAS_VFX_PLATELOCK_PEOPLE",
    "ATLAS_LIGHT_RIG": "ATLAS_VFX_LIGHT_RIG",
}
_PAT = re.compile('"(' + "|".join(map(re.escape, RENAMES)) + ')"')


def rename_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def sub(m):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        return '"' + RENAMES[m.group(1)] + '"'

    return _PAT.sub(sub, text), counts


def rename_file(path: Path, dry_run: bool = False) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    new, counts = rename_text(text)
    if counts:
        json.loads(new)  # still valid JSON, or nothing is written
        if not dry_run:
            path.write_text(new, encoding="utf-8", newline="")
    return counts


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    targets = [Path(a) for a in argv if a != "--dry-run"]
    if not targets:
        print(__doc__)
        return 2
    files = []
    for t in targets:
        files += sorted(t.glob("*.json")) if t.is_dir() else [t]
    for f in files:
        counts = rename_file(f, dry)
        if counts:
            print(("would rename " if dry else "renamed ") + f"{f}: " +
                  ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
