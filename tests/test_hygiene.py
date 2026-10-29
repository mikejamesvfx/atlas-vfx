"""The tracked tree carries nothing that belongs to one machine or one person (spec panel 2026-10-26, Hygiene).

Rules over `git ls-files`:
1. No tracked file names the LAN (192.168.) or the farm token (FARM_REMOTE_TOKEN). No exceptions.
2. The shipped parts (atlas_vfx/, workflows/, config/, web/, README, pyproject, the root __init__, the licence register)
   carry no user folder (C:\\Users or C:/Users) and no owner username: the package runs from any home folder, and
   the workflows say `~` for the home folder and leave the OCIO config blank (= the shipped one).
3. Everything else (docs/, tools/, tests/, prototypes/) may still hold the owner's paths as evidence or
   one-machine scripts, but only as much as the BASELINE below: counts may go down, never up, and a new file with a
   hit fails. Lower a number here when you remove some.

Run: <V141 python> tests\\test_hygiene.py   (plain script, exits non-zero on failure; no pytest)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_hygiene.py"      # holds the patterns it bans, so it is the one file not scanned

LAN = re.compile(r"192\.168\.|FARM_REMOTE_TOKEN")
PERSONAL = re.compile(r"[Cc]:[\\/]+Users\b|miike", re.I)
SHIPPED = ("atlas_vfx/", "workflows/", "config/", "web/", "README.md", "pyproject.toml", "requirements.txt",
           "__init__.py", "atlas_components.json", ".gitignore", "LICENSE", "AGENTS.md", "SECURITY.md")
TEXT = (".py", ".md", ".json", ".toml", ".ocio", ".txt", ".gitignore", "LICENSE", ".cfg", ".ini", ".yaml", ".yml",
        ".js", ".html")

# personal-path hits still allowed, per file (rule 3). Filled from the tree on 2026-10-26; only ever lower it.
BASELINE: dict[str, int] = {
    "docs/dev/IMPLEMENTATION_PLAN.md": 6,
    "docs/dev/NOTES.md": 8,
    "docs/PICKING.md": 2,
    "docs/reviews/SPEC_PANEL_2026-10-26.md": 10,
    "prototypes/howling_yowies/nuke_export_for_claude.py": 6,
    "prototypes/vfx_probe/align_nuke.py": 2,
    "tests/test_camera_io.py": 4,
    "tests/test_compare.py": 4,
    "tests/test_plate.py": 2,
    "tests/test_shot_camera.py": 4,
    "tools/gpu_turn.py": 4,
    "tools/platelock_crossview.py": 2,
    "tools/rs_clip_solve.py": 4,
    "tools/rs_deck_proof.py": 4,
    "tools/rs_readout_profile.py": 2,
    "tools/rs_solve_dscf3000.py": 2,
    "tools/run_guess_s2s3.py": 2,
    "tools/run_t2_s2s3.py": 4,
    "tools/run_t3_s2s3.py": 4,
    "tools/run_t4_proto.py": 4,
    "tools/run_t6_grass.py": 2,
    "tools/run_t8_nuke.py": 2,
    "tools/sensor_workflow.py": 4,
    "tools/track_picks.py": 2,
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True)
    return [f for f in out.stdout.decode("utf-8").split("\0") if f]


def is_text(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(TEXT) or name in ("LICENSE", ".gitignore")


def hits(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def main() -> int:
    files = [f for f in tracked_files() if is_text(f) and f != SELF]
    assert files, "git ls-files returned nothing; run from a checkout"
    problems: list[str] = []
    personal: dict[str, int] = {}
    for rel in files:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        if hits(LAN, text):
            problems.append(f"{rel}: names the LAN or the farm token ({hits(LAN, text)} hits)")
        n = hits(PERSONAL, text)
        if not n:
            continue
        personal[rel] = n
        if rel.startswith(SHIPPED) or rel in SHIPPED:
            problems.append(f"{rel}: shipped file with a user folder or username ({n} hits)")
        elif rel not in BASELINE:
            problems.append(f"{rel}: new file with a user folder or username ({n} hits); scrub it or add it to BASELINE")
        elif n > BASELINE[rel]:
            problems.append(f"{rel}: {n} personal-path hits, baseline {BASELINE[rel]}; scrub the new ones")
    stale = [f for f in BASELINE if f not in personal]
    if problems:
        print("hygiene problems:\n  " + "\n  ".join(problems))
        print("\ncurrent personal-path counts (for BASELINE):")
        for rel, n in sorted(personal.items()):
            print(f'    "{rel}": {n},')
        return 1
    if stale:
        print(f"note: BASELINE entries now clean, remove them: {stale}")
    total = sum(personal.values())
    print(f"ok hygiene: {len(files)} tracked text files, no LAN addresses or token names, shipped files clean, "
          f"{total} personal-path hits left in {len(personal)} allowed files (baseline {sum(BASELINE.values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
