"""Run every tests/test_*.py as its own process and print one line per test, with the time.

Run: <V141 python> tests\\run_all.py [substring ...] [--fail-fast]
    substrings filter the test names (any match keeps the test); no pytest, each test is a plain script.
Exit code: the number of failed tests (0 = all passed). A failing test's last lines are shown under the table.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).name


def main(argv: list[str]) -> int:
    fail_fast = "--fail-fast" in argv
    want = [a for a in argv if not a.startswith("--")]
    tests = sorted(p for p in HERE.glob("test_*.py") if p.name != SELF)
    if want:
        tests = [t for t in tests if any(w in t.stem for w in want)]
    if not tests:
        print("no tests match", want)
        return 1
    width = max(len(t.stem) for t in tests)
    failed: list[tuple[str, str]] = []
    t_all = time.perf_counter()
    for t in tests:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(t)], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(HERE.parent))
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {t.stem:<{width}} {dt:7.1f} s", flush=True)
        if not ok:
            tail = "\n".join(((r.stdout or "") + "\n" + (r.stderr or "")).strip().splitlines()[-15:])
            failed.append((t.stem, tail))
            if fail_fast:
                break
    print(f"\n{len(tests) - len(failed)} of {len(tests)} passed in {time.perf_counter() - t_all:.0f} s")
    for name, tail in failed:
        print(f"\n--- {name} (last lines) ---\n{tail}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
