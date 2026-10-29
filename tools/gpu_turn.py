"""Take turns on ws's GPU / ComfyUI V141 with other sessions (Mike: "take turns with the other agent").

    python tools/gpu_turn.py take <who> <what>   # waits until V141's queue is empty and no one holds the turn
    python tools/gpu_turn.py release <who>
    python tools/gpu_turn.py status

The turn is a plain text file, C:/Users/miike/atlas-vfx-work/GPU_TURN.txt, so any session (or Mike) can read it. A
turn older than 3 hours is treated as stale. Restarting V141 also needs the turn: never restart under someone's job.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

TURN = Path(r"C:\Users\miike\atlas-vfx-work\GPU_TURN.txt")
API = "http://127.0.0.1:8288"
STALE_S = 3 * 3600


def queue_busy() -> int:
    try:
        q = json.loads(urllib.request.urlopen(f"{API}/queue", timeout=10).read())
        return len(q["queue_running"]) + len(q["queue_pending"])
    except Exception:
        return 0                                    # V141 down: nothing running on it


def holder():
    if not TURN.exists():
        return None
    if time.time() - TURN.stat().st_mtime > STALE_S:
        return None
    return TURN.read_text(encoding="utf-8").strip()


def take(who: str, what: str, wait_s: float = 6 * 3600) -> None:
    t0 = time.time()
    while True:
        h, busy = holder(), queue_busy()
        if (h is None or h.startswith(who + " ")) and busy == 0:
            TURN.write_text(f"{who} since {datetime.now():%Y-%m-%d %H:%M} : {what}\n", encoding="utf-8")
            print("turn taken:", TURN.read_text(encoding="utf-8").strip())
            return
        if time.time() - t0 > wait_s:
            sys.exit(f"gave up waiting: holder={h!r} queue={busy}")
        print(f"waiting: holder={h!r} queue={busy}", flush=True)
        time.sleep(60)


def release(who: str) -> None:
    h = holder()
    if h and h.startswith(who + " "):
        TURN.unlink()
        print("turn released")
    else:
        print("not holding the turn:", h)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "take":
        take(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "release":
        release(sys.argv[2])
    else:
        print("holder:", holder(), "| V141 queue:", queue_busy())
