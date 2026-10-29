"""Put sha256 and size of every weights file into atlas_components.json (spec panel 2026-10-26, defence 2).

Each model component names its files in its note (`something.safetensors`). This finds each file under the model
roots, hashes it and writes a `files` list on the component: name, bytes, sha256, the model subfolder it was found
in, and the date. A sha256 already quoted in the note is checked against the computed one. Files already in
`files` are skipped unless --refresh.

Run: <V141 python> tools\\register_hashes.py [--roots D:/models;~/ComfyUI_V141/ComfyUI/models] [--refresh] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "atlas_components.json"
DEFAULT_ROOTS = ["D:/models", "~/ComfyUI_V141/ComfyUI/models"]
FILE_RE = re.compile(r"[\w][\w.\-]*\.(?:safetensors|pth|pt|ckpt|onnx|gguf)\b")
SHA_RE = re.compile(r"\b[0-9a-f]{64}\b")


def sha256(path: Path, chunk: int = 32 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def index(roots: list[Path]) -> dict[str, list[Path]]:
    """Every file under the roots by name (weights folders hold a few thousand files; a walk is cheap)."""
    out: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            print(f"root missing, skipped: {root}")
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                out.setdefault(name, []).append(Path(dirpath) / name)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", default=os.pathsep.join(DEFAULT_ROOTS), help="model roots, os.pathsep separated")
    ap.add_argument("--refresh", action="store_true", help="re-hash files already listed")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args(argv)
    roots = [Path(os.path.expandvars(os.path.expanduser(r))) for r in a.roots.split(os.pathsep) if r]
    by_name = index(roots)
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    today = dt.date.today().isoformat()
    problems = 0
    for comp in reg["components"]:
        if comp.get("usage") != "model":
            continue
        note = comp.get("note", "")
        names = list(dict.fromkeys(FILE_RE.findall(note)))
        if not names:
            print(f"{comp['id']}: no file named in the note")
            problems += 1
            continue
        quoted = SHA_RE.findall(note)
        files = {f["name"]: f for f in comp.get("files", [])}
        for name in names:
            if name in files and files[name].get("sha256") and not a.refresh:
                print(f"{comp['id']}: {name} already hashed, kept")
                continue
            hits = by_name.get(name, [])
            if not hits:
                print(f"{comp['id']}: {name} NOT FOUND under {[str(r) for r in roots]}")
                problems += 1
                continue
            if len(hits) > 1:
                sizes = {p.stat().st_size for p in hits}
                print(f"{comp['id']}: {name} found {len(hits)} times" + (" (same size)" if len(sizes) == 1 else " WITH DIFFERENT SIZES"))
                if len(sizes) > 1:
                    problems += 1
            p = hits[0]
            t0 = time.perf_counter()
            digest = sha256(p)
            size = p.stat().st_size
            root = next(r for r in roots if r in p.parents)
            sub = str(p.parent.relative_to(root)).replace("\\", "/")
            mark = ""
            if quoted:
                mark = " (matches the note)" if digest in quoted else " (DIFFERS from the sha256 in the note)"
                if digest not in quoted:
                    problems += 1
            print(f"{comp['id']}: {name} {size:,} bytes sha256 {digest} in {sub} ({time.perf_counter() - t0:.0f} s){mark}")
            files[name] = {"name": name, "bytes": size, "sha256": digest, "folder": sub, "hashed": today}
        if files:
            comp["files"] = [files[n] for n in names if n in files] + [f for n, f in files.items() if n not in names]
    if not a.dry_run:
        REGISTER.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {REGISTER}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
