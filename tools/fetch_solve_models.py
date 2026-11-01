"""Fetch the two Solve Mask models (catalogue 67) into ComfyUI's models folder and hash them into the register.

  grounding-dino-base   IDEA-Research/grounding-dino-base   -> <models>/detection/grounding-dino-base
  sam2.1-hiera-large    facebook/sam2.1-hiera-large         -> <models>/sam2/sam2.1-hiera-large

Only config/tokenizer JSON, text and *.safetensors are downloaded (no pickles, no code). Both are Apache-2.0; see
atlas_components.json. The sha256 and size of each weights file go into that component's `files` list
(both repos name their weights model.safetensors, so the name is written with its folder).

Run: <V141 python> tools\\fetch_solve_models.py [--models ~/ComfyUI_V141/ComfyUI/models] [--hash-only]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from register_hashes import REGISTER, sha256  # noqa: E402

MODELS = {
    "grounding-dino-base": ("IDEA-Research/grounding-dino-base", ("detection", "grounding-dino-base")),
    "sam2.1-hiera-large": ("facebook/sam2.1-hiera-large", ("sam2", "sam2.1-hiera-large")),
}
PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="~/ComfyUI_V141/ComfyUI/models")
    ap.add_argument("--hash-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.models).expanduser()
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    comps = {c["id"]: c for c in reg["components"]}
    for cid, (repo, sub) in MODELS.items():
        dst = root.joinpath(*sub)
        if not a.hash_only:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, local_dir=str(dst), allow_patterns=PATTERNS)
        files = []
        for f in sorted(dst.glob("*.safetensors")):
            files.append({"name": f"{sub[1]}/{f.name}", "bytes": f.stat().st_size, "sha256": sha256(f),
                          "folder": sub[0], "hashed": dt.date.today().isoformat()})
            print(cid, files[-1]["name"], files[-1]["bytes"], files[-1]["sha256"])
        if not files:
            print(f"{cid}: no safetensors in {dst}")
            return 1
        comps[cid]["files"] = files
    REGISTER.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
