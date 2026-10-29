"""Node contracts (spec panel 2026-10-29, items 3-5, 8): every node's sockets are pinned, the committed workflows
link only sockets that exist, and docs/NODE_CATALOGUE.md describes each node the way the code does.

1. Snapshot. Every loaded node's CATEGORY, module folder, inputs (name, type, required or optional) and outputs (name,
   type, in order) must equal tests/node_contracts.json. A saved graph links by name, type and slot, so any change
   here breaks somebody's graph. Change it on purpose: run with --update and commit the json with the code.
2. Workflows. In workflows/*.json every atlas-vfx node's inputs and outputs exist on the node with the same type,
   in the same output slot, and every link carries its output's type (or '*').
3. Catalogue. Each loaded node has exactly one contract block (`### <row> <name>` + a line with the class key in
   backticks); the folder on that line is the node's subpackage; its `in:` / `out:` lines equal the generated ones;
   the Types table equals the generated one; the `test:` line names files in tests/ that exist; every `licences:`
   id is in atlas_components.json.

Run: <V141 python> tests\\test_node_contracts.py [--update]
     --update rewrites tests/node_contracts.json, the in/out lines and the Types table in the catalogue, then checks.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SNAPSHOT = ROOT / "tests" / "node_contracts.json"
CATALOGUE = ROOT / "docs" / "NODE_CATALOGUE.md"
REGISTER = ROOT / "atlas_components.json"
WORKFLOWS = ROOT / "workflows"
TYPES_BEGIN, TYPES_END = "<!-- types:begin (generated) -->", "<!-- types:end -->"
EXTERNAL = {"MOGE_GEOMETRY", "MOGE_MODEL"}  # ComfyUI core (MoGe nodes); listed in the Types table, not namespaced here
BUILTIN = {"STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE", "MASK", "COMBO", "LATENT", "MODEL", "CLIP", "VAE",
           "CONDITIONING", "AUDIO", "VIDEO"}
COMBO_SHOW = 4  # combos with up to this many options list them in the catalogue


def _spec_type(spec) -> str:
    t = spec[0]
    return "COMBO" if isinstance(t, (list, tuple)) else str(t)


def contracts() -> tuple[dict, dict]:
    """(contracts by class key, disabled) from the live registry, no ComfyUI needed."""
    from atlas_vfx import registry
    classes, _names, disabled = registry.load_nodes()
    out = {}
    for key in sorted(classes):
        cls = classes[key]
        it = cls.INPUT_TYPES()
        inputs = []
        for group in ("required", "optional"):
            for name, spec in (it.get(group) or {}).items():
                entry = {"name": name, "type": _spec_type(spec), "group": group}
                if entry["type"] == "COMBO" and len(spec[0]) <= COMBO_SHOW:
                    entry["options"] = [str(o) for o in spec[0]]
                inputs.append(entry)
        rt = list(cls.RETURN_TYPES)
        rn = list(getattr(cls, "RETURN_NAMES", None) or [t.lower() for t in rt])
        out[key] = {
            "folder": cls.__module__.split(".")[1],
            "category": cls.CATEGORY,
            "inputs": inputs,
            "outputs": [{"name": n, "type": t} for n, t in zip(rn, rt)],
        }
    return out, disabled


# ---- generated catalogue text ------------------------------------------------------------------------------------

def _fmt_input(i) -> str:
    t = "|".join(i["options"]) if "options" in i else i["type"]
    return f"{i['name']} {t}"


def in_out_lines(c) -> list[str]:
    req = [_fmt_input(i) for i in c["inputs"] if i["group"] == "required"]
    opt = [_fmt_input(i) for i in c["inputs"] if i["group"] == "optional"]
    lines = [f"- in: {', '.join(req) or '(none)'}"]
    if opt:
        lines.append(f"- in, optional: {', '.join(opt)}")
    lines.append(f"- out: {', '.join(o['name'] + ' ' + o['type'] for o in c['outputs']) or '(none)'}")
    return lines


def types_table(cons: dict, rows: dict[str, str]) -> str:
    made, read = {}, {}
    for key, c in cons.items():
        label = rows.get(key, key)
        for o in c["outputs"]:
            if o["type"] not in BUILTIN:
                made.setdefault(o["type"], []).append(label)
        for i in c["inputs"]:
            if i["type"] not in BUILTIN:
                read.setdefault(i["type"], []).append(label)
    lines = ["| Type | Made by | Read by |", "|---|---|---|"]
    for t in sorted(set(made) | set(read)):
        lines.append(f"| `{t}` | {', '.join(sorted(set(made.get(t, [])), key=_rowkey)) or '-'} | "
                     f"{', '.join(sorted(set(read.get(t, [])), key=_rowkey)) or '-'} |")
    return "\n".join(lines)


def _rowkey(label: str):
    m = re.match(r"(\d+)", label)
    return (int(m.group(1)) if m else 999, label)


# ---- catalogue parsing -------------------------------------------------------------------------------------------

BLOCK_HEAD = re.compile(r"^### (\d+) (.+)$")
CLASS_LINE = re.compile(r"^`(AtlasVFX\w+)` · (\w+) · (.+)$")


def parse_blocks(text: str) -> dict[str, dict]:
    """class key -> {row, name, folder, category, start, end, lines} for every contract block."""
    lines = text.split("\n")
    blocks, cur = {}, None
    for idx, line in enumerate(lines):
        h = BLOCK_HEAD.match(line)
        if h or line.startswith("## "):
            if cur:
                cur["end"] = idx
                cur = None
        if h:
            nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
            m = CLASS_LINE.match(nxt)
            if not m:
                continue
            key = m.group(1)
            if key in blocks:
                raise AssertionError(f"catalogue: {key} has two contract blocks")
            cur = blocks[key] = {"row": h.group(1), "name": h.group(2), "folder": m.group(2),
                                 "category": m.group(3), "start": idx, "end": len(lines)}
    for b in blocks.values():
        b["lines"] = lines[b["start"]:b["end"]]
    return blocks


def field(block, name) -> str | None:
    for line in block["lines"]:
        if line.startswith(f"- {name}:"):
            return line[len(name) + 3:].strip()
    return None


def update_catalogue(cons: dict) -> None:
    text = CATALOGUE.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    lines = text.split("\n")
    # rewrite in/out lines, last block first so indexes stay valid
    for key, b in sorted(blocks.items(), key=lambda kv: -kv[1]["start"]):
        if key not in cons:
            continue
        body = lines[b["start"]:b["end"]]
        keep = [l for l in body if not re.match(r"^- (in|in, optional|out):", l)]
        new = keep[:2] + in_out_lines(cons[key]) + keep[2:]
        lines[b["start"]:b["end"]] = new
    text = "\n".join(lines)
    rows = {k: f"{b['row']} {b['name']}" for k, b in parse_blocks(text).items()}
    a, z = text.index(TYPES_BEGIN) + len(TYPES_BEGIN), text.index(TYPES_END)
    text = text[:a] + "\n" + types_table(cons, rows) + "\n" + text[z:]
    CATALOGUE.write_text(text, encoding="utf-8", newline="")


# ---- checks ------------------------------------------------------------------------------------------------------

def check_snapshot(cons, fails):
    if not SNAPSHOT.exists():
        fails.append(f"{SNAPSHOT.name} missing: run with --update")
        return
    pinned = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for key in sorted(set(pinned) | set(cons)):
        if key not in cons:
            fails.append(f"snapshot: {key} pinned but not loaded (removed, renamed or disabled)")
        elif key not in pinned:
            fails.append(f"snapshot: {key} loads but is not pinned (new node: --update)")
        elif pinned[key] != cons[key]:
            for part in ("folder", "category", "inputs", "outputs"):
                if pinned[key][part] != cons[key][part]:
                    fails.append(f"snapshot: {key}.{part} changed\n      was {pinned[key][part]}\n      now "
                                 f"{cons[key][part]}")


def check_workflows(cons, fails) -> int:
    n = 0
    for wf in sorted(WORKFLOWS.glob("*.json")):
        g = json.loads(wf.read_text(encoding="utf-8"))
        nodes = {nd["id"]: nd for nd in g.get("nodes", [])}
        for nd in nodes.values():
            c = cons.get(nd["type"])
            if c is None:
                if nd["type"].startswith("AtlasVFX"):
                    fails.append(f"{wf.name}: node {nd['id']} {nd['type']} is not a loaded node")
                continue
            n += 1
            ins = {i["name"]: i["type"] for i in c["inputs"]}
            for i in nd.get("inputs") or []:
                if i["name"] not in ins:
                    fails.append(f"{wf.name}: {nd['type']} has no input '{i['name']}'")
                elif i.get("type") not in (ins[i["name"]], "*") and not (
                        ins[i["name"]] == "COMBO" and "widget" in i):
                    fails.append(f"{wf.name}: {nd['type']}.{i['name']} is {i.get('type')}, node says "
                                 f"{ins[i['name']]}")
            for slot, o in enumerate(nd.get("outputs") or []):
                if slot >= len(c["outputs"]):
                    fails.append(f"{wf.name}: {nd['type']} has no output slot {slot} ('{o['name']}')")
                    continue
                want = c["outputs"][slot]
                if (o["name"], o["type"]) != (want["name"], want["type"]):
                    fails.append(f"{wf.name}: {nd['type']} output {slot} is {o['name']} {o['type']}, node says "
                                 f"{want['name']} {want['type']}")
        for link in g.get("links", []):
            _id, src, slot, _dst, _dslot, ltype = link[:6]
            s = nodes.get(src)
            if s and s["type"] in cons and ltype != "*" and slot < len(cons[s["type"]]["outputs"]):
                if ltype != cons[s["type"]]["outputs"][slot]["type"]:
                    fails.append(f"{wf.name}: link {_id} carries {ltype}, {s['type']} output {slot} is "
                                 f"{cons[s['type']]['outputs'][slot]['type']}")
    return n


def check_catalogue(cons, fails) -> int:
    text = CATALOGUE.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    register = {c["id"] for c in json.loads(REGISTER.read_text(encoding="utf-8"))["components"]}
    tests = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    for key, c in cons.items():
        b = blocks.get(key)
        if b is None:
            fails.append(f"catalogue: no contract block for {key}")
            continue
        if b["folder"] != c["folder"]:
            fails.append(f"catalogue: {key} folder says {b['folder']}, the module is in {c['folder']}")
        if b["category"] != c["category"]:
            fails.append(f"catalogue: {key} category says {b['category']}, node says {c['category']}")
        got = [l for l in b["lines"] if re.match(r"^- (in|in, optional|out):", l)]
        if got != in_out_lines(c):
            fails.append(f"catalogue: {key} in/out lines differ from the code (--update)")
        t = field(b, "test")
        if not t:
            fails.append(f"catalogue: {key} has no test line")
        else:
            for name in re.findall(r"test_\w+\.py", t):
                if name not in tests:
                    fails.append(f"catalogue: {key} test {name} does not exist")
            if not re.findall(r"test_\w+\.py", t):
                fails.append(f"catalogue: {key} test line names no tests/test_*.py")
        lic = field(b, "licences")
        if lic is None:
            fails.append(f"catalogue: {key} has no licences line")
        else:
            for lid in re.findall(r"`([a-z0-9][a-z0-9.\-]*)`", lic):
                if lid not in register:
                    fails.append(f"catalogue: {key} licence id {lid} is not in atlas_components.json")
        if field(b, "needs") is None:
            fails.append(f"catalogue: {key} has no needs line")
        if field(b, "proof") is None:
            fails.append(f"catalogue: {key} has no proof line")
    for key in blocks:
        if key not in cons:
            fails.append(f"catalogue: contract block for {key}, which does not load")
    rows = {k: f"{b['row']} {b['name']}" for k, b in blocks.items()}
    if TYPES_BEGIN not in text or TYPES_END not in text:
        fails.append("catalogue: Types table markers missing")
    else:
        a, z = text.index(TYPES_BEGIN) + len(TYPES_BEGIN), text.index(TYPES_END)
        if text[a:z].strip() != types_table(cons, rows):
            fails.append("catalogue: Types table differs from the code (--update)")
    return len(blocks)


def check_type_names(cons, fails):
    """Custom types carry a namespace (panel item 1): camera_side ATLAS_ (they move into Atlas Camera), the rest
    ATLAS_VFX_. Atlas Camera's own names are taken: its payloads differ."""
    taken = {"ATLAS_SHOT_CAM", "ATLAS_PLATE_REF", "ATLAS_CAMERA", "ATLAS_SOLVE", "ATLAS_REAL_PLATE"}
    for key, c in cons.items():
        for s in c["outputs"] + c["inputs"]:
            t = s["type"]
            if t in BUILTIN or t in EXTERNAL:
                continue
            if t in taken:
                fails.append(f"types: {key} uses {t}, which is Atlas Camera's (different payload)")
            elif not t.startswith("ATLAS_"):
                fails.append(f"types: {key} uses {t} with no ATLAS_ prefix")
            elif c["folder"] != "camera_side" and not t.startswith("ATLAS_VFX_") and not _made_by_camera_side(
                    cons, t):
                fails.append(f"types: {key} uses {t}; non-camera-side types are ATLAS_VFX_")


def _made_by_camera_side(cons, t) -> bool:
    return any(c["folder"] == "camera_side" and any(o["type"] == t for o in c["outputs"]) for c in cons.values())


def main(argv) -> int:
    cons, disabled = contracts()
    if "--update" in argv:
        if disabled:
            print(f"refusing --update with disabled nodes (their contracts would be dropped): {disabled}")
            return 1
        SNAPSHOT.write_text(json.dumps(cons, indent=1) + "\n", encoding="utf-8", newline="\n")
        update_catalogue(cons)
        print(f"updated {SNAPSHOT.name} and the catalogue's in/out lines and Types table ({len(cons)} nodes)")
    fails: list[str] = []
    if disabled:
        print(f"note: {len(disabled)} disabled here, not checked: {', '.join(disabled)}")
    check_snapshot(cons, fails)
    n_wf = check_workflows(cons, fails)
    n_blocks = check_catalogue(cons, fails)
    check_type_names(cons, fails)
    if fails:
        print(f"FAIL node contracts: {len(fails)} problem(s)")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"ok snapshot: {len(cons)} nodes pinned (inputs, outputs, category, folder)")
    print(f"ok workflows: {n_wf} atlas-vfx nodes in {len(list(WORKFLOWS.glob('*.json')))} workflows link real sockets")
    print(f"ok catalogue: {n_blocks} contract blocks match the code; tests, licences, needs and proof lines present")
    print("ok type names: every custom type namespaced, none of Atlas Camera's")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
