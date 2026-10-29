"""Save an API-format graph as a GUI (litegraph) workflow Mike can open in ComfyUI.
Ported from howling-yowies/scripts/ui_workflow.py. Converts with the live node definitions from V141 (/object_info),
lays the graph out left to right with the comfyui skill's workflow_layout.auto_layout, and writes it to:
  ComfyUI_V141/ComfyUI/user/default/workflows/atlas-vfx/<name>.json   (Workflows sidebar)
  atlas-vfx/workflows/<name>.json                                      (repo copy)
Use from a script:   import ui_workflow; ui_workflow.save(graph, "t2_plate_read_s2s3")
Backfill from history: ui_workflow.py history <filename_prefix_substring> <name>"""
import json, sys, urllib.request
from pathlib import Path

API = "http://127.0.0.1:8288"
SIDEBAR = Path.home() / "ComfyUI_V141" / "ComfyUI" / "user" / "default" / "workflows" / "atlas-vfx"
REPO = Path(__file__).resolve().parents[1] / "workflows"
SHIPPED_OCIO = str(REPO.parent / "config" / "atlas_aces2.0_studio_flog2.ocio")
sys.path.insert(0, str(Path.home() / ".claude" / "skills" / "comfyui"))
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3", "FLOAT,INT", "INT,FLOAT"}
_INFO = {}


def _info(cls):
    if cls not in _INFO:
        _INFO[cls] = json.loads(urllib.request.urlopen(f"{API}/object_info/{cls}", timeout=30).read())[cls]
    return _INFO[cls]


def _is_widget(spec):
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):          # a FLOAT/INT/STRING the node only takes as a socket (no widget)
        return False
    return isinstance(t, list) or t in WIDGET_TYPES


def _has_control(name, spec):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return bool(opts.get("control_after_generate")) or (spec[0] == "INT" and name in ("seed", "noise_seed"))


def _default(spec):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return opts["default"]
    if isinstance(spec[0], list):
        return spec[0][0] if spec[0] else ""
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec[0], "")


def convert(graph, title=""):
    nodes, links, link_id = [], [], 0
    out_links = {}  # (node_id, slot) -> [link ids]
    ui = {}
    order = sorted(graph, key=lambda k: int(k) if str(k).isdigit() else 10 ** 9)
    for nid in order:
        node = graph[nid]
        info = _info(node["class_type"])
        specs = list(info["input"].get("required", {}).items()) + list(info["input"].get("optional", {}).items())
        inputs, widgets = [], []
        for name, spec in specs:
            val = node["inputs"].get(name)
            linked = isinstance(val, list) and len(val) == 2 and isinstance(val[1], int) and str(val[0]) in graph
            if _is_widget(spec):
                widgets.append(_default(spec) if (val is None or linked) else val)
                if _has_control(name, spec):
                    widgets.append("fixed")
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if opts.get("image_upload") or opts.get("video_upload"):
                    widgets.append("image")  # the upload button's widget slot
                if linked:
                    t = "COMBO" if isinstance(spec[0], list) else spec[0]
                    inputs.append({"name": name, "type": t, "widget": {"name": name}, "link": None, "_src": val})
            elif linked or name in info["input"].get("required", {}):
                inputs.append({"name": name, "type": spec[0], "link": None, "_src": val if linked else None})
        names = info.get("output_name") or info["output"]
        outputs = [{"name": n, "type": t, "links": []} for n, t in zip(names, info["output"])]
        ui[nid] = {"id": int(nid), "type": node["class_type"], "pos": [0, 0], "size": [300, 100], "flags": {},
                   "order": len(nodes), "mode": 0, "inputs": inputs, "outputs": outputs,
                   "properties": {"Node name for S&R": node["class_type"]}, "widgets_values": widgets}
        nodes.append(ui[nid])
    for n in nodes:
        for slot, inp in enumerate(n["inputs"]):
            src = inp.pop("_src")
            if not src:
                continue
            link_id += 1
            sid, sslot = int(src[0]), src[1]
            links.append([link_id, sid, sslot, n["id"], slot, inp["type"]])
            inp["link"] = link_id
            ui[str(src[0])]["outputs"][sslot]["links"].append(link_id)
    wf = {"last_node_id": max(n["id"] for n in nodes), "last_link_id": link_id, "nodes": nodes, "links": links,
          "groups": [], "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}}, "version": 0.4}
    try:
        from workflow_layout import auto_layout, fit_group
        wf = auto_layout(wf)
        if title:
            wf = fit_group(wf, title=title)
    except Exception as e:  # layout is a nicety; the graph is still valid without it
        print("layout skipped:", e)
    return wf


def portable(text: str) -> str:
    """The repo copy of a workflow: the home folder written as ~ (both slash styles, as JSON escapes them) and the
    shipped OCIO config blank (the nodes take blank as the shipped config). Path widgets expand ~ when they run.
    tests/test_hygiene.py fails on a committed workflow that still names the home folder."""
    home = str(Path.home())
    for cfg in (SHIPPED_OCIO, SHIPPED_OCIO.replace("\\", "/")):
        text = text.replace(json.dumps(cfg)[1:-1], "")
    for h in (home, home.replace("\\", "/")):
        text = text.replace(json.dumps(h)[1:-1], "~")
    return text


def save(graph, name, title=None):
    wf = convert(graph, title or name)
    text = json.dumps(wf, indent=1)
    SIDEBAR.mkdir(parents=True, exist_ok=True)
    (SIDEBAR / f"{name}.json").write_text(text, encoding="utf-8")         # the live copy keeps real paths
    REPO.mkdir(parents=True, exist_ok=True)
    (REPO / f"{name}.json").write_text(portable(text), encoding="utf-8")  # the committed copy is machine-neutral
    return SIDEBAR / f"{name}.json"


def from_history(match):
    h = json.loads(urllib.request.urlopen(f"{API}/history?max_items=500", timeout=60).read())
    for pid, rec in reversed(list(h.items())):
        g = rec["prompt"][2]
        if any(match in str(n["inputs"].get("filename_prefix", "")) for n in g.values()):
            return g
    return None


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == "history":
    g = from_history(sys.argv[2])
    if not g:
        sys.exit(f"no history entry matching {sys.argv[2]}")
    print("saved", save(g, sys.argv[3]))
