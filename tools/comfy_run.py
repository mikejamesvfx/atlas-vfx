"""Queue an API graph on V141, wait, print outputs; saves the GUI workflow first (ui_workflow.save)."""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_workflow  # noqa: E402

API = "http://127.0.0.1:8288"


def _post(path, body):
    req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def run(graph: dict, name: str, title: str, timeout: float = 1800) -> dict:
    print("gui workflow:", ui_workflow.save(graph, name, title))
    t0 = time.time()
    pid = _post("/prompt", {"prompt": graph, "client_id": f"atlas-vfx-{name}"})["prompt_id"]
    while True:
        time.sleep(2)
        h = json.loads(urllib.request.urlopen(f"{API}/history/{pid}", timeout=30).read())
        if pid in h:
            break
        if time.time() - t0 > timeout:
            sys.exit("timed out")
    rec = h[pid]
    st = rec["status"]
    print(f"status {st['status_str']} in {time.time() - t0:.0f}s")
    if st["status_str"] != "success":
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                print(m[1].get("exception_message"), "\n", "".join(m[1].get("traceback", []))[-3000:])
        sys.exit(1)
    for nid, out in rec["outputs"].items():
        shown = {k: (v if k == "text" else len(v) if isinstance(v, list) else v) for k, v in out.items()}
        print(nid, graph[nid]["class_type"], shown)
    return rec
