"""Clean-install check: a ComfyUI that has never seen this machine loads atlas-vfx and renders the starter sky.

What it does
  1. a scratch base folder: custom_nodes/atlas-vfx (a fresh `git clone` of --repo at --ref, so only committed files),
     an empty home folder, a work folder. Nothing else is in custom_nodes.
  2. starts ComfyUI (`main.py --base-directory <scratch>`) on its own port with HOME / USERPROFILE pointed at the empty
     home and every ATLAS_* variable removed, so Atlas Camera, Atlas Scene and the owner's plate can't be found.
  3. checks the log has no import failure for atlas-vfx and /object_info serves all the nodes in
     tests/node_contracts.json.
  4. queues the starter graph (tools/physical_sky_workflow.py, width 256 for speed) and checks it finished with the
     Hosek-Wilkie sky, an EXR and its info JSON in the scratch work folder.
  5. stops ComfyUI and deletes the scratch folder (--keep leaves it).

Runs with any ComfyUI checkout and its python. On ws (shares V141's code and venv, never its custom_nodes):
    <V141 python> tools\\clean_install_check.py --comfy %USERPROFILE%\\ComfyUI_V141\\ComfyUI
In CI (.github/workflows/tests.yml) against a fresh ComfyUI clone, with --cpu.
Exit 0 = pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

PACK = "atlas-vfx"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def clean_env(home: Path, work: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("ATLAS_")}
    env.update({"HOME": str(home), "USERPROFILE": str(home), "ATLAS_VFX_WORK": str(work),
                "PYTHONIOENCODING": "utf-8"})
    return env


def wait_up(api: str, proc: subprocess.Popen, log: Path, seconds: float) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            fail(f"ComfyUI exited ({proc.returncode}) while starting; log tail:\n{tail(log)}")
        try:
            get(f"{api}/system_stats", 3)
            return
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(2)
    fail(f"ComfyUI did not answer on {api} within {seconds:.0f} s; log tail:\n{tail(log)}")


def tail(log: Path, n: int = 30) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def run_prompt(api: str, graph: dict, seconds: float) -> dict:
    pid = post(f"{api}/prompt", {"prompt": graph, "client_id": "atlas-vfx-clean-install"})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < seconds:
        h = get(f"{api}/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            return h[pid]
        time.sleep(2)
    fail(f"the starter prompt did not finish within {seconds:.0f} s")
    return {}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--comfy", required=True, help="ComfyUI checkout (the folder with main.py)")
    ap.add_argument("--python", default=sys.executable, help="ComfyUI's python (default: this one)")
    ap.add_argument("--repo", default=str(REPO), help="git URL or path to clone (default: this checkout)")
    ap.add_argument("--ref", default="HEAD", help="branch or tag to clone (default: the checkout's HEAD)")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--cpu", action="store_true", help="pass --cpu to ComfyUI (CI runners have no GPU)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds for start-up and for the render, each")
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder")
    a = ap.parse_args(argv)

    comfy = Path(os.path.expandvars(os.path.expanduser(a.comfy))).resolve()
    if not (comfy / "main.py").is_file():
        fail(f"no main.py in {comfy}")
    scratch = Path(tempfile.mkdtemp(prefix="atlas_vfx_clean_"))
    home, work, nodes = scratch / "home", scratch / "work", scratch / "custom_nodes"
    for d in (home, work, nodes):
        d.mkdir(parents=True)
    api = f"http://127.0.0.1:{a.port}"
    log = scratch / "comfyui.log"
    proc = None
    try:
        # 1. a fresh clone: only what git tracks reaches the pack
        ref, dest = a.ref, nodes / PACK
        if Path(a.repo).is_dir():          # a local checkout (also CI's detached one): clone it, check out the commit
            sha = subprocess.run(["git", "-C", a.repo, "rev-parse", ref], capture_output=True, text=True,
                                 check=True).stdout.strip()
            subprocess.run(["git", "clone", "-q", "--no-checkout", a.repo, str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)
        else:
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", ref, a.repo, str(dest)], check=True)
        commit = subprocess.run(["git", "-C", str(nodes / PACK), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        print(f"ok   cloned {a.repo} @ {ref} ({commit}) into a custom_nodes of its own")

        # 2. ComfyUI with nothing of this machine in reach
        cmd = [a.python, "main.py", "--base-directory", str(scratch), "--port", str(a.port), "--listen", "127.0.0.1",
               "--disable-auto-launch"] + (["--cpu"] if a.cpu else [])
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(comfy), env=clean_env(home, work), stdout=fh,
                                    stderr=subprocess.STDOUT)
        wait_up(api, proc, log, a.timeout)
        print(f"ok   ComfyUI up on {api} (empty home, no ATLAS_* variables)")

        # 3. the pack loaded, every node registered
        text = log.read_text(encoding="utf-8", errors="replace")
        bad = [ln for ln in text.splitlines() if "IMPORT FAILED" in ln and PACK in ln]
        if bad:
            fail("atlas-vfx failed to import:\n" + "\n".join(bad) + f"\n{tail(log)}")
        want = sorted(json.loads((REPO / "tests" / "node_contracts.json").read_text(encoding="utf-8")))
        info = get(f"{api}/object_info", 60)
        missing = [n for n in want if n not in info]
        if missing:
            fail(f"{len(missing)} nodes not registered: {missing}\n{tail(log)}")
        print(f"ok   all {len(want)} atlas-vfx nodes registered, no import failure")

        # 4. the starter graph renders the shipped sky
        import physical_sky_workflow as psw
        graph = psw.graph(256, "clean_install_starter")
        h = run_prompt(api, graph, a.timeout)
        st = h.get("status", {})
        if st.get("status_str") != "success":
            msgs = json.dumps(st.get("messages", []))[-2000:]
            fail(f"starter prompt status {st.get('status_str')!r}: {msgs}\n{tail(log)}")
        out = work / "physical_sky" / "clean_install_starter"
        exrs, infos = sorted(out.glob("*.exr")), sorted(out.glob("*.json"))
        if not exrs:
            fail(f"no EXR under {out}: {sorted(p.name for p in out.glob('*')) if out.exists() else 'no folder'}")
        used = [p.name for p in infos if '"hosek-wilkie"' in p.read_text(encoding="utf-8", errors="replace")]
        if not used:
            fail(f"no info JSON under {out} records the hosek-wilkie sky: {[p.name for p in infos]}")
        print(f"ok   starter sky rendered with Hosek-Wilkie: {exrs[0].name} ({exrs[0].stat().st_size} bytes), {used[0]}")
        print("\nclean install check passed")
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if a.keep:
            print(f"kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
