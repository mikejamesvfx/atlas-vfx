"""HDR spike (flow review 2026-11-01, Phase F step 2): can the LTX-2.5 SDR-to-HDR IC-LoRA give back the highlights
the SDR round trip clips? Measured against ground truth: the plate itself.

  1. DSCF3000 S2-S3, 33 frames from 1130, decoded to ACEScg (the truth) and shown through the ACES 2.0 SDR view
     LTX Prep uses (what LTX sees and gives back), written as a near-lossless mp4 into ComfyUI's input folder.
  2. The owner's SDR->HDR lookdev graph (~/Downloads/ltx25_hdr_workflows/..._fullhd_lookdev_API.json: LTX-2.5
     distilled + sdr-to-hdr IC-LoRA + scene embedding, tiled fusion sampler, ACEScct decode -> ACEScg EXR) on it,
     through V141's API.
  3. HDR EXRs vs the truth, and the plain inverse view (LTX Return today) vs the truth:
       mids        median log2(out / truth) where truth luminance is 0.05-0.5 (exposure match)
       highlights  where the SDR view clips (display >= 0.99): median log2(out / truth), share within 1/2 stop
     A number per metric, both methods, into atlas-vfx-work/hdr_spike/report.json. Nothing is built on it until
     Mike has seen the numbers.

The plate is only read. Run: <V141 python> tools\\hdr_spike.py [--frames 33] [--width 1920] [--api ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_vfx.camera_side import plate as pl  # noqa: E402
from atlas_vfx.layers import ltx_integration as li  # noqa: E402

PLATE = Path.home() / "Videos" / "atlas-vfx_footage" / "DSCF3000.MOV"
GRAPH = Path.home() / "Downloads" / "ltx25_hdr_workflows" / "2026-10-20_ltx25_sdr2hdr_fullhd_lookdev_API.json"
COMFY = Path.home() / "ComfyUI_V141" / "ComfyUI"
AP1_Y = np.array([0.2722287, 0.6740818, 0.0536895], np.float32)


def lum(x):
    return x[..., :3] @ AP1_Y


def make_inputs(first: int, n: int, width: int, out: Path):
    ref = pl.plate_ref(PLATE, first - 1001, first - 1001 + n - 1)
    fwd, inv = li.view_processors(ref["ocio_config"])
    truth, sdr = [], []
    for _i, img in pl.decode_rgb(ref, width=width):
        lin = pl.to_working(img, ref)
        truth.append(lin)
        sdr.append(li.to_view(lin, fwd))
    truth, sdr = np.stack(truth), np.stack(sdr)
    h, w = sdr.shape[1:3]
    mp4 = COMFY / "input" / "atlas_hdr_spike_s2s3.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
                            "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "4", "-preset", "slow", "-pix_fmt",
                            "yuv444p", "-color_range", "pc", str(mp4)], stdin=subprocess.PIPE)
    enc.stdin.write((np.clip(sdr, 0, 1) * 65535 + 0.5).astype("<u2").tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    np.save(out / "truth_acescg.npy", truth.astype(np.float16))
    return mp4.name, truth, sdr, inv


def queue(api: str, mp4_name: str, prefix: str) -> str:
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    for n in g.values():
        if n["class_type"] == "LoadVideo":
            n["inputs"]["file"] = mp4_name
        if n["class_type"] == "LTXVHDRDecodePostprocess":
            n["inputs"].update({"output_dir": "atlas-vfx/hdr_spike/exr", "filename_prefix": prefix, "save_exr": True,
                                "exr_color_space": "acescg"})
        if n["class_type"] == "ComfyMathExpression" and n["inputs"].get("expression") == "a if a <= 30 else 30":
            n["inputs"]["expression"] = "a"                       # the plate's own 24 fps
        if n["class_type"] in ("SaveVideo", "LTXVSaveHLG") and "filename_prefix" in n["inputs"]:
            n["inputs"]["filename_prefix"] = f"atlas-vfx/hdr_spike/{prefix}_{n['class_type'].lower()}"
    body = json.dumps({"prompt": g, "client_id": "atlas-vfx-hdr-" + uuid.uuid4().hex[:6]}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{api}/prompt", data=body,
                                                                 headers={"Content-Type": "application/json"}),
                                          timeout=60).read())
    if r.get("node_errors"):
        raise SystemExit(f"node errors: {r['node_errors']}")
    return r["prompt_id"]


def wait(api: str, pid: str, timeout: float = 3 * 3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(urllib.request.urlopen(f"{api}/history/{pid}", timeout=30).read())
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                errs = [m[1] for m in st.get("messages", []) if m[0] == "execution_error"]
                raise SystemExit(f"run failed: {errs[:1]}")
            return round(time.time() - t0)
        time.sleep(15)
    raise SystemExit("timed out")


def read_exrs(prefix: str, n: int, size) -> np.ndarray:
    import cv2

    from atlas_vfx.camera_side import exr_io

    d = COMFY / "output" / "atlas-vfx" / "hdr_spike" / "exr"
    out = []
    for i in range(n):
        img = exr_io.read_rgba(d / f"{prefix}_{i:05d}.exr")[0][..., :3]
        if img.shape[1::-1] != tuple(size):
            img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
        out.append(img)
    return np.stack(out)


def stats(out, truth, sdr):
    L, Lt = lum(out), lum(truth)
    ok = (Lt > 1e-4) & (L > 1e-6)
    r = np.log2(np.where(ok, L, 1) / np.where(ok, Lt, 1))
    mids = ok & (Lt > 0.05) & (Lt < 0.5)
    hi = ok & (sdr.max(-1) >= 0.99)
    res = {"mids_median_stops": float(np.median(r[mids])) if mids.any() else None,
           "highlight_pixels_pct": round(100 * float(hi.mean()), 3)}
    if hi.any():
        res.update({"highlights_median_stops": float(np.median(r[hi])),
                    "highlights_within_half_stop_pct": round(100 * float((np.abs(r[hi]) < 0.5).mean()), 1),
                    "highlights_truth_lum_p50_p99": [float(np.percentile(Lt[hi], 50)),
                                                     float(np.percentile(Lt[hi], 99))],
                    "highlights_out_lum_p50_p99": [float(np.percentile(L[hi], 50)), float(np.percentile(L[hi], 99))]})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1130)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    out = pl.resolve_out_dir("", pl.WORK_ROOT / "hdr_spike")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mp4, truth, sdr, inv = make_inputs(a.first, a.frames, a.width, out)
    print(f"inputs: {len(truth)} frames {truth.shape[2]}x{truth.shape[1]}, {mp4}", flush=True)
    prefix = f"spike_{a.first}"
    pid = queue(a.api, mp4, prefix)
    print("queued", pid, flush=True)
    secs = wait(a.api, pid)
    hdr = read_exrs(prefix, len(truth), (truth.shape[2], truth.shape[1]))
    base = np.stack([li.from_view(s, inv) for s in sdr])        # LTX Return today: the plain inverse view
    rep = {"plate": str(PLATE), "frames": [a.first, a.first + len(truth) - 1], "size": [truth.shape[2], truth.shape[1]],
           "graph": str(GRAPH), "run_seconds": secs, "total_seconds": round(time.time() - t0),
           "inverse_view": stats(base, truth, sdr), "sdr_to_hdr": stats(hdr, truth, sdr)}
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
