"""Decode the F-Log2 plate (legal-range 10-bit Y'CbCr -> full-range RGB), OCIO F-Log2 F-Gamut -> ACES 2.0 SDR
(sRGB display) proxy frames, and per-frame stats for splitting the shot. The plate is read in place, never copied.

usage: plate_proxy.py PLATE OCIO_CONFIG OUT_DIR [width]
"""
import json, os, subprocess, sys
import numpy as np
import PyOpenColorIO as ocio
from PIL import Image

plate, config, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
H = int(round(W * 2160 / 4096 / 2)) * 2
os.makedirs(os.path.join(out, "proxy"), exist_ok=True)

cfg = ocio.Config.CreateFromFile(config)
disp = cfg.getDefaultDisplay()
view = "ACES 2.0 - SDR 100 nits (Rec.709)"
to_disp = cfg.getProcessor(ocio.DisplayViewTransform(src="F-Log2 F-Gamut", display=disp, view=view)).getDefaultCPUProcessor()
to_acescg = cfg.getProcessor("F-Log2 F-Gamut", "ACEScg").getDefaultCPUProcessor()

cmd = ["ffmpeg", "-v", "error", "-i", plate, "-vf", f"scale={W}:{H}:in_range=tv:out_range=pc:flags=area,format=rgb48le",
       "-f", "rawvideo", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
n = W * H * 3 * 2
stats, prev = [], None
i = 0
while True:
    buf = p.stdout.read(n)
    if len(buf) < n:
        break
    log = np.frombuffer(buf, np.uint16).reshape(H, W, 3).astype(np.float32) / 65535.0
    lin = log.copy()
    to_acescg.applyRGB(lin)
    d = log.copy()
    to_disp.applyRGB(d)
    d8 = (np.clip(d, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(d8).save(os.path.join(out, "proxy", f"plate_{i:04d}.png"))
    lum = 0.2722 * lin[..., 0] + 0.6741 * lin[..., 1] + 0.0537 * lin[..., 2]  # ACEScg luminance weights
    small = d8[::8, ::8].astype(np.float32)
    diff = float(np.mean(np.abs(small - prev))) if prev is not None else 0.0
    prev = small
    stats.append({"frame": i, "lin_mean": float(lum.mean()), "lin_p99": float(np.percentile(lum, 99)),
                  "disp_mean": float(d8.mean()), "change": diff})
    i += 1
p.wait()
json.dump({"plate": plate, "config": config, "view": view, "display": disp, "size": [W, H], "frames": stats},
          open(os.path.join(out, "plate_stats.json"), "w"), indent=1)
print(f"{i} frames -> {out}\\proxy  ({W}x{H}), stats in plate_stats.json")
