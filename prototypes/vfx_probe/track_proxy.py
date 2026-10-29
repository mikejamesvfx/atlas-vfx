"""Tracking proxies (not for viewing): F-Log2 code values straight from the plate (log already lifts the shadows),
luma only, then CLAHE so the dark deck and doorway keep texture. 16-bit decode, 8-bit out. The plate is read in place.
usage: track_proxy.py PLATE OUT_DIR [first] [last] [width]"""
import os, subprocess, sys
import numpy as np
import cv2

plate, out = sys.argv[1], sys.argv[2]
first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
last = int(sys.argv[4]) if len(sys.argv) > 4 else 297
W = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
H = int(round(W * 2160 / 4096 / 2)) * 2
os.makedirs(out, exist_ok=True)
cmd = ["ffmpeg", "-v", "error", "-i", plate, "-vf",
       f"select='between(n\\,{first}\\,{last - 1})',scale={W}:{H}:in_range=tv:out_range=pc:flags=area,format=gray16le",
       "-vsync", "0", "-f", "rawvideo", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
n, i = W * H * 2, first
while True:
    buf = p.stdout.read(n)
    if len(buf) < n:
        break
    y = np.frombuffer(buf, np.uint16).reshape(H, W)
    lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)       # per-frame stretch of the log signal
    y = np.clip((y.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    y16 = (y * 65535).astype(np.uint16)
    y8 = (clahe.apply(y16) >> 8).astype(np.uint8)
    cv2.imwrite(os.path.join(out, f"plate_{i:04d}.png"), y8)
    i += 1
p.wait()
print(f"{i - first} tracking proxies -> {out} ({W}x{H})")
