"""Light transfer: keep a clip's detail, take only the light from a relit version of it (e.g. Light-A-Video's
SD1.5 output, which relights well but softens and loses identity detail).

    shade = blur(relit_linear) / blur(mean(source_linear))      (coloured, low frequency, sigma px at source res)
    out   = source_linear * shade

Display sRGB in and out (H3 / Light-A-Video clips are display-referred mp4s); linearised with the sRGB curve for the
ratio. Experimental, on the character walk test (NOTES 2026-10-19).

Run: <V141 python> tools\light_transfer.py <source.mp4> <relit.mp4> <out.mp4> [--every 3] [--count 32] [--sigma 12]
"""
import argparse
import json
import subprocess

import numpy as np


def probe(p):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", p], capture_output=True).stdout)["streams"][0]
    return s["width"], s["height"]


def read(p):
    w, h = probe(p)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def transfer(src, relit, sigma=12.0):
    import cv2

    h, w = src.shape[:2]
    s = srgb_to_lin(src)
    r = srgb_to_lin(cv2.resize(relit, (w, h), interpolation=cv2.INTER_CUBIC))
    shade = cv2.GaussianBlur(r, (0, 0), sigma) / (cv2.GaussianBlur(s.mean(2), (0, 0), sigma)[..., None] + 1e-3)
    return np.clip(lin_to_srgb(s * shade), 0, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("source"); a.add_argument("relit"); a.add_argument("out")
    a.add_argument("--every", type=int, default=3); a.add_argument("--count", type=int, default=32)
    a.add_argument("--sigma", type=float, default=12.0); a.add_argument("--fps", type=float, default=8.0)
    o = a.parse_args()
    src = read(o.source)[0::o.every][:o.count]
    rel = read(o.relit)
    out = np.stack([transfer(src[i], rel[i], o.sigma) for i in range(min(len(src), len(rel)))])
    h, w = out.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
                          str(o.fps), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", o.out],
                         stdin=subprocess.PIPE)
    p.stdin.write((out * 255).astype(np.uint8).tobytes()); p.stdin.close(); p.wait()
    print(o.out, out.shape)


if __name__ == "__main__":
    main()
