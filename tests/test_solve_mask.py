"""camera_side/solve_mask.py (Solve Mask, catalogue 67).

  1. Pure parts: prompt_text cleans and refuses, windows cover every frame once with a middle keyframe.
  2. masks_for_clip with a stand-in masker (known boxes, box-shaped masks): windows are unioned, dilation grows the
     mask, the report counts boxes per window.
  3. With the weights fetched (tools/fetch_solve_models.py): SAM 2.1 tracks a textured block sliding over a
     plain background from one box on the middle frame, IoU >= 0.85 on every frame; Grounding DINO runs on a frame
     and returns boxes as (K, 4). Without the weights this part prints SKIP.
Run: <V141 python> tests/test_solve_mask.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_vfx.camera_side import solve_mask as sm  # noqa: E402

MODELS = Path(os.environ.get("ATLAS_VFX_MODELS", str(Path.home() / "ComfyUI_V141" / "ComfyUI" / "models")))
fails: list[str] = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def block_clip(n=12, H=256, W=384):
    rng = np.random.default_rng(4)
    tex = gaussian_filter(rng.random((90, 90)), 1.5)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    frames, truth = [], []
    for i in range(n):
        img = np.full((H, W, 3), 200, np.uint8)
        x0, y0 = 30 + 18 * i, 80 + 3 * i
        img[y0:y0 + 90, x0:x0 + 90] = (np.stack([tex * 180, tex * 60 + 20, tex * 40 + 10], -1)).astype(np.uint8)
        m = np.zeros((H, W), bool)
        m[y0:y0 + 90, x0:x0 + 90] = True
        frames.append(img)
        truth.append((m, (x0, y0, x0 + 90, y0 + 90)))
    return frames, truth


class FakeMasker:
    def __init__(self, truth):
        self.truth = truth
        self.offset = 0

    def detect(self, rgb, prompt, bt, tt):
        for i, f in enumerate(self.frames):
            if f is rgb:
                return np.array([self.truth[i][1]], float)
        return np.zeros((0, 4))

    def track(self, frames, key, boxes):
        start = next(i for i, f in enumerate(self.frames) if f is frames[0])
        return [self.truth[start + i][0] if len(boxes) else np.zeros_like(self.truth[0][0])
                for i in range(len(frames))]


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> int:
    check(sm.prompt_text("Person, car.tree") == "person. car. tree.", "prompt_text cleans the class list")
    for bad in ("", " . , ", "x" * 50, ",".join(["a"] * 40)):
        try:
            sm.prompt_text(bad)
            check(False, f"prompt_text refuses {bad[:12]!r}")
        except sm.MaskError:
            pass
    w = sm.windows(37, 16)
    cover = sorted(f for a, b, _k in w for f in range(a, b))
    check(cover == list(range(37)) and all(a <= k < b for a, b, k in w), f"windows cover 37 frames once: {w}")

    frames, truth = block_clip()
    fm = FakeMasker(truth)
    fm.frames = frames
    masks, rep = sm.masks_for_clip(frames, fm, "block", window=5, dilate_px=0)
    check(all(iou(m > 0.5, t[0]) == 1.0 for m, t in zip(masks, truth)), "union of window masks = the truth")
    check([x["boxes"] for x in rep["windows"]] == [1, 1, 1], f"one box per window ({rep['windows']})")
    dil, _ = sm.masks_for_clip(frames, fm, "block", window=5, dilate_px=4)
    check(all(d.sum() > m.sum() for d, m in zip(dil, masks)), "dilation grows every mask")

    have = all((MODELS.joinpath(*sub) / "config.json").is_file() for sub in (sm.DETECTOR_DIR, sm.SAM_DIR))
    if not have:
        print(f"SKIP model checks: no weights under {MODELS} (tools/fetch_solve_models.py)")
    else:
        mk = sm.Masker(MODELS)
        try:
            key = len(frames) // 2
            got = mk.track(frames, key, np.array([truth[key][1]], float))
            ious = [iou(g, t[0]) for g, t in zip(got, truth)]
            check(min(ious) >= 0.85, f"SAM 2.1 tracks the block from one box: IoU min {min(ious):.3f}, "
                                     f"mean {np.mean(ious):.3f}")
            boxes = mk.detect(frames[key], sm.prompt_text("square. box."), 0.2, 0.2)
            check(boxes.ndim == 2 and boxes.shape[1] == 4, f"Grounding DINO returns (K, 4) boxes ({boxes.shape})")
        finally:
            mk.free()
    print("PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
