"""Solve Mask: masks of things that move (people, cars, trees, water) so the solver ignores them.

Grounding DINO (Apache-2.0) finds boxes for a text prompt on the middle frame of each window; SAM 2.1 (Apache-2.0)
turns the boxes into masks and tracks them forwards and backwards through the window. The masks of all windows are
unioned per frame, so an object only has to be found once per window. Both models are transformers-native classes
loaded from local safetensors folders (never trust_remote_code); see atlas_components.json.

Masks: float32 HxW per frame, 1 = moving thing (the solver puts no features there).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("atlas_vfx")

DETECTOR_DIR = ("detection", "grounding-dino-base")
SAM_DIR = ("sam2", "sam2.1-hiera-large")
DEFAULT_PROMPT = "person. car. bicycle. dog. tree. bush. water. flag."


class MaskError(RuntimeError):
    pass


def prompt_text(classes: str) -> str:
    """'person, car' / 'person. car.' -> 'person. car.' (Grounding DINO wants lower case, dot separated)."""
    parts = [p.strip().lower() for p in classes.replace(",", ".").split(".")]
    parts = [p for p in parts if p]
    if not parts or len(parts) > 32 or any(len(p) > 40 for p in parts):
        raise MaskError("classes: 1-32 names, each under 40 characters")
    return " ".join(f"{p}." for p in parts)


def windows(n: int, size: int) -> list[tuple[int, int, int]]:
    """(start, stop, keyframe) per window; the keyframe is the window's middle frame."""
    size = max(1, int(size))
    return [(a, min(a + size, n), (a + min(a + size, n) - 1) // 2) for a in range(0, n, size)]


def model_dir(models_root: Path, sub: tuple[str, str]) -> Path:
    p = Path(models_root, *sub)
    if not (p / "config.json").is_file():
        raise MaskError(f"model folder {p} is missing; run tools/fetch_solve_models.py")
    return p


class Masker:
    def __init__(self, models_root, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.root = Path(models_root)
        self._det = self._sam = None

    def _load(self):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, Sam2VideoModel, Sam2VideoProcessor

        if self._det is None:
            d = model_dir(self.root, DETECTOR_DIR)
            self._det = (AutoProcessor.from_pretrained(d),
                         GroundingDinoForObjectDetection.from_pretrained(d, use_safetensors=True).to(self.device).eval())
        if self._sam is None:
            s = model_dir(self.root, SAM_DIR)
            self._sam = (Sam2VideoProcessor.from_pretrained(s),
                         Sam2VideoModel.from_pretrained(s, use_safetensors=True).to(self.device, self.dtype).eval())

    def free(self):
        self._det = self._sam = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def detect(self, rgb: np.ndarray, prompt: str, box_threshold: float, text_threshold: float) -> np.ndarray:
        """Boxes (K, 4) x1 y1 x2 y2 in image px."""
        from PIL import Image

        self._load()
        proc, model = self._det
        h, w = rgb.shape[:2]
        with self.torch.inference_mode():
            inputs = proc(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to(self.device)
            out = model(**inputs)
            res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_threshold,
                                                              text_threshold=text_threshold, target_sizes=[(h, w)])[0]
        return res["boxes"].float().cpu().numpy().reshape(-1, 4)

    def track(self, frames: list[np.ndarray], key: int, boxes: np.ndarray) -> list[np.ndarray]:
        """SAM 2.1 masks for `boxes` on frames[key], propagated both ways; union per frame (bool HxW)."""
        self._load()
        proc, model = self._sam
        h, w = frames[0].shape[:2]
        union = [np.zeros((h, w), bool) for _ in frames]
        if len(boxes) == 0:
            return union
        with self.torch.inference_mode():
            sess = proc.init_video_session(video=frames, inference_device=self.device, dtype=self.dtype)
            proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=key,
                                                 obj_ids=list(range(1, len(boxes) + 1)),
                                                 input_boxes=[[[float(v) for v in b] for b in boxes]])
            for reverse in (False, True):
                for o in model.propagate_in_video_iterator(sess, start_frame_idx=key, reverse=reverse):
                    m = proc.post_process_masks([o.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                    union[o.frame_idx] |= m.any(dim=0).reshape(h, w).cpu().numpy().astype(bool)
        return union


def masks_for_clip(frames: list[np.ndarray], masker, classes: str = DEFAULT_PROMPT, window: int = 16,
                   box_threshold: float = 0.35, text_threshold: float = 0.25, dilate_px: int = 0,
                   progress=None) -> tuple[list[np.ndarray], dict]:
    """frames: uint8 RGB HxWx3 (display referred). Returns (masks float32 HxW, report)."""
    import cv2

    prompt = prompt_text(classes)
    n = len(frames)
    out = [np.zeros(frames[0].shape[:2], bool) for _ in range(n)]
    rep = {"prompt": prompt, "window": window, "windows": []}
    wins = windows(n, window)
    for k, (a, b, key) in enumerate(wins):
        boxes = masker.detect(frames[key], prompt, box_threshold, text_threshold)
        m = masker.track(frames[a:b], key - a, boxes)
        for i, mm in enumerate(m):
            out[a + i] |= mm
        rep["windows"].append({"frames": [a, b - 1], "key": key, "boxes": len(boxes)})
        if progress:
            progress(k + 1, len(wins))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px > 0 else None
    res = []
    for m in out:
        a = m.astype(np.uint8)
        if kern is not None:
            a = cv2.dilate(a, kern)
        res.append(a.astype(np.float32))
    rep["coverage"] = float(np.mean([r.mean() for r in res]))
    return res, rep
