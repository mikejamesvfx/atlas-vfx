"""Atlas VFX Viewer: look at a frame the way Nuke's viewer shows it. A and B from EXR sequences on disk (or IMAGE
inputs), A/B wipe, side by side or difference, an f-stop and gamma that change only the view, the OCIO display /
view, zoom to the element's alpha, and false colour for the log2 gain EXRs Relight Transfer writes. Reads one frame
per input, so it is cheap next to the shot."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import plate as pl

from . import chunks as ch
from . import ltx_integration as li
from .node_sequence_read_chunk import read_image

MODES = ("a", "b", "wipe", "side by side", "difference", "gain false colour")
GAIN_RANGE = 3.0


def views(ocio_config: str = "") -> list[str]:
    """"display / view" for every view of the config, the default first."""
    default = f"{pl.DEFAULT_DISPLAY} / {pl.DEFAULT_VIEW}"
    try:
        cfg = pl.ocio_config(ocio_config)
        out = [f"{d} / {v}" for d in cfg.getDisplays() for v in cfg.getViews(d)]
    except Exception:  # noqa: BLE001 (no OCIO: the default is still offered, run() reports the real error)
        out = []
    return [default] + [v for v in out if v != default]


def read_frame(path: str, frame: int) -> np.ndarray:
    """RGBA float of one frame: path is a single file, a sequence pattern (####, %04d) or a folder of one sequence."""
    p = pl.clean_path(path)
    if not p:
        raise ValueError("no path")
    if Path(p).is_file():
        f = Path(p)
    else:
        files = ch.scan(p)
        if not files:
            raise ValueError(f"no frames at {p}")
        if frame not in files:
            raise ValueError(f"frame {frame} not in {Path(p).name} ({min(files)}-{max(files)})")
        f = files[frame]
    img, _ = read_image(f)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    if img.shape[-1] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], np.float32)])
    return np.ascontiguousarray(img[..., :4], np.float32)


def false_colour(gain: np.ndarray, rng: float = GAIN_RANGE) -> np.ndarray:
    """log2 gain (H, W) -> RGB 0-1: blue below 0 (darker), white at 0, orange above (brighter), saturated at +-rng."""
    t = np.clip(gain / rng, -1.0, 1.0)[..., None]
    lo, mid, hi = (np.array(c, np.float32) for c in ((0.15, 0.35, 1.0), (1.0, 1.0, 1.0), (1.0, 0.45, 0.05)))
    return np.where(t < 0, mid + (-t) * (lo - mid), mid + t * (hi - mid)).astype(np.float32)


def legend(width: int, rng: float = GAIN_RANGE, height: int = 28) -> np.ndarray:
    import cv2

    ramp = np.linspace(-rng, rng, width, dtype=np.float32)[None].repeat(height, 0)
    out = np.ascontiguousarray((false_colour(ramp) * 255 + 0.5).astype(np.uint8))     # putText draws on 8-bit only
    for x, label in ((4, f"-{rng:g}"), (width // 2 - 6, "0"), (width - 34, f"+{rng:g}")):
        cv2.putText(out, label, (x, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


def bbox(alpha: np.ndarray, pad_frac: float = 0.05):
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) == 0:
        return None
    H, W = alpha.shape
    py, px = int((ys.max() - ys.min()) * pad_frac) + 2, int((xs.max() - xs.min()) * pad_frac) + 2
    return max(0, ys.min() - py), min(H, ys.max() + 1 + py), max(0, xs.min() - px), min(W, xs.max() + 1 + px)


def render(a: np.ndarray, b: np.ndarray | None, mode: str, wipe=0.5, exposure=0.0, gamma=1.0, view_proc=None,
           zoom_to_alpha=False, width=1280) -> tuple[np.ndarray, dict]:
    """The viewer image (H, W, 3) 0-1 and stats. a, b RGBA linear."""
    import cv2

    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    if mode in ("b", "wipe", "side by side", "difference") and b is None:
        raise ValueError(f"mode {mode} needs B")
    if b is not None and b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    alpha = a[..., 3]
    stats = {}
    inside = alpha > 0.5
    if inside.any():
        luma = a[..., :3] @ np.array([0.2722, 0.6741, 0.0537], np.float32)
        v = luma[inside]
        stats = {"alpha_px": int(inside.sum()), "mean": round(float(v.mean()), 5),
                 "min": round(float(v.min()), 5), "max": round(float(v.max()), 5)}
        if mode == "gain false colour":
            gv = a[..., :3].mean(-1)[inside]
            stats.update(gain_mean=round(float(gv.mean()), 3), gain_p1=round(float(np.percentile(gv, 1)), 3),
                         gain_p99=round(float(np.percentile(gv, 99)), 3))
    box = bbox(alpha) if zoom_to_alpha else None
    crop = (lambda x: x if (x is None or box is None) else x[box[0]:box[1], box[2]:box[3]])
    a, b = crop(a), crop(b)

    def show(rgb):
        lin = np.ascontiguousarray(rgb, np.float32) * np.float32(2.0 ** exposure)
        disp = li.to_view(lin, view_proc) if view_proc is not None else np.clip(lin, 0, 1)
        if gamma != 1.0:
            disp = np.power(np.clip(disp, 0, 1), 1.0 / gamma)
        return disp

    if mode == "gain false colour":
        img = false_colour(a[..., :3].mean(-1))
    elif mode == "a":
        img = show(a[..., :3])
    elif mode == "b":
        img = show(b[..., :3])
    elif mode == "difference":
        img = show(np.abs(a[..., :3] - b[..., :3]))
    elif mode == "side by side":
        img = np.hstack([show(a[..., :3]), show(b[..., :3])])
    else:
        img = show(a[..., :3])
        x = int(round(np.clip(wipe, 0, 1) * img.shape[1]))
        img[:, x:] = show(b[..., :3])[:, x:]
        if 0 < x < img.shape[1]:
            img[:, max(0, x - 1):x + 1] = (1.0, 0.85, 0.2)
    if width and img.shape[1] != width:
        h = max(1, round(img.shape[0] * width / img.shape[1]))
        interp = cv2.INTER_AREA if width < img.shape[1] else cv2.INTER_LINEAR
        img = cv2.resize(img, (width, h), interpolation=interp)
    if mode == "gain false colour":
        img = np.vstack([img, legend(img.shape[1])])
    return np.clip(img, 0, 1).astype(np.float32), stats


class AtlasVFXViewer:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    DESCRIPTION = ("A Nuke-style viewer for one frame: A and B from EXR sequences (a pattern like shot.####.exr, a "
                   "folder, or Layer Picker's pattern) or from IMAGE inputs. wipe / side by side / difference compare "
                   "them; exposure (f-stops) and gamma change only the view; view is the OCIO display and view; "
                   "zoom_to_alpha frames the element; gain false colour shows a Relight Transfer gain EXR "
                   "(blue darker, orange brighter, +-3 stops). info gives the values inside A's alpha.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a_path": ("STRING", {"default": "", "tooltip": "sequence pattern, folder or file; blank = a_image"}),
                "b_path": ("STRING", {"default": "", "tooltip": "the same for B; blank = b_image or none"}),
                "frame": ("INT", {"default": 1001, "min": 0, "max": 1_000_000, "tooltip": "frame number on disk"}),
                "mode": (list(MODES), {"default": "wipe"}),
                "wipe": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.25,
                                       "tooltip": "f-stops, view only (Nuke's viewer gain)"}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "view only"}),
                "view": (views(), {"tooltip": "OCIO display / view of the shipped config"}),
                "zoom_to_alpha": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 16, "tooltip": "0 = full size"}),
            },
            "optional": {
                "a_image": ("IMAGE", {"tooltip": "used when a_path is blank (index 0, or image_index)"}),
                "b_image": ("IMAGE",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 100_000}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
            },
        }

    def run(self, a_path, b_path, frame, mode, wipe, exposure, gamma, view, zoom_to_alpha, width, a_image=None,
            b_image=None, image_index=0, ocio_config=""):
        def get(path, image, label):
            if str(path).strip():
                return read_frame(path, frame)
            if image is None:
                return None
            arr = image.float().cpu().numpy()
            f = arr[min(image_index, len(arr) - 1)]
            if f.shape[-1] == 3:
                f = np.dstack([f, np.ones(f.shape[:2], np.float32)])
            return np.ascontiguousarray(f[..., :4], np.float32)

        a = get(a_path, a_image, "A")
        if a is None:
            raise ValueError("no A: give a_path or a_image")
        b = get(b_path, b_image, "B")
        display, _, v = str(view).partition(" / ")
        proc = None
        if mode != "gain false colour":
            proc, _ = li.view_processors(ocio_config, display=display.strip(), view=v.strip())
        img, stats = render(a, b, mode, wipe, exposure, gamma, proc, zoom_to_alpha, width)
        stats.update(frame=frame, mode=mode, exposure=exposure, view=view)
        text = json.dumps(stats)
        ui = {"text": [text]}
        saved = _save_preview(img)
        if saved:
            ui["images"] = [saved]
        import torch
        return {"ui": ui, "result": (torch.from_numpy(img[None]), text)}


def _save_preview(img: np.ndarray):
    """Write the view to ComfyUI's temp folder as PreviewImage does; nothing outside ComfyUI."""
    try:
        import folder_paths
    except ImportError:
        return None
    import cv2

    name = f"atlas_vfx_viewer_{uuid.uuid4().hex[:12]}.png"
    out = Path(folder_paths.get_temp_directory())
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / name), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1])
    return {"filename": name, "subfolder": "", "type": "temp"}


NODE_CLASS_MAPPINGS = {"AtlasVFXViewer": AtlasVFXViewer}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXViewer": "Atlas VFX Viewer"}
