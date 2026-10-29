"""Atlas VFX Plate Lock Cross-View: lock an AI EXR sequence onto its plate, frame for frame, per ID."""
from __future__ import annotations

import json
import logging

import numpy as np

from ..camera_side import plate as pl
from . import _comfy, colour
from .core import crossview as cv

log = logging.getLogger("atlas_vfx")

DEFAULT_OUT = str(pl.WORK_ROOT / "platelock" / "crossview")


def parse_ids(text: str) -> dict:
    """'name=path/pattern.####.exr' per line (blank lines and # comments ignored)."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"ID line needs name=pattern: {line!r}")
        k, v = line.split("=", 1)
        k = k.strip()
        if not k.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"ID name {k!r}: letters, digits, _ and - only (it becomes a folder name)")
        out[k] = _comfy.clean_path(v)
    return out


def summary_text(side: dict) -> str:
    lines = [f"frames {side['frames'][0]}-{side['frames'][1]} in {side['seconds']}s, "
             f"solve width {side['config']['proc_width']}, match {side['inputs']['match_space']}"]
    for k, s in side["summary"].items():
        bits = []
        if "edge_px_raw" in s:
            bits.append(f"raw edge {s['edge_px_raw']['mean']:.1f}px")
        if "residual_px_median" in s:
            bits.append(f"residual {s['residual_px_median']['mean']:.2f}px (worst {s['residual_px_median']['worst']:.2f})")
        if "confident_fraction" in s:
            bits.append(f"trusted {100 * s['confident_fraction']['mean']:.0f}%")
        lines.append(f"  {k}: " + ", ".join(bits))
    return "\n".join(lines)


class AtlasVFXPlateLockCrossView:
    CATEGORY = _comfy.CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("summary", "out_dir", "preview")
    OUTPUT_NODE = True
    DESCRIPTION = ("Frame-for-frame cross-view warp: plate frame N and AI frame N, one dense warp per ID matte, "
                   "matched in ACEScct through OCIO (relight-proof), solved at proc_width and scaled to full "
                   "resolution. Writes warped AI per ID (A = plate ID matte), float STMaps into the AI frames "
                   "(A = confidence), a comp where the plate's own mattes are the alphas, a JSON sidecar and "
                   "a Nuke script of Reads. Pixels are only resampled, never colour converted.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("STRING", {"default": "", "tooltip": "plate EXR sequence (shot.####.exr / %04d / folder)"}),
                "ai": ("STRING", {"default": "", "tooltip": "AI take EXR sequence, same frame numbers"}),
                "plate_ids": ("STRING", {"multiline": True, "default": "person_a=\nperson_b=",
                                         "tooltip": "one ID per line: name=matte sequence"}),
                "out_dir": ("STRING", {"default": DEFAULT_OUT}),
                "plate_space": ("STRING", {"default": "ACEScg", "tooltip": "OCIO colorspace of the plate EXRs"}),
                "ai_space": ("STRING", {"default": "ACEScg", "tooltip": "OCIO colorspace of the AI EXRs"}),
                "proc_width": ("INT", {"default": 2048, "min": 0, "max": 8192, "step": 64,
                                       "tooltip": "solve width; 0 = full resolution"}),
                "first": ("INT", {"default": -1, "min": -1, "max": 10_000_000, "tooltip": "-1 = from the start"}),
                "last": ("INT", {"default": -1, "min": -1, "max": 10_000_000, "tooltip": "-1 = to the end"}),
            },
            "optional": {
                "ai_ids": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "same names as plate_ids, mattes of the AI take (better warps)"}),
                "matte_channel": (["A", "R", "G", "B", "Y"], {"default": "A"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "base": (["ai", "plate"], {"default": "ai", "tooltip": "what sits under the IDs in the comp"}),
                "quality": (["medium", "fast", "ultrafast"], {"default": "medium"}),
                "write_comp": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, plate, ai, plate_ids, out_dir, plate_space, ai_space, proc_width, first, last,
            ai_ids="", matte_channel="A", ocio_config="", base="ai", quality="medium", write_comp=True):
        from . import crossview_job
        cfg = cv.CVConfig(proc_width=proc_width, preset=quality)
        try:
            import comfy.utils  # type: ignore
            bar = comfy.utils.ProgressBar(100)

            def progress(n, total):
                bar.update_absolute(int(100 * n / max(total, 1)))
        except ImportError:
            progress = None
        out_dir = str(pl.resolve_out_dir(out_dir, DEFAULT_OUT))     # under the work root, never the repo
        side = crossview_job.run(
            _comfy.clean_path(plate), _comfy.clean_path(ai), parse_ids(plate_ids), out_dir,
            ai_ids=parse_ids(ai_ids) or None, plate_space=plate_space.strip(), ai_space=ai_space.strip(),
            ocio_config=_comfy.clean_path(ocio_config) or None, first=None if first < 0 else first,
            last=None if last < 0 else last, matte_channel=matte_channel, cfg=cfg, base=base,
            write_comp=write_comp, progress=progress)
        text = summary_text(side)
        preview = self._preview(side, out_dir, ai_space, ocio_config)
        return {"ui": {"text": [text]}, "result": (text, _comfy.clean_path(out_dir), preview)}

    @staticmethod
    def _preview(side, out_dir, space, ocio_config):
        """Middle comp frame through the config's sRGB display, about 1K wide."""
        import os
        from .core import seqio
        f0, f1 = side["frames"]
        mid = (f0 + f1) // 2
        path = os.path.join(_comfy.clean_path(out_dir), "comp", f"comp.{mid:04d}.exr")
        try:
            img = seqio._read_pixels(path)
            h, w = img.shape[:2]
            if w > 1024:
                img = cv.resize(img, 1024, max(2, int(h * 1024 / w)))
            try:
                disp = colour.convert(img, space, "sRGB - Display", _comfy.clean_path(ocio_config) or None)
            except Exception as e:  # noqa: BLE001
                log.warning("platelock: preview display transform failed (%s), showing clamped linear", e)
                disp = np.clip(img, 0, 1) ** (1 / 2.2)
            return _comfy.to_image(np.clip(disp, 0, 1)[None])
        except Exception as e:  # noqa: BLE001
            log.warning("platelock: no preview (%s)", e)
            return _comfy.to_image(np.zeros((1, 64, 64, 3), np.float32))


NODE_CLASS_MAPPINGS = {"AtlasVFXPlateLockCrossView": AtlasVFXPlateLockCrossView}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXPlateLockCrossView": "Atlas VFX Plate Lock Cross-View"}
