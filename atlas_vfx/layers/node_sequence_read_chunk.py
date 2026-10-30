"""Atlas VFX Sequence Read Chunk: frames start..start+length-1 of an image sequence on disk (EXR, DPX, PNG, TIFF, JPEG)
as an IMAGE batch, for one chunk from Chunk Plan. Read only; the sequence is never written."""
from __future__ import annotations

import numpy as np

from atlas_vfx.camera_side import plate as pl

from . import chunks as ch

WARN_BYTES = 8 * 1024 ** 3     # an IMAGE batch bigger than this gets a note (list mode holds every chunk at once)


def read_image(path) -> tuple[np.ndarray, dict]:
    """Any OIIO-readable still as float32 HxWxC, plus the header facts the chunk nodes carry along."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    s = inp.spec()
    data = inp.read_image(oiio.FLOAT)
    info = {"width": s.width, "height": s.height, "channels": list(s.channelnames),
            "colorspace": s.get_string_attribute("oiio:ColorSpace"),
            "premultiplied": s.get_int_attribute("atlas_vfx:premultiplied", -1)}
    inp.close()
    if data is None:
        raise RuntimeError(f"cannot read pixels of {path}: {oiio.geterror()}")
    data = np.asarray(data, np.float32)
    if data.ndim == 2:
        data = data[..., None]
    return data, info


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    return np.ascontiguousarray(img[..., :3])


class AtlasVFXSequenceReadChunk:
    CATEGORY = "Atlas VFX/Layers/Chunks"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "first_frame", "info")
    DESCRIPTION = ("Reads `length` frames from `start` of an image sequence: shot.####.exr, shot.%04d.exr, "
                   "shot.{frame}.exr or a folder holding one sequence. numbering frame = file numbers are shot frames "
                   "(1001...); sequential = the files in order are start, start+1, ... RGB out (alpha dropped). "
                   "width > 0 resizes (keeps the aspect). Colour converts with OCIO only when input_colorspace is "
                   "set; blank passes the values through as stored.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "~/atlas-vfx-work/shot/shot.####.exr"}),
                "start": ("INT", {"default": 1001, "min": 0, "max": 1_000_000}),
                "length": ("INT", {"default": 121, "min": 1, "max": 10_000}),
                "numbering": (["frame", "sequential"],),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 = as stored"}),
                "input_colorspace": ("STRING", {"default": "", "tooltip": "OCIO name; blank = no conversion"}),
                "colorspace": ("STRING", {"default": "sRGB Encoded Rec.709 (sRGB)",
                                          "tooltip": "OCIO name out, used only with input_colorspace"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
            },
        }

    def run(self, path, start, length, numbering, width, input_colorspace, colorspace, ocio_config):
        import cv2
        import torch

        src = pl.clean_path(path)
        files = ch.pick(ch.scan(src), start, length, numbering)
        proc = None
        if input_colorspace.strip():
            proc = pl.ocio_config(ocio_config).getProcessor(input_colorspace.strip(),
                                                            colorspace.strip()).getDefaultCPUProcessor()
        frames, size, info0 = [], None, None
        for f in files:
            img, info = read_image(f)
            rgb = to_rgb(img)
            if size is None:
                size, info0 = rgb.shape[:2], info
            elif rgb.shape[:2] != size:
                raise ValueError(f"{f.name}: {rgb.shape[1]}x{rgb.shape[0]}, the chunk started at {size[1]}x{size[0]}")
            if proc is not None:
                rgb = np.ascontiguousarray(rgb, np.float32)
                proc.applyRGB(rgb)
            if width and width != rgb.shape[1]:
                h = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
                interp = cv2.INTER_AREA if width < rgb.shape[1] else cv2.INTER_CUBIC
                rgb = cv2.resize(rgb, (width, h), interpolation=interp)
            frames.append(rgb)
        batch = np.stack(frames).astype(np.float32)
        lines = [f"{len(files)} frames {start}-{start + length - 1} of {src} ({numbering} numbering)",
                 f"{info0['width']}x{info0['height']} {info0['channels']} stored colorspace "
                 f"{info0['colorspace'] or 'unknown'}"
                 + (f", converted {input_colorspace} -> {colorspace}" if proc else ", values as stored"),
                 f"out {batch.shape[2]}x{batch.shape[1]}, {batch.nbytes / 1024 ** 3:.2f} GB"]
        if batch.nbytes > WARN_BYTES:
            lines.append("note: big batch; set width, or queue chunks one at a time (tools/queue_chunks.py) rather "
                         "than chunk -1")
        return (torch.from_numpy(batch), int(start), "\n".join(lines))


NODE_CLASS_MAPPINGS = {"AtlasVFXSequenceReadChunk": AtlasVFXSequenceReadChunk}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSequenceReadChunk": "Atlas VFX Sequence Read Chunk"}
