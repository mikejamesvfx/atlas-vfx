"""Relight Transfer from files, one frame at a time: frame sources that read an image sequence on demand, MoGe normals
computed per frame and cached on disk, and an EXR sink. light_transfer.transfer runs on these exactly as on arrays,
so a 4K shot of any length needs a few GB of RAM instead of ~10 GB per IMAGE link per 100 frames."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import relight as rl
from .node_sequence_read_chunk import read_image

KINDS = ("rgb", "alpha", "mask")
_RAW = OrderedDict()          # (path, mtime, size) -> decoded file: an element's RGB and its alpha matte share a read
_RAW_KEEP = 2


def _read(path: Path):
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _RAW:
        _RAW.move_to_end(key)
        return _RAW[key]
    img, _ = read_image(path)
    img.flags.writeable = False                   # shared: _convert copies before changing anything
    _RAW[key] = img
    while len(_RAW) > _RAW_KEEP:
        _RAW.popitem(last=False)
    return img


class SeqFrames:
    """Frames of a sequence (a list of files), read when asked for and kept for the last `keep` frames.

    kind rgb: the first three channels (one channel repeated). alpha: channel 4 (an error without one). mask: the
    first channel (a matte stored as grey or RGB). proc: an OCIO CPU processor applied to rgb. srgb: decode sRGB to
    linear. flip: left-right."""

    def __init__(self, files, kind="rgb", proc=None, srgb=False, flip=False, keep=2):
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}: one of {KINDS}")
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("no files")
        self.kind, self.proc, self.srgb, self.flip, self.keep = kind, proc, srgb, flip, max(1, int(keep))
        self._cache = OrderedDict()
        first = self[0]
        self.shape = (len(self.files),) + first.shape

    def __len__(self):
        return len(self.files)

    def _convert(self, img, name):
        if self.kind == "alpha":
            if img.shape[-1] < 4:
                raise ValueError(f"{name}: no alpha channel ({img.shape[-1]} channels)")
            out = img[..., 3]
        elif self.kind == "mask":
            out = img[..., 0]
        else:
            out = np.repeat(img, 3, -1) if img.shape[-1] == 1 else img[..., :3]
            out = np.ascontiguousarray(out, np.float32)
            if np.shares_memory(out, img):            # the decoded file is shared (_read): change a copy
                out = out.copy()
            if self.proc is not None:
                self.proc.applyRGB(out)
            if self.srgb:
                out = rl.srgb_to_lin(out)
        if self.flip:
            out = out[:, ::-1]
        out = np.ascontiguousarray(out, np.float32)
        return out.copy() if np.shares_memory(out, img) else out

    def __getitem__(self, t):
        t = int(t)
        if t < 0:
            t += len(self.files)
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]
        img = _read(self.files[t])
        out = self._convert(img, self.files[t].name)
        if hasattr(self, "shape") and out.shape != self.shape[1:]:
            raise ValueError(f"{self.files[t].name}: {out.shape[1]}x{out.shape[0]}, the sequence started at "
                             f"{self.shape[2]}x{self.shape[1]}")
        self._cache[t] = out
        while len(self._cache) > self.keep:
            self._cache.popitem(last=False)
        return out


class MoGeNormals:
    """OpenGL camera-space normals of the element, from a MoGe model run one frame at a time, then smoothed as Atlas
    VFX Normals does (+-temporal frames, sigma px inside the matte). Each finished frame is cached in cache_dir as
    float16 .npy (plain arrays, loaded without pickle), so the SH fit and the full-res pass share one MoGe run.

    model is the MOGE_MODEL ComfyUI's Load MoGe Model gives (anything with .infer(bchw, ...) returning "normal"
    and/or "points", and "depth"). display turns a linear frame into what MoGe expects (sRGB, 0-1)."""

    def __init__(self, element, model, cache_dir, display=None, matte=None, temporal=1, sigma=1.5,
                 resolution_level=9, refine_steps=3):
        self.element, self.model, self.display, self.matte = element, model, display, matte
        self.temporal, self.sigma = int(temporal), float(sigma)
        self.resolution_level, self.refine_steps = int(resolution_level), int(refine_steps)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(element.shape[:3]) + (3,)
        self._raw = {}
        self.runs = 0
        self.source = None
        self.depth = _Depth(self)

    def __len__(self):
        return self.shape[0]

    def _infer(self, t):
        import torch

        rgb = np.array(self.element[t], np.float32)
        if self.display is not None:
            rgb = self.display(rgb)
        x = torch.from_numpy(np.ascontiguousarray(np.clip(rgb, 0.0, 1.0))).movedim(-1, 0)[None]
        r = self.model.infer(x, resolution_level=self.resolution_level, fov_x=None, force_projection=True,
                             apply_mask=False, refine_steps=self.refine_steps)
        self.runs += 1
        if r.get("normal") is not None:
            n = r["normal"][0].float().cpu().numpy()
            self.source = "predicted"
        elif r.get("points") is not None:
            n = rl.normals_from_points(r["points"].float().cpu().numpy())[0]
            self.source = "from points"
        else:
            raise ValueError("MoGe gave neither normals nor points")
        d = r["depth"][0].float().cpu().numpy() if r.get("depth") is not None else None
        return rl.opencv_to_opengl(n), d

    def _raw_at(self, t):
        if t not in self._raw:
            self._raw[t] = self._infer(t)
        return self._raw[t]

    def _paths(self, t):
        return self.cache / f"normal.{t:05d}.npy", self.cache / f"depth.{t:05d}.npy"

    def _compute(self, t):
        import cv2

        T, k = self.shape[0], self.temporal
        lo, hi = max(0, t - k), min(T, t + k + 1)
        for old in [j for j in self._raw if j < lo - 1 or j > hi]:
            del self._raw[old]
        n = np.mean([self._raw_at(j)[0] for j in range(lo, hi)], 0)
        al = None
        if self.matte is not None:
            al = np.asarray(self.matte[t], np.float32)
            if al.shape != n.shape[:2]:
                al = cv2.resize(al, (n.shape[1], n.shape[0]), interpolation=cv2.INTER_LINEAR)
            al = al[None]
        n = rl.smooth_normals(n[None], al, 0, self.sigma)[0]
        pn, pd = self._paths(t)
        np.save(pn, n.astype(np.float16))
        d = self._raw_at(t)[1]
        if d is not None:
            np.save(pd, d.astype(np.float32))
        return n

    def __getitem__(self, t):
        t = int(t) % self.shape[0]
        pn, _ = self._paths(t)
        if pn.exists():
            return np.load(pn).astype(np.float32)
        return self._compute(t)

    def clear(self):
        for f in self.cache.glob("*.npy"):
            f.unlink()
        try:
            self.cache.rmdir()
        except OSError:
            pass


class _Depth:
    """MoGe depth of the same run, as a frame source (computed with the normals)."""

    def __init__(self, normals):
        self.n = normals

    @property
    def shape(self):
        return self.n.shape[:3]

    def __len__(self):
        return self.n.shape[0]

    def __getitem__(self, t):
        t = int(t) % self.n.shape[0]
        _, pd = self.n._paths(t)
        if not pd.exists():
            self.n[t]
        if not pd.exists():
            raise ValueError("MoGe gave no depth")
        return np.load(pd)
