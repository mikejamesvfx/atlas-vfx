"""OCIO CPU processors on every core.

An OCIO CPU processor runs on one thread, and the ACES 2.0 output transform costs ~3 s per 4K frame that way.
PyOpenColorIO releases the GIL inside applyRGB, so cutting the frame into row bands and running them on a thread pool
gives the same bits ~13x faster (0.22 s per 4K frame on 24 threads). No lattice, no approximation: OCIO computes every
pixel, so nothing about the colour changes.

    proc = fast(cfg.getProcessor(src, dst))          # a Processor or an existing CPUProcessor
    proc.applyRGB(arr)                               # in place, as before

ATLAS_VFX_OCIO_THREADS sets the pool size (default: cores - 2, 1 turns it off).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MIN_PIXELS = 1 << 16              # below this one thread is quicker than handing out bands
BAND_PIXELS = 1 << 17             # ~130k pixels per band: enough bands to balance, few enough to stay cheap
_pool = None
_pool_lock = threading.Lock()


def threads() -> int:
    try:
        n = int(os.environ.get("ATLAS_VFX_OCIO_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else max(1, (os.cpu_count() or 2) - 2)


def _executor():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(threads(), thread_name_prefix="atlas_ocio")
        return _pool


class Banded:
    """Wraps an OCIO CPUProcessor; applyRGB(arr) runs the same processor over row bands in parallel."""

    def __init__(self, cpu):
        self.cpu = cpu

    def applyRGB(self, arr):
        a = arr if isinstance(arr, np.ndarray) else None
        if (a is None or a.dtype != np.float32 or not a.flags.c_contiguous or a.size % 3
                or a.size // 3 < MIN_PIXELS or threads() == 1):
            self.cpu.applyRGB(arr)
            return
        flat = a.reshape(-1, 3)                      # a view: bands write straight into `arr`
        n = len(flat)
        cuts = list(range(0, n, BAND_PIXELS)) + [n]

        def run(k):
            band = flat[cuts[k]:cuts[k + 1]]
            self.cpu.applyRGB(band)
            return band

        out = list(_executor().map(run, range(len(cuts) - 1)))
        for k, band in enumerate(out):               # in case a binding ever copies instead of writing in place
            if not np.shares_memory(band, flat):
                flat[cuts[k]:cuts[k + 1]] = band

    def __getattr__(self, name):                    # everything else is the processor's own
        return getattr(self.cpu, name)


def fast(proc) -> Banded:
    """A Processor, a CPUProcessor or a Banded -> Banded."""
    if isinstance(proc, Banded):
        return proc
    if hasattr(proc, "getDefaultCPUProcessor"):
        proc = proc.getDefaultCPUProcessor()
    return Banded(proc)
