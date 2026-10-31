"""Atlas VFX Shot Range: set the frame range once, in Nuke frames, and wire it everywhere. Gives the LTX length (8k+1),
the plate index Plate Read wants and what the range costs in RAM, in plain words."""
from __future__ import annotations

import json

from atlas_vfx.camera_side import plate as pl

SNAPS = ("as is", "8k+1 shorter", "8k+1 longer")


def snap_length(n: int, snap: str) -> int:
    """n frames as LTX needs them: a multiple of 8 plus 1 (9, 17, ... 97, 105, 121)."""
    if snap not in SNAPS:
        raise ValueError(f"ltx_length {snap!r}: one of {SNAPS}")
    if snap == "as is" or (n - 1) % 8 == 0:
        return n
    if snap == "8k+1 shorter":
        return max(9, (n - 1) // 8 * 8 + 1)
    return -(-(n - 1) // 8) * 8 + 1


def link_gb(frames: int, width: int, height: int) -> float:
    """RAM one ComfyUI IMAGE link of this many RGB float frames holds."""
    return frames * width * height * 3 * 4 / 1024 ** 3


class AtlasVFXShotRange:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("first_frame", "length", "last_frame", "plate_first", "plate_last", "info")
    DESCRIPTION = ("Type the shot's first and last frame once, as Nuke numbers them (1001 = the plate's first frame). "
                   "Wire first_frame / length into LTX Prep and Relight Transfer Files, plate_first / plate_last into "
                   "Plate Read (right-click a widget > convert to input). ltx_length rounds the length to what LTX "
                   "accepts (8k+1). info says what the range costs in RAM.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("INT", {"default": 1130, "min": 0, "max": 1_000_000, "tooltip": "first frame, Nuke numbers"}),
                "last": ("INT", {"default": 1226, "min": 0, "max": 1_000_000, "tooltip": "last frame, Nuke numbers"}),
                "ltx_length": (list(SNAPS), {"default": "8k+1 shorter",
                                             "tooltip": "LTX runs 8k+1 frames (97, 105, 121...): shorter drops the "
                                                        "end, longer runs past it"}),
                "frame_offset": ("INT", {"default": pl.NUKE_FRAME_OFFSET, "min": 0, "max": 1_000_000,
                                         "tooltip": "the Nuke frame of the plate's first frame"}),
                "width": ("INT", {"default": 4096, "min": 16, "max": 16384, "tooltip": "for the RAM estimate"}),
                "height": ("INT", {"default": 2160, "min": 16, "max": 16384, "tooltip": "for the RAM estimate"}),
            }
        }

    def run(self, first, last, ltx_length="8k+1 shorter", frame_offset=pl.NUKE_FRAME_OFFSET, width=4096,
            height=2160):
        if last < first:
            raise ValueError(f"last {last} is before first {first}")
        if first < frame_offset:
            raise ValueError(f"first {first} is before the plate starts (frame {frame_offset})")
        asked = last - first + 1
        n = snap_length(asked, ltx_length)
        end = first + n - 1
        gb = link_gb(n, width, height)
        lines = [f"{n} frames {first}-{end}" + ("" if n == asked else f" (asked {asked}: {ltx_length})")
                 + (", 8k+1 ok for LTX" if (n - 1) % 8 == 0 else ", not 8k+1: LTX Prep will refuse it"),
                 f"plate index {first - frame_offset}-{end - frame_offset} (Plate Read first / last)",
                 f"one {width}x{height} RGB link holds {gb:.1f} GB"]
        if gb > 8:
            lines.append("that is a lot for a graph of IMAGE links: use Relight Transfer Files, or Chunk Plan")
        info = {"first_frame": first, "length": n, "last_frame": end, "asked": asked,
                "plate_first": first - frame_offset, "plate_last": end - frame_offset, "link_gb": round(gb, 2),
                "text": "\n".join(lines)}
        return {"ui": {"text": ["\n".join(lines)]},
                "result": (first, n, end, first - frame_offset, end - frame_offset, json.dumps(info))}


NODE_CLASS_MAPPINGS = {"AtlasVFXShotRange": AtlasVFXShotRange}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXShotRange": "Atlas VFX Shot Range"}
