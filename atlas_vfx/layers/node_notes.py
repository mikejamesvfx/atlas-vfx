"""Atlas VFX Notes: turns a node's info / report JSON into a few readable lines, with anything worth a look flagged.
Wire any info or report output into it."""
from __future__ import annotations

import json

# key: label, in the order they are shown; the rest follow as key: value
LABELS = {
    "text": None,
    "mode": "mode", "frames": "frames", "frames_range": "frames", "first_frame": "first frame", "length": "length",
    "size": "size", "analysis_size": "measured at", "seconds": "took (s)", "link_gb": "RAM per IMAGE link (GB)",
    "sh_frames": "frames fitted with sh", "lift_limit": "lift limit (stops)",
}
LIST_MAX = 8


def warnings(d: dict) -> list[str]:
    w = []
    c = d.get("clamped_fraction_max")
    if isinstance(c, (int, float)) and c > 0.25:
        w.append(f"the gain hit max_stops on up to {c:.0%} of the matte: the take changes the light more than "
                 f"max_stops allows (raise it, or check the take)")
    fb = d.get("fallback_frames")
    if isinstance(fb, list) and fb and isinstance(d.get("frames"), int):
        w.append(f"{len(fb)} of {d['frames']} frames fell back from sh to guided (normals explain the light poorly)")
    r = d.get("relit_pixels_fraction_max")
    if isinstance(r, (int, float)) and r > 0.1:
        w.append(f"up to {r:.0%} of the matte uses the take's own (soft) pixels: the plate was too dark there "
                 f"(lift_limit)")
    g = d.get("link_gb")
    if isinstance(g, (int, float)) and g > 8:
        w.append(f"{g:.1f} GB per IMAGE link: use Relight Transfer Files or chunks")
    return w


def short(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list) and len(v) > LIST_MAX:
        return f"[{', '.join(map(str, v[:LIST_MAX]))}, ... {len(v)} items]"
    if isinstance(v, dict):
        return ", ".join(f"{k} {short(x)}" for k, x in v.items())
    return str(v)


def notes(text: str) -> str:
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return str(text)
    if not isinstance(d, dict):
        return short(d)
    lines = []
    if isinstance(d.get("text"), str):
        lines.append(d["text"])
    w = warnings(d)
    lines += [f"! {x}" for x in w]
    shown = set()
    for k, label in LABELS.items():
        if k in d and label:
            lines.append(f"{label}: {short(d[k])}")
            shown.add(k)
    for k, v in d.items():
        if k not in shown and k != "text":
            lines.append(f"{k}: {short(v)}")
    return "\n".join(lines)


class AtlasVFXNotes:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    DESCRIPTION = ("Shows a node's info / report as readable lines, with warnings first (lines starting with !): "
                   "clamped gain, sh fallbacks, soft pixels, RAM. Wire any info or report output in.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": ("STRING", {"forceInput": True})}}

    def run(self, info):
        t = notes(info)
        return {"ui": {"text": [t]}, "result": (t,)}


NODE_CLASS_MAPPINGS = {"AtlasVFXNotes": AtlasVFXNotes}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXNotes": "Atlas VFX Notes"}
