"""Write Atlas Comp's recipes (recipes/*.json) from the graph builders, so a recipe is always the same graph as the
matching starter workflow:

  matte    tools/alpha_matte_workflow.graph()   the element layer read as sRGB instead of a Load Image
  relight  tools/ltx_relight_workflow.graph_4k()  the element layer instead of Layer Picker
  sky      tools/physical_sky_workflow.graph()

Each recipe says which graph inputs the artist's fields, the shot and the version folder set, and what Preview
changes (fewer frames, the tiny VAE instead of the full decode, a smaller sky). See atlas_vfx/comp/recipes.py.

Run: <V141 python> tools\\comp_recipes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ui_workflow  # noqa: E402
from atlas_vfx.layers.node_sun_direction import DIRECTIONS, HEIGHTS  # noqa: E402

OUT = HERE.parent / "recipes"
SRGB = "sRGB Encoded Rec.709 (sRGB)"
TINY = {"class_type": "AtlasVFXTinyVAE", "inputs": {"tae_name": "taeltx2_3.safetensors"}}


def latent_guide(g: dict, guide: str = "43", pre: str = "90") -> None:
    """Swap LTX's image guide for Latent Pre-render -> Guide From Latent (same conditioning, bit-identical output on
    V141): the reference is encoded once at float32 into the layer's cache folder and reused while its frames are
    unchanged. Wire the VAE through LTXVVAEForceFloat32 (both recipes already do)."""
    old = g[guide]["inputs"]
    g[pre] = {"class_type": "AtlasVFXLatentPrerender", "inputs": {
        "images": old["image"], "vae": old["vae"], "name": "reference", "out_dir": ""}}
    g[guide] = {"class_type": "AtlasVFXGuideFromLatent", "inputs": {
        "positive": old["positive"], "negative": old["negative"], "vae": old["vae"], "latent": old["latent"],
        "guide": [pre, 0], "strength": old["strength"], "latent_downscale_factor": old["latent_downscale_factor"]}}


def seed_field(node: str) -> dict:
    return {"key": "seed", "label": "Seed", "type": "int", "default": 42, "min": 0, "max": 2 ** 31 - 1,
            "targets": [[node, "noise_seed"]]}


def element_field(targets) -> dict:
    return {"key": "element", "label": "Element layer", "type": "layer", "default": "", "targets": targets}


def matte() -> dict:
    import alpha_matte_workflow as am

    g = am.graph()
    g["1"] = {"class_type": "AtlasVFXSequenceReadChunk", "inputs": {
        "path": "", "start": 1001, "length": 9, "numbering": "frame", "width": 0, "input_colorspace": "ACEScg",
        "colorspace": SRGB, "ocio_config": ""}}
    g["65"]["inputs"]["first_frame"] = ["1", 1]
    latent_guide(g)
    return {"name": "matte", "label": "Matte (Alpha-Gen)",
            "description": "A matte for the element layer from LTX Alpha-Gen: no green screen, soft edges kept. "
                           "Writes RGB + alpha EXRs.",
            "graph": g,
            "fields": [element_field([["1", "path"]]),
                       {"key": "still_frames", "label": "Hold a still for (frames)", "type": "choice", "default": 9,
                        "choices": [9, 17, 25], "targets": [["2", "still_frames"]]},
                       seed_field("49")],
            "context": {"first_frame": [["1", "start"]], "length": [["1", "length"]], "cache_dir": [["90", "out_dir"]]},
            "outputs": [["65", "out_dir", ""]], "names": [["65", "name"]],
            "preview": {"frames": 9,
                        "nodes": {"53": {"class_type": "VAEDecode", "inputs": {"samples": ["52", 2],
                                                                              "vae": ["61", 0]}}},
                        "drop": ["62", "63", "64", "66", "67"]}}


def relight() -> dict:
    import ltx_relight_workflow as lr

    g = lr.graph_4k()
    del g["71"]                                                     # Layer Picker: the element layer comes instead
    latent_guide(g)
    return {"name": "relight", "label": "Relight (LTX-2.3 + 4K transfer)",
            "description": "Relights the element layer with the LTX-2.3 Relight IC-LoRA from the sun you pick, then "
                           "carries the light onto the full-res element. A sun on the left is done mirrored.",
            "graph": g,
            "fields": [element_field([["2", "layer_folder", "folder"], ["57", "element_path", "pattern"],
                                      ["59", "b_path", "pattern"]]),
                       {"key": "direction", "label": "Sun from", "type": "choice", "default": "the left",
                        "choices": list(DIRECTIONS), "targets": [["72", "direction"]]},
                       {"key": "height", "label": "Sun height", "type": "choice", "default": "low",
                        "choices": list(HEIGHTS), "targets": [["72", "height"]]},
                       {"key": "look", "label": "Look", "type": "text", "default": "hard warm golden sunset sunlight",
                        "targets": [["72", "look"]]},
                       {"key": "mode", "label": "Transfer", "type": "choice", "default": "sh",
                        "choices": ["sh", "guided"], "targets": [["57", "mode"]]},
                       seed_field("49")],
            "context": {"plate": [["1", "path"]], "first": [["70", "first"]], "last": [["70", "last"]],
                        "frame_offset": [["70", "frame_offset"], ["1", "frame_offset"]],
                        "cache_dir": [["90", "out_dir"]]},
            "outputs": [["54", "out_dir", "ltx"], ["57", "out_dir", "beauty"]],
            "names": [["54", "name"], ["57", "name"]],
            "preview": {"frames": 17,
                        "nodes": {"61": TINY,
                                  "53": {"class_type": "VAEDecode", "inputs": {"samples": ["52", 2],
                                                                              "vae": ["61", 0]}}},
                        "drop": ["56", "57", "58", "59"], "outputs": [["54", "out_dir", ""]]}}


def sky() -> dict:
    import physical_sky_workflow as ps

    g = ps.graph(2048, "sky")
    f = lambda key, label, t, d, lo=None, hi=None: {k: v for k, v in {  # noqa: E731
        "key": key, "label": label, "type": t, "default": d, "min": lo, "max": hi,
        "targets": [["1", key]]}.items() if v is not None}
    return {"name": "sky", "label": "Sky (physical)",
            "description": "A physically based sky and sun for a place and time, as an ACEScg lat-long EXR.",
            "graph": g,
            "fields": [f("latitude", "Latitude", "float", -34.1906, -90.0, 90.0),
                       f("longitude", "Longitude", "float", 151.0425, -180.0, 180.0),
                       f("capture_utc", "Time (UTC)", "text", "2025-12-25T07:39:25Z"),
                       f("heading_deg", "Camera heading", "float", 180.0, -360.0, 360.0),
                       f("fov_x", "Lens field of view", "float", 69.5, 1.0, 179.0),
                       {**f("width", "Width", "choice", 2048), "choices": [1024, 2048, 4096]}],
            "context": {}, "outputs": [["1", "out_dir", ""]], "names": [["1", "name"]],
            "preview": {"set": [["1", "width", 512]]}}


def presets() -> list[dict]:
    """The Presets gallery: recipes with settings filled in, grouped. The element layer is the artist's pick."""
    def sun(pid, group, name, what, direction, height, look):
        return {"id": pid, "group": group, "name": name, "what": what, "recipe": "relight",
                "params": {"direction": direction, "height": height, "look": look}}
    return [
        sun("right-low", "Right sun", "Low sun from frame right", "Golden hour, long shadows falling left.",
            "the right", "low", "hard warm golden sunset sunlight"),
        sun("right-mid", "Right sun", "Afternoon sun from frame right", "Warm, shorter shadows, more fill.",
            "the right", "mid", "warm late afternoon sunlight"),
        sun("right-high", "Right sun", "High sun from frame right", "Midday, crisp and short-shadowed.",
            "the right", "high", "bright clean midday sunlight"),
        sun("front-right-low", "Right sun", "Low sun over the camera's right shoulder",
            "Face lit, shadows behind the subject.", "the front-right", "low", "soft warm golden light"),
        sun("back-right-low", "Right sun", "Rim light from back right", "Backlit edge, the front falls into shade.",
            "the back-right", "low", "warm golden backlight rim"),
        sun("left-low", "Left sun", "Low sun from frame left", "The DSCF3000 take: rendered mirrored.",
            "the left", "low", "hard warm golden sunset sunlight"),
        sun("front-left-low", "Left sun", "Low sun over the camera's left shoulder",
            "Face lit from the left, rendered mirrored.", "the front-left", "low", "soft warm golden light"),
        {"id": "matte", "group": "Matte", "name": "Alpha from anything",
         "what": "LTX Alpha-Gen: soft edges, no green screen; a still works too.", "recipe": "matte", "params": {}},
        {"id": "sky-burning-palms", "group": "Sky", "name": "Burning Palms, 6:39 pm",
         "what": "Physical sky for DSCF2977's place and time.", "recipe": "sky", "params": {}},
    ]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for r in (matte(), relight(), sky()):
        p = OUT / f"{r['name']}.json"
        p.write_text(ui_workflow.portable(json.dumps(r, indent=1)), encoding="utf-8")
        print("wrote", p)
    (OUT / "presets").mkdir(exist_ok=True)
    p = OUT / "presets" / "presets.json"
    p.write_text(json.dumps({"presets": presets()}, indent=1), encoding="utf-8")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
