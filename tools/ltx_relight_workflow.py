"""Build the LTX-2.3 Relight starter: workflows/ltx_relight_mirror.json

  Shot Range + Layer Picker + Sun Direction (the three things an artist sets) -> Plate Read -> LTX Prep (reference =
  comp, 1280x704, mirror from Sun Direction) -> Light Ball -> LTX-2.3 Relight IC-LoRA (LTX-2.5 distilled base, single
  stage, prompt from Sun Direction) -> LTX Return (flips the take back, ACEScg layer at plate res)

Also writes workflows/ltx_relight_mirror_4k.json, the whole job in one graph: the same LTX pass, then LTX Return's
folder straight into Relight Transfer Files (catalogue 54, MoGe-3 run per frame inside it), which puts the light on
the full-res element and writes the 4K EXRs one frame at a time. The element is the Card Project layer LTX Prep
comps in (its alpha is the matte).

Why mirror: on DSCF3000 the Relight LoRA kept the sun on frame right whatever the ball or prompt said (left, front-left,
back-left, strength 1.5 all tried, 2026-10-30). Prep flips what LTX sees, the ball and prompt ask for the right, and
Return flips the take back, so the sun lands on frame left. For a sun on frame right, set mirror off and keep the same
ball and prompt.

Sun Direction (catalogue 56) does the mirror for any sun with "left" in it and writes the prompt; Light Ball (57)
pastes the matching ball (shipped in assets/relight_balls) at the training layout, after Prep, so on the mirrored
frame. Shot Range (55) feeds every frame number (Plate Read's index, Prep, the latent lengths, the transfer); Layer
Picker (58) feeds the element. The 4K graph ends in Notes (61) and a Viewer (60) wiping the result over the element.

graph() is the API graph. Uses tools/ui_workflow.py's convert + portable with this checkout's node definitions for
the atlas-vfx nodes and a running ComfyUI's /object_info for the core and LTX ones. Writes only the repo copy.

Run: <V141 python> tools\\ltx_relight_workflow.py [--api http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ui_workflow  # noqa: E402
from physical_sky_workflow import local_info  # noqa: E402
from atlas_vfx.camera_side.node_plate_read import AtlasVFXPlateRead  # noqa: E402
from atlas_vfx.layers.node_ltx_prep import AtlasVFXLTXPrep  # noqa: E402
from atlas_vfx.layers.node_ltx_return import AtlasVFXLTXReturn  # noqa: E402
from atlas_vfx.layers.node_relight_transfer_files import AtlasVFXRelightTransferFiles  # noqa: E402
from atlas_vfx.layers.node_layer_picker import AtlasVFXLayerPicker  # noqa: E402
from atlas_vfx.layers.node_light_ball import AtlasVFXLightBall  # noqa: E402
from atlas_vfx.layers.node_notes import AtlasVFXNotes  # noqa: E402
from atlas_vfx.layers.node_shot_range import AtlasVFXShotRange  # noqa: E402
from atlas_vfx.layers.node_sun_direction import AtlasVFXSunDirection  # noqa: E402
from atlas_vfx.layers.node_viewer import AtlasVFXViewer  # noqa: E402

PLATE = "~/Videos/atlas-vfx_footage/DSCF3000.MOV"
LAYER = "DSCF3000/layers/showcase_abc_showcase_element"            # Layer Picker: relative to the work root
LOOK = "hard warm golden sunset sunlight"
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, deformed"
W, H, N, FPS = 1280, 704, 97, 24


def graph() -> dict:
    return {
        "70": {"class_type": "AtlasVFXShotRange", "inputs": {
            "first": 1130, "last": 1226, "ltx_length": "8k+1 shorter", "frame_offset": 1001, "width": 4096,
            "height": 2160}},
        "71": {"class_type": "AtlasVFXLayerPicker", "inputs": {"layer": LAYER}},
        "72": {"class_type": "AtlasVFXSunDirection", "inputs": {
            "direction": "the left", "height": "low", "look": LOOK, "mirror_trick": "auto"}},
        "1": {"class_type": "AtlasVFXPlateRead", "inputs": {
            "path": PLATE, "first": ["70", 3], "last": ["70", 4], "ocio_config": "", "input_space": "auto",
            "range": "auto", "frame_offset": 1001, "preview_width": 512, "preview_every": 56}},
        "2": {"class_type": "AtlasVFXLTXPrep", "inputs": {
            "plate_ref": ["1", 0], "layer_folder": ["71", 0], "first_frame": ["70", 0], "length": ["70", 1],
            "width": W, "height": H, "mask_thresh": 0.05, "shadow_px": 24, "dilate_px": 12, "keyframe_every": 24,
            "mode": "full", "crop_margin": 96, "extra_layers": "", "reference": "comp", "mask_mode": "layer",
            "sensor_rs_folder": "", "sensor_lens_folder": "", "mirror": ["72", 0]}},
        "5": {"class_type": "AtlasVFXLightBall", "inputs": {"image": ["2", 0], "sun": ["72", 2]}},
        "12": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "type": "ltxv",
            "device": "default"}},
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
        "15": {"class_type": "LTXVVAEForceFloat32", "inputs": {"vae": ["14", 0]}},
        "16": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "17": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
            "model": ["12", 0], "lora_name": "ltx23\\ltx-2.3-22b-ic-lora-relight-1.0.safetensors",
            "strength_model": 1.0}},
        "39": {"class_type": "CLIPTextEncode", "inputs": {"text": ["72", 1], "clip": ["13", 0]}},
        "40": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
        "41": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["39", 0], "negative": ["40", 0], "frame_rate": float(FPS)}},
        "42": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": ["70", 1],
                                                                "batch_size": 1}},
        "43": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["41", 0], "negative": ["41", 1], "vae": ["15", 0], "latent": ["42", 0], "image": ["5", 0],
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": ["17", 1], "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "use_streaming": False,
            "tile_frames": ["70", 1]}},
        "44": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": ["70", 1], "frame_rate": FPS, "batch_size": 1, "audio_vae": ["16", 0]}},
        "45": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["43", 2], "audio_latent": ["44", 0]}},
        "46": {"class_type": "CFGGuider", "inputs": {
            "model": ["17", 0], "positive": ["43", 0], "negative": ["43", 1], "cfg": 1.0}},
        "47": {"class_type": "ManualSigmas", "inputs": {
            "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "48": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "49": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "50": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["49", 0], "guider": ["46", 0], "sampler": ["48", 0], "sigmas": ["47", 0],
            "latent_image": ["45", 0]}},
        "51": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["50", 0]}},
        "52": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["43", 0], "negative": ["43", 1],
                                                          "latent": ["51", 0]}},
        "53": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["52", 2], "vae": ["15", 0], "tile_size": 768, "overlap": 192, "temporal_size": 64,
            "temporal_overlap": 16}},
        "54": {"class_type": "AtlasVFXLTXReturn", "inputs": {
            "images": ["53", 0], "ltx_prep": ["2", 5], "feather_px": 3.0, "name": "element_relit",
            "source_model": "ltx-2.5-22b-distilled int8 + ltx-2.3-22b-ic-lora-relight-1.0", "preview_every": 24,
            "out_dir": ""}},
        "55": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    }


def graph_4k() -> dict:
    """graph() plus the 4K transfer: LTX Return's folder is the relit take of Relight Transfer Files."""
    g = graph()
    g["56"] = {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_3_vitl_fp16.safetensors"}}
    g["57"] = {"class_type": "AtlasVFXRelightTransferFiles", "inputs": {
        "element_path": ["71", 1], "relit_path": ["54", 0], "matte_path": "", "first_frame": ["70", 0],
        "length": ["70", 1],
        "relit_numbering": "frame", "relit_encoding": "linear", "mode": "sh", "analysis_width": W, "blur_px": 2.0,
        "max_stops": 3.0, "temporal": 1, "eps": 0.01, "subsample": 2, "geo_weight": 0.5, "min_r2": 0.3,
        "lift_limit": 3.0, "name": "element_relit_4k", "out_dir": "", "moge_model": ["56", 0],
        "resolution_level": 9, "refine_steps": 3, "normals_temporal": 1, "normals_sigma": 1.5, "depth_guide": True,
        "relit_mirrored": False, "holdout_path": "", "holdout_grow": 2, "ocio_config": ""}}
    g["58"] = {"class_type": "AtlasVFXNotes", "inputs": {"info": ["57", 2]}}
    g["59"] = {"class_type": "AtlasVFXViewer", "inputs": {
        "a_path": ["57", 0], "b_path": ["71", 1], "frame": ["70", 0], "mode": "wipe", "wipe": 0.5, "exposure": 0.0,
        "gamma": 1.0, "view": "sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)", "zoom_to_alpha": True,
        "width": 1280}}
    return g


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    for cls in (AtlasVFXPlateRead, AtlasVFXLTXPrep, AtlasVFXLTXReturn, AtlasVFXRelightTransferFiles,
                AtlasVFXShotRange, AtlasVFXLayerPicker, AtlasVFXSunDirection, AtlasVFXLightBall, AtlasVFXNotes,
                AtlasVFXViewer):
        ui_workflow._INFO[cls.__name__] = local_info(cls)
    wf = ui_workflow.convert(graph(), "LTX-2.3 Relight with mirror: sun on frame left on DSCF3000, then Relight "
                                      "Transfer")
    path = ui_workflow.REPO / "ltx_relight_mirror.json"
    path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", path)
    wf = ui_workflow.convert(graph_4k(), "LTX-2.3 Relight with mirror, then Relight Transfer Files: the relit "
                                         "element at 4K in one graph")
    path = ui_workflow.REPO / "ltx_relight_mirror_4k.json"
    path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
