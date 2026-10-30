"""Build the LTX-2.3 Relight starter: workflows/ltx_relight_mirror.json

  Plate Read -> LTX Prep (reference = comp, mirror on, 1280x704, 97 frames) -> light ball top-right -> LTX-2.3 Relight
  IC-LoRA (LTX-2.5 distilled base, single stage) -> LTX Return (flips the take back, ACEScg layer at plate res)

The relit layer LTX Return writes is the `relit` input of workflows/relight_transfer_element.json (Relight Transfer,
catalogue 53), which puts the light back on the full-res element.

Why mirror: on DSCF3000 the Relight LoRA kept the sun on frame right whatever the ball or prompt said (left, front-left,
back-left, strength 1.5 all tried, 2026-10-30). Prep flips what LTX sees, the ball and prompt ask for the right, and
Return flips the take back, so the sun lands on frame left. For a sun on frame right, set mirror off and keep the same
ball and prompt.

The ball is a plain LoadImage: put the light-direction ball PNG (Sphere-Light-Render-ComfyUI, MIT; named
"<direction>__<band>.png", see the README next to the balls) in ComfyUI's input folder. It is scaled to 143 px and
pasted 22 px from the top-right corner of the 1280x704 reference, on every frame, after Prep (so on the mirrored frame).

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

PLATE = "~/Videos/atlas-vfx_footage/DSCF3000.MOV"
LAYER = "~/atlas-vfx-work/DSCF3000/layers/showcase_abc_showcase_element"
BALL = "the right__low.png"
PROMPT = ("relight the video to match the light-direction ball. hard warm golden sunset sunlight entering from frame "
          "right, long shadows falling to the left from the right, low sun")
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, deformed"
W, H, N, FPS = 1280, 704, 97, 24
BALL_PX, BALL_PAD = 143, 22


def graph() -> dict:
    return {
        "1": {"class_type": "AtlasVFXPlateRead", "inputs": {
            "path": PLATE, "first": 129, "last": 296, "ocio_config": "", "input_space": "auto", "range": "auto",
            "frame_offset": 1001, "preview_width": 512, "preview_every": 56}},
        "2": {"class_type": "AtlasVFXLTXPrep", "inputs": {
            "plate_ref": ["1", 0], "layer_folder": LAYER, "first_frame": 1130, "length": N, "width": W, "height": H,
            "mask_thresh": 0.05, "shadow_px": 24, "dilate_px": 12, "keyframe_every": 24, "mode": "full",
            "crop_margin": 96, "extra_layers": "", "reference": "comp", "mask_mode": "layer",
            "sensor_rs_folder": "", "sensor_lens_folder": "", "mirror": True}},
        "3": {"class_type": "LoadImage", "inputs": {"image": BALL}},
        "4": {"class_type": "ImageScale", "inputs": {
            "image": ["3", 0], "upscale_method": "area", "width": BALL_PX, "height": BALL_PX, "crop": "disabled"}},
        "5": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["2", 0], "source": ["4", 0], "x": W - BALL_PAD - BALL_PX, "y": BALL_PAD,
            "resize_source": False}},
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
        "39": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["13", 0]}},
        "40": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["13", 0]}},
        "41": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["39", 0], "negative": ["40", 0], "frame_rate": float(FPS)}},
        "42": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": N,
                                                                "batch_size": 1}},
        "43": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["41", 0], "negative": ["41", 1], "vae": ["15", 0], "latent": ["42", 0], "image": ["5", 0],
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": ["17", 1], "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "use_streaming": False,
            "tile_frames": N}},
        "44": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": N, "frame_rate": FPS, "batch_size": 1, "audio_vae": ["16", 0]}},
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    for cls in (AtlasVFXPlateRead, AtlasVFXLTXPrep, AtlasVFXLTXReturn):
        ui_workflow._INFO[cls.__name__] = local_info(cls)
    wf = ui_workflow.convert(graph(), "LTX-2.3 Relight with mirror: sun on frame left on DSCF3000, then Relight "
                                      "Transfer")
    path = ui_workflow.REPO / "ltx_relight_mirror.json"
    path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
