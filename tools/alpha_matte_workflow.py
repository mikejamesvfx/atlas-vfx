"""Build the Alpha-Gen starter: workflows/alpha_matte.json

  Load Image (a still, or swap in Sequence Read Chunk for a clip) -> Alpha Prep (fit 1920x1088, pad /32, hold a still
  to 9 frames) -> LTX-2.5 Alpha-Gen IC-LoRA (distilled int8, single stage, empty prompt, model-card settings) ->
  VAE Decode -> Alpha Return (matte at the source size, a still's mattes averaged) -> EXR Write (RGB + matte alpha)
  + Viewer. A Tiny VAE decode of the same latent gives a quick look at the matte in a second.

graph() is the API graph. Uses tools/ui_workflow.py's convert + portable with this checkout's node definitions for
the atlas-vfx nodes and a running ComfyUI's /object_info for the core and LTX ones. Writes only the repo copy.

Run: <V141 python> tools\\alpha_matte_workflow.py [--api http://127.0.0.1:8188]
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

VIEW = "sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)"
SRGB = "sRGB Encoded Rec.709 (sRGB)"


def graph(image: str = "example.png") -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "AtlasVFXAlphaPrep", "inputs": {
            "images": ["1", 0], "still_frames": 9, "max_width": 1920, "max_height": 1088}},
        "12": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "type": "ltxv",
            "device": "default"}},
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
        "15": {"class_type": "LTXVVAEForceFloat32", "inputs": {"vae": ["14", 0]}},
        "16": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "17": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
            "model": ["12", 0], "lora_name": "ltx25\\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
            "strength_model": 1.0}},
        "39": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["13", 0]}},
        "41": {"class_type": "LTXVConditioning", "inputs": {"positive": ["39", 0], "negative": ["39", 0],
                                                            "frame_rate": 24.0}},
        "42": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": ["2", 2], "height": ["2", 3], "length": ["2", 1], "batch_size": 1}},
        "43": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["41", 0], "negative": ["41", 1], "vae": ["15", 0], "latent": ["42", 0], "image": ["2", 0],
            "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": ["17", 1], "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64, "use_streaming": False,
            "tile_frames": ["2", 1]}},
        "44": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": ["2", 1], "frame_rate": 24, "batch_size": 1, "audio_vae": ["16", 0]}},
        "45": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["43", 2], "audio_latent": ["44", 0]}},
        "46": {"class_type": "CFGGuider", "inputs": {
            "model": ["17", 0], "positive": ["43", 0], "negative": ["43", 1], "cfg": 1.0}},
        "47": {"class_type": "ManualSigmas", "inputs": {
            "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "48": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "49": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1234}},
        "50": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["49", 0], "guider": ["46", 0], "sampler": ["48", 0], "sigmas": ["47", 0],
            "latent_image": ["45", 0]}},
        "51": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["50", 0]}},
        "52": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["43", 0], "negative": ["43", 1],
                                                          "latent": ["51", 0]}},
        "53": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["52", 2], "vae": ["15", 0], "tile_size": 1024, "overlap": 256, "temporal_size": 128,
            "temporal_overlap": 48}},
        "60": {"class_type": "AtlasVFXAlphaReturn", "inputs": {
            "images": ["53", 0], "alpha_prep": ["2", 4], "channel": "green", "still_average": True}},
        "61": {"class_type": "AtlasVFXTinyVAE", "inputs": {"tae_name": "taeltx2_3.safetensors"}},
        "62": {"class_type": "VAEDecode", "inputs": {"samples": ["52", 2], "vae": ["61", 0]}},
        "63": {"class_type": "AtlasVFXAlphaReturn", "inputs": {
            "images": ["62", 0], "alpha_prep": ["2", 4], "channel": "green", "still_average": True}},
        "64": {"class_type": "PreviewImage", "inputs": {"images": ["63", 1]}},
        "65": {"class_type": "AtlasVFXEXRWrite", "inputs": {
            "images": ["1", 0], "first_frame": 1001, "input_colorspace": SRGB, "colorspace": "ACEScg",
            "input_premultiplied": False, "ocio_config": "", "name": "alpha_matte", "out_dir": "", "mask": ["60", 0]}},
        "66": {"class_type": "AtlasVFXViewer", "inputs": {
            "a_path": "", "b_path": "", "frame": 1001, "mode": "side by side", "wipe": 0.5, "exposure": 0.0,
            "gamma": 1.0, "view": VIEW, "zoom_to_alpha": False, "width": 1280, "a_image": ["1", 0],
            "b_image": ["60", 1]}},
        "67": {"class_type": "AtlasVFXNotes", "inputs": {"info": ["60", 2]}},
    }


def register_local():
    from atlas_vfx.registry import load_nodes

    classes = load_nodes()[0]
    for cls_name in ("AtlasVFXAlphaPrep", "AtlasVFXAlphaReturn", "AtlasVFXTinyVAE", "AtlasVFXEXRWrite",
                     "AtlasVFXViewer", "AtlasVFXNotes"):
        ui_workflow._INFO[cls_name] = local_info(classes[cls_name])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8188", help="a running ComfyUI, for the core node definitions")
    ui_workflow.API = ap.parse_args(argv).api
    register_local()
    wf = ui_workflow.convert(graph(), "LTX Alpha-Gen matte: a still or a clip to RGB + alpha EXR, quick look with "
                                      "the tiny VAE")
    path = ui_workflow.REPO / "alpha_matte.json"
    path.write_text(ui_workflow.portable(json.dumps(wf, indent=1)), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
