"""T6 on real data: generate tall grass (Z-Image-Turbo, 2 seeds), matte it (core BiRefNet), place two cards on the
lawn in front of the deck (1.0 m left and 2.0 m right of the camera path at 1130, 1.35-1.4 m tall), render both through the GUESS camera for S2-S3.

Run: <V141 python> tools\\run_t6_grass.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_run import run  # noqa: E402
from run_guess_s2s3 import GRAPH as G  # noqa: E402

PROMPT = ("A clump of tall wild meadow grass with thin seed-head stalks and a few dry blades, about one metre tall, "
          "growing straight up from a small patch of soil at the bottom edge, photographed straight on at eye level "
          "against a seamless plain light grey studio backdrop. Soft overcast daylight from the upper left, muted "
          "natural greens and straw colours, crisp detail on every blade, the whole clump in frame from the roots to "
          "the tips, nothing else in the scene.")


def fg(batch, distance, lateral, ground, name, height=0.9):
    return {"class_type": "AtlasVFXForegroundElement", "inputs": {
        "image": ["14", 0], "mask": ["16", 0], "shot_camera": ["3", 0], "batch_index": batch, "frame": 1130,
        "distance_m": distance, "lateral_m": lateral, "height_m": height, "ground_y": ground, "yaw_deg": 0.0,
        "erode_px": 1, "mask_is_background": False,
        "ocio_config": r"C:\Users\miike\atlas-vfx\config\atlas_aces2.0_studio_flog2.ocio", "name": name,
        "out_dir": ""}}


def proj(card_node):
    return {"class_type": "AtlasVFXCardProject", "inputs": {
        "plate_ref": ["1", 0], "shot_camera": ["3", 0], "card": [card_node, 0], "scale": 1.0, "frame_step": 1,
        "preview_every": 12, "preview_width": 1024, "out_dir": ""}}


GRAPH = {
    "1": G["1"], "2": G["2"], "3": G["3"],
    # Z-Image-Turbo (template image_z_image_turbo: 8 steps, cfg 1, res_multistep / simple, AuraFlow shift 3)
    "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "Z_Image_Turbo_BF16.safetensors",
                                                   "weight_dtype": "default"}},
    "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2",
                                                   "device": "default"}},
    "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "13": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["10", 0], "shift": 3.0}},
    "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": PROMPT}},
    "21": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["20", 0]}},
    "22": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 2}},
    "23": {"class_type": "KSampler", "inputs": {
        "model": ["13", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["22", 0], "seed": 1130,
        "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
    "14": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["12", 0]}},
    "24": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "atlas-vfx/T6_grass_zimage"}},
    "15": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
    "16": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["15", 0], "image": ["14", 0]}},
    # the camera is ~1.37 m above the lawn: grass this close only enters from the bottom of frame, so it is tall
    "30": fg(0, 1.0, -0.6, "-0.76", "grass_near", 1.35),
    "31": fg(1, 2.0, 1.3, "-0.76", "grass_far", 1.4),
    "32": {"class_type": "PreviewImage", "inputs": {"images": ["30", 1]}},
    "33": {"class_type": "PreviewImage", "inputs": {"images": ["31", 1]}},
    "40": proj("30"),
    "41": proj("31"),
    "42": {"class_type": "PreviewImage", "inputs": {"images": ["40", 1]}},
    "43": {"class_type": "PreviewImage", "inputs": {"images": ["41", 1]}},
}

if __name__ == "__main__":
    run(GRAPH, "t6_grass_cards_s2s3", "T6 Foreground grass: Z-Image, BiRefNet matte, two cards, Card Project",
        timeout=3600)
