"""Atlas VFX Guide From Latent: add a cached latent (Latent Pre-render, catalogue 65) as the LTX IC-LoRA guide,
exactly as LTX's own "Add Video IC-LoRA Guide" adds the same frames after encoding them.

That node encodes the image, then hands the latent to LTXVAddGuide.append_keyframe and records the guide in the
attention entries. This node does those two steps with the cached latent, calling LTX's own code from the loaded
ComfyUI-LTXVideo pack, so the conditioning is the same by construction (tests/test_latent_guide.py compares the two).
Covered: the whole clip from frame 0, latent_downscale_factor 1 (what the Relight and Alpha-Gen LoRAs declare).
Anything else is refused rather than guessed."""
from __future__ import annotations

import sys


def _ltx_iclora():
    """ComfyUI-LTXVideo's iclora module as ComfyUI loaded it (found by what it holds, not by its folder name)."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("iclora") and hasattr(mod, "LTXAddVideoICLoRAGuide"):
            return mod
    raise ValueError("Guide From Latent needs the ComfyUI-LTXVideo pack (its IC-LoRA guide node) loaded")


def _ltx_attention():
    for name, mod in list(sys.modules.items()):
        if name.endswith("iclora_attention") and hasattr(mod, "append_guide_attention_entry"):
            return mod
    raise ValueError("Guide From Latent needs ComfyUI-LTXVideo's iclora_attention module loaded")


class AtlasVFXGuideFromLatent:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    DESCRIPTION = ("LTX's Add Video IC-LoRA Guide for a reference already encoded by Latent Pre-render: the same "
                   "conditioning, without encoding the frames again. Whole clip from frame 0, downscale 1 (wire the "
                   "IC-LoRA loader's latent_downscale_factor in).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE", {"tooltip": "the job's video VAE: its layout and output dtype, nothing is encoded"}),
                "latent": ("LATENT", {"tooltip": "the empty video latent the sampler fills"}),
                "guide": ("LATENT", {"tooltip": "Latent Pre-render's latent (or Load Latent of its file)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 1.0}),
            }
        }

    def run(self, positive, negative, vae, latent, guide, strength=1.0, latent_downscale_factor=1.0):
        import comfy_extras.nodes_lt as nodes_lt

        if float(latent_downscale_factor) != 1.0:
            raise ValueError(f"latent_downscale_factor {latent_downscale_factor}: Guide From Latent covers downscale "
                             f"1 only (the Relight and Alpha-Gen LoRAs); use LTX's image guide for others")
        icl = _ltx_iclora()
        att = _ltx_attention()
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        g = guide["samples"]
        if g.ndim != 5 or g.shape[1] != latent_image.shape[1]:
            raise ValueError(f"guide latent {tuple(g.shape)}: expected (1, {latent_image.shape[1]}, frames, h, w)")
        if tuple(g.shape[3:]) != tuple(latent_image.shape[3:]):
            raise ValueError(f"guide size {tuple(g.shape[3:])} is not the job's latent size "
                             f"{tuple(latent_image.shape[3:])}: pre-render the reference at the job's work size")
        out_dtype = vae.vae_output_dtype() if hasattr(vae, "vae_output_dtype") else g.dtype
        g = g.to(dtype=out_dtype, device=latent_image.device)
        _, _, latent_length, _, _ = latent_image.shape
        image_frames = (g.shape[2] - 1) * scale_factors[0] + 1          # the clip length this latent encodes
        guide_orig_shape = list(g.shape[2:])
        tokens = g.shape[2] * g.shape[3] * g.shape[4]
        frame_idx = icl.LTXAddVideoICLoRAGuide.resolve_frame_idx(positive, 0, latent_length, image_frames, g,
                                                                 scale_factors)
        positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx, latent_image, noise_mask, g, strength, scale_factors,
            guide_mask=None, latent_downscale_factor=1.0, causal_fix=True)
        positive = att.append_guide_attention_entry(positive, tokens, guide_orig_shape)
        negative = att.append_guide_attention_entry(negative, tokens, guide_orig_shape)
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


NODE_CLASS_MAPPINGS = {"AtlasVFXGuideFromLatent": AtlasVFXGuideFromLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXGuideFromLatent": "Atlas VFX Guide From Latent"}
