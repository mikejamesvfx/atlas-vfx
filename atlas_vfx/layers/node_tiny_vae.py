"""Atlas VFX Tiny VAE: loads the LTX-2.3 / 2.5 tiny autoencoder (taeltx2_3, madebyollin/taehv, MIT) from vae_approx
as a VAE, for quick-look decodes: wire it to a VAE Decode next to the real one to see a take in seconds, blurry but
right in framing, motion and light. Core's VAE Loader does not list this file (it only knows taeltx_2).

Loading this pack also turns on live sampler previews for LTX through it (ltx_live_preview.py): set ComfyUI's live
preview method to TAESD."""
from __future__ import annotations

import logging

from . import ltx_live_preview as llp

NONE = "(put taeltx2_3.safetensors in models/vae_approx)"


def choices() -> list[str]:
    try:
        import folder_paths
    except ImportError:
        return [NONE]
    found = [f for f in folder_paths.get_filename_list("vae_approx") if f.startswith("taeltx")
             and f.endswith(".safetensors")]
    return found or [NONE]


class AtlasVFXTinyVAE:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    DESCRIPTION = ("The LTX tiny VAE (taeltx2_3) for a quick look: VAE Decode with it decodes a take in a second or "
                   "two at low cost, blurry but right in framing, motion and light. For previews only, never for "
                   "delivery. It also drives live sampler previews for LTX when ComfyUI's live preview method is "
                   "TAESD.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tae_name": (choices(), {"tooltip": "safetensors only (no pickled .pth)"})}}

    def run(self, tae_name):
        import comfy.sd
        import comfy.utils
        import folder_paths

        if tae_name == NONE or not tae_name.startswith("taeltx") or not tae_name.endswith(".safetensors"):
            raise ValueError("no taeltx*.safetensors in models/vae_approx (see atlas_components.json, taeltx2-3)")
        path = folder_paths.get_full_path_or_raise("vae_approx", tae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path, safe_load=True))
        if vae.latent_channels != 128:
            raise ValueError(f"{tae_name}: {vae.latent_channels} latent channels, LTX needs 128")
        return (vae,)


logging.info("atlas-vfx LTX live preview: %s", llp.install())

NODE_CLASS_MAPPINGS = {"AtlasVFXTinyVAE": AtlasVFXTinyVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXTinyVAE": "Atlas VFX Tiny VAE"}
