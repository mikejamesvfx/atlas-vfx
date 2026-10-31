"""Live sampler previews for LTX-2.3 / 2.5 with the tiny VAE taeltx2_3 (madebyollin/taehv, MIT).

ComfyUI shows a live preview while a sampler runs. For LTX it only has a rough colour projection (Latent2RGB): core
lists taeltx_2 as a video tiny VAE but no LTX latent format names one. This wraps latent_preview.get_previewer so that,
when the preview method is TAESD (ComfyUI settings > Live preview method) and vae_approx holds taeltx2_3, LTX latents
are previewed through it: the first latent frame, decoded at full frame size, in a fraction of a second.

Safe by construction: anything unexpected (another latent layout, a missing file, a decode error) falls back to
ComfyUI's own previewer, and a preview can never stop a render. ATLAS_VFX_LTX_LIVE_PREVIEW=0 turns the hook off."""
from __future__ import annotations

import logging
import os

TAE_NAME = "taeltx2_3"
_state = {"installed": False, "reason": "not tried"}


def tae_file() -> str | None:
    import folder_paths

    for f in folder_paths.get_filename_list("vae_approx"):
        if f.startswith(TAE_NAME) and f.endswith(".safetensors"):
            return folder_paths.get_full_path("vae_approx", f)
    return None


def _is_ltx(latent_format) -> bool:
    import comfy.latent_formats as lf

    return isinstance(latent_format, lf.LTXV)


class _SafeLTXPreviewer:
    """Decodes the first latent frame of an LTX video latent with the tiny VAE; otherwise asks the fallback."""

    def __init__(self, tae, fallback):
        self.tae, self.fallback, self.failed = tae, fallback, False

    def decode_latent_to_preview_image(self, preview_format, x0):
        import latent_preview

        if not self.failed and x0.ndim == 5 and x0.shape[1] == 128:
            try:
                img = self.tae.decode(x0[:1, :, :1])[0][0]
                pil = latent_preview.preview_to_image(img, do_scale=False)
                return "JPEG", pil, latent_preview.MAX_PREVIEW_RESOLUTION
            except Exception as e:  # noqa: BLE001 (a preview must never stop the render)
                self.failed = True
                logging.warning("atlas-vfx LTX live preview: %s; using ComfyUI's preview", e)
        if self.fallback is None:
            return None
        return self.fallback.decode_latent_to_preview_image(preview_format, x0)


def install() -> str:
    """Wrap ComfyUI's get_previewer once. Returns why it is or is not active (logged at load)."""
    if _state["installed"]:
        return _state["reason"]
    if os.environ.get("ATLAS_VFX_LTX_LIVE_PREVIEW", "1") == "0":
        _state["reason"] = "off (ATLAS_VFX_LTX_LIVE_PREVIEW=0)"
        return _state["reason"]
    try:
        import latent_preview
    except ImportError:
        _state["reason"] = "not inside ComfyUI"
        return _state["reason"]
    original = latent_preview.get_previewer
    cache = {}

    def get_previewer(device, latent_format):
        fallback = original(device, latent_format)
        try:
            from comfy.cli_args import LatentPreviewMethod, args

            if args.preview_method != LatentPreviewMethod.TAESD or not _is_ltx(latent_format):
                return fallback
            path = tae_file()
            if path is None:
                return fallback
            if path not in cache:
                import comfy.sd
                import comfy.utils

                tae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path, safe_load=True))
                tae.first_stage_model.show_progress_bar = False
                cache[path] = tae
            return _SafeLTXPreviewer(cache[path], fallback)
        except Exception as e:  # noqa: BLE001
            logging.warning("atlas-vfx LTX live preview off for this run: %s", e)
            return fallback

    latent_preview.get_previewer = get_previewer
    _state.update(installed=True, reason=f"on when the live preview method is TAESD and vae_approx has {TAE_NAME}")
    return _state["reason"]
