"""Atlas VFX Character Insert (catalogue 16): a generated character clip on flat grey -> keyed ACEScg RGBA texture
sequence -> an animated ATLAS_VFX_CARD standing on the deck at a picked point, scaled to the figure's height. Render it with
Card Project."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import character_insert as ci

log = logging.getLogger("atlas_vfx")

# Source models whose licence limits where the result may be used. The node cannot see the model, so the graph says
# which one made the clip; the warning lands in the log and the sidecar.
LICENCE_WARNINGS = {
    "minimax-h3": ("MiniMax H3 Community Licence: no use in the US, EU, UK or Korea, outputs may not train other models. "
                   "Own tests only; never on client work."),
}


class AtlasVFXCharacterInsert:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_VFX_CARD", "IMAGE", "STRING")
    RETURN_NAMES = ("card", "preview", "report")
    DESCRIPTION = ("Keys a character clip generated on flat grey (distance key + erode + edge decontamination, OCIO "
                   "sRGB -> ACEScg) into a premultiplied RGBA texture sequence and stands it on a card whose feet line "
                   "sits on the deck plane (y = deck_y) where the pick ray (pick_frame, pick_u, pick_v; plate pixels, "
                   "top-left origin) meets it, scaled so the figure is height_m tall. 2.5D: billboard faces the camera "
                   "every frame.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "first_frame": ("INT", {"default": 1130, "min": 0, "max": 1_000_000}),
                "last_frame": ("INT", {"default": 1297, "min": 0, "max": 1_000_000}),
                "pick_frame": ("INT", {"default": 1130, "min": 0, "max": 1_000_000}),
                "pick_u": ("FLOAT", {"default": 2048.0, "min": -100000.0, "max": 100000.0, "step": 0.5}),
                "pick_v": ("FLOAT", {"default": 1600.0, "min": -100000.0, "max": 100000.0, "step": 0.5}),
                "deck_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "height_m": ("FLOAT", {"default": 2.13, "min": 0.1, "max": 50.0, "step": 0.01}),
                "key_lo": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.005}),
                "key_hi": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.5, "step": 0.005}),
                "erode_px": ("INT", {"default": 1, "min": 0, "max": 20}),
                "mode": (["billboard", "fixed"], {"default": "billboard"}),
                "clip_start": ("INT", {"default": 0, "min": 0, "max": 100000,
                                       "tooltip": "clip frame shown on first_frame"}),
                "end_mode": (["hold", "pingpong", "loop"], {"default": "pingpong",
                                                            "tooltip": "past the clip's end: hold the last frame, "
                                                                       "play it back and forth, or loop from clip_start"}),
                "source_model": ("STRING", {"default": "",
                                            "tooltip": "register id of the model that made the clip (licence check)"}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "name": ("STRING", {"default": "element"}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "per-frame matte, 1 = figure (e.g. core RemoveBackground / BiRefNet). "
                                             "Replaces the distance key; key_lo/key_hi are then unused."}),
                "exposure_stops": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.25,
                                             "tooltip": "brighten (+) or darken (-) the character, in stops, "
                                                        "applied in linear ACEScg before the card is written"}),
            },
        }

    def run(self, images, shot_camera, first_frame, last_frame, pick_frame, pick_u, pick_v, deck_y, height_m, key_lo,
            key_hi, erode_px, mode, clip_start, end_mode, source_model, ocio_config, name, out_dir, mask=None,
            exposure_stops=0.0):
        import torch

        warn = LICENCE_WARNINGS.get(source_model.strip().lower())
        if warn:
            log.warning("atlas-vfx Character Insert (%s): %s", source_model, warn)
        cam = sc.ShotCamera(shot_camera)
        frames = [n for n in cam.frames if first_frame <= n <= last_frame]
        if not frames:
            raise ValueError(f"no camera frames in {first_frame}-{last_frame}")
        if pick_frame not in cam.frames:
            raise ValueError(f"pick_frame {pick_frame} not in camera {cam.name}")

        clip = images.cpu().numpy()[..., :3].astype(np.float32)
        if mask is not None:
            alpha, B, plate_info = ci.mask_key(clip, mask.cpu().numpy(), erode_px)
            rgba, rep = ci.make_sequence(clip, alpha, B, ocio_config, method=ci.MASK_METHOD)
        else:
            alpha, B, plate_info = ci.grey_key(clip, key_lo, key_hi, erode_px)
            rgba, rep = ci.make_sequence(clip, alpha, B, ocio_config)
        rep["backdrop_plate"] = plate_info
        if exposure_stops:
            rgba[..., :3] *= np.float32(2.0 ** float(exposure_stops))     # premultiplied linear: scale RGB only
        rep["exposure_stops"] = float(exposure_stops)
        foot = ci.deck_point_from_pick(cam, pick_frame, pick_u, pick_v, deck_y)
        th, tw = rgba.shape[1:3]
        card = ci.place_character(cam, frames, foot, (tw, th), rep["figure_px"], rep["feet_v"], height_m, deck_y,
                                  mode=mode, face_frame=pick_frame)

        plate_path = Path(cam.data["plate"]["path"]) if cam.data["plate"]["path"] else Path("plate")
        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / plate_path.stem / "elements" / cam.name / name)
        for i in range(len(rgba)):
            exr_io.write_rgba(out / f"{name}.tex.{i:04d}.exr", rgba[i],
                              attrs={"element": name, "camera": cam.name, "kind": "character texture", "clip_frame": i})
        card.update({"name": name, "camera": cam.name, "texture_pattern": str(out / f"{name}.tex.%04d.exr"),
                     "texture_by_frame": ci.time_map(len(rgba), frames, clip_start, end_mode),
                     "texture": str(out / f"{name}.tex.0000.exr"), "texture_size": [int(tw), int(th)],
                     "colorspace": "ACEScg", "premultiplied": True, "key": rep,
                     "pick": {"frame": pick_frame, "u": pick_u, "v": pick_v}, "source_model": source_model,
                     "licence_warning": warn})
        (out / f"{name}.card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXCharacterInsert",
                         inputs={"camera": cam.name, "frames": [frames[0], frames[-1]], "clip_shape": list(clip.shape),
                                 "pick": card["pick"], "deck_y": deck_y, "height_m": height_m, "key_lo": key_lo,
                                 "key_hi": key_hi, "erode_px": erode_px, "matte": "mask input" if mask is not None else "distance key", "mode": mode, "clip_start": clip_start,
                                 "end_mode": end_mode, "source_model": source_model, "ocio_config": ocio_config,
                                 "exposure_stops": exposure_stops},
                         outputs={"texture_pattern": card["texture_pattern"], "foot": card["foot"],
                                  "size_m": card["size_m"], "key": rep, "licence_warning": warn})

        # preview: a few keyed frames over a checker, display-referred
        disp_proc = pl._cpu(ocio_config, "ACEScg", None, pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW)
        yy, xx = np.mgrid[0:th, 0:tw]
        chk = np.where(((xx // 32) + (yy // 32)) % 2, 0.18, 0.05)[..., None].astype(np.float32)
        prev = []
        for i in np.linspace(0, len(rgba) - 1, min(4, len(rgba))).astype(int):
            comp = np.ascontiguousarray(rgba[i, ..., :3] + chk * (1 - rgba[i, ..., 3:4]))
            disp_proc.applyRGB(comp)
            fr = int(round(rep["feet_row"]))
            comp[max(fr - 1, 0):fr + 1] = [1, 0, 0]                 # feet line in red
            prev.append(np.clip(comp, 0, 1))
        line = (f"{name}: {card['size_m'][0]:.2f} x {card['size_m'][1]:.2f} m card, figure {height_m} m "
                f"({rep['figure_px']:.0f} px of {th}), foot at ({foot[0]:+.2f}, {foot[1]:+.2f}, {foot[2]:+.2f}), "
                f"{len(rgba)} clip frames over {len(frames)} plate frames ({end_mode} from clip frame {clip_start}), "
                f"{mode}" + (f", exposure {exposure_stops:+.2f} stops" if exposure_stops else "")
                + (f". LICENCE: {warn}" if warn else ""))
        return {"ui": {"text": [line]},
                "result": (card, torch.from_numpy(np.stack(prev)), json.dumps({k: v for k, v in card.items()
                                                                               if k != "corners_by_frame"}, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXCharacterInsert": AtlasVFXCharacterInsert}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXCharacterInsert": "Atlas VFX Character Insert"}
