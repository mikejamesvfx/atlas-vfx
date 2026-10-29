"""Atlas VFX Foreground Element (catalogue 15): generated element + matte -> ACEScg RGBA texture -> ATLAS_VFX_CARD placed on the
ground at a distance from the camera at a given frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import measure as ms
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import foreground_element as fe


class AtlasVFXForegroundElement:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_VFX_CARD", "IMAGE", "STRING")
    RETURN_NAMES = ("card", "preview", "report")
    DESCRIPTION = ("Keys a generated element (display sRGB + foreground mask) into a premultiplied ACEScg RGBA "
                   "texture (OCIO 'sRGB Encoded Rec.709 (sRGB)' -> ACEScg, edge decontamination) and stands it on a "
                   "card on the ground, distance_m ahead and lateral_m right of the camera at `frame`, facing it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "frame": ("INT", {"default": 1130, "min": 0, "max": 1_000_000}),
                "distance_m": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 100.0, "step": 0.01}),
                "lateral_m": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "height_m": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 50.0, "step": 0.01}),
                "ground_y": ("STRING", {"default": "auto", "tooltip": "auto = from the solve's points near the card"}),
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5}),
                "erode_px": ("INT", {"default": 1, "min": 0, "max": 20}),
                "mask_is_background": ("BOOLEAN", {"default": False}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "name": ("STRING", {"default": "grass"}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, image, mask, shot_camera, batch_index, frame, distance_m, lateral_m, height_m, ground_y, yaw_deg,
            erode_px, mask_is_background, ocio_config, name, out_dir):
        import torch

        cam = sc.ShotCamera(shot_camera)
        if frame not in cam.frames:
            raise ValueError(f"frame {frame} not in camera {cam.name} ({cam.frames[0]}-{cam.frames[-1]})")
        img = image[min(batch_index, image.shape[0] - 1)].cpu().numpy()[..., :3]
        m = mask[min(batch_index, mask.shape[0] - 1)].cpu().numpy()
        if m.shape != img.shape[:2]:
            import cv2
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask_is_background:
            m = 1.0 - m
        rgba, rep = fe.make_texture(img, m, ocio_config, erode_px=erode_px)
        aspect = rgba.shape[1] / rgba.shape[0]

        provisional = fe.place_card(cam, frame, distance_m, lateral_m, height_m, aspect, 0.0, yaw_deg)
        if ground_y.strip().lower() == "auto":
            pts, note = ms.points_for_camera(cam)
            gy, how = fe.ground_height(pts["xyz"] if pts else None, provisional["base"][0], provisional["base"][2])
        else:
            gy, how = float(ground_y), "given"
        card = fe.place_card(cam, frame, distance_m, lateral_m, height_m, aspect, gy, yaw_deg)

        plate_path = Path(cam.data["plate"]["path"]) if cam.data["plate"]["path"] else Path("plate")
        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / plate_path.stem / "elements" / cam.name)
        tex = exr_io.write_rgba(out / f"{name}.texture.exr", rgba,
                                attrs={"element": name, "camera": cam.name, "kind": "card texture"})
        card.update({"name": name, "texture": str(tex), "camera": cam.name, "ground_from": how,
                     "texture_size": [int(rgba.shape[1]), int(rgba.shape[0])], "colorspace": "ACEScg",
                     "premultiplied": True, "matte": rep})
        (out / f"{name}.card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXForegroundElement",
                         inputs={"camera": cam.name, "batch_index": batch_index, "frame": frame,
                                 "distance_m": distance_m, "lateral_m": lateral_m, "height_m": height_m,
                                 "ground_y": ground_y, "yaw_deg": yaw_deg, "erode_px": erode_px,
                                 "image_shape": list(img.shape), "ocio_config": ocio_config},
                         outputs={"texture": str(tex), "card": card})
        # preview: element over a checker, display-referred
        h, w = rgba.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        chk = np.where(((xx // 32) + (yy // 32)) % 2, 0.18, 0.05)[..., None].astype(np.float32)
        comp = rgba[..., :3] + chk * (1 - rgba[..., 3:4])
        disp = np.ascontiguousarray(comp.copy())
        pl._cpu(ocio_config, "ACEScg", None, pl.DEFAULT_DISPLAY, pl.DEFAULT_VIEW).applyRGB(disp)
        line = (f"{name}: {card['size_m'][0]:.2f} x {card['size_m'][1]:.2f} m card at {distance_m} m / {lateral_m:+} m "
                f"from frame {frame}, ground y {gy:+.3f} ({how}), texture {card['texture_size']}")
        return {"ui": {"text": [line]},
                "result": (card, torch.from_numpy(np.clip(disp, 0, 1)[None]), json.dumps(card, indent=1))}


NODE_CLASS_MAPPINGS = {"AtlasVFXForegroundElement": AtlasVFXForegroundElement}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXForegroundElement": "Atlas VFX Foreground Element"}
