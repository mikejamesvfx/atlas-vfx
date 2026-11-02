"""Atlas VFX Nuke Builder (catalogue 24): plate + exported camera + layer EXR folders -> build_<shot>.py for the Nuke
Script Editor (Indie-safe, idempotent by node-name prefix)."""
from __future__ import annotations

import json
from pathlib import Path

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl

from . import nuke_builder as nb


class AtlasVFXNukeBuilder:
    CATEGORY = "Atlas VFX/Deliver"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_path", "script")
    DESCRIPTION = ("Writes build_<shot>.py: paste into Nuke's Script Editor. Read (plate, its input space, e.g. F-Log F-Gamut, via the "
                   "shipped OCIO config), Camera from the chan, a Read per layer folder (back to front) merged over, a "
                   "Write. Checks every layer EXR sequence (complete, RGBA half, ACEScg, plate size) first.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_ref": ("ATLAS_PLATE_SEQ",),
                "nuke_camera_json": ("STRING", {"default": "", "forceInput": True}),
                "layer_folders": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Card Project output folders, one per line, BACK to FRONT"}),
                "shot": ("STRING", {"default": "DSCF3000_S2S3"}),
                "plate_kind": (["dpx", "prores", "original"], {"default": "dpx"}),
                "set_proxy_obj": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "AVFX_"}),
                "set_root": ("BOOLEAN", {"default": True, "tooltip": "switch the root to the shipped OCIO config, "
                                                                     "frame range, fps and format"}),
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "x_offset": ("INT", {"default": 0, "min": -100000, "max": 100000,
                                     "tooltip": "shift this build's nodes in the DAG (A/B two builds side by side)"}),
                "lens_delivery": (list(nb.LENS_DELIVERY), {"default": "plate", "tooltip": (
                    "plate: each layer's plate version (redistorted) over the plate. clean + STMaps: the clean "
                    "(pinhole) layers stacked, redistorted once with the Sensor Lens Maps STMap and merged over the "
                    "untouched plate, plus an undistorted plate to line the camera up on")}),
                "sensor_lens_folder": ("STRING", {"default": "", "tooltip": "Sensor Lens Maps of the shot camera "
                                                  "(clean + STMaps only)"}),
            },
        }

    def run(self, plate_ref, nuke_camera_json, layer_folders, shot, plate_kind, set_proxy_obj, prefix, set_root,
            out_dir, x_offset=0, lens_delivery="plate", sensor_lens_folder=""):
        shot = pl.safe_name(shot, "shot")
        cam = json.loads(Path(pl.clean_path(nuke_camera_json)).read_text(encoding="utf-8"))
        if lens_delivery not in nb.LENS_DELIVERY:
            raise ValueError(f"lens_delivery {lens_delivery!r}")
        clean = lens_delivery != "plate"
        layers = [nb.find_layer(pl.clean_path(f), "clean" if clean else "plate")
                  for f in layer_folders.splitlines() if f.strip()]
        lens_maps = None
        if clean:
            lf = Path(pl.clean_path(sensor_lens_folder))
            if not sensor_lens_folder.strip() or not (lf / "sensor_lens.json").is_file():
                raise ValueError("Nuke Builder: clean + STMaps needs sensor_lens_folder (Sensor Lens Maps of the shot "
                                 "camera)")
            # G is the reference channel: lateral CA (R/B) stays in the maps folder, not in this one STMap
            lens_maps = {"undistort": lf / "undistort_g.exr", "redistort": lf / "redistort_g.exr"}
        w, h = plate_ref["width"], plate_ref["height"]
        if clean:
            # the camera lines up on the undistorted plate: same focal, aperture widened to the clean frame
            rep = json.loads((lf / "sensor_lens.json").read_text(encoding="utf-8"))
            W0, H0 = rep["plate_size"]
            Wc, Hc = rep.get("clean_size") or rep["plate_size"]
            cam = dict(cam, haperture=cam["haperture"] * Wc / W0, vaperture=cam["vaperture"] * Hc / H0,
                       win_translate=[v * W0 / Wc for v in cam["win_translate"]],
                       units=f"{cam.get('units', 'metres')}, clean frame {Wc}x{Hc}")
            for L in layers:
                if L["size"] and list(L["size"]) != [Wc, Hc]:
                    raise ValueError(f"Nuke Builder: layer {L['name']} clean frames are {L['size']}, the lens maps' "
                                     f"clean frame is {Wc}x{Hc} (made with other lens maps?)")
        checks = [exr_io.check_sequence(L["pattern"], L["first"], L["last"], *(L["size"] or (w, h)), every=24)
                  for L in layers]
        f0, f1 = plate_ref["nuke_first"], plate_ref["nuke_last"]
        plate = (nb.plate_source("", "virtual") if pl.is_virtual(plate_ref)
                 else nb.plate_source(plate_ref["path"], plate_kind, plate_ref["frame_offset"]))
        stem = pl.plate_stem(plate_ref)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / stem / "nuke")
        data = nb.build_data(
            shot=shot, plate=plate, plate_colorspace=plate_ref["colorspace"], plate_frames=(f0, f1),
            fmt=(w, h, 1.0), ocio_config=plate_ref["ocio_config"],
            camera_chan=cam["chan"], camera_settings=cam, layers=layers,
            write_pattern=str(pl.WORK_ROOT / stem / "comp" / f"{shot}_comp.%04d.exr"),
            set_proxy_obj=pl.clean_path(set_proxy_obj) or None, prefix=prefix, set_root=set_root, x_offset=x_offset,
            fps=plate_ref["fps"][0] / plate_ref["fps"][1], lens_maps=lens_maps)
        path = nb.write_script(data, out / f"build_{shot}.py")
        nk = nb.write_nk(data, out / f"build_{shot}.nk")         # Indie: paste this (Python may touch 10 nodes)
        pl.write_sidecar(out / f"build_{shot}.sidecar.json", "AtlasVFXNukeBuilder",
                         inputs={"camera": nuke_camera_json, "layers": [L["pattern"] for L in layers],
                                 "plate_kind": plate_kind, "set_root": set_root},
                         outputs={"script": str(path), "nk_paste": str(nk), "layer_checks": checks,
                                  "reads": data["reads"]})
        text = path.read_text(encoding="utf-8")
        line = (f"{path}: {len(layers)} layers ({', '.join(L['name'] for L in layers)}), plate {plate_kind}, frames "
                f"{f0}-{f1}. Nuke (full): exec(open(r'{path.as_posix()}').read()). Nuke Indie (Python may touch only "
                f"10 nodes): open {nk.as_posix()} in a text editor, copy all, Ctrl+V in the Node Graph")
        return {"ui": {"text": [line]}, "result": (str(path), text)}


NODE_CLASS_MAPPINGS = {"AtlasVFXNukeBuilder": AtlasVFXNukeBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXNukeBuilder": "Atlas VFX Nuke Builder"}
