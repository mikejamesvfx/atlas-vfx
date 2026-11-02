"""Atlas VFX Relight Transfer Files: Relight Transfer (catalogue 53) that reads its sequences from disk and writes
EXRs one frame at a time, with MoGe run per frame inside the node. Same maths as the IMAGE node (light_transfer.py);
RAM stays at a few GB whatever the length, where the IMAGE graph holds ~10 GB per 4K link per 100 frames."""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from atlas_vfx.camera_side import exr_io
from atlas_vfx.camera_side import plate as pl
from atlas_vfx.contracts import shot_camera as sc

from . import chunks as ch
from . import element_colour as ec
from . import light_transfer as lt
from . import light_transfer_files as ltf

SRGB = "sRGB Encoded Rec.709 (sRGB)"


class AtlasVFXRelightTransferFiles:
    CATEGORY = "Atlas VFX/Layers"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "gain", "info")
    DESCRIPTION = ("Relight Transfer for long or 4K shots: give it the element, matte and relit sequences as paths "
                   "and it streams them, one frame at a time, writing <name>.<frame>.exr (ACEScg, alpha = the matte) "
                   "and <name>_gain (log2 gain for QC). Wire Load MoGe Model for mode sh: MoGe runs on each frame "
                   "inside the node. Its normals and depth depend on the element only, so with geometry_cache on "
                   "they are kept in <work>/geometry_cache/<key> and later runs on the same element, matte and "
                   "MoGe settings skip MoGe entirely. Same result as Relight Transfer fed by Sequence Read + MoGe + "
                   "Normals, in a few GB of RAM.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "element_path": ("STRING", {"default": "~/atlas-vfx-work/element/element.####.exr",
                                            "tooltip": "linear ACEScg sequence (EXR), full res"}),
                "relit_path": ("STRING", {"default": "~/atlas-vfx-work/DSCF3000/layers/element_relit/"
                                                     "element_relit.####.exr",
                                          "tooltip": "the relit take, same framing, any size (LTX Return output)"}),
                "matte_path": ("STRING", {"default": "", "tooltip": "matte sequence (first channel); blank = the "
                                                                    "element's alpha"}),
                "first_frame": ("INT", {"default": 1001, "min": 0, "max": 1_000_000}),
                "length": ("INT", {"default": 97, "min": 1, "max": 100_000}),
                "relit_numbering": (["frame", "sequential"], {"default": "frame",
                                                              "tooltip": "sequential: a raw SaveImage take numbered "
                                                                         "from 1"}),
                "relit_encoding": (["linear", "srgb", ec.VIEW_INVERSE], {"default": "linear", "tooltip": (
                    "how the relit take is encoded. A raw LTX PNG / video take is display-referred through the ACES "
                    "2.0 SDR view LTX Prep used: pick the inverse display view so it matches the ACEScg element. "
                    "srgb is the plain sRGB curve (no tone map, Rec.709 primaries): kept for old takes")}),
                "mode": (list(lt.MODES), {"default": "sh"}),
                "analysis_width": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 16}),
                "blur_px": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "max_stops": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 10.0, "step": 0.25}),
                "temporal": ("INT", {"default": 1, "min": 0, "max": 12}),
                "eps": ("FLOAT", {"default": 0.01, "min": 1e-5, "max": 1.0, "step": 0.001}),
                "subsample": ("INT", {"default": 2, "min": 1, "max": 16}),
                "geo_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05}),
                "min_r2": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lift_limit": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.25}),
                "name": ("STRING", {"default": "element_relit"}),
                "out_dir": ("STRING", {"default": "", "tooltip": "blank = <work>/exr_write/<name>"}),
            },
            "optional": {
                "moge_model": ("MOGE_MODEL", {"tooltip": "Load MoGe Model (moge_3_vitl): normals for mode sh and "
                                                         "depth / normals as guides"}),
                "resolution_level": ("INT", {"default": 9, "min": 0, "max": 9}),
                "refine_steps": ("INT", {"default": 3, "min": 0, "max": 16}),
                "normals_temporal": ("INT", {"default": 1, "min": 0, "max": 8}),
                "normals_sigma": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "depth_guide": ("BOOLEAN", {"default": True, "tooltip": "MoGe depth as an extra guide"}),
                "shot_camera": ("ATLAS_TRACKED_CAMERA", {"tooltip": "sh fit in world space"}),
                "relit_mirrored": ("BOOLEAN", {"default": False,
                                               "tooltip": "flip a raw mirrored take back (LTX Return already does)"}),
                "holdout_path": ("STRING", {"default": "", "tooltip": "roto sequence over what the relight "
                                                                      "invented (first channel); blank = none"}),
                "holdout_grow": ("INT", {"default": 2, "min": 0, "max": 64}),
                "ocio_config": ("STRING", {"default": "", "tooltip": "blank = the shipped config"}),
                "geometry_cache": ("BOOLEAN", {"default": True, "tooltip": (
                    "keep MoGe's normals and depth in <work>/geometry_cache, keyed on the element and matte files "
                    "and the MoGe settings, so relighting the same plate again skips MoGe (the slow part). Off = "
                    "compute them in the output folder and delete them at the end, as before")}),
            },
        }

    def run(self, element_path, relit_path, matte_path, first_frame, length, relit_numbering, relit_encoding, mode,
            analysis_width, blur_px, max_stops, temporal, eps, subsample, geo_weight, min_r2, lift_limit, name,
            out_dir, moge_model=None, resolution_level=9, refine_steps=3, normals_temporal=1, normals_sigma=1.5,
            depth_guide=True, shot_camera=None, relit_mirrored=False, holdout_path="", holdout_grow=2,
            ocio_config="", geometry_cache=True):
        t0 = time.time()
        if mode == "sh" and moge_model is None:
            raise ValueError("mode sh needs moge_model (Load MoGe Model)")
        files = lambda p, numbering="frame": ch.pick(ch.scan(pl.clean_path(p)), first_frame, length, numbering)
        element = ltf.SeqFrames(files(element_path))
        relit = ltf.SeqFrames(files(relit_path, relit_numbering), srgb=relit_encoding == "srgb", flip=relit_mirrored,
                              proc=ec.to_acescg_processor(ocio_config, ec.VIEW_INVERSE)
                              if relit_encoding == ec.VIEW_INVERSE else None)
        matte = (ltf.SeqFrames(files(matte_path), kind="mask") if matte_path.strip()
                 else ltf.SeqFrames(files(element_path), kind="alpha"))
        holdout = ltf.SeqFrames(files(holdout_path), kind="mask") if holdout_path.strip() else None

        name = pl.safe_name(name)
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / "exr_write" / name)
        out_gain = out.parent / f"{out.name}_gain"
        pl.resolve_out_dir(str(out_gain), out_gain)
        out.mkdir(parents=True, exist_ok=True)
        out_gain.mkdir(parents=True, exist_ok=True)

        normals = depth = None
        geo_dir, geo_hit = None, False
        if moge_model is not None:
            proc = pl.fast_cpu(pl.ocio_config(ocio_config).getProcessor("ACEScg", SRGB))

            def display(rgb):
                rgb = np.ascontiguousarray(rgb, np.float32).copy()
                proc.applyRGB(rgb)
                return rgb
            if geometry_cache:
                key = ltf.geometry_key(element, matte, {
                    "resolution_level": int(resolution_level), "refine_steps": int(refine_steps),
                    "temporal": int(normals_temporal), "sigma": float(normals_sigma), "display": ["ACEScg", SRGB],
                    "ocio_config": pl.clean_path(ocio_config), "model": ltf.model_tag(moge_model)})
                geo_dir = pl.resolve_out_dir("", pl.WORK_ROOT / ltf.GEOMETRY_CACHE / key)
            normals = ltf.MoGeNormals(element, moge_model, geo_dir or (out / ".normals_cache"), display=display,
                                      matte=matte,
                                      temporal=normals_temporal, sigma=normals_sigma,
                                      resolution_level=resolution_level, refine_steps=refine_steps)
            depth = normals.depth if depth_guide else None
            geo_hit = geo_dir is not None and normals.complete()
        rots = None
        if shot_camera is not None:
            cam = sc.ShotCamera(shot_camera)
            try:
                rots = np.stack([cam.cam_to_world(first_frame + t)[:3, :3] for t in range(length)]).astype(np.float32)
            except KeyError as e:
                raise ValueError(f"shot_camera: {e.args[0]} (first_frame {first_frame}, {length} frames)") from None

        def sink(t, o, g):
            n = first_frame + t
            a = np.clip(matte[t], 0.0, 1.0)
            if a.shape != o.shape[:2]:
                import cv2
                a = cv2.resize(a, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)
            exr_io.write_rgba(out / f"{name}.{n:04d}.exr", np.dstack([o, a]), colorspace="ACEScg",
                              attrs={"layer": name, "frame": n, "source": "Atlas VFX Relight Transfer Files"})
            exr_io.write_rgba(out_gain / f"{name}_gain.{n:04d}.exr", np.dstack([g, np.ones_like(a)]),
                              colorspace="Raw", attrs={"layer": f"{name}_gain", "frame": n, "unit": "log2 stops"},
                              premultiplied=False)

        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(length)
        except ImportError:  # outside ComfyUI (tests)
            pass
        try:
            _, _, rep = lt.transfer(element, relit, matte, mode=mode, analysis_width=analysis_width, blur_px=blur_px,
                                    max_stops=max_stops, temporal=temporal, eps=eps, subsample=subsample,
                                    normals=normals, depth=depth, cam_rotations=rots, geo_weight=geo_weight,
                                    min_r2=min_r2, holdout=holdout, holdout_grow=holdout_grow,
                                    lift_limit=lift_limit, sink=sink,
                                    progress=(lambda t: pbar.update(1)) if pbar else None)
        finally:
            if normals is not None and geo_dir is None:
                shutil.rmtree(normals.cache, ignore_errors=True)
        rep.update(streamed=True, frames_range=[first_frame, first_frame + length - 1],
                   relit_mirrored=bool(relit_mirrored), seconds=round(time.time() - t0, 1),
                   moge=None if normals is None else {"runs": normals.runs, "normals": normals.source,
                                                      "resolution_level": resolution_level,
                                                      "refine_steps": refine_steps, "temporal": normals_temporal,
                                                      "sigma": normals_sigma, "depth_guide": bool(depth_guide),
                                                      "seconds": round(normals.seconds, 1),
                                                      "cache": None if geo_dir is None else
                                                      {"dir": str(geo_dir), "hit": geo_hit}})
        pattern, gpattern = str(out / f"{name}.%04d.exr"), str(out_gain / f"{name}_gain.%04d.exr")
        inputs = {"element": element_path, "relit": relit_path, "matte": matte_path or "(element alpha)",
                  "holdout": holdout_path, "first_frame": first_frame, "length": length, "mode": mode,
                  "relit_encoding": relit_encoding, "ocio_config": ocio_config}
        pl.write_sidecar(out / f"{name}.sidecar.json", "AtlasVFXRelightTransferFiles", inputs=inputs, outputs=rep)
        info = json.dumps(rep)
        return {"ui": {"text": [info]}, "result": (pattern, gpattern, info)}


NODE_CLASS_MAPPINGS = {"AtlasVFXRelightTransferFiles": AtlasVFXRelightTransferFiles}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXRelightTransferFiles": "Atlas VFX Relight Transfer Files"}
