"""Atlas VFX Measure & Scale (catalogue 7): picks JSON -> metres, up, ground -> scaled, gravity-aligned shot camera."""
from __future__ import annotations

import json
from pathlib import Path

from atlas_vfx.contracts import shot_camera as sc

from . import measure as ms
from . import plate as pl


class AtlasVFXMeasureScale:
    CATEGORY = "Atlas VFX/Camera"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ATLAS_TRACKED_CAMERA", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shot_camera", "path", "report", "summary")
    DESCRIPTION = ("Known lengths picked in the plate (docs/PICKING.md) -> metres per unit (refused if the "
                   "measurements disagree by more than max_spread_pct unless forced), up from vertical picks, "
                   "ground from solve points in a region or picked points. Output: metres, Y up, ground y = 0.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_camera": ("ATLAS_TRACKED_CAMERA",),
                "picks_path": ("STRING", {"default": ""}),
                "solve_points": ("STRING", {"default": "", "tooltip": "COLMAP folder for plane picks; empty = the "
                                                                      "camera's own solve if it is one"}),
                "max_spread_pct": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "force": ("BOOLEAN", {"default": False}),
                "name": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, picks_path, **kw):
        """Re-run when the picks file changes, not only its path."""
        import hashlib

        try:
            return hashlib.sha256(Path(pl.clean_path(picks_path)).read_bytes()).hexdigest()
        except OSError:
            return float("nan")

    def run(self, shot_camera, picks_path, solve_points, max_spread_pct, force, name, out_dir):
        cam = sc.ShotCamera(shot_camera)
        picks_file = Path(pl.clean_path(picks_path))
        picks = ms.load_picks(picks_file, cam.data["plate"]["height"])
        pts, pts_note = ms.points_for_camera(cam, solve_points)
        r = ms.measure(cam, picks, pts, max_spread_pct=max_spread_pct, force=force)
        name = pl.safe_name(name.strip() or f"{cam.name}_m")
        new = ms.apply(cam, r, name=name)
        plate_path = Path(cam.data["plate"]["path"]) if cam.data["plate"]["path"] else Path("plate")
        out = pl.resolve_out_dir(out_dir, pl.WORK_ROOT / plate_path.stem / "camera")
        path = sc.save(new, out / f"{name}.shot_camera.json")
        r["solve_points"] = pts_note
        (out / f"{name}.measure.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        pl.write_sidecar(out / f"{name}.shot_camera.sidecar.json", "AtlasVFXMeasureScale",
                         inputs={"camera": cam.name, "picks_path": str(picks_file),
                                 "picks": json.loads(picks_file.read_text(encoding="utf-8")),
                                 "solve_points": pts_note, "max_spread_pct": max_spread_pct, "force": force},
                         outputs={"path": str(path), "measure_report": str(out / f"{name}.measure.json")})
        lines = [f"{name}: {r['m_per_unit']:.5f} m/unit, spread {r['spread_pct']:.2f} %"
                 + (" (FORCED)" if r["forced"] else "")]
        lines += [f"  {m['label']}: {m['measured_units']:.4f} u -> {m['length_after_m']:.3f} m (given {m['length_m']})"
                  for m in r["measurements"]]
        lines.append(f"  up: {r['up_method']}" + (f", verticals within {r['up_spread_deg']:.2f} deg"
                                                   if r["up_spread_deg"] is not None else ""))
        lines.append(f"  ground: {r['ground_method']}" + (f", normal vs up {r['ground_normal_vs_up_deg']:.2f} deg"
                                                           if r["ground_normal_vs_up_deg"] is not None else ""))
        lines.append("  camera height: " + ", ".join(f"{f} {h:.2f} m" for f, h in r["camera_height_m"].items()))
        return (new.data, str(path), json.dumps(r, indent=1), "\n".join(lines))


NODE_CLASS_MAPPINGS = {"AtlasVFXMeasureScale": AtlasVFXMeasureScale}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXMeasureScale": "Atlas VFX Measure & Scale"}
