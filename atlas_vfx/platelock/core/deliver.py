"""Write a lock to disk for Nuke.

Layout under <out>/:
  bg/bg.####.exr              re-shot background RGBA (A = 1 real, 0.5 filled, 0 none)
  bg/bg_stmap.####.exr        STMap into the AI frame of the same number (A = valid)
  <person>/<person>.####.exr  warped performer RGBA
  <person>/<person>_stmap.####.exr  STMap into the AI frame named in its header
                              (platelock_source_frame) and in shot.json
  comp/comp.####.exr          the restack, for review
  shot.json                   retime curves, gauge, camera delta, QC, conventions
  platelock_lock.nk           Nuke script: AI read -> TimeWarp per performer ->
                              STMap -> Merge over the STMapped background

Comp keeps control: every correction is a map on the original AI frames,
so it can be dialled back, masked or re-filtered in Nuke.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ...camera_side import nuke_text as nt
from ...camera_side.plate import safe_name
from . import exr, warp
from .pipeline import LockResult


def check_people_names(names) -> list[str]:
    """Performer names become folder names and Nuke node names: one path component each, or a ValueError before
    anything is written."""
    return [safe_name(n, "performer name") for n in names]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write_lock(result: LockResult, out_dir: str, *, first_frame: int = 1001, ai_first_frame: int = 1001,
               ai_path: str = "ai/ai.####.exr", ai_size: tuple | None = None,
               write_comp: bool = True, extra: dict | None = None) -> dict:
    check_people_names(result.people)
    os.makedirs(out_dir, exist_ok=True)
    N, H, W = result.comp.shape[:3]
    aw, ah = ai_size or (W, H)
    frames = result.frames or list(range(N))
    paths = {"bg": os.path.join(out_dir, "bg"), "comp": os.path.join(out_dir, "comp")}
    os.makedirs(paths["bg"], exist_ok=True)
    if write_comp:
        os.makedirs(paths["comp"], exist_ok=True)

    for i in frames:
        fn = first_frame + i
        exr.write_rgba(f"{paths['bg']}/bg.{fn:04d}.exr", result.bg[i], result.bg_valid[i],
                       {"platelock_source_frame": ("int", ai_first_frame + i)})
        mx, my = result.bg_maps[i]
        valid = ((mx > -1e3) & (result.bg_valid[i] >= 1.0)).astype(np.float32)
        exr.write_stmap(f"{paths['bg']}/bg_stmap.{fn:04d}.exr", warp.to_stmap((mx, my), aw, ah), valid,
                        {"platelock_source_frame": ("int", ai_first_frame + i)})
        if write_comp:
            exr.write_rgba(f"{paths['comp']}/comp.{fn:04d}.exr", result.comp[i])

    people = {}
    for name, pr in result.people.items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        for i in frames:
            fn = first_frame + i
            src = ai_first_frame + int(pr.ai_frames[i])
            attrs = {"platelock_source_frame": ("int", src), "platelock_tau": ("float", float(pr.retime.tau[i]))}
            exr.write_rgba(f"{d}/{name}.{fn:04d}.exr", pr.rgb[i], pr.alpha[i], attrs)
            exr.write_stmap(f"{d}/{name}_stmap.{fn:04d}.exr", warp.to_stmap(pr.maps[i], aw, ah), None, attrs)
        people[name] = {
            "source_frames": [ai_first_frame + int(j) for j in pr.ai_frames],
            "tau": [ai_first_frame + float(t) for t in pr.retime.tau],
            "retime": pr.retime.offset_stats(),
        }

    g = result.gauge
    shot = {
        "schema": "platelock.shot/1",
        "first_frame": first_frame, "last_frame": first_frame + N - 1,
        "ai_first_frame": ai_first_frame, "ai_path": ai_path,
        "conventions": {
            "camera": "OpenCV world->camera, pixel centres on integers, origin top-left",
            "stmap": "Nuke STMap: s=(x+0.5)/W, t=1-(y+0.5)/H in the AI frame; A=valid",
            "bg_valid": "1 real AI pixels, 0.5 push-pull filled, 0 nothing",
        },
        "gauge_ai_to_plate": {"scale": g.s, "R": g.R.tolist(), "t": g.t.tolist()},
        "camera_delta": result.camera_delta,
        "people": people,
        "qc_before": result.report_before,
        "qc_after": result.report_after,
        **(extra or {}),
    }
    with open(os.path.join(out_dir, "shot.json"), "w") as f:
        json.dump(_jsonable(shot), f, indent=1)
    with open(os.path.join(out_dir, "platelock_lock.nk"), "w") as f:
        f.write(nuke_script(shot, out_dir))
    return shot


def _curve(frames, values) -> str:
    # K = constant (step) interpolation: each plate frame reads exactly one AI frame
    return "{curve K " + " ".join(f"x{f} {v}" for f, v in zip(frames, values)) + "}"


def nuke_script(shot: dict, out_dir: str) -> str:
    """A small .nk that only uses single-input chains, so it opens cleanly.

    Per performer: Read_AI -> TimeWarp (step lookup = AI frame per plate frame),
    plus Reads of the delivered STMap and baked layer. Hook TimeWarp into an
    STMap node's src and the STMap Read into its stmap input (uv = rgb) to get
    the live version, then Merge over the background.
    """
    f0, f1 = shot["first_frame"], shot["last_frame"]
    od = out_dir.replace("\\", "/")
    ai = shot["ai_path"].replace("\\", "/")
    out = ["# platelock generated script. Check paths on first open.",
           f"Root {{\n inputs 0\n first_frame {f0}\n last_frame {f1}\n}}"]

    def read(path, name, first, last, x, y, label=""):
        lab = f"\n label {nt.nk_str(label)}" if label else ""
        return (f"Read {{\n inputs 0\n file {nt.nk_str(path)}\n first {int(first)}\n last {int(last)}\n on_error black{lab}"
                f"\n name {nt.nk_name(name)}\n xpos {x}\n ypos {y}\n}}")

    out.append(read(f"{od}/bg/bg.####.exr", "Read_bg", f0, f1, 0, 0, "re-shot background (A: 1 real, 0.5 filled)"))
    out.append(read(f"{od}/bg/bg_stmap.####.exr", "Read_bg_stmap", f0, f1, 150, 0, "STMap into AI frame of same number"))
    out.append(read(f"{od}/comp/comp.####.exr", "Read_comp", f0, f1, 300, 0, "platelock review comp"))
    frames = list(range(f0, f1 + 1))
    for k, (name, p) in enumerate(shot["people"].items()):
        x = 500 + 400 * k
        out.append(read(ai, f"Read_AI_{name}", shot["ai_first_frame"], 99999, x, 0, f"AI take for {name}"))
        out.append(f"TimeWarp {{\n lookup {_curve(frames, [int(v) for v in p['source_frames']])}\n"
                   f" label {nt.nk_str(f'{name}: AI frame per plate frame')}\n name {nt.nk_name(f'TimeWarp_{name}')}\n"
                   f" xpos {x}\n ypos 80\n}}")
        out.append(read(f"{od}/{name}/{name}_stmap.####.exr", f"Read_{name}_stmap", f0, f1, x + 150, 0,
                        f"STMap for {name} (wire to STMap.stmap, uv=rgb)"))
        out.append(read(f"{od}/{name}/{name}.####.exr", f"Read_{name}", f0, f1, x + 300, 0, f"baked {name} RGBA"))
    return "\n".join(out) + "\n"
