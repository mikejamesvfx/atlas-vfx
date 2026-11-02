"""Atlas VFX Sun Direction: where the sun is, in words, for the LTX-2.3 Relight IC-LoRA. Gives the prompt, the light
ball to paste (Light Ball) and whether LTX Prep should mirror the job.

Why mirror: on DSCF3000 the LoRA kept the sun on frame right whatever the ball or prompt said for a left sun
(2026-10-30: the left, front-left, back-left, strength 1.5 all tried). Mirrored, asked for the right, then flipped back
by LTX Return, the sun lands on frame left. mirror_trick auto does that for any direction with "left" in it."""
from __future__ import annotations

import json
import math

import numpy as np

DIRECTIONS = ("the front", "the front-right", "the right", "the back-right", "behind", "the back-left", "the left",
              "the front-left")
HEIGHTS = ("low", "mid", "high")
BASE = "relight the video to match the light-direction ball."


def from_light_rig(rig: dict, which: str = "middle") -> dict:
    """The LoRA's words for a real sun: a light rig (Physical Sky with the shot camera: the sun's direction in
    OpenGL camera space per frame, x right, y up, the camera looking down -z) -> direction and height as seen in the
    finished shot. A sun behind the camera lights the scene from the front; one in view is "behind" (backlight).
    The 8 sectors are 45 deg wide, clockwise from front to right. Height: the sun's elevation above the horizon
    (sun_elevation_deg; an older rig without it uses the camera-relative angle)."""
    dirs = np.asarray(rig.get("sun_dir") or [], np.float64).reshape(-1, 3)
    if not len(dirs):
        raise ValueError("light rig has no sun direction")
    i = {"first": 0, "middle": len(dirs) // 2, "last": len(dirs) - 1}.get(which)
    if i is None:
        raise ValueError(f"which {which!r}: first, middle or last")
    x, y, z = dirs[i]
    ang = math.degrees(math.atan2(x, z))                      # 0 = behind the camera (front light), +90 = right
    direction = DIRECTIONS[int(round(ang / 45.0)) % 8]
    el = rig.get("sun_elevation_deg")
    if el is None:
        el = math.degrees(math.asin(max(-1.0, min(1.0, y / max(float(np.linalg.norm(dirs[i])), 1e-12)))))
    el = float(el)
    height = "low" if el < 32.5 else "mid" if el < 57.5 else "high"
    spread = [DIRECTIONS[int(round(math.degrees(math.atan2(d[0], d[2])) / 45.0)) % 8] for d in dirs]
    return {"direction": direction, "height": height, "elevation_deg": round(el, 2),
            "azimuth_in_camera_deg": round(ang, 2), "frame_index": i, "frames": len(dirs),
            "directions_over_shot": sorted(set(spread)), "sun_visible": bool(rig.get("sun_visible", True))}


def mirrored(direction: str) -> str:
    return direction.replace("left", "\0").replace("right", "left").replace("\0", "right")


def sun(direction: str, height: str, look: str, mirror_trick: str = "auto") -> dict:
    """The ball, prompt and mirror flag for a sun from `direction` (as seen in the final shot)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}: one of {DIRECTIONS}")
    if height not in HEIGHTS:
        raise ValueError(f"height {height!r}: one of {HEIGHTS}")
    if mirror_trick not in ("auto", "off"):
        raise ValueError(f"mirror_trick {mirror_trick!r}: auto or off")
    mirror = mirror_trick == "auto" and "left" in direction
    ball = mirrored(direction) if mirror else direction
    side = ""
    if "right" in ball:
        side = " entering from frame right, long shadows falling to the left"
    elif "left" in ball:
        side = " entering from frame left, long shadows falling to the right"
    look = " ".join(str(look).split())
    prompt = f"{BASE} {look}{side} from {ball}, {height} sun"
    return {"asked": direction, "direction": ball, "height": height, "mirror": mirror, "prompt": prompt}


class AtlasVFXSunDirection:
    CATEGORY = "Atlas VFX/Artist"
    FUNCTION = "run"
    RETURN_TYPES = ("BOOLEAN", "STRING", "ATLAS_VFX_SUN")
    RETURN_NAMES = ("mirror", "prompt", "sun")
    DESCRIPTION = ("Pick where the sun is in the finished shot. prompt goes to the LTX text encode, mirror to LTX "
                   "Prep's mirror (convert the widget to an input), sun to Light Ball, which pastes the matching "
                   "light-direction ball on the reference. A sun on the left is done mirrored (the Relight LoRA only "
                   "lit from the right on our tests); LTX Return flips the take back.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (list(DIRECTIONS), {"default": "the left",
                                                 "tooltip": "where the sun is, as seen in the finished shot"}),
                "height": (list(HEIGHTS), {"default": "low", "tooltip": "low ~20 deg, mid ~45, high ~70"}),
                "look": ("STRING", {"default": "hard warm golden sunset sunlight", "multiline": True,
                                    "tooltip": "the light, in words; direction and height are added for you"}),
                "mirror_trick": (["auto", "off"], {"default": "auto",
                                                   "tooltip": "auto: a sun on the left is rendered mirrored"}),
            },
            "optional": {
                "light_rig": ("ATLAS_VFX_LIGHT_RIG", {"tooltip": (
                    "Physical Sky's rig (with the shot camera): the real sun, placed relative to the tracked camera, "
                    "replaces direction and height")}),
                "rig_frame": (["middle", "first", "last"], {"default": "middle",
                                                            "tooltip": "which frame of the rig the sun is read at"}),
            },
        }

    def run(self, direction, height, look, mirror_trick="auto", light_rig=None, rig_frame="middle"):
        rig_info = None
        if light_rig is not None:
            rig_info = from_light_rig(light_rig, rig_frame)
            direction, height = rig_info["direction"], rig_info["height"]
        s = sun(direction, height, look, mirror_trick)
        if rig_info is not None:
            s["from_light_rig"] = rig_info
        text = (f"ball: {s['direction']}, {height}" + (" (mirrored: LTX Prep mirror on)" if s["mirror"] else "")
                + f"\nprompt: {s['prompt']}")
        if rig_info is not None:
            text += (f"\nfrom the sky: sun {rig_info['azimuth_in_camera_deg']:+.0f} deg from the camera's back, "
                     f"{rig_info['elevation_deg']:.0f} deg up -> {rig_info['direction']}, {height}"
                     + ("" if len(rig_info["directions_over_shot"]) == 1 else
                        f"; over the shot it moves through {rig_info['directions_over_shot']}")
                     + ("" if rig_info["sun_visible"] else "; the sun is below the horizon"))
        return {"ui": {"text": [text]}, "result": (s["mirror"], s["prompt"], s)}


NODE_CLASS_MAPPINGS = {"AtlasVFXSunDirection": AtlasVFXSunDirection}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSunDirection": "Atlas VFX Sun Direction"}

if __name__ == "__main__":
    print(json.dumps(sun("the left", "low", "hard warm golden sunset sunlight"), indent=1))
