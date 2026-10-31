"""Atlas VFX Sun Direction: where the sun is, in words, for the LTX-2.3 Relight IC-LoRA. Gives the prompt, the light
ball to paste (Light Ball) and whether LTX Prep should mirror the job.

Why mirror: on DSCF3000 the LoRA kept the sun on frame right whatever the ball or prompt said for a left sun
(2026-10-30: the left, front-left, back-left, strength 1.5 all tried). Mirrored, asked for the right, then flipped back
by LTX Return, the sun lands on frame left. mirror_trick auto does that for any direction with "left" in it."""
from __future__ import annotations

import json

DIRECTIONS = ("the front", "the front-right", "the right", "the back-right", "behind", "the back-left", "the left",
              "the front-left")
HEIGHTS = ("low", "mid", "high")
BASE = "relight the video to match the light-direction ball."


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
            }
        }

    def run(self, direction, height, look, mirror_trick="auto"):
        s = sun(direction, height, look, mirror_trick)
        text = (f"ball: {s['direction']}, {height}" + (" (mirrored: LTX Prep mirror on)" if s["mirror"] else "")
                + f"\nprompt: {s['prompt']}")
        return {"ui": {"text": [text]}, "result": (s["mirror"], s["prompt"], s)}


NODE_CLASS_MAPPINGS = {"AtlasVFXSunDirection": AtlasVFXSunDirection}
NODE_DISPLAY_NAME_MAPPINGS = {"AtlasVFXSunDirection": "Atlas VFX Sun Direction"}

if __name__ == "__main__":
    print(json.dumps(sun("the left", "low", "hard warm golden sunset sunlight"), indent=1))
