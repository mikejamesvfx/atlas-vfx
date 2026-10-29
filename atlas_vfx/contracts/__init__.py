"""Data contracts shared between atlas-vfx, Atlas Camera and twentyfive.

shot-camera@0.1: schema in shot_camera_v0_1.schema.json, load/validate/save + helpers in shot_camera.py.
"""
from pathlib import Path

SHOT_CAMERA_SCHEMA = Path(__file__).with_name("shot_camera_v0_1.schema.json")
