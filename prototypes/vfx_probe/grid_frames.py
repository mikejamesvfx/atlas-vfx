"""Tracking-proxy frames with a labelled 32 px grid, for picking door corners by eye. usage: grid_frames.py f1 f2 ..."""
import os, sys
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
for f in map(int, sys.argv[1:]):
    im = Image.open(os.path.join(HERE, "DSCF3000", "track", f"plate_{f:04d}.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    for x in range(0, im.width, 32):
        d.line([(x, 0), (x, im.height)], fill=(255, 0, 0) if x % 128 == 0 else (90, 0, 0), width=1)
        if x % 128 == 0:
            d.text((x + 2, 2), str(x), fill="yellow")
    for y in range(0, im.height, 32):
        d.line([(0, y), (im.width, y)], fill=(255, 0, 0) if y % 128 == 0 else (90, 0, 0), width=1)
        if y % 128 == 0:
            d.text((2, y + 2), str(y), fill="yellow")
    d.text((im.width - 60, 2), f"f{f}", fill="cyan")
    im.save(os.path.join(HERE, f"grid_{f:04d}.png"))
    print("grid", f)
