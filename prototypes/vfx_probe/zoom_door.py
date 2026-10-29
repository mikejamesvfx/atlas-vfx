"""3x zoom of a region of tracking-proxy frames with a 4 px grid (labels in proxy pixels). usage: zoom_door.py x0 y0 x1 y1 f..."""
import os, sys
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
x0, y0, x1, y1 = map(int, sys.argv[1:5])
Z = 3
tiles = []
for f in map(int, sys.argv[5:]):
    im = Image.open(os.path.join(HERE, "DSCF3000", "track", f"plate_{f:04d}.png")).convert("RGB").crop((x0, y0, x1, y1))
    im = im.resize((im.width * Z, im.height * Z), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for x in range(x0 - x0 % 8, x1, 8):
        X = (x - x0) * Z
        d.line([(X, 0), (X, im.height)], fill=(255, 0, 0) if x % 32 == 0 else (110, 0, 0))
        if x % 32 == 0:
            d.text((X + 2, 2), str(x), fill="yellow")
    for y in range(y0 - y0 % 8, y1, 8):
        Y = (y - y0) * Z
        d.line([(0, Y), (im.width, Y)], fill=(255, 0, 0) if y % 32 == 0 else (110, 0, 0))
        if y % 32 == 0:
            d.text((2, Y + 2), str(y), fill="yellow")
    d.text((im.width - 40, im.height - 14), f"f{f}", fill="cyan")
    tiles.append(im)
W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
out = Image.new("RGB", (W, tiles[0].height), "black")
x = 0
for t in tiles:
    out.paste(t, (x, 0)); x += t.width + 6
name = f"zoom_{'_'.join(sys.argv[5:])}.png"
out.save(os.path.join(HERE, name)); print(name, out.size)
