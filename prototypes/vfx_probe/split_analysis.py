"""Suggest where to split the plate: exposure (linear ACEScg luminance) and frame-to-frame change, smoothed.
Writes a luminance strip and a contact sheet with frame numbers."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

d = json.load(open("DSCF3000/plate_stats.json"))
f = d["frames"]
lum = np.array([r["lin_mean"] for r in f])
chg = np.array([r["change"] for r in f])
k = np.ones(13) / 13
lum_s = np.convolve(lum, k, "same")
stops = np.log2(np.maximum(lum_s, 1e-6) / np.median(lum_s[:72]))
print("exposure relative to the lawn opening, in stops, every 24 frames (1 s):")
for i in range(0, len(f), 24):
    print(f"  f{i:3d} {i/24:5.1f}s  {stops[i]:+.2f} stops   change {np.mean(chg[max(0,i-6):i+6]):5.1f}")
# biggest exposure falls (doorway / interior)
dl = np.gradient(stops)
for i in np.argsort(dl)[:6]:
    print(f"steep fall at f{i} ({i/24:.1f}s): {dl[i]*24:+.2f} stops/s")
frames = [0, 48, 96, 132, 156, 180, 204, 228, 252, 300, 360, 420, 480, 527]
tiles = []
for i in frames:
    im = Image.open(f"DSCF3000/proxy/plate_{i:04d}.png").resize((384, 202))
    ImageDraw.Draw(im).text((6, 4), f"f{i}  {i/24:.1f}s  {stops[i]:+.1f}st", fill=(255, 255, 0))
    tiles.append(im)
cols = 5
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 384, rows * 202))
for j, t in enumerate(tiles):
    sheet.paste(t, ((j % cols) * 384, (j // cols) * 202))
sheet.save("DSCF3000_split_sheet.jpg", quality=85)
print("sheet: DSCF3000_split_sheet.jpg")
