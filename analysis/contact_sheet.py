"""Contact sheet of registration crops + verification frames with candidate boxes."""
import json, sys, os
from PIL import Image, ImageDraw, ImageFont

root, out = sys.argv[1], sys.argv[2]
recs = json.load(open(os.path.join(root, "records.json")))
events = [json.loads(l) for l in open(os.path.join(root, "events.jsonl"))]
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)

T = 220
tiles = []
for r in recs:
    im = Image.open(os.path.join(root, r["cropFile"])).convert("RGB")
    im.thumbnail((T, T))
    tile = Image.new("RGB", (T, T + 50), "white")
    tile.paste(im, ((T - im.width) // 2, 0))
    d = ImageDraw.Draw(tile)
    d.text((4, T + 2), f'{r["id"]} "{r["name"]}"', fill="black", font=font)
    d.text((4, T + 26), f'{r["hitType"]} {r["captureDistance"]:.2f}m', fill="gray", font=font)
    tiles.append(tile)

cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet1 = Image.new("RGB", (cols * T, rows * (T + 50)), "white")
for i, t in enumerate(tiles):
    sheet1.paste(t, ((i % cols) * T, (i // cols) * (T + 50)))

frames = []
for e in events:
    if e["type"] != "verify":
        continue
    im = Image.open(os.path.join(root, e["frameFile"])).convert("RGB")
    H = 1080  # sensor height; rotation 90: upright (x,y) = (H - sy, sx)
    d = ImageDraw.Draw(im)
    for s in e["scores"]:
        l, t, r_, b = s["rect"]
        box = [H - b, l, H - t, r_]
        color = "red" if s["kind"] == "projected" else ("lime" if s["sim"] == max(x["sim"] for x in e["scores"]) else "yellow")
        d.rectangle(box, outline=color, width=6)
        d.text((box[0] + 8, box[1] + 8), f'{s["kind"][:4]} {s["sim"]:.2f}', fill=color, font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40))
    px, py = e["projectedUpright"]
    d.ellipse([px - 15, py - 15, px + 15, py + 15], outline="cyan", width=6)
    im.thumbnail((540, 960))
    frames.append((e, im))

W = max(sheet1.width, 540 * len(frames))
Hs = sheet1.height + 970
sheet = Image.new("RGB", (W, Hs), "white")
sheet.paste(sheet1, (0, 0))
for i, (e, im) in enumerate(frames):
    sheet.paste(im, (i * 540, sheet1.height + 10))
sheet.save(out)
print(out, sheet.size)
