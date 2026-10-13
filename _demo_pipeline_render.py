"""Build a report-ready side-by-side grid: original patent structure on
the left, DECIMER canonical SMILES re-rendered with linearize_macrocycle
on the right. OCR entropy stats shown in the title.

Usage:  .venv\Scripts\python.exe -u _demo_pipeline_render.py
Output: demo_pipeline_render/grid_<N>.png + grid_<N>_manifest.txt
"""
import json, sys
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
out_dir = ROOT / "data" / "processed"
img_dir = ROOT / "data" / "raw" / "WO2025162428" / "golden_tables"
demo_dir = ROOT / "demo_pipeline_render"
demo_dir.mkdir(exist_ok=True)

# Load the existing combined CSV (or, if not present, just use the cache)
csv_path = out_dir / "compounds_combined.csv"
if csv_path.exists():
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by_id = {r["cmpd_id"]: r for r in rows}
    print(f"loaded {len(rows)} rows from {csv_path.name}")
else:
    cache = json.load(open(out_dir / "decimer_cache.json", encoding="utf-8"))["entries"]
    by_id = {cid: {"cmpd_id": cid, "smiles": "", "smiles_decimer": e.get("raw",""),
                   "decimer_min_conf": e.get("min_conf","")} for cid, e in cache.items()}
    print(f"no CSV; built {len(by_id)} rows from cache")

# Pick 12 cmpds that have both an original structure crop AND a DECIMER render.
cells_dir = out_dir / "cells"
structs = {p.stem.split("_")[-2]: p for p in cells_dir.glob("*_struct.png")}
print(f"available structures: {len(structs)}")

rendered_dir = out_dir / "rendered"
renders = {p.stem: p for p in rendered_dir.glob("*.png")}
print(f"available renders: {len(renders)}")

both = sorted(set(structs) & set(renders))
both = both[:12]  # 12-cmpd grid
print(f"showing {len(both)} cmpds with both crops")

# Layout: 4 cols x 3 rows, each cell 540x440 (left = orig, right = render)
COLS, ROWS = 4, 3
CELL_W, CELL_H = 540, 440
PAD = 8
LBL_H = 24
W = COLS * (CELL_W + PAD) + PAD
H = ROWS * (CELL_H + LBL_H + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except OSError:
    font = ImageFont.load_default()

manifest = []
for i, cid in enumerate(both):
    r, c = divmod(i, COLS)
    x = PAD + c * (CELL_W + PAD)
    y = PAD + r * (CELL_H + LBL_H + PAD)

    orig = Image.open(structs[cid]).convert("RGB").resize((CELL_W // 2 - 4, CELL_H - LBL_H - 4))
    rend = Image.open(renders[cid]).convert("RGB").resize((CELL_W // 2 - 4, CELL_H - LBL_H - 4))
    cell = Image.new("RGB", (CELL_W, CELL_H - LBL_H), "white")
    cell.paste(orig, (2, 2))
    cell.paste(rend, (CELL_W // 2 + 2, 2))
    canvas.paste(cell, (x, y + LBL_H))

    info = by_id.get(cid, {})
    src = info.get("source_image", "")
    dec_conf = info.get("decimer_min_conf", "")
    ocr_ent = info.get("ocr_avg_entropy", "")
    lbl = f"{cid}  orig | DECIMER  (min_conf={dec_conf or 'n/a'}, ocr_H={ocr_ent or 'n/a'})"
    draw.text((x, y + 2), lbl, fill="black", font=font)
    manifest.append(f"  {cid}  source={src}  decimer_min_conf={dec_conf}  ocr_avg_entropy={ocr_ent}")

# Footer
draw.text((PAD, H - 18), "left: patent structure crop | right: DECIMER canonical SMILES re-rendered with linearize_macrocycle=True",
          fill="#666", font=font)

out_png = demo_dir / "grid_12.png"
canvas.save(out_png)
(demo_dir / "grid_12_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
print(f"\n-> {out_png}")
print(f"-> {demo_dir / 'grid_12_manifest.txt'}")
print("done")
