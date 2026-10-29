from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data" / "processed" / "cells"
RENDERED = ROOT / "data" / "processed" / "rendered"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_grid_12.png"

PICK = [
    "1001", "1005", "1010", "1014",
    "1036", "1039", "1045", "1080",
    "2001", "2020", "2035", "2042",
]

TILE = 220
HEAD = 24
ROW_LBL = 0
COLS = 6

def find_cell(cmpd_id):
    for p in CELLS.iterdir():
        if cmpd_id in p.stem:
            return p
    return None

def find_rendered(cmpd_id):
    p = RENDERED / (cmpd_id + ".png")
    return p if p.exists() else None

def fit(p, w, h):
    img = Image.open(p)
    img.thumbnail((w, h), Image.LANCZOS)
    return img

def main():
    rows = list(csv.DictReader(open("data/processed/compounds_combined.csv", encoding="utf-8-sig")))
    by_id = {r["cmpd_id"]: r for r in rows}

    n = len(PICK)
    rows_per_col = (n + COLS - 1) // COLS
    cell_w = ROW_LBL + 2 * COLS * TILE
    cell_h = HEAD + rows_per_col * TILE
    canvas = Image.new("RGB", (cell_w, cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        fontb = ImageFont.truetype("arialbd.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        fontb = font

    draw.text((ROW_LBL + TILE // 2, 4), "Original (patent)", fill="black", font=fontb, anchor="mt")
    draw.text((ROW_LBL + TILE + TILE // 2, 4), "DECIMER re-render", fill="black", font=fontb, anchor="mt")

    for i, cid in enumerate(PICK):
        col = (i // rows_per_col) * 2
        row = i % rows_per_col
        x0 = ROW_LBL + col * TILE
        y0 = HEAD + row * TILE
        record = by_id.get(cid, {})

        cell = find_cell(cid)
        if cell:
            img = fit(cell, TILE - 6, TILE - 22)
            canvas.paste(img, (x0 + (TILE - img.width) // 2, y0 + 16))
        draw.text((x0 + 4, y0 + 2), "cmpd " + cid, fill="black", font=fontb, anchor="lt")

        x1 = x0 + TILE
        ren = find_rendered(cid)
        if ren:
            img = fit(ren, TILE - 6, TILE - 22)
            canvas.paste(img, (x1 + (TILE - img.width) // 2, y0 + 16))
        intact = record.get("decimer_macrocycle_intact", "")
        if intact == "False":
            draw.text((x1 + TILE - 6, y0 + 4), "broken", fill="red", font=font, anchor="rt")
            if record.get("is_repaired") == "True":
                draw.text((x1 + TILE - 6, y0 + 22), "repaired", fill="darkgreen", font=font, anchor="rt")
        elif intact == "True":
            draw.text((x1 + TILE - 6, y0 + 4), "intact", fill="green", font=font, anchor="rt")

    canvas.save(OUT)
    print("wrote " + str(OUT) + " (" + str(cell_w) + "x" + str(cell_h) + ")")

if __name__ == "__main__":
    main()
