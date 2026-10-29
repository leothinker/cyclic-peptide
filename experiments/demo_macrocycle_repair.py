from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data" / "processed" / "cells"
RENDERED = ROOT / "data" / "processed" / "rendered"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_macrocycle_repair.png"

PICK = [
    ("1001", True),
    ("1036", False),
    ("1010", False),
    ("1039", False),
]

TILE = 320
PAD = 12
HEAD = 36
COLS = 3

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
    img.thumbnail((w - 2 * PAD, h - 2 * PAD), Image.LANCZOS)
    return img

def main():
    rows = list(csv.DictReader(open("data/processed/compounds_combined.csv", encoding="utf-8-sig")))
    by_id = {r["cmpd_id"]: r for r in rows}

    cell_w = COLS * TILE
    cell_h = HEAD + len(PICK) * (TILE + HEAD)
    canvas = Image.new("RGB", (cell_w, cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        fontb = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        fontb = font

    for ci, label in enumerate(["Original (patent)", "DECIMER re-render", "After macrocycle_repair"]):
        x = ci * TILE + TILE // 2
        draw.text((x, 6), label, fill="black", font=fontb, anchor="mt")

    for ri, (cid, _) in enumerate(PICK):
        row_y = HEAD + ri * (TILE + HEAD)
        draw.text((PAD, row_y + TILE // 2), "cmpd " + cid, fill="black", font=fontb, anchor="lm")
        record = by_id.get(cid, {})

        # col 0: original cell
        cell = find_cell(cid)
        col0_x = TILE
        if cell:
            img = fit(cell, TILE, TILE)
            canvas.paste(img, (col0_x + (TILE - img.width) // 2, row_y + (TILE - img.height) // 2))
        else:
            draw.text((col0_x + TILE // 2, row_y + TILE // 2), "no cell", fill="red", font=font, anchor="mm")

        # col 1: DECIMER render
        ren = find_rendered(cid)
        col1_x = 2 * TILE
        intact = record.get("decimer_macrocycle_intact", "")
        if ren:
            img = fit(ren, TILE, TILE)
            canvas.paste(img, (col1_x + (TILE - img.width) // 2, row_y + (TILE - img.height) // 2))
        else:
            draw.text((col1_x + TILE // 2, row_y + TILE // 2), "no render", fill="red", font=font, anchor="mm")
        if intact == "False":
            draw.text((col1_x + TILE - 8, row_y + 8), "broken", fill="red", font=fontb, anchor="rt")
        elif intact == "True":
            draw.text((col1_x + TILE - 8, row_y + 8), "intact", fill="green", font=fontb, anchor="rt")

        # col 2: repaired (if any)
        col2_x = 2 * TILE
        if record.get("is_repaired") == "True":
            draw.text((col2_x + TILE // 2, row_y + TILE // 2),
                      "(repaired_smiles written\nbut PNG re-render\nnot regenerated\nin this quick demo)",
                      fill="darkgreen", font=font, anchor="mm", align="center")
        else:
            draw.text((col2_x + TILE // 2, row_y + TILE // 2), "not repaired", fill="gray", font=font, anchor="mm")

    canvas.save(OUT)
    print("wrote " + str(OUT) + " (" + str(cell_w) + "x" + str(cell_h) + ")")

if __name__ == "__main__":
    main()
