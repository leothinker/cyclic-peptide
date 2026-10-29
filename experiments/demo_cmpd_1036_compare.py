"""Side-by-side comparison for cmpd 1036: original vs DECIMER re-render.

Horizontal 4-panel layout so the 924x649 patent crop is not cropped and
the 10 fused rings stay readable in the linearised views.
"""
from __future__ import annotations

import csv
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpd.render import render_mol_pil

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data" / "processed" / "cells"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_cmpd_1036_compare.png"

CMPD = "1036"
PANEL_W = 540
PANEL_H = 460
HEAD = 60
PAD = 12


def _load_original():
    for p in CELLS.iterdir():
        if CMPD in p.stem:
            return Image.open(p).convert("RGB")
    return None


def _render(smi, size=(520, 380), bond_w=1.0, font_scale=1.0):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = bond_w
    opts.minFontSize = int(10 * font_scale)
    opts.maxFontSize = int(18 * font_scale)
    opts.padding = 0.04
    opts.addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def _fit(img, w, h):
    canvas = Image.new("RGB", (w, h), "white")
    img2 = img.copy()
    img2.thumbnail((w, h), Image.LANCZOS)
    canvas.paste(img2, ((w - img2.width) // 2, (h - img2.height) // 2))
    return canvas


def main():
    rows = list(csv.DictReader(open("data/processed/compounds_combined.csv",
                                    encoding="utf-8-sig")))
    r = next(rr for rr in rows if rr["cmpd_id"] == CMPD)
    smi_raw = r.get("smiles_decimer") or ""
    smi_can = r.get("canonical_smiles_decimer") or ""

    print("cmpd", CMPD)
    print("  largest ring :", r.get("decimer_largest_ring_size"))
    print("  num rings    :", r.get("decimer_num_rings"))
    print("  intact       :", r.get("decimer_macrocycle_intact"))

    img_orig = _load_original()
    img_closed = _render(smi_raw, (520, 380), bond_w=1.2, font_scale=1.0)
    img_linear = render_mol_pil(smi_raw, size=(520, 380), linearize_macrocycle=True)
    img_canon = render_mol_pil(smi_can, size=(520, 380), linearize_macrocycle=True)

    panels = [
        ("1. Original (patent crop)",
         img_orig,
         "PCTCN2025075389-ftappb-I100297_1036_struct.png  (full 924x649)"),
        ("2. DECIMER raw SMILES, closed macrocycle",
         img_closed,
         "22-atom ring intact; rdDepictor 2D layout, 10 fused rings"),
        ("3. DECIMER raw SMILES, linearised",
         img_linear,
         "same molecule, 1 amide C-N bond opened; all 10 rings visible"),
        ("4. DECIMER canonical SMILES, linearised",
         img_canon,
         "canonical atom ordering, same connectivity"),
    ]

    cols = len(panels)
    W = PAD + cols * (PANEL_W + PAD)
    H = HEAD + PANEL_H + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        fontb = ImageFont.truetype("arialbd.ttf", 18)
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        fontb = font = ImageFont.load_default()

    draw.text((PAD, 8),
              f"cmpd {CMPD}: original vs DECIMER re-render  "
              f"(largest_ring={r.get('decimer_largest_ring_size')}, "
              f"num_rings={r.get('decimer_num_rings')}, "
              f"intact={r.get('decimer_macrocycle_intact')})",
              fill="black", font=fontb)

    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        draw.text((x, HEAD - 22), title, fill="black", font=fontb, anchor="lt")
        if note:
            draw.text((x, HEAD - 6), note, fill="gray", font=font, anchor="lt")
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 28), (x, HEAD + 18))

    canvas.save(OUT)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    main()
