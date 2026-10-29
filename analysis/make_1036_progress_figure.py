"""Build a single-page progress figure for cmpd 1036.

Layout::

  +-----------------+  patent_1036 (hand-transcribed reference)
  | patent original |
  +-----------------+
  RDKit default          RDKit ChemDraw-ish
  (closed macrocycle)    (closed macrocycle, thick lines)

with mass-check verdict and formula annotations below each panel.

Run::
    .venv/Scripts/python.exe analysis/make_1036_progress_figure.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdCoordGen
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "data" / "processed" / "cells"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_cmpd_1036_progress.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MANUAL_SMI = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
    "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)


def _load_original():
    for p in CELLS.iterdir():
        if "1036" in p.stem:
            return Image.open(p).convert("RGB")
    return None


def _render(smi, size=(640, 460), bond_w=1.2, padding=0.05):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        try:
            rdCoordGen.SetCoords(mol)
        except Exception:
            pass
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = bond_w
    opts.padding = padding
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def _fit(img, w, h):
    canvas = Image.new("RGB", (w, h), "white")
    img2 = img.copy()
    img2.thumbnail((w, h), Image.LANCZOS)
    canvas.paste(img2, ((w - img2.width) // 2, (h - img2.height) // 2))
    return canvas


def main() -> None:
    mol = Chem.MolFromSmiles(MANUAL_SMI)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
    rings = mol.GetRingInfo().NumRings()
    largest = max((len(r) for r in mol.GetRingInfo().AtomRings()), default=0)

    panels = [
        (("Original patent crop", "I100297"), _load_original(),
         "hand-transcribed reference"),
        (("RDKit default", "closed macrocycle"),
         _render(MANUAL_SMI, bond_w=1.0, padding=0.05),
         "bondLineWidth=1.0"),
        (("RDKit ChemDraw-ish", "closed macrocycle"),
         _render(MANUAL_SMI, bond_w=1.7, padding=0.04),
         "bondLineWidth=1.7"),
    ]

    PANEL_W, PANEL_H = 540, 460
    HEAD, PAD = 80, 16
    cols = len(panels)
    W = PAD + cols * (PANEL_W + PAD)
    H = HEAD + PANEL_H + PAD + 80  # annotation strip
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        fontb = ImageFont.truetype("arialbd.ttf", 22)
        fontm = ImageFont.truetype("arial.ttf", 16)
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        fontb = fontm = font = ImageFont.load_default()

    draw.text((PAD, 8),
              f"cmpd 1036  -  hand transcription vs RDKit render    "
              f"formula={formula}  mass={exact}  "
              f"rings={rings}  largest={largest}",
              fill="black", font=fontb)
    draw.text((PAD, 36),
              "Mass gate: target 1609.84..1609.94 (patent [M+H]+=1610.9) "
              "- transcription is 0.20 Da low; DECIMER is 66 Da off.",
              fill="dimgray", font=fontm)

    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        # titles above panel (each line on its own)
        line1, line2 = title
        draw.text((x, HEAD - 50), line1, fill="black", font=fontb,
                  anchor="lt")
        draw.text((x, HEAD - 26), line2, fill="dimgray", font=font,
                  anchor="lt")
        if note:
            draw.text((x, HEAD - 6), note, fill="gray", font=font,
                      anchor="lt")
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 4), (x, HEAD + 18))

    canvas.save(OUT)
    print(f"wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
