r"""Render the manually-verified 1036 SMILES for direct visual comparison.

The user hand-transcribed 1036's SMILES from the patent image
(I100297). This script renders that exact string in 4 forms:

  1. Original (patent crop)
  2. Closed macrocycle (default RDKit layout)
  3. Linearised chain (1 amide bond opened, all 10 rings visible)
  4. Canonical SMILES, linearised

Run::
    .venv\\Scripts\\python.exe -u demo_1036_manual_render.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

from cpd.render import render_mol_pil

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data" / "processed" / "cells"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_1036_manual_render.png"

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


def _render(smi, size=(520, 380), bond_w=1.0, font_scale=1.0):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
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
    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    return img, mol


def _fit(img, w, h):
    canvas = Image.new("RGB", (w, h), "white")
    img2 = img.copy()
    img2.thumbnail((w, h), Image.LANCZOS)
    canvas.paste(img2, ((w - img2.width) // 2, (h - img2.height) // 2))
    return canvas


def main():
    print("=== manual 1036 SMILES ===")
    print(f"raw        : {MANUAL_SMI}")
    print(f"len        : {len(MANUAL_SMI)}")
    mol = Chem.MolFromSmiles(MANUAL_SMI)
    if mol is None:
        print("RDKit parse: FAILED")
        return 1
    n_atoms = mol.GetNumAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    largest = max((len(r) for r in mol.GetRingInfo().AtomRings()), default=0)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    print(f"RDKit parse: OK ({n_atoms} atoms, {n_rings} rings, "
          f"largest={largest}, formula={formula})")
    can = Chem.MolToSmiles(mol)
    print(f"canonical  : {can}")
    print()

    img_orig = _load_original()
    img_closed, _ = _render(MANUAL_SMI, (520, 380), bond_w=1.2, font_scale=1.0)
    img_linear = render_mol_pil(MANUAL_SMI, size=(520, 380),
                                linearize_macrocycle=True)
    img_canon = render_mol_pil(can, size=(520, 380),
                               linearize_macrocycle=True)

    panels = [
        ("1. Original (patent crop)",
         img_orig,
         "I100297_1036_struct.png  (manually-transcribed reference)"),
        ("2. Manual SMILES, closed macrocycle",
         img_closed,
         f"{n_atoms} atoms, {n_rings} rings, largest={largest}"),
        ("3. Manual SMILES, linearised",
         img_linear,
         "1 amide C-N bond opened; all rings visible"),
        ("4. Manual SMILES, canonical, linearised",
         img_canon,
         "RDKit canonical atom ordering, same connectivity"),
    ]

    PANEL_W, PANEL_H, HEAD, PAD = 540, 460, 60, 12
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
              f"cmpd 1036: manual SMILES (271 chars) -> RDKit re-render  "
              f"({n_atoms} atoms, {n_rings} rings, largest_ring={largest})",
              fill="black", font=fontb)
    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        draw.text((x, HEAD - 22), title, fill="black", font=fontb, anchor="lt")
        if note:
            draw.text((x, HEAD - 6), note, fill="gray", font=font, anchor="lt")
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 28), (x, HEAD + 18))
    canvas.save(OUT)
    print(f"wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
