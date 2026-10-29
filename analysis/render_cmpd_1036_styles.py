"""Try multiple rendering styles for cmpd 1036's hand-transcribed SMILES.

The patent's structure is drawn in a ChemDraw style: thick bond lines,
angular layout, the central fluorinated aromatic ring sits inside the
macrocycle cavity, stereo wedges are visible. RDKit's default
rendering tends to produce a smooth circle that loses the visual
hierarchy, and linearizing one amide bond (the old default) flattens
the structure into a chain.

This script renders the same SMILES with several drawing options and
saves a comparison panel, so we can pick the best one for the final
report.

Run::
    .venv/Scripts/python.exe analysis/render_cmpd_1036_styles.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdCoordGen

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "data" / "processed" / "cells"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_cmpd_1036_render_styles.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MANUAL_SMI = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
    "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)


def _render(smi, size=(520, 380), bond_w=1.2,
            use_coordgen=True, padding=0.05):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        try:
            if use_coordgen:
                rdCoordGen.SetCoords(mol)
            else:
                rdDepictor.Compute2DCoords(mol)
        except Exception:
            rdDepictor.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = bond_w
    opts.padding = padding
    opts.fixedBondLength = -1  # let RDKit decide
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def _load_original():
    for p in CELLS.iterdir():
        if "1036" in p.stem:
            return Image.open(p).convert("RGB")
    return None


def _fit(img, w, h):
    canvas = Image.new("RGB", (w, h), "white")
    img2 = img.copy()
    img2.thumbnail((w, h), Image.LANCZOS)
    canvas.paste(img2, ((w - img2.width) // 2, (h - img2.height) // 2))
    return canvas


def main():
    variants = [
        ("default", dict(bond_w=1.0, padding=0.05)),
        ("chemdraw-ish", dict(bond_w=1.6, padding=0.04)),
        ("thick+small pad", dict(bond_w=2.0, padding=0.02)),
        ("rdkit-depictor",
            dict(bond_w=1.2, padding=0.05, use_coordgen=False)),
    ]

    panels = [("0. Original (patent crop)", _load_original(),
               "I100297_1036_struct.png")]
    for label, kwargs in variants:
        img = _render(MANUAL_SMI, **kwargs)
        if img is None:
            continue
        panels.append((f"{label}", img,
                       ", ".join(f"{k}={v}" for k, v in kwargs.items())))

    PANEL_W, PANEL_H, HEAD, PAD = 540, 440, 50, 12
    cols = len(panels)
    W = PAD + cols * (PANEL_W + PAD)
    H = HEAD + PANEL_H + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        fontb = ImageFont.truetype("arialbd.ttf", 18)
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        fontb = font = ImageFont.load_default()

    draw.text((PAD, 8),
              "cmpd 1036: render-style comparison (same SMILES, "
              "different RDKit options)",
              fill="black", font=fontb)
    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        draw.text((x, HEAD - 24), title, fill="black", font=fontb,
                  anchor="lt")
        if note:
            draw.text((x, HEAD - 8), note, fill="gray", font=font,
                      anchor="lt")
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 24),
                          (x, HEAD + 22))
    canvas.save(OUT)
    print(f"wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
