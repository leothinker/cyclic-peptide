"""Render cmpd 1036 with multiple RDKit options to find ChemDraw-like style."""
from __future__ import annotations
import sys
from io import BytesIO
from pathlib import Path
sys.path.insert(0, "src")
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdCoordGen, rdDepictor
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "data" / "processed" / "cells"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "demo_cmpd_1036_render_sweep.png"
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
def _render(smi, *, coordgen, padding, bond_length, fixed_bond, size=(600, 440)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        if coordgen == "coordgen":
            try:
                rdCoordGen.SetCoords(mol)
            except Exception:
                rdDepictor.Compute2DCoords(mol)
        else:
            rdDepictor.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.padding = padding
    if fixed_bond > 0:
        opts.fixedBondLength = fixed_bond
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
    variants = [
        ("CoordGen / default", dict(coordgen="coordgen", padding=0.05, bond_length=1.0, fixed_bond=-1)),
        ("CoordGen / small pad", dict(coordgen="coordgen", padding=0.02, bond_length=1.0, fixed_bond=-1)),
        ("rdDepictor (old)", dict(coordgen="old", padding=0.05, bond_length=1.0, fixed_bond=-1)),
        ("CoordGen / fixed 28", dict(coordgen="coordgen", padding=0.03, bond_length=1.0, fixed_bond=28)),
        ("CoordGen / fixed 35", dict(coordgen="coordgen", padding=0.03, bond_length=1.0, fixed_bond=35)),
        ("CoordGen / fixed 45", dict(coordgen="coordgen", padding=0.03, bond_length=1.0, fixed_bond=45)),
    ]
    panels = [("Original", _load_original(), "patent crop")]
    for label, kwargs in variants:
        img = _render(MANUAL_SMI, **kwargs)
        if img is None:
            continue
        panels.append((label, img,
                       ", ".join(f"{k}={v}" for k, v in kwargs.items())))
    PANEL_W, PANEL_H, HEAD, PAD = 540, 480, 60, 12
    cols = len(panels)
    W = PAD + cols * (PANEL_W + PAD)
    H = HEAD + PANEL_H + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        fontb = ImageFont.truetype("arialbd.ttf", 16)
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        fontb = font = ImageFont.load_default()
    draw.text((PAD, 8),
              "cmpd 1036: render-option sweep (algorithm, padding, fixed bond length)",
              fill="black", font=fontb)
    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        draw.text((x, HEAD - 30), title, fill="black", font=fontb, anchor="lt")
        if note:
            draw.text((x, HEAD - 12), note, fill="gray", font=font, anchor="lt")
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 28), (x, HEAD + 18))
    canvas.save(OUT)
    print(f"wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")
if __name__ == "__main__":
    main()
