'''cmpd 1036 review deck for the report: original vs every SMILES candidate.

Produces a horizontally-stacked panel:
  1. Original patent crop (ground truth)
  2. Reference SMILES as RDKit draws it
  3. DECIMER raw SMILES as RDKit draws it
  4. DECIMER canonical SMILES as RDKit draws it
'''
from __future__ import annotations

import csv
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdCoordGen
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '"data" / "audit" / "demos" / "figures" / "demo_cmpd_1036_deck.png'
CELLS = ROOT / 'data' / 'processed' / 'cells'
SMILES_PATH = ROOT / 'data' / 'processed' / 'compounds_combined.csv'

REF_SMI = (
    'CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)'
    'C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)'
    'C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)'
    '[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)'
    'C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C'
)
CMPD = '1036'


def _load_original():
    for p in CELLS.iterdir():
        if CMPD in p.stem:
            return Image.open(p).convert('RGB')
    return None


def _render_rdkit(smi, size=(540, 400)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rdCoordGen.AddCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 1.4
    opts.padding = 0.04
    opts.addStereoAnnotation = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert('RGB')


def _fit(img, w, h):
    canvas = Image.new('RGB', (w, h), 'white')
    img2 = img.copy()
    img2.thumbnail((w, h), Image.LANCZOS)
    canvas.paste(img2, ((w - img2.width) // 2, (h - img2.height) // 2))
    return canvas


def _decimer_smiles():
    with SMILES_PATH.open(encoding='utf-8-sig') as f:
        row = next((r for r in csv.DictReader(f) if r['cmpd_id'] == CMPD), None)
    if row is None:
        return '', ''
    return row.get('smiles_decimer') or '', row.get('canonical_smiles_decimer') or ''


def _reference_properties():
    mol = Chem.MolFromSmiles(REF_SMI)
    ri = mol.GetRingInfo()
    sizes = sorted((len(r) for r in ri.AtomRings()), reverse=True)
    return {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'exact_mass': round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        'rings': len(sizes),
        'ring_sizes': sizes,
        'largest_ring': sizes[0] if sizes else 0,
        'n_atoms': mol.GetNumAtoms(),
        'n_heavy_atoms': mol.GetNumHeavyAtoms(),
    }


def main():
    properties = _reference_properties()
    decimer_raw, decimer_can = _decimer_smiles()

    panels = []
    panels.append((
        '1. Patent original',
        _load_original(),
        'PCTCN2025075389-ftappb-I100297_1036_struct.png -- 33-atom macrocyclic peptide + bridged fluorinated aromatic core',
    ))
    panels.append((
        '2. Manual SMILES (RDKit)',
        _render_rdkit(REF_SMI),
        f'formula = {properties["formula"]}; exact mass = {properties["exact_mass"]} Da; [M+H]+ = {round(properties["exact_mass"] + 1.0073, 4)}',
    ))
    decimer_img = _render_rdkit(decimer_raw) if decimer_raw else None
    if decimer_img is None:
        decimer_img = Image.new('RGB', (540, 400), 'white')
        ImageDraw.Draw(decimer_img).text((20, 20), 'DECIMER raw SMILES did not parse', fill='black')
    n_rings_dec = '?'
    if decimer_raw:
        m_d = Chem.MolFromSmiles(decimer_raw)
        if m_d is not None:
            n_rings_dec = len(m_d.GetRingInfo().AtomRings())
    panels.append((
        '3. DECIMER raw SMILES',
        decimer_img,
        f'rings = {n_rings_dec}  (macrocycle often fragmented)',
    ))
    canon_img = _render_rdkit(decimer_can) if decimer_can else None
    panels.append((
        '4. DECIMER canonical SMILES',
        canon_img,
        'canonical atom re-ordering; same connectivity',
    ))

    PANEL_W, PANEL_H, HEAD, PAD = 540, 480, 80, 14
    cols = len(panels)
    W = PAD + cols * (PANEL_W + PAD)
    H = HEAD + PANEL_H + 30
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    try:
        fontb = ImageFont.truetype('arialbd.ttf', 18)
        font = ImageFont.truetype('arial.ttf', 13)
    except OSError:
        fontb = font = ImageFont.load_default()

    d.text((PAD, 8),
           f'cmpd 1036 review deck -- formula {properties["formula"]}, largest ring {properties["largest_ring"]} atoms, {properties["rings"]} rings total',
           fill='black', font=fontb)
    d.text((PAD, 32),
           f'manual SMILES: {len(REF_SMI)} chars, exact mass {properties["exact_mass"]} Da',
           fill='gray', font=font)

    for i, (title, img, note) in enumerate(panels):
        x = PAD + i * (PANEL_W + PAD)
        d.text((x, HEAD - 30), title, fill='black', font=fontb, anchor='lt')
        if note:
            d.text((x, HEAD - 12), note, fill='gray', font=font, anchor='lt')
        if img is not None:
            canvas.paste(_fit(img, PANEL_W, PANEL_H - 28), (x, HEAD + 18))

    canvas.save(OUT)
    print(f'wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})')
    for k, v in properties.items():
        print(f'  {k:14s} : {v}')


if __name__ == '__main__':
    main()
