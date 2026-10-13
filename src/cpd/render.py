"""Render a SMILES string to a publication-style 2D structure PNG.

Wraps ``rdkit.Chem.Draw.rdMolDraw2D`` with the knobs we want for
cyclic peptide visual review:

  * CoordGen layout (less "macrocycle donut", more "ChemDraw" -- open
    backbone with side chains extending outward)
  * Loaded default ring-system templates so aromatic rings become
    hexagons with explicit double bonds
  * Coloured heteroatoms (N blue, O red, S yellow, F cyan)
  * Stereo wedges honoured (predict_SMILES returns chirality)

Cyclic peptide macrocycles are still rendered as a closed loop by
CoordGen, which can look quite different from a patent's hand-drawn
depiction. ``linearize_macrocycle=True`` adds an optional pass: it
identifies the largest ring and breaks one amide bond in it before
laying out 2D coords, so the macrocycle is depicted as an open chain
with side chains extending outward. The returned image is for visual
review only -- the underlying SMILES is not modified.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

if rdDepictor.IsCoordGenSupportAvailable():
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.LoadDefaultRingSystemTemplates()


def _open_one_amide_bond(mol: Chem.Mol) -> Chem.Mol | None:
    """Break the largest ring at one amide bond; return a new mol.

    Returns None if no suitable ring/amide is found.
    """
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if not rings:
        return None
    largest = max(rings, key=len)
    if len(largest) < 12:  # not a macrocycle
        return None
    ring_atoms = set(largest)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetIdx() not in ring_atoms or b.GetIdx() not in ring_atoms:
            continue
        c, n = (a, b) if a.GetSymbol() == "C" else (b, a) if b.GetSymbol() == "C" else (None, None)
        if c is None or n.GetSymbol() != "N":
            continue
        if not any(
            ob.GetBondType() == Chem.BondType.DOUBLE and ob.GetOtherAtom(c).GetSymbol() == "O"
            for ob in c.GetBonds()
        ):
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveBond(c.GetIdx(), n.GetIdx())
        rw.AddAtom(Chem.Atom(1))
        last = rw.GetNumAtoms() - 1
        rw.AddBond(c.GetIdx(), last, Chem.BondType.SINGLE)
        rw.AddAtom(Chem.Atom(1))
        last = rw.GetNumAtoms() - 1
        rw.AddBond(n.GetIdx(), last, Chem.BondType.SINGLE)
        try:
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            return None
    return None


def _draw(mol: Chem.Mol, size: tuple[int, int], line_width: float, coloured: bool, stereo: bool) -> Optional[bytes]:
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.bondLineWidth = line_width
        opts.fixedBondLength = -1
        opts.useBWAtomPalette = not coloured
        opts.addStereoAnnotation = stereo
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def render_mol(
    smi: str,
    size: tuple[int, int] = (520, 360),
    line_width: float = 1.4,
    fixed_bond_length: int = -1,
    coloured_atoms: bool = True,
    stereo: bool = True,
    linearize_macrocycle: bool = False,
) -> Optional[bytes]:
    """Render SMILES to PNG bytes (Cairo). Returns None on parse failure.

    If ``linearize_macrocycle`` is True and the molecule has a ring
    of 12+ atoms, the largest ring is opened at one amide bond before
    laying out 2D coords -- produces a more open-chain ChemDraw-like
    layout for visual review. The underlying SMILES is not modified.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    target = mol
    if linearize_macrocycle:
        opened = _open_one_amide_bond(mol)
        if opened is not None:
            target = opened
    try:
        AllChem.Compute2DCoords(target)
    except Exception:
        return None
    return _draw(target, size, line_width, coloured_atoms, stereo)


def render_mol_pil(smi: str, size: tuple[int, int] = (520, 360), linearize_macrocycle: bool = False):
    """Convenience: return a PIL RGBA Image (or None)."""
    from io import BytesIO
    from PIL import Image

    png = render_mol(smi, size=size, linearize_macrocycle=linearize_macrocycle)
    if png is None:
        return None
    return Image.open(BytesIO(png)).convert("RGBA")
