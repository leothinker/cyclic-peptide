"""2D structure rendering for cyclic-peptide SMILES.

This module converts a SMILES string into a 2D structure image.  For
intact macrocycles (the largest ring has 12 or more atoms) we
*linearise* the molecule by opening one amide C-N bond before
generating coordinates; this flattens the cycle into a chain that is
much easier to read in a small thumbnail.  When DECIMER has produced
a SMILES whose macrocycle is already broken (largest ring < 12) we
render it as-is, since there is no single ring to open.

Public API
----------

* :func:`is_macrocycle_intact`  -- True iff a SMILES parses to a molecule
  whose largest ring is at least :data:`MACROCYCLE_MIN_SIZE` atoms.
* :func:`ring_closure_audit`    -- diagnostic counts of ring-closure
  digits in the raw SMILES (helps spot the "same digit = two bonds"
  failure mode DECIMER occasionally exhibits on fused bicyclic peptides).
* :func:`render_mol`            -- SMILES -> :class:`PIL.Image.Image`
  (or ``None`` on parse / render failure).
* :func:`render_mol_pil`        -- convenience wrapper returning a
  PIL image of the given size.
"""

from __future__ import annotations

import re
from collections import Counter
from io import BytesIO
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is in the env but guard anyway
    Image = None  # type: ignore[assignment]


# A real cyclic peptide backbone has 8-30+ atoms in the macrocycle.  Anything
# smaller than this after RDKit parses the SMILES is almost always a sign
# that the macrocycle was broken into a fused system of small rings, e.g. by
# DECIMER re-using a ring-closure digit in close proximity.  Used by
# :func:`is_macrocycle_intact` and to gate :func:`render_mol`.
MACROCYCLE_MIN_SIZE: int = 12


def _largest_ring_size(mol: Chem.Mol) -> int:
    """Return the size of the largest ring in ``mol`` (0 if acyclic)."""
    if mol is None:
        return 0
    ring_info = mol.GetRingInfo()
    sizes = [len(r) for r in ring_info.AtomRings()]
    return max(sizes) if sizes else 0


def is_macrocycle_intact(
    smiles: str, min_ring_size: int = MACROCYCLE_MIN_SIZE
) -> bool:
    """``True`` iff ``smiles`` parses to a molecule whose largest ring is
    at least ``min_ring_size`` atoms.

    The intended use is the binary "is this cyclic peptide still in one
    piece" gate that callers apply to DECIMER output.  Acyclic or
    fragmented SMILES (largest ring < threshold) are considered
    "broken" because the macrocycle is no longer a single connected
    cycle.
    """
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return _largest_ring_size(mol) >= min_ring_size


# SMILES ring-closure digits are %10, %11, ... for two-digit numbers.
# We count how often each digit / %nn token appears in the raw string.
# Single-digit ring closures (1-9) are valid SMILES as long as they are
# not the start of a multi-digit number, i.e. they are not immediately
# preceded by '%'.  The only place a plain digit appears in SMILES
# outside of a ring closure is inside square brackets (e.g. an isotope
# like [13C] or a charge like [Fe+2]); we skip those by tracking
# bracket depth below.
_MULTI_DIGIT_RE = re.compile(r"%(\d{2})")
_SINGLE_DIGITS = set("123456789")


def _scan_ring_closures(smiles: str) -> Counter:
    """Walk the SMILES and tally every ring-closure digit/token.

    Bracket-balanced: digits inside ``[ ... ]`` (isotopes, charges,
    explicit hydrogens) are ignored because they are not ring
    closures.  Outside brackets, ``%NN`` counts as the two-digit
    token ``NN``; any single digit 1-9 is a one-digit ring closure.
    A single digit preceded by '%' is part of the multi-digit token
    and not counted again.
    """
    counts: Counter[str] = Counter()
    depth = 0
    i = 0
    n = len(smiles)
    while i < n:
        c = smiles[i]
        if c == "[":
            depth += 1
            i += 1
            continue
        if c == "]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth > 0:
            i += 1
            continue
        if c == "%" and i + 2 < n and smiles[i + 1 : i + 3].isdigit():
            counts[smiles[i + 1 : i + 3]] += 1
            i += 3
            continue
        if c in _SINGLE_DIGITS:
            counts[c] += 1
        i += 1
    return counts


def ring_closure_audit(smiles: str) -> dict:
    """Return a tiny diagnostic dict about ring-closure digit usage.

    Example::

        >>> ring_closure_audit("C1CCCCC1")
        {'counts': {'1': 2}, 'unclosed': {}, 'len': 6}

    ``counts`` is a dict mapping each ring-closure digit to the number
    of times it appears in the SMILES.  SMILES semantics require
    every digit to appear **exactly twice** (the two ends of the bond
    it closes).  A digit that appears 0 times is unused, 1 time is
    unclosed (a parse error), and 3+ times indicates the digit was
    re-used to close a second ring -- the failure mode DECIMER
    exhibits on fused bicyclic peptides, where it accidentally uses
    the same digit twice in close proximity to create a small ring
    instead of carrying the original macrocycle digit through the
    full backbone.
    """
    if not smiles:
        return {"counts": {}, "unclosed": {}, "len": 0}
    counts = _scan_ring_closures(smiles)
    unclosed = {digit: n for digit, n in counts.items() if n != 2}
    return {"counts": dict(counts), "unclosed": unclosed, "len": len(smiles)}


def _open_one_amide_bond(mol: Chem.Mol) -> Chem.Mol | None:
    """Break one amide C-N bond in a macrocycle so it renders as a chain.

    Walks the molecule's atoms looking for a bond ``C(=O)-N`` that sits
    inside the macrocycle (i.e. removing it increases the number of
    fragments by 1 and the resulting largest ring becomes <= the
    macrocycle size).  When found, the bond is deleted and a copy of
    the molecule is returned with the ring opened.  Returns ``None``
    when no suitable amide bond is present (caller should fall back to
    rendering the molecule as-is).
    """
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    macrocycle = max((len(r) for r in ri.AtomRings()), default=0)
    if macrocycle < MACROCYCLE_MIN_SIZE:
        return None

    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        # amide: C(=O)-N
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7:
            carbon = a
        elif a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6:
            carbon = b
        else:
            continue
        # only break C-N where the carbon is a carbonyl carbon
        has_carbonyl = False
        for nb in carbon.GetBonds():
            if nb.GetIdx() == bond.GetIdx():
                continue
            if (
                nb.GetBondType() == Chem.BondType.DOUBLE
                and nb.GetOtherAtom(carbon).GetAtomicNum() == 8
            ):
                has_carbonyl = True
                break
        if not has_carbonyl:
            continue
        # test: removing this bond should drop the largest ring below threshold
        test = Chem.RWMol(rw)
        test.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        test_mol = test.GetMol()
        try:
            Chem.SanitizeMol(test_mol)
        except Exception:
            continue
        new_largest = _largest_ring_size(test_mol)
        if new_largest < macrocycle:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            result = rw.GetMol()
            try:
                Chem.SanitizeMol(result)
            except Exception:
                continue
            return result
    return None


def _draw(
    mol: Chem.Mol,
    size: tuple[int, int] = (520, 360),
) -> Optional["Image.Image"]:
    """Render a molecule to a PIL image using RDKit's 2D drawer.

    Uses CoordGen for layout (gives nicer bond geometry for fused
    macrocycles than RDKit's built-in engine) and a transparent
    background.  Returns ``None`` if anything goes wrong.
    """
    if mol is None or Image is None:
        return None
    try:
        if mol.GetNumConformers() == 0:
            try:
                rdCoordGen.SetCoords(mol)
            except Exception:
                rdDepictor.Compute2DCoords(mol)
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        if not png:
            return None
        return Image.open(BytesIO(png)).convert("RGBA")
    except Exception:
        return None


def render_mol(
    smiles: str,
    size: tuple[int, int] = (520, 360),
    linearize_macrocycle: bool = True,
) -> Optional["Image.Image"]:
    """Render ``smiles`` to a 2D structure image.

    When ``linearize_macrocycle`` is True (default) and the molecule's
    largest ring is >= :data:`MACROCYCLE_MIN_SIZE` (an intact
    macrocycle), one amide C-N bond is broken before drawing so the
    cyclic peptide appears as a linear chain -- the standard
    rendering style for cyclic peptides in this project.

    If the macrocycle is already broken (largest ring < threshold),
    the SMILES is rendered as-is: there is no single ring to open and
    the structure already shows the fused / fragmented topology.

    Returns ``None`` for empty input, parse errors, or rendering
    failures -- callers can treat None as "skip this structure".
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    if linearize_macrocycle:
        largest = _largest_ring_size(mol)
        if largest >= MACROCYCLE_MIN_SIZE:
            opened = _open_one_amide_bond(mol)
            if opened is not None:
                mol = opened
        # else: macrocycle is broken; render as-is, no linearization.

    return _draw(mol, size=size)


def render_mol_pil(
    smiles: str,
    size: tuple[int, int] = (520, 360),
    linearize_macrocycle: bool = True,
) -> Optional["Image.Image"]:
    """Convenience wrapper around :func:`render_mol`.

    Identical behaviour; exists so callers that want a PIL image can
    be explicit about the return type.  Returns ``None`` on failure.
    """
    return render_mol(smiles, size=size, linearize_macrocycle=linearize_macrocycle)


__all__ = [
    "MACROCYCLE_MIN_SIZE",
    "is_macrocycle_intact",
    "ring_closure_audit",
    "render_mol",
    "render_mol_pil",
]
