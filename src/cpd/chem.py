"""Chemical structure validation, auto-repair, and descriptor calculation for cyclic peptides."""

from __future__ import annotations

import re
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class ChemInfo:
    """Validated SMILES and physical descriptors for a cyclic peptide."""

    canonical_smiles: str
    molecular_weight: float
    heavy_atom_count: int
    logp: float
    tpsa: float
    rotatable_bonds: int
    num_rings: int
    largest_ring_size: int
    num_stereocenters: int
    formula: str


def largest_fragment_canonical(smiles: str) -> str | None:
    """Drop small phantom fragments from a SMILES string.

    DECIMER occasionally appends tiny fragments like ``CC`` or ``CCO``
    when it gets confused on a structure, separated from the real
    peptide by ``.``. RDKit then parses the dot as a multi-fragment
    molecule and canonicalises differently, so the SMILES never
    matches the OCR's single-fragment output. Keeping only the
    largest fragment by atom count recovers the real peptide.

    Returns the canonical SMILES of the largest fragment, or ``None``
    if the input cannot be parsed.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    if len(frags) <= 1:
        return Chem.MolToSmiles(mol)
    frags.sort(key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0])


def close_unbalanced_brackets(smiles: str) -> str | None:
    """Append the missing ``)`` and ``]`` so a truncated SMILES parses.

    The DECIMER transformer decoder has a positional-encoding maxlen of
    ~300, so cyclic-peptide SMILES longer than that get truncated mid-
    branch (e.g. ``...C(=O)N(C)C)F)F)F``). RDKit cannot parse them
    because of the dangling parens. Counting ``(`` minus ``)`` and
    ``[`` minus ``]`` and appending the missing closers recovers a
    surprising fraction of these cases (~33% in our benchmark).

    Returns the patched SMILES, or ``None`` if it has nothing to fix.
    """
    if not smiles:
        return None
    open_p = smiles.count("(") - smiles.count(")")
    open_b = smiles.count("[") - smiles.count("]")
    if open_p <= 0 and open_b <= 0:
        return None
    return smiles + ")" * max(0, open_p) + "]" * max(0, open_b)


def auto_repair_smiles(raw_smiles: str) -> str:
    """Auto-heal common OCR syntax flaws in peptide SMILES strings."""
    s = raw_smiles
    # 1. Strip non-ASCII artifacts (e.g. table border line fragments)
    s = re.sub(r"[^\x00-\x7F]+", "", s)

    # 2. Repair wrapped amide carbonyls: C(O)N or C(=0)N -> C(=O)N
    s = re.sub(r"C\(O\)N", "C(=O)N", s)
    s = re.sub(r"C\(=[0O]\)?N", "C(=O)N", s)
    # Same flavour for generic C(=0) -> C(=O) anywhere in the string
    s = re.sub(r"C\(=0\)", "C(=O)", s)

    # 3. Repair unclosed chiral brackets: [C@H( or [C@@H( -> [C@H]( or [C@@H](
    s = re.sub(r"(\[C@{1,2}H)\(", r"\1](", s)

    # 4. Re-attach the "CC[C@" prefix the OCR tends to lop off when the
    #    first SMILES characters touch the left column boundary.
    if s.startswith("H](") or s.startswith("@H]("):
        s = "CC[C@" + s
    elif s.startswith("[C@H](") and not s.startswith("CC[C@H]("):
        s = "CC" + s

    return s.strip()


def validate_and_enrich(smiles: str) -> ChemInfo | None:
    """Validate a SMILES string with RDKit and compute cyclic peptide descriptors.

    Returns None if the SMILES string is chemically invalid.
    """
    cleaned_smiles = auto_repair_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned_smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    ring_info = mol.GetRingInfo()

    # Count explicitly assigned chiral centers
    num_stereo = sum(
        1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )

    return ChemInfo(
        canonical_smiles=canonical,
        molecular_weight=round(Descriptors.MolWt(mol), 2),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        logp=round(Descriptors.MolLogP(mol), 2),
        tpsa=round(Descriptors.TPSA(mol), 2),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        num_rings=ring_info.NumRings(),
        largest_ring_size=max((len(r) for r in ring_info.AtomRings()), default=0),
        num_stereocenters=num_stereo,
        formula=rdMolDescriptors.CalcMolFormula(mol),
    )
