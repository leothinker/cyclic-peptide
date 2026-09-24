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


def auto_repair_smiles(raw_smiles: str) -> str:
    """Auto-heal common OCR syntax flaws in peptide SMILES strings."""
    s = raw_smiles
    # 1. Strip non-ASCII artifacts (e.g. table border line fragments)
    s = re.sub(r"[^\x00-\x7F]+", "", s)

    # 2. Repair wrapped amide carbonyls: C(O)N or C(=0)N -> C(=O)N
    s = re.sub(r"C\(O\)N", "C(=O)N", s)
    s = re.sub(r"C\(=[0O]\)?N", "C(=O)N", s)

    # 3. Repair unclosed chiral brackets: [C@H( or [C@@H( -> [C@H]( or [C@@H](
    s = re.sub(r"(\[C@{1,2}H)\(", r"\1](", s)

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
