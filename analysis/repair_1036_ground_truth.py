"""Run the mass-constrained repair on cmpd 1036 and grade it.

1036 is the only row where the correct answer is known independently: the
SMILES was transcribed by hand from the patent image. That makes it the one
place a repair search can actually be *scored* instead of merely run.

This reports, for the search's output: whether RDKit parses it and the
largest ring is a real macrocycle; whether the formula matches the reference
formula; whether the mass lands in the window the patent's m/z implies; and
most importantly whether the canonical SMILES equals the reference's -- i.e.
whether the connectivity is genuinely right and not merely plausible.

Run from the repo root::

    .venv/Scripts/python.exe analysis/repair_1036_ground_truth.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from cpd.mass_constrained_search import mass_window, repair_from_ocr
from cpd.ocr_edit_model import edit_model_report, normalise_lexical
from cpd.ocr_mass_gate import MassGate

RDLogger.DisableLog("rdApp.*")

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

# Hand transcription of 1036, read off PCTCN2025075389-ftappb-I100297.
# Stored verbatim; no heuristic is allowed to modify it.
REFERENCE_1036 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
    "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)


def describe(smiles):
    """Formula, mass, ring data and canonical form of a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parses": False}
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        "parses": True,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        "rings": len(rings),
        "largest_ring": max(rings) if rings else 0,
        "atoms": mol.GetNumAtoms(),
        "canonical": Chem.MolToSmiles(mol),
    }


def main():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    row = next(r for r in rows if r["cmpd_id"] == "1036")
    ocr = row["smiles"].strip()
    mz = float(row["ms_mz"].replace(",", ""))
    pol = row["ms_polarity"]

    print("=" * 74)
    print("cmpd 1036 -- mass-constrained repair, graded against the reference")
    print("=" * 74)
    print(f"reported m/z     : {row['ms_mz']} {pol}")
    lo, hi = mass_window(mz, pol)
    print(f"neutral window   : {lo:.4f} .. {hi:.4f}")
    print(f"OCR length       : {len(ocr)}")
    print(f"reference length : {len(REFERENCE_1036)}")
    print()

    prepared, notes = normalise_lexical(ocr)
    print(f"lexical pre-pass : {notes}")
    rep = edit_model_report(prepared)
    print(f"bracket deficit  : {rep['insertions']}")
    print(f"hyphens present  : {rep['hybrid']['hyphen']}")
    print()

    gate = MassGate.from_row(row["ms_mz"], pol)
    print("mass gate on the OCR text, before any repair:")
    print(f"  {gate.score_text(prepared)}")
    print()

    ref = describe(REFERENCE_1036)
    print("reference:")
    for k, v in ref.items():
        print(f"  {k:<12}: {v}")
    print()

    print("searching...")
    t0 = time.time()
    res = repair_from_ocr(ocr, mz, pol, max_seconds=120.0)
    dt = time.time() - t0
    print(f"done in {dt:.1f}s")
    print()
    print("search result:")
    for k, v in res.items():
        if k == "lexical_notes":
            continue
        print(f"  {k:<12}: {v}")
    print()

    if not res["smiles"]:
        print("VERDICT: no mass-verified macrocycle found.")
        return 1

    got = describe(res["smiles"])
    print("repaired structure:")
    for k, v in got.items():
        if k != "canonical":
            print(f"  {k:<12}: {v}")
    print()

    same = got["canonical"] == ref["canonical"]
    print("comparison with the reference:")
    print(f"  formula match : {got['formula'] == ref['formula']}"
          f"   ({got['formula']} vs {ref['formula']})")
    print(f"  mass delta    : {got['exact_mass'] - ref['exact_mass']:+.4f} Da")
    print(f"  in window     : {lo <= got['exact_mass'] <= hi}")
    print(f"  IDENTICAL     : {same}")
    print()
    if same:
        print("VERDICT: exact recovery. The search returned the same molecule")
        print("as the hand transcription, from OCR text alone, with the mass")
        print("gate as the only chemical check.")
    else:
        print("VERDICT: mass-verified but not identical to the reference.")
        print("This is the important limitation: the mass constrains the atom")
        print("inventory, not the connectivity, so more than one bracket")
        print("placement can satisfy it. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
