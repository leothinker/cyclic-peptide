"""Decisive check: does the patent's printed SMILES match the patent's m/z?

Every repair heuristic in this project has been judged on whether it
"produces a macrocycle".  That criterion is self-referential: a heuristic
that bolts a bond between two distant atoms always produces *some* ring, so
the check passes even when the chemistry is invented.

The reported m/z is not self-referential.  It was measured on an
instrument, and this patent lists the same compound twice under different
ionisation -- cmpd 1036 as [M+H]+ 1610.9 and cmpd 1046 as [M+3H]3+ 537.6 --
which gives an internal consistency check on the m/z column itself.

So this script asks: does the hand-transcribed structure for 1036
reproduce the mass, and does DECIMER's output for the same cell?

Run from the repo root::

    .venv/Scripts/python.exe analysis/verify_1036_mass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger, Chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.ms_validation import neutral_mass_from_mz, validate_against_ms  # noqa: E402

# The transcription is imported from the demo script so there is exactly one
# copy of the string and no chance of the two drifting apart.
sys.path.insert(0, str(ROOT))

from demo_1036_manual_render import MANUAL_SMI  # noqa: E402

# DECIMER's output for the 1036 cell, as stored in the combined CSV.
DECIMER_1036 = (
    "CC[C@H](C)[C@@H](C(=O)N(C)[C@@H](C)C(=O)N1CC[C@H]1C(=O)N(C)"
    "[C@@H](C)C(=O)N(C)[C@@H](C)C(=O)N[C@@H](CCC2=CC(=C(C(=C2)F)"
    "C(F)(F)F)F)C(=O)N3CCC[C@H]3C(=O)NC4(CCCC4)C(=O)N(C)[C@@H]"
    "(C5CCCC5)C(=O)N(C)[C@@H](CC(=O)N(C)CCNC(=O)C6=C(C(=C(C(=C6F)"
    "F)SC)F)F)C(=O)N(C)C)NC=O"
)


def _report(label: str, smiles: str, mz: float, polarity: str) -> dict:
    res = validate_against_ms(smiles, mz, polarity)
    mol = Chem.MolFromSmiles(smiles)
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()] if mol else []
    print(f"-- {label}")
    print(f"   reported m/z          : {mz}  {polarity}")
    print(f"   implied neutral mass  : {res['expected_mass']}")
    print(f"   formula from SMILES   : {res['formula']}")
    print(f"   monoisotopic mass     : {res['observed_mass']}")
    print(f"   largest ring          : {max(rings) if rings else 0}")
    print(f"   mass error            : {res['error_ppm']} ppm")
    print(f"   verdict               : {res['verdict']}")
    print()
    return res


def main() -> int:
    print("=" * 72)
    print("CMPD 1036: hand-transcribed patent SMILES vs reported LC-MS mass")
    print("=" * 72)
    print()

    print("step 1 -- internal consistency of the m/z column itself")
    a = neutral_mass_from_mz(1610.9, "[M+H]+")
    b = neutral_mass_from_mz(537.6, "[M+3H]3+")
    print("   cmpd 1036 [M+H]+   1610.9 -> "
          f"{a:.4f}" if a else "   cmpd 1036 -> None")
    print("   cmpd 1046 [M+3H]3+  537.6 -> "
          f"{b:.4f}" if b else "   cmpd 1046 -> None")
    if a and b:
        print(f"   difference: {abs(a - b):.4f} Da ({abs(a - b) / a * 1e6:.1f} ppm)")
    print()

    print("step 2 -- hand-transcribed patent structure")
    print(f"   transcription length: {len(MANUAL_SMI)} chars")
    hand = _report("manual transcription", MANUAL_SMI, 1610.9, "[M+H]+")

    print("step 3 -- DECIMER output for the same cell")
    dec = _report("DECIMER", DECIMER_1036, 1610.9, "[M+H]+")

    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    if hand["verdict"] == "match" and dec["verdict"] == "mismatch":
        print("  The hand-transcribed structure reproduces the reported mass")
        print("  and DECIMER's output does not. The m/z column is trustworthy")
        print("  ground truth, so 'did the repair produce a macrocycle' is the")
        print("  wrong acceptance criterion. A candidate should instead be")
        print("  required to match the measured mass.")
    elif dec["verdict"] == "match":
        print("  DECIMER reproduces the reported mass; the transcription and")
        print("  the structure image need another look.")
    else:
        print("  Neither structure reproduces the reported mass. Investigate the")
        print("  m/z column, the ion label, or both transcriptions first.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
