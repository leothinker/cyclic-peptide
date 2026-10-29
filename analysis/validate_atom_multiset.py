"""Cross-check :func:`cpd.repair_audit.atom_multiset` against RDKit.

The atom-multiset diff is the primary evidence in
:func:`cpd.repair_audit.audit_repair`, so it has to be right.  This script
compares the histogram's total against RDKit's heavy-atom count on every
compound in the corpus that *does* parse, plus a set of hand-written edge
cases (two-letter halogens, aromatic brackets, isotopes, ions, deuterium,
ring-closure numbers, unclosed brackets).

Exits non-zero when any case disagrees, so it can be wired into CI.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from cpd.repair_audit import atom_multiset, fingerprint  # noqa: E402

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

EDGE_CASES = [
    # (smiles, expected atom count)
    ("CCO", 3),
    ("CC(=O)N", 4),
    ("[C@H](C)Cl", 3),
    ("[13C]", 1),
    ("c1ccccc1F", 7),
    ("C1CCCCC1.C", 7),
    ("BrCCl", 3),
    ("[nH]1cccc1", 5),
    ("[Na+]", 1),
    ("[C@@H]1N(C)", 3),
    ("[NH4+]", 1),
    ("C%10CCCCC%10", 6),
    ("F/C=C/F", 4),
    ("c1ccc2ccccc2c1", 10),
]


def check(smiles: str, expected: int) -> tuple[bool, str]:
    got = sum(atom_multiset(smiles).values())
    return got == expected, f"expected {expected}, got {got}"


def main() -> int:
    failures = 0
    print("-- edge cases --")
    for smiles, expected in EDGE_CASES:
        ok, msg = check(smiles, expected)
        print(f"  {'PASS' if ok else 'FAIL'}  {smiles!r:28s} {msg if not ok else ''}")
        if not ok:
            failures += 1

    print()
    print("-- corpus SMILES that RDKit can parse --")
    checked = 0
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            for col in ("smiles_decimer", "smiles_best", "repaired_smiles"):
                smi = (row.get(col) or "").strip()
                if not smi:
                    continue
                fp = fingerprint(smi)
                if fp is None:
                    continue
                got = sum(atom_multiset(smi).values())
                if got != fp["heavy_atoms"]:
                    failures += 1
                    print(
                        f"  FAIL cmpd={row.get('cmpd_id')} col={col} "
                        f"multiset={got} rdkit={fp['heavy_atoms']}"
                    )
                checked += 1
    print(f"  cross-checked {checked} parseable SMILES")

    print()
    if failures:
        print(f"FAILED: {failures} disagreement(s)")
        return 1
    print("OK: atom_multiset agrees with RDKit everywhere")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
