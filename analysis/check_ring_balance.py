"""Are the OCR text's ring-closure digits balanced, independently of brackets?

The repair search has to close the macrocycle, and the macrocycle is closed by
ring-closure digits -- not by parentheses. So before blaming the search, it is
worth checking whether the digits survive OCR at all. If a digit is dropped,
no amount of bracket repair can close the ring, and the failure lies in the
input rather than in the algorithm.

This reports the digit histogram of the reference and of the OCR text, and
whether each number of opens is matched by the same number of closes. Digits
inside square brackets (isotopes, charges) are excluded, since those are not
ring closures.

Run from the repo root::

    .venv/Scripts/python.exe analysis/check_ring_balance.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpd.ocr_edit_model import normalise_lexical
from cpd.parsers.prefix_validator import PrefixValidator

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

REFERENCE_1036 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
    "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)


def ring_digits(text: str) -> Counter:
    """Count ring-closure digits, ignoring any inside square brackets."""
    counts: Counter = Counter()
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch.isdigit():
                counts[ch] += 1
            elif ch == "%":
                counts["%"] += 1
                i += 2
        i += 1
    return counts


def grammar_end_state(text: str) -> str:
    """Where the incremental walk stops, and in what bracket state."""
    v = PrefixValidator()
    for i, ch in enumerate(text):
        nxt = v.advance(ch)
        if nxt is None:
            return f"breaks at {i} on {ch!r}"
        v = nxt
    return (f"depth={v.depth} open_rings={sorted(v.rings)} "
            f"complete={v.is_complete()}")


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    row = next(r for r in rows if r["cmpd_id"] == "1036")
    ocr, _ = normalise_lexical(row["smiles"].strip())

    ref_digits = ring_digits(REFERENCE_1036)
    ocr_digits = ring_digits(ocr)
    keys = sorted(set(ref_digits) | set(ocr_digits))
    print(f"{'digit':>6} {'reference':>10} {'ocr':>6} {'ref odd?':>9} "
          f"{'ocr odd?':>9}")
    print("-" * 48)
    for k in keys:
        r, o = ref_digits.get(k, 0), ocr_digits.get(k, 0)
        print(f"{k:>6} {r:>10} {o:>6} {str(r % 2 == 1):>9} "
              f"{str(o % 2 == 1):>9}")
    print()
    print(f"total ring-closure digits: reference {sum(ref_digits.values())}, "
          f"OCR {sum(ocr_digits.values())}")
    print(f"all counts even, reference: "
          f"{all(v % 2 == 0 for v in ref_digits.values())}")
    print(f"all counts even, OCR      : "
          f"{all(v % 2 == 0 for v in ocr_digits.values())}")
    print()
    print("grammar at end of reference:", grammar_end_state(REFERENCE_1036))
    print("grammar at end of OCR      :", grammar_end_state(ocr))
    print()

    missing = {k: (ref_digits.get(k, 0), ocr_digits.get(k, 0)) for k in keys
               if ref_digits.get(k, 0) != ocr_digits.get(k, 0)}
    if missing:
        print(f"digits whose count differs (reference, OCR): {missing}")
        print()
        print("This matters more than parentheses: a macrocycle is closed by a")
        print("ring-closure digit, so a lost digit means no bracket repair can")
        print("close the ring. The search would have to invent a digit, which")
        print("changes the topology -- exactly the kind of guess that produced")
        print("the invalid molecules audited in PROGRESS.md.")
    else:
        print("OCR kept every ring-closure digit. The macrocycle is therefore")
        print("still closeable, and the damage is confined to parentheses,")
        print("which do not change the topology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
