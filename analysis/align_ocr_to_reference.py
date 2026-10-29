"""Learn the OCR edit model by aligning damaged text to a known answer.

We have exactly one row where the correct answer is not in doubt: cmpd 1036,
whose SMILES was transcribed by hand from the patent image. Everything else
is inference. That makes 1036 a Rosetta stone for the OCR damage -- and so
far every repair strategy has been a guess because nobody looked at what
the recogniser actually did to the characters.

This script diffs the OCR string against the reference and prints every edit
with its context, grouped by kind:

* **deletion** -- a character present in the reference, absent from OCR.
  This is the dominant failure and the one we have never quantified.
* **substitution** -- a character the recogniser replaced.
* **insertion** -- a character the recogniser invented.

It then reports the frequency of each deleted character, which tells us
what an insertion-only repair must be able to put back.

Run from the repo root::

    .venv/Scripts/python.exe analysis/align_ocr_to_reference.py
"""

from __future__ import annotations

import csv
import difflib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

# Hand transcription of 1036, read off PCTCN2025075389-ftappb-I100297.
# Stored verbatim and never altered by any heuristic.
REFERENCE_1036 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
    "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
    "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)


def context(text: str, index: int, width: int = 10) -> str:
    lo = max(0, index - width)
    hi = min(len(text), index + width)
    return f"{text[lo:index]}[{text[index:index + 1]}]{text[index + 1:hi]}"


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    row = next(r for r in rows if r["cmpd_id"] == "1036")
    ocr = row["smiles"].strip()
    ref = REFERENCE_1036

    print("=" * 78)
    print("cmpd 1036: OCR text aligned to the hand transcription")
    print("=" * 78)
    print(f"reference length  : {len(ref)}")
    print(f"OCR length        : {len(ocr)}")
    print(f"OCR is {len(ref) - len(ocr)} characters short")
    print()

    sm = difflib.SequenceMatcher(a=ref, b=ocr, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]

    dels: Counter = Counter()
    subs: Counter = Counter()
    ins: Counter = Counter()

    print(f"{'#':>3}  {'kind':<13} {'ref':<9} {'ocr':<9} context in reference")
    print("-" * 78)
    for n, (tag, i1, i2, j1, j2) in enumerate(ops, 1):
        r = ref[i1:i2]
        o = ocr[j1:j2]
        if tag == "delete":
            for ch in r:
                dels[ch] += 1
            kind = f"delete x{len(r)}"
        elif tag == "replace":
            subs[(r, o)] += 1
            kind = "substitute"
        elif tag == "insert":
            for ch in o:
                ins[ch] += 1
            kind = f"insert x{len(o)}"
        else:
            continue
        print(f"{n:>3}  {kind:<13} {r!r:<9} {o!r:<9} {context(ref, i1)}")

    print()
    print("=" * 78)
    print("summary of the edit model")
    print("=" * 78)
    print(f"edit sites         : {len(ops)}")
    print(f"characters deleted : {sum(dels.values())}")
    print(f"characters inserted: {sum(ins.values())}")
    print(f"substitution sites : {sum(subs.values())}")
    print()

    print("deleted characters (what OCR dropped):")
    total_del = max(sum(dels.values()), 1)
    for ch, n in dels.most_common():
        print(f"  {ch!r:>5} : {n:>3}  ({n * 100 // total_del:>2}%)")
    print()

    print("inserted characters (what OCR invented):")
    for ch, n in ins.most_common():
        print(f"  {ch!r:>5} : {n:>3}")
    print()

    print("substitutions (reference -> OCR):")
    for (r, o), n in subs.most_common(20):
        print(f"  {r!r:>9} -> {o!r:<9} : {n}")
    print()

    parens = dels["("] + dels[")"]
    brackets = dels["["] + dels["]"]
    print(f"deleted parentheses : {parens}  ({dels['(']} open, {dels[')']} close)")
    print(f"deleted square brkt : {brackets}  ({dels['[']} open, {dels[']']} close)")
    others = {c: n for c, n in dels.items() if c not in "()[]"}
    print(f"other deletions     : {others}")
    print()
    print("Reading: brackets are missing, not wrong. An insertion-only repair")
    print("that can put back '(' ')' '[' ']' is the minimal sufficient edit set")
    print("for this row, provided the substitutions above are resolved from the")
    print("recogniser's own top-k alternatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
