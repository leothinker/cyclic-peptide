"""How often is a bracket read as its mirror image?

Compound 28 of WO2022251372 is read with ``)`` where the page prints
``[`` -- the opening bracket of a chiral atom is read as a closing
one.  The string then holds one ``]`` too many, so it cannot parse.

This script measures how often that happens in the delivered data, and
whether the fix is unambiguous: a ``]`` that has no partner can be
turned back into a ``[`` in exactly one place.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_bracket_flip_scan.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

OUT = REPO / "data" / "audit" / "phase9"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"


def _flip_candidates(text: str) -> list[int]:
    """Positions of a ``]`` that could be a misread ``[``.

    A candidate is a ``]`` that sits where the string has no spare
    opening bracket.  Turning it into a ``[`` balances the string.
    Only the positions where the flip makes the brackets balance are
    returned, so an ambiguous string yields no candidate at all.
    """
    if not text:
        return []
    if text.count("]") <= text.count("["):
        return []
    surplus = text.count("]") - text.count("[")
    if surplus != 1:
        return []
    candidates: list[int] = []
    for index, char in enumerate(text):
        if char != "]":
            continue
        patched = text[:index] + "[" + text[index + 1:]
        if patched.count("[") == patched.count("]"):
            candidates.append(index)
    return candidates


def main() -> int:
    # The delivered CSV carries a byte-order mark, so it is read with
    # ``utf-8-sig``; otherwise the first column is called
    # "﻿cmpd_id" and cannot be looked up.
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    printed = [
        row for row in rows if (row.get("smiles") or "").strip()
    ]
    unbalanced = [
        row for row in printed
        if row["smiles"].count("]") != row["smiles"].count("[")
    ]
    flipped = [
        row for row in unbalanced
        if len(_flip_candidates(row["smiles"])) == 1
    ]
    ambiguous = [
        row for row in unbalanced
        if len(_flip_candidates(row["smiles"])) != 1
    ]

    out_csv = OUT / "bracket_flip_scan.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cmpd_id", "closing", "opening", "candidates",
                "context", "smiles",
            ],
        )
        writer.writeheader()
        for row in unbalanced:
            candidates = _flip_candidates(row["smiles"])
            index = candidates[0] if len(candidates) == 1 else -1
            context = (
                row["smiles"][max(0, index - 20):index + 20]
                if index >= 0
                else ""
            )
            writer.writerow({
                "cmpd_id": row["cmpd_id"],
                "closing": row["smiles"].count("]"),
                "opening": row["smiles"].count("["),
                "candidates": len(candidates),
                "context": context,
                "smiles": row["smiles"],
            })
    print(f"-> {out_csv}")
    print()
    print(f"rows with a printed SMILES      : {len(printed)}")
    print(f"rows with unbalanced brackets    : {len(unbalanced)}")
    print(f"  exactly one unambiguous flip   : {len(flipped)}")
    print(f"  ambiguous or no candidate      : {len(ambiguous)}")
    print()
    for row in flipped:
        index = _flip_candidates(row["smiles"])[0]
        print(
            f"  cmpd {row['cmpd_id']:>5}  "
            f"...{row['smiles'][max(0, index - 14):index + 14]}..."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
