"""Can a surplus ``]`` be turned back into a ``[``?

Compound 28 of WO2022251372 is read with ``)`` followed by ``]``
where the page prints ``)`` followed by ``[``: the recogniser confused
an opening bracket with a closing one.  The string then holds one
``]`` too many, so it cannot parse.

The repair is only allowed where it is unambiguous: the string must
hold exactly one ``]`` too many, and exactly one position may be
flipped to make the brackets balance.  This script measures that on
the whole corpus of WO2022251372 and on the delivered CSV of
WO2025162428, so the rule is measured on both patents before it is
considered for use.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_bracket_flip_patent2.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

RDLogger.DisableLog("rdApp.*")

OUT = REPO / "data" / "audit" / "phase9"


def _parses(text: str) -> bool:
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _flip_positions(text: str) -> list[int]:
    """Positions of a ``]`` whose flip to ``[`` yields a parsable string.

    Counting alone cannot pick a position: when the string holds one
    ``]`` too many, *every* ``]`` balances the count once flipped.
    What distinguishes them is whether the result is a well-formed
    SMILES, so the parse is what selects the candidate.
    """
    if not text:
        return []
    if text.count("]") != text.count("[") + 1:
        return []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char != "]":
            continue
        patched = text[:index] + "[" + text[index + 1:]
        if patched.count("[") == patched.count("]") and _parses(patched):
            positions.append(index)
    return positions


def _report(name: str, items: list[tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for key, text in items:
        positions = _flip_positions(text)
        repaired = (
            text[:positions[0]] + "[" + text[positions[0] + 1:]
            if len(positions) == 1
            else ""
        )
        rows.append({
            "patent": name,
            "key": key,
            "reading": text,
            "flip_positions": len(positions),
            "repaired": repaired,
            "parses_before": _parses(text),
            "parses_after": _parses(repaired) if repaired else False,
        })
    return rows


def main() -> int:
    # --- WO2022251372: the 28 compounds of the route comparison ------
    with (
        REPO / "data" / "audit" / "phase8" / "patent2_vote_effect.csv"
    ).open(encoding="utf-8", newline="") as handle:
        patent2 = [
            (row["cmpd_id"], row["text_after"])
            for row in csv.DictReader(handle)
        ]

    # --- WO2025162428: the delivered CSV ------------------------------
    with (
        REPO / "data" / "processed" / "compounds_combined.csv"
    ).open(encoding="utf-8-sig", newline="") as handle:
        patent1 = [
            (row["cmpd_id"], row["smiles"])
            for row in csv.DictReader(handle)
            if (row.get("smiles") or "").strip()
        ]

    rows = _report("WO2022251372", patent2) + _report("WO2025162428", patent1)

    out_csv = OUT / "bracket_flip_evidence.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_csv}")

    for name in ("WO2022251372", "WO2025162428"):
        subset = [row for row in rows if row["patent"] == name]
        unbalanced = [
            row for row in subset
            if row["reading"].count("]") != row["reading"].count("[")
        ]
        unique = [
            row for row in unbalanced if row["flip_positions"] == 1
        ]
        ambiguous = [
            row for row in unbalanced if row["flip_positions"] != 1
        ]
        fixed = [row for row in unique if row["parses_after"]]
        broken = [
            row for row in unique
            if row["parses_before"] and not row["parses_after"]
        ]
        print()
        print(f"=== {name}")
        print(f"  readings                  : {len(subset)}")
        print(f"  unbalanced brackets       : {len(unbalanced)}")
        print(f"  exactly one flip position : {len(unique)}")
        print(f"  ambiguous / none          : {len(ambiguous)}")
        print(f"  parse after the flip      : {len(fixed)}")
        print(f"  parse before AND after    : {len(broken)}  (must be 0)")
        for row in unique:
            index = row["reading"].find("]")
            print(
                f"    cmpd {row['key']:>4}  "
                f"{row['parses_before']} -> {row['parses_after']}  "
                f"...{row['reading'][max(0, index - 12):index + 12]}..."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
