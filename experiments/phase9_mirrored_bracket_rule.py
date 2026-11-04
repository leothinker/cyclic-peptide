"""Can a bracket read as its mirror image be repaired?

Compound 28 of WO2022251372 is read with ``C(=O)]C@@H](C)`` where the
page prints ``C(=O)[C@@H](C)``: the recogniser turned the opening
bracket of a chiral atom into a closing one.

This script measures the repair rule that now lives in
:func:`cpd.chem.auto_repair_smiles` against both patents' corpora, so
that the rule's reach and its blast radius are both on record.

The rule is deliberately narrow.  It only fires when

* the string holds **two** more ``]`` than ``[`` -- a bracket read as
  its mirror image moves the difference by two, so a difference of one
  cannot come from this mistake and must not be "fixed" by flipping a
  bracket;
* the parentheses still balance, so the confusion is not wider than a
  single bracket;
* the string does not parse as it stands;
* **exactly one** position can be flipped to make the brackets balance
  and parse; and
* the flip does not pull a digit inside a bracket, where it would
  change from a ring closure into an isotope.

Every condition is checked on the string itself -- no reference
transcription and no mass value is consulted.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_mirrored_bracket_rule.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9"


def _corpora() -> list[tuple[str, list[tuple[str, str]]]]:
    with (
        REPO / "data" / "audit" / "phase8" / "patent2_vote_effect.csv"
    ).open(encoding="utf-8", newline="") as handle:
        patent2 = [
            (row["cmpd_id"], row["text_after"])
            for row in csv.DictReader(handle)
        ]
    with (
        REPO / "data" / "processed" / "compounds_combined.csv"
    ).open(encoding="utf-8-sig", newline="") as handle:
        patent1 = [
            (row["cmpd_id"], row["smiles"])
            for row in csv.DictReader(handle)
            if (row.get("smiles") or "").strip()
        ]
    return [
        ("WO2022251372", patent2),
        ("WO2025162428", patent1),
    ]


def main() -> int:
    rows: list[dict[str, object]] = []
    for name, items in _corpora():
        for key, text in items:
            repaired = auto_repair_smiles(text)
            changed = repaired != text
            rows.append({
                "patent": name,
                "cmpd_id": key,
                "reading": text,
                "repaired": repaired,
                "changed": changed,
                "bracket_surplus_before": text.count("]") - text.count("["),
            })

    out_csv = OUT / "mirrored_bracket_rule_evidence.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_csv}")

    for name in ("WO2022251372", "WO2025162428"):
        subset = [row for row in rows if row["patent"] == name]
        changed = [row for row in subset if row["changed"]]
        print()
        print(f"=== {name}")
        print(f"  readings            : {len(subset)}")
        print(f"  repaired by the rule: {len(changed)}")
        for row in changed:
            print(
                f"    cmpd {row['cmpd_id']:>4}  "
                f"bracket surplus {row['bracket_surplus_before']:+d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
