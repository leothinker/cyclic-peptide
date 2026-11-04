"""Compare the WO2022251372 readings with the original and the corrected
reference.

The hand transcriptions in ``wo2022251372_references.py`` were checked
against the printed pixels one character at a time, and three of them
turned out to be transcription mistakes rather than reading errors:

``compound 10``
    the reference read ``[C@@@H]`` where the patent prints
    ``[C@@H]`` -- three chiral markers where the page has two.  The
    printed text is what the image shows: two circled ``a`` glyphs.

``compound 28``
    the reference read ``[C@@H]`` on line 3 where the patent prints
    ``[C@H]``, and it read ``NC(C`` on line 5 where the patent prints
    ``N(C``.  Both were checked against the magnified pixels.

Those mistakes were in the *ground truth*, so every number derived
from it was wrong in the pipeline's disfavour.  The references have
since been corrected -- the ``CORRECTIONS`` table in
``wo2022251372_references.py`` records each one with its evidence.

This script reproduces both sides of that correction, so the size of
the correction stays visible rather than asserted: the readings are
compared against the original transcription and against the corrected
one.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_corrected_references.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from wo2022251372_references import (  # noqa: E402
    CORRECTIONS,
    REFERENCE_LINES,
    edit_distance,
)

OUT = REPO / "data" / "audit" / "phase9"

def _original() -> dict[int, tuple[str, ...]]:
    """The references as they were transcribed before the correction."""
    result: dict[int, tuple[str, ...]] = {}
    for compound_id, lines in REFERENCE_LINES.items():
        fixed = list(lines)
        for correction in CORRECTIONS:
            if correction["compound"] != compound_id:
                continue
            index = int(correction["line"])
            if fixed[index] == correction["now"]:
                fixed[index] = correction["was"]
        result[compound_id] = tuple(fixed)
    return result


def main() -> int:
    original = _original()
    with (
        REPO / "data" / "audit" / "phase8"
        / "patent2_vote_effect.csv"
    ).open(encoding="utf-8", newline="") as handle:
        readings = {
            row["cmpd_id"]: row["text_after"]
            for row in csv.DictReader(handle)
        }

    rows: list[dict[str, object]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        original_text = "".join(original[compound_id])
        corrected_text = "".join(lines)
        read = readings.get(str(compound_id), "")
        rows.append({
            "cmpd_id": compound_id,
            "reference_original": original_text,
            "reference_corrected": corrected_text,
            "reading": read,
            "edits_against_original": edit_distance(original_text, read),
            "edits_against_corrected": edit_distance(corrected_text, read),
            "exact_against_original": original_text == read,
            "exact_against_corrected": corrected_text == read,
        })

    out_csv = OUT / "patent2_corrected_reference_comparison.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_csv}")

    old_edits = sum(int(row["edits_against_original"]) for row in rows)
    new_edits = sum(int(row["edits_against_corrected"]) for row in rows)
    old_exact = sum(1 for row in rows if row["exact_against_original"])
    new_exact = sum(1 for row in rows if row["exact_against_corrected"])
    total_chars = sum(len(str(row["reference_corrected"])) for row in rows)

    print()
    print(f"{'cmpd':>5}  {'old':>4}  {'new':>4}   exact(old/new)")
    for row in rows:
        print(
            f"{row['cmpd_id']:>5}  "
            f"{row['edits_against_original']:>4}  "
            f"{row['edits_against_corrected']:>4}   "
            f"{'yes' if row['exact_against_original'] else 'no '}/"
            f"{'yes' if row['exact_against_corrected'] else 'no'}"
        )
    print()
    print(f"edits: {old_edits} -> {new_edits}")
    print(f"exact: {old_exact}/{len(rows)} -> {new_exact}/{len(rows)}")
    print(
        f"CER   : {100.0 * old_edits / total_chars:.3f}% -> "
        f"{100.0 * new_edits / total_chars:.3f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
