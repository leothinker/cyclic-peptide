"""Does the bond guard still earn its place, on the corrected cells?

The claim in the docs -- "without the guard, a vote turns a double
bond into a single bond in 6 of the 17 cells it changes" -- was
measured on the flawed corpus, the one that cut only the first band
of a printed row.  This script re-measures it on the cells the
pipeline actually cuts.

It runs the vote twice on the same lines: once with the bond guard
on (the shipped behaviour) and once with it off.  A cell "loses a
bond" when the unguarded vote declares fewer double or triple bonds
than the primary reading did.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_bond_guard_evidence.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402
from cpd.ocr.vote import _bond_count, vote_lines  # noqa: E402
from phase8_pipeline_cells_corpus import _capture_cells  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9"


def main() -> int:
    print("cutting the pipeline's SMILES cells ...")
    cells = _capture_cells()
    print(f"{len(cells)} compounds")

    reader = PrintedSmilesOCR(route="validated_vote")
    lines: list[dict[str, object]] = []
    unguarded_lost = 0
    guarded_lost = 0
    unguarded_changed = 0
    guarded_changed = 0

    for compound_id, bands in sorted(cells.items()):
        for band in bands:
            primary = reader._read_cell_ink_bands(
                band, margin=0, scale=2, engine=reader.engine, as_lines=True,
            )
            secondary = reader._read_cell_ink_bands(
                band, margin=0, scale=2, engine=reader.second_engine,
                as_lines=True,
            )
            if not primary:
                continue
            before = _bond_count("".join(primary))
            guarded = vote_lines(primary, secondary, keep_bonds=True)
            unguarded = vote_lines(primary, secondary, keep_bonds=False)
            guarded_bonds = _bond_count(guarded.text)
            unguarded_bonds = _bond_count(unguarded.text)
            guarded_changed += guarded.text != "".join(primary)
            unguarded_changed += unguarded.text != "".join(primary)
            guarded_lost += guarded_bonds < before
            unguarded_lost += unguarded_bonds < before
            if guarded.text != unguarded.text:
                lines.append(
                    {
                        "cmpd_id": compound_id,
                        "primary_bonds": before,
                        "guarded_bonds": guarded_bonds,
                        "unguarded_bonds": unguarded_bonds,
                        "unguarded_lost_a_bond": unguarded_bonds < before,
                        "guarded_lost_a_bond": guarded_bonds < before,
                    }
                )

    summary = {
        "cells_with_two_readings": len(lines),
        "guarded_vote_changed": guarded_changed,
        "unguarded_vote_changed": unguarded_changed,
        "guarded_lost_a_bond": guarded_lost,
        "unguarded_lost_a_bond": unguarded_lost,
        "cells_where_the_guard_matters": lines,
    }
    print(
        f"guarded: {guarded_changed} changed, {guarded_lost} lost a bond; "
        f"unguarded: {unguarded_changed} changed, {unguarded_lost} lost a bond"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bond_guard_evidence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (OUT / "bond_guard_evidence.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "cmpd_id", "primary_bonds", "guarded_bonds", "unguarded_bonds",
            "unguarded_lost_a_bond", "guarded_lost_a_bond",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    print(f"-> {OUT / 'bond_guard_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
