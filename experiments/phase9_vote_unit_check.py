"""How many changes does the vote make -- bands or compounds?

The route comparison counts the compounds whose *text* changed.  The
bond-guard measurement counts the printed *lines* that changed.  A
compound is printed over several lines, so the two numbers are not
the same thing and the documents must not mix them.

This reconciles them: for every compound it counts the lines the
guarded vote changes, and reports the two totals side by side.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_vote_unit_check.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402
from cpd.ocr.vote import vote_lines  # noqa: E402
from phase8_pipeline_cells_corpus import _capture_cells  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9"


def main() -> int:
    print("cutting the pipeline's SMILES cells ...")
    cells = _capture_cells()
    reader = PrintedSmilesOCR(route="validated_vote")

    changed_compounds = 0
    changed_bands = 0
    per_compound = []
    for compound_id, bands in sorted(cells.items()):
        compound_changed = False
        compound_bands = 0
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
            result = vote_lines(primary, secondary, keep_bonds=True)
            if result.text != "".join(primary):
                compound_changed = True
                compound_bands += 1
        if compound_changed:
            changed_compounds += 1
            changed_bands += compound_bands
            per_compound.append(
                {"cmpd_id": compound_id, "lines_changed": compound_bands}
            )

    summary = {
        "compounds": len(cells),
        "compounds_whose_text_changed": changed_compounds,
        "printed_lines_changed": changed_bands,
        "per_compound": per_compound,
    }
    print(
        f"{len(cells)} compounds: the vote changes the text of "
        f"{changed_compounds} of them, over {changed_bands} printed lines"
    )
    for entry in per_compound:
        print(f"  {entry['cmpd_id']}: {entry['lines_changed']} line(s)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vote_unit_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
