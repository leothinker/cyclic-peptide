"""Run the mass-constrained OCR-repair search across the corpus.

For each row in compounds_combined.csv that has an OCR SMILES and a
usable m/z, run :func:`cpd.mass_constrained_search.repair_from_ocr`
with a tight per-row budget, and record the result in five new
columns independent of the existing best-SMILES pipeline:

* ``smiles_ocr_repaired``     -- the search's best SMILES, or empty
* ``ocr_repair_trace``        -- one-line summary: ok / no_answer / failed_X
* ``ocr_repair_formula``      -- formula if it parses
* ``ocr_repair_largest_ring`` -- largest ring size if it parses
* ``ocr_repair_mass_ppm``     -- signed ppm error vs the patent's window

The search itself lives in :mod:`cpd.ocr_search`; this script is the
thin wrapper that reads the combined CSV, runs it and writes the
result back.

Run from the repo root::

    .venv/Scripts/python.exe analysis/run_ocr_repair_corpus.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpd.ocr_repair_costs import expansion_cost  # noqa: F401  (warm import)
from cpd.ocr_search import SEARCH_COLUMNS, run_search
from cpd.postprocess import apply_derived_columns

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

OUT_COLS = list(SEARCH_COLUMNS) + [
    "ocr_bracket_deficit",
    "ocr_difficulty",
]


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())
    for c in OUT_COLS:
        if c not in fieldnames:
            fieldnames.append(c)

    n_total = len(rows)
    n_with_ocr = sum(1 for r in rows if (r.get("smiles") or "").strip())
    rows = run_search(rows)
    # The bracket-deficit columns are a property of the text, not of the
    # search, so they are recomputed from the text rather than copied.
    rows = apply_derived_columns(rows)
    n_ok = sum(
        1 for r in rows if (r.get("ocr_repair_trace") or "").startswith("ok ")
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {n_total} rows; {n_ok}/{n_with_ocr} produced an "
          f"OCR-repaired SMILES.", flush=True)


if __name__ == "__main__":
    main()
