"""External check: compound 2173, a compound the pipeline never read.

Compound 2173 is the most interesting compound in the shipped CSV.
Its row carries a printed SMILES on patent page I100376 (PDF page 240),
but the pipeline's OCR output for it is the *broken* production-route
string -- every bracket of every ``C(=O)`` is missing -- and
``smiles_best`` is empty, so the compound contributes no usable
structure to the dataset at all.

2173 is therefore a free generalisation test:

* it is a page the pipeline *has* seen, but a compound it never
  managed to read, so the failure is not a layout question;
* its printed SMILES is long (9 lines) and runs to the right-hand
  table rule, which is exactly the geometry that was fixed in
  ``_widen_text_column_to_border``;
* its DECIMER result is available in the CSV, so the OCR output can
  be compared against an independent structure source.

The script runs both OCR routes on the same crop and reports:

* the character-level difference between the two routes;
* whether each route's string parses with RDKit;
* whether the routes agree with the DECIMER string in the CSV.

This is a *text and parse* comparison.  It says nothing about whether
the recovered structure is the right chemical structure -- that would
need a human check against the drawing.
"""

from __future__ import annotations

import csv
import difflib
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "external"

TARGETS = {
    "2173": "I100376_2173_text",
    "1036": "I100297_1036_text",
    "1037": "I100297_1037_text",
    "1038": "I100298_1038_text",
}


def csv_row(cmpd_id: str) -> dict[str, str]:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("cmpd_id", "")).strip() == cmpd_id:
                return row
    return {}


def rdkit_ok(text: str) -> bool:
    if not text:
        return False
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(text) is not None
    except Exception:  # pragma: no cover - RDKit is optional here
        return None


def bracket_balance(text: str) -> tuple[int, int]:
    return text.count("("), text.count(")")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    production = PrintedSmilesOCR(route="production")
    validated = PrintedSmilesOCR(route="validated")

    for cmpd_id, stem in TARGETS.items():
        path = CELLS / f"{stem}.png"
        image = cv2.imread(str(path))
        if image is None:
            print(f"{cmpd_id}: no crop at {path}")
            continue

        prod = production.read_cell(image) or ""
        val = validated.read_cell(image) or ""
        row = csv_row(cmpd_id)

        decimer = (row.get("smiles_decimer") or "").strip()
        shipped = (row.get("smiles") or "").strip()

        rows.append(
            {
                "cmpd_id": cmpd_id,
                "crop": path.name,
                "shipped_smiles_len": len(shipped),
                "production_len": len(prod),
                "validated_len": len(val),
                "validated_matches_shipped": val == shipped,
                "production_matches_shipped": prod == shipped,
                "validated_bracket_open": bracket_balance(val)[0],
                "validated_bracket_close": bracket_balance(val)[1],
                "production_bracket_open": bracket_balance(prod)[0],
                "production_bracket_close": bracket_balance(prod)[1],
                "validated_rdkit": rdkit_ok(val),
                "production_rdkit": rdkit_ok(prod),
                "decimer_rdkit": rdkit_ok(decimer),
                "decimer_len": len(decimer),
                "validated_vs_decimer_sim": round(
                    difflib.SequenceMatcher(None, val, decimer).ratio(), 4
                ),
                "shipped_vs_decimer_sim": round(
                    difflib.SequenceMatcher(None, shipped, decimer).ratio(), 4
                ),
            }
        )

        print(f"--- compound {cmpd_id} ({path.name}) ---")
        print(f"  shipped CSV smiles : {len(shipped):4d} chars")
        print(f"  production route   : {len(prod):4d} chars")
        print(f"  validated route    : {len(val):4d} chars")
        print(f"  validated == shipped: {val == shipped}")
        print(f"  decimer (independent): {len(decimer):4d} chars")
        print(f"  validated vs decimer similarity: {rows[-1]['validated_vs_decimer_sim']}")
        print(f"  shipped  vs decimer similarity: {rows[-1]['shipped_vs_decimer_sim']}")
        print(f"  validated: {val}")
        print()

    out_csv = OUT / "external_2173_routes.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
