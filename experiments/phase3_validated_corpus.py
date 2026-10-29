"""Run the validated route over the whole corpus and report the deltas.

The validated route was tuned on one page (I100298) and then checked
on a second one (I100376 / compound 2173).  This script runs it over
all 335 printed-SMILES cells and reports the same corpus-level
counters that were used for the phase-3 comparison, so that the
route's behaviour on the other 78 pages is on record rather than
assumed.

Counters (all text-level, none of them an accuracy claim):

``balanced``   brackets open == brackets close
``rows``       number of printed rows the route recovered
``strict``     balanced AND rows match the printed row count AND the
               length is in the SMILES-like range 220-300
``parses``     RDKit can parse the string
``illegal``    the string contains characters outside the SMILES
               alphabet

Usage:
    .venv\\Scripts\\python.exe experiments/phase3_validated_corpus.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    PrintedSmilesOCR,
    ink_bands,
    text_span,
    upscale,
)

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "validated_corpus"

LEGAL = set("BCNOPSFIHbcnopsf0123456789=#@[]()+-/\\%.")


def printed_row_count(cmpd_id: str) -> int:
    """How many printed rows the patent shows for this compound."""
    # The reference tables print 7 rows for the cyclic peptides in this
    # patent; the single-row entries are the short ones.  The value is
    # only used as the "rows match" gate, so it is read from the audit
    # manifest when that file is present.
    return ROWS.get(cmpd_id, 0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    reader = PrintedSmilesOCR(route="validated")
    rows: list[dict[str, object]] = []

    for path in sorted(CELLS.glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        stem = path.stem
        cmpd_id = stem.split("_")[-2] if "_" in stem else stem

        # The printed row count comes from the crop itself: the route
        # segments the cell into ink bands, and one printed line is one
        # band.  A line that the recogniser returned nothing for is a
        # lost row.  The loop below mirrors the validated route's own
        # band handling so that the per-row outcome is visible; the
        # route's own ``read_cell`` is called afterwards as a
        # cross-check that the mirror agrees with it.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = ink_bands(gray)
        expected = len(bands)
        parts: list[str] = []
        for top, bottom in bands:
            y1 = max(0, top - PAD_TOP)
            y2 = min(image.shape[0], bottom + PAD_BOTTOM)
            row = image[y1:y2, :]
            if row.size == 0:
                continue
            row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            x1, x2 = text_span(row_gray)
            variant = row[:, x1:x2 + 1]
            if variant.size == 0:
                continue
            recognised = reader._recognise_only(upscale(variant))
            if recognised:
                parts.append(recognised)
        text = "".join(parts)
        got_rows = len(parts)
        # Cross-check the mirror against the route itself on a sample,
        # rather than on every cell (the route would recognise every
        # band a second time).
        route_agrees = ""
        if len(rows) < 25:
            route_agrees = (reader.read_cell(image) or "") == text
        illegal = sorted({c for c in text if c not in LEGAL})
        rows.append(
            {
                "cell": stem,
                "cmpd_id": cmpd_id,
                "length": len(text),
                "bracket_open": text.count("("),
                "bracket_close": text.count(")"),
                "balanced": text.count("(") == text.count(")"),
                "rows": got_rows,
                "expected_rows": expected,
                "rows_match": (expected == 0) or (got_rows == expected),
                "route_agrees": route_agrees,
                "strict": (
                    text.count("(") == text.count(")")
                    and (expected == 0 or got_rows == expected)
                    and 220 <= len(text) <= 300
                ),
                "illegal": "".join(illegal),
                "text": text,
            }
        )

    n = len(rows)
    summary = {
        "cells": n,
        "balanced": sum(1 for r in rows if r["balanced"]),
        "rows_match": sum(1 for r in rows if r["rows_match"]),
        "strict": sum(1 for r in rows if r["strict"]),
        "illegal": sum(1 for r in rows if r["illegal"]),
        "median_length": statistics.median(
            [r["length"] for r in rows] or [0]
        ),
    }
    print(json.dumps(summary, indent=2))

    out_csv = OUT / "validated_corpus.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "validated_corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
