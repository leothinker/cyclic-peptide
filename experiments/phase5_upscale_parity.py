"""Does the shipped ``padded`` route match the experiment it came from?

The 6 px margin was measured in ``phase4_wo2022251372_candidate_config``
and ``..._padding_matrix``, both of which pad first and then resize
2x.  ``..._repair_ladder`` pads but does *not* resize.

This script runs all three readings over the same 28 Table 1a cells
and reports, per compound, whether they produce the same string and
the same RDKit verdict.  It exists so the headline numbers in the
write-up can be traced to the code that actually ships.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_upscale_parity.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PADDED_MARGIN_PX,
    WHITE,
    PrintedSmilesOCR,
    ink_bands,
    text_span,
    upscale,
)
from cpd.ocr.printed_smiles import english_recogniser_only  # noqa: E402

from wo2022251372_references import REFERENCE_LINES, edit_distance  # noqa: E402

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
OUT = REPO / "data" / "audit" / "phase5" / "upscale_parity"

PAD_TOP = 14
PAD_BOTTOM = 15


def _read_no_upscale(cell, engine) -> str:
    """The ``repair_ladder`` reading: pad, but do not resize."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        band = row[:, x1:x2 + 1]
        if band.size == 0:
            continue
        band = cv2.copyMakeBorder(
            band,
            0,
            0,
            PADDED_MARGIN_PX,
            PADDED_MARGIN_PX,
            cv2.BORDER_CONSTANT,
            value=WHITE,
        )
        result = engine(band, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        text = (txts[0] or "").strip() if txts else ""
        if text:
            parts.append(text)
    return "".join(parts)


def _read_pad_then_upscale(cell, engine) -> str:
    """The ``candidate_config`` reading: pad, then resize 2x."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        band = row[:, x1:x2 + 1]
        if band.size == 0:
            continue
        band = cv2.copyMakeBorder(
            band,
            0,
            0,
            PADDED_MARGIN_PX,
            PADDED_MARGIN_PX,
            cv2.BORDER_CONSTANT,
            value=WHITE,
        )
        scaled = cv2.resize(
            band,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        result = engine(scaled, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        text = (txts[0] or "").strip() if txts else ""
        if text:
            parts.append(text)
    return "".join(parts)


def _parses(text: str) -> bool:
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engine = english_recogniser_only()
    shipped = PrintedSmilesOCR(route="padded")

    rows = []
    for path in sorted(CELLS.glob("*.png")):
        compound_id = int(path.stem.split("cmpd")[-1])
        cell = cv2.imread(str(path))
        reference = "".join(REFERENCE_LINES.get(compound_id, ()))

        readings = {
            "shipped_route": shipped.read_cell(cell) or "",
            "pad_then_upscale": _read_pad_then_upscale(cell, engine),
            "pad_no_upscale": _read_no_upscale(cell, engine),
        }
        row = {"compound_id": compound_id, "has_reference": bool(reference)}
        for name, text in readings.items():
            row[f"{name}_parses"] = _parses(text)
            row[f"{name}_edit"] = (
                edit_distance(reference, text) if reference else ""
            )
        row["shipped_matches_pad_then_upscale"] = (
            readings["shipped_route"] == readings["pad_then_upscale"]
        )
        row["shipped_matches_pad_no_upscale"] = (
            readings["shipped_route"] == readings["pad_no_upscale"]
        )
        rows.append(row)

    with (OUT / "upscale_parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cells": len(rows),
        "shipped_parses": sum(1 for r in rows if r["shipped_route_parses"]),
        "pad_then_upscale_parses": sum(
            1 for r in rows if r["pad_then_upscale_parses"]
        ),
        "pad_no_upscale_parses": sum(
            1 for r in rows if r["pad_no_upscale_parses"]
        ),
        "shipped_identical_to_pad_then_upscale": sum(
            1 for r in rows if r["shipped_matches_pad_then_upscale"]
        ),
        "shipped_identical_to_pad_no_upscale": sum(
            1 for r in rows if r["shipped_matches_pad_no_upscale"]
        ),
    }
    with_ref = [r for r in rows if r["has_reference"]]
    summary["shipped_edits"] = [r["shipped_route_edit"] for r in with_ref]
    summary["pad_then_upscale_edits"] = [
        r["pad_then_upscale_edit"] for r in with_ref
    ]
    summary["pad_no_upscale_edits"] = [
        r["pad_no_upscale_edit"] for r in with_ref
    ]

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
