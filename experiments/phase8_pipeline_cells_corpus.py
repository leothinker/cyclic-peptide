"""Compare the printed-SMILES routes on the cells the *pipeline* reads.

``phase8_corpus_route_compare.py`` compares the routes on the cell
crops of ``data/audit/phase2/cells_text``.  Those crops were cut by
an earlier version of the extractor, so they are not the cells the
pipeline feeds to the recogniser.

This script runs the **real extractor** -- the same column detection,
the same row boundaries, the same right-hand-rule widening and the
same handling of a printed row that continues over the bottom of a
page -- and compares the routes on what it produces.

How the cells are captured
--------------------------
The extractor does not expose the SMILES cell it hands to the
recogniser, so this script subclasses it and overrides
``_ocr_smiles_with_stats``.  That method receives exactly the crop the
route receives, before any route-specific processing.  The subclass
records the crop and returns an empty reading, which keeps the
extraction pass free of OCR: the two routes are then run on the
captured crops afterwards, on identical pixels.

Both routes under test (``validated`` and ``validated_vote``) are
ink-band routes, and the ink-band routes share their crop geometry --
the right-hand-rule widening in ``_widen_last_column`` applies to all
of them.  So one extraction pass yields cells that are valid for both.

The continuation matters more than it looks.  A printed SMILES that
runs over the bottom of a page is read as *two* cells, and the
pipeline appends the second one to the first.  A measurement that
cuts only the first cell sees a truncated molecule, which is why an
earlier version of this script reported 29 well-formed readings where
the deliverable has 170.

The reading is post-processed with ``auto_repair_smiles`` exactly as
the pipeline does, so the numbers here line up with the numbers in
the shipped CSV.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_pipeline_cells_corpus.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.chem import auto_repair_smiles  # noqa: E402
from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402
from cpd.ocr.vote import SMILES_ALPHABET  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import (  # noqa: E402
    OcrStats,
    RapidOcrTableExtractor,
)
from phase7_route_verify import REFERENCES, edit_distance  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT = REPO / "data" / "audit" / "phase8"
IMAGE_DIR = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"

LEGAL = set(SMILES_ALPHABET)
ROUTES = ("validated", "validated_vote")


def _parses(text: str) -> bool:
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


class _CellCapturingExtractor(RapidOcrTableExtractor):
    """An extractor that records the SMILES cell it would recognise.

    ``_ocr_smiles_with_stats`` is the single point where the pipeline
    hands a SMILES crop to the recogniser, so overriding it captures
    exactly the right pixels.  The method returns an empty reading:
    the routes are run on the captured crops afterwards, which keeps
    the extraction pass free of OCR and guarantees that both routes
    see identical pixels.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cells: dict[str, list[np.ndarray]] = {}
        self._current_id: str | None = None

    def _read_cmpd_id(self, cell: np.ndarray) -> str | None:
        result = super()._read_cmpd_id(cell)
        if result:
            # A continuation band carries no number, so the id of the
            # compound that is still open stays current -- which is
            # exactly the compound the band belongs to.
            self._current_id = result
        return result

    def _ocr_smiles_with_stats(self, cell: np.ndarray):
        if self._current_id and cell.size:
            self.cells.setdefault(self._current_id, []).append(cell.copy())
        return None, OcrStats()


def _capture_cells() -> dict[str, list[np.ndarray]]:
    """Cut the printed-SMILES cell of every compound, as the pipeline does."""
    profile = get_profile("WO2025162428")
    extractor = _CellCapturingExtractor(
        rec_top_k=0, smiles_route="validated", profile=profile,
    )
    prev_columns: list = []
    for path in sorted(IMAGE_DIR.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        _records, _assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
        print(
            f"  {path.name}: {len(extractor.cells)} compounds so far",
            flush=True,
        )
    extractor.flush_pending()
    return extractor.cells


def _read_route(cells: dict[str, list[np.ndarray]], route: str) -> dict[str, str]:
    """Read every captured cell with one route; join continuation bands."""
    reader = PrintedSmilesOCR(route=route)
    texts: dict[str, str] = {}
    for compound_id, bands in cells.items():
        parts = [reader.read_cell(band) or "" for band in bands]
        texts[compound_id] = "".join(parts)
    return texts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("cutting the pipeline's SMILES cells ...")
    cells = _capture_cells()
    print(
        f"{len(cells)} compounds, "
        f"{sum(len(b) for b in cells.values())} cells"
    )

    readings = {route: _read_route(cells, route) for route in ROUTES}
    rows: list[dict[str, object]] = []
    summary: dict[str, dict] = {}

    for route in ROUTES:
        texts = readings[route]
        repaired = {
            compound_id: auto_repair_smiles(text)
            for compound_id, text in texts.items()
        }
        parses = sum(1 for t in repaired.values() if _parses(t))
        balanced = sum(
            1 for t in repaired.values()
            if t.count("(") == t.count(")") and t.count("[") == t.count("]")
        )
        illegal = sum(1 for t in repaired.values() if set(t) - LEGAL)
        edits = 0
        exact = 0
        for compound_id, reference in REFERENCES.items():
            distance = edit_distance(reference, repaired.get(compound_id, ""))
            edits += distance
            exact += distance == 0
        summary[route] = {
            "compounds": len(repaired),
            "bracket_balanced": balanced,
            "rdkit_parses": parses,
            "cells_with_illegal_characters": illegal,
            "references_exact": exact,
            "references_total": len(REFERENCES),
            "reference_edits": edits,
        }
        print(
            f"{route}: balanced {balanced}/{len(repaired)}, "
            f"parses {parses}/{len(repaired)}, illegal {illegal}, "
            f"references {exact}/{len(REFERENCES)} exact ({edits} edits)"
        )
        for compound_id, text in texts.items():
            fixed = repaired[compound_id]
            rows.append(
                {
                    "route": route,
                    "cmpd_id": compound_id,
                    "cells": len(cells[compound_id]),
                    "length": len(fixed),
                    "balanced": (
                        fixed.count("(") == fixed.count(")")
                        and fixed.count("[") == fixed.count("]")
                    ),
                    "parses": _parses(fixed),
                    "illegal": "".join(sorted(set(fixed) - LEGAL)),
                    "edit_vs_reference": (
                        edit_distance(REFERENCES[compound_id], fixed)
                        if compound_id in REFERENCES
                        else ""
                    ),
                    "text_raw": text,
                    "text": fixed,
                }
            )

    # --- what the vote changed ---------------------------------------
    primary, vote = readings["validated"], readings["validated_vote"]
    changed = [
        compound_id for compound_id in primary
        if auto_repair_smiles(primary[compound_id])
        != auto_repair_smiles(vote[compound_id])
    ]
    broke = [
        compound_id for compound_id in primary
        if _parses(auto_repair_smiles(primary[compound_id]))
        and not _parses(auto_repair_smiles(vote[compound_id]))
    ]
    became_parseable = [
        compound_id for compound_id in primary
        if not _parses(auto_repair_smiles(primary[compound_id]))
        and _parses(auto_repair_smiles(vote[compound_id]))
    ]
    summary["vote_effect"] = {
        "changed": len(changed),
        "broke_a_parsing_reading": len(broke),
        "became_parseable": len(became_parseable),
        "changed_cells": changed,
        "became_parseable_cells": became_parseable,
    }
    print(
        f"the vote changed {len(changed)} strings, "
        f"{len(became_parseable)} became parseable, "
        f"{len(broke)} broke a parsing reading"
    )

    out_csv = OUT / "pipeline_cells_route_compare.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "pipeline_cells_route_compare.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
