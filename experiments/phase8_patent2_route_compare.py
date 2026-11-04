"""Compare the printed-SMILES routes of WO2022251372 end to end.

The ``padded`` route is what the WO2022251372 profile uses.  The
``padded_vote`` route reads every printed line with a second
recogniser as well and lets the two readings settle the characters
they disagree on.

This script runs the *real* pipeline -- the same extractor, the same
repair chain and the same merge code -- with each route, and reports:

* the character accuracy against the six hand transcriptions; and
* the reference-free measures over all 28 compounds of Table 1a
  (bracket balance and RDKit parseability).

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_patent2_route_compare.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.merge import merge_compound_records  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import (  # noqa: E402
    RapidOcrTableExtractor,
)
from wo2022251372_references import (  # noqa: E402
    REFERENCE_LINES,
    edit_distance,
)

RDLogger.DisableLog("rdApp.*")

OUT = REPO / "data" / "audit" / "phase8"
IMAGE_DIR = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"

ROUTES = ("padded", "padded_vote")


def _parses(text: str) -> bool:
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def _extract(route: str) -> dict:
    """Run the pipeline with one route and return the SMILES per id."""
    profile = get_profile("WO2022251372")
    extractor = RapidOcrTableExtractor(
        rec_top_k=0,
        smiles_route=route,
        profile=profile,
    )
    cells_dir = OUT / "cells_patent2"
    cells_dir.mkdir(parents=True, exist_ok=True)

    compounds = []
    prev_columns = []
    for path in sorted(IMAGE_DIR.iterdir()):
        page_compounds, _assays, columns = extractor.extract_records(
            path,
            cells_dir=cells_dir,
            prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
        compounds.extend(page_compounds)
    compounds.extend(extractor.flush_pending())

    merged = merge_compound_records(compounds, prefer_structure=True)
    return {
        cmpd_id: (record.smiles or "")
        for cmpd_id, record in merged.items()
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    corpus_rows = []
    corpus = {}
    readings: dict[str, dict[str, str]] = {}

    for route in ROUTES:
        found = _extract(route)
        print(f"\n=== {route}: {len(found)} compounds ===")
        readings[route] = found

        exact = 0
        edits = 0
        length = 0
        for compound_id, lines in REFERENCE_LINES.items():
            reference = "".join(lines)
            text = found.get(str(compound_id), "")
            distance = edit_distance(reference, text)
            edits += distance
            length += len(reference)
            if reference == text:
                exact += 1
            rows.append({
                "route": route,
                "cmpd_id": compound_id,
                "reference_length": len(reference),
                "read_length": len(text),
                "edit_distance": distance,
                "exact": reference == text,
                "parses": _parses(text),
                "balanced": _balanced(text),
            })

        parses = sum(1 for text in found.values() if _parses(text))
        balanced = sum(1 for text in found.values() if _balanced(text))
        print(
            f"  transcribed: {exact}/{len(REFERENCE_LINES)} exact, "
            f"{edits} edits, CER {100.0 * edits / max(length, 1):.3f}%"
        )
        print(
            f"  all compounds: {parses}/{len(found)} parse, "
            f"{balanced}/{len(found)} bracket balanced"
        )
        corpus[route] = {
            "compounds": len(found),
            "parses": parses,
            "bracket_balanced": balanced,
            "transcribed_exact": exact,
            "transcribed_total": len(REFERENCE_LINES),
            "transcribed_edits": edits,
            "transcribed_chars": length,
            "transcribed_cer_percent": round(
                100.0 * edits / max(length, 1), 4
            ),
        }

    # --- what the vote changed, compound by compound ------------------
    # The rule keeps the primary reading whenever it parses, so a
    # changed string must be a string that did *not* parse before.
    primary_route, vote_route = ROUTES[0], ROUTES[-1]
    changed = 0
    changed_but_parsed_before = 0
    for compound_id, text in readings[vote_route].items():
        before = readings[primary_route].get(compound_id, "")
        is_changed = before != text
        changed += is_changed
        changed_but_parsed_before += is_changed and _parses(before)
        corpus_rows.append(
            {
                "cmpd_id": compound_id,
                "text_unchanged": not is_changed,
                "parsed_before": _parses(before),
                "parses_after": _parses(text),
                "length_before": len(before),
                "length_after": len(text),
                # The two readings, so that a changed cell can be
                # read rather than merely counted.
                "text_before": before,
                "text_after": text,
            }
        )
    print(
        f"\n=== the vote changed {changed} of {len(corpus_rows)} strings; "
        f"{changed_but_parsed_before} of them parsed before ==="
    )

    out_csv = OUT / "patent2_route_compare.csv"
    fields = [
        "route", "cmpd_id", "reference_length", "read_length",
        "edit_distance", "exact", "parses", "balanced",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_csv}")

    corpus_csv = OUT / "patent2_vote_effect.csv"
    with corpus_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(corpus_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(corpus_rows)
    print(f"-> {corpus_csv}")

    (OUT / "patent2_route_compare.json").write_text(
        json.dumps(
            {
                "routes": corpus,
                "vote_effect": {
                    "compounds": len(corpus_rows),
                    "changed": changed,
                    "changed_but_parsed_before": changed_but_parsed_before,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
