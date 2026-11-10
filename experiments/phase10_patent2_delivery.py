"""Produce the delivery CSV of WO2022251372.

WO2025162428 ships ``data/processed/compounds_combined.csv``.  The
second patent needs the same kind of artefact -- one CSV that carries
the printed SMILES *and* the activity data of every compound -- but
its activity data is six letter grades rather than five numbers.

Before this script the second patent only existed as an extraction
CSV inside ``data/audit/phase6/``, and that file had been written by
the older ``padded`` route.  This script runs the *same* pipeline the
first patent uses, with the second patent's profile, and writes:

* ``data/processed/WO2022251372/compounds_combined.csv`` -- the
  deliverable, with the six grade columns after the canonical block;
* ``data/audit/phase10/patent2_delivery_summary.json`` -- the counts
  and the reference-based accuracy, recomputed from this run.

DECIMER stays shelved: the printed text is the primary source, so the
structure columns are left empty rather than filled with a second
opinion that has never been checked against the drawing.

Usage::

    .venv\\Scripts\\python.exe experiments/phase10_patent2_delivery.py
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

import main as pipeline  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.schema import combined_columns  # noqa: E402
from wo2022251372_references import (  # noqa: E402
    REFERENCE_LINES,
    edit_distance,
)

RDLogger.DisableLog("rdApp.*")

PATENT = "WO2022251372"
OUT_DIR = REPO / "data" / "processed" / PATENT
OUT_CSV = OUT_DIR / "compounds_combined.csv"
AUDIT_DIR = REPO / "data" / "audit" / "phase10"


def _parses(text: str) -> bool:
    """Whether RDKit can parse the string.  Not a structure check."""
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    """Whether the brackets of the string are balanced."""
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def main() -> int:
    profile = get_profile(PATENT)
    route = profile.smiles_route
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # The deliverable is produced by the *standard* pipeline, not by a
    # parallel copy of it.  That is the point of this script: the same
    # `main.py` that serves the first patent produces the second
    # patent's CSV, and the only thing that differs is the profile.
    print(f"[1/3] running the standard pipeline for {PATENT} "
          f"(route={route}) ...")
    pipeline.run_pipeline(
        img_dir=str(REPO / "data" / "raw" / PATENT / "golden_tables"),
        out_dir=str(OUT_DIR),
        skip_decimer=True,
        rec_top_k=0,
        smiles_route=route,
        profile_name=PATENT,
    )

    print("[2/3] reading the deliverable back ...")
    columns = combined_columns(profile.extra_assay_fields)
    with OUT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n = len(rows)
    if list(rows[0].keys()) != list(columns):
        raise SystemExit(
            f"{OUT_CSV} has the wrong header; expected {len(columns)} "
            f"columns, found {len(rows[0])}"
        )

    # ---- the numbers, recomputed from this very run ----------------
    print("[3/3] checking the result ...")
    by_id = {row["cmpd_id"]: row for row in rows}
    printed = [
        (cid, row.get("smiles") or "") for cid, row in by_id.items()
    ]
    n_printed = sum(1 for _, text in printed if text.strip())
    n_parse = sum(1 for _, text in printed if _parses(text))
    n_balanced = sum(1 for _, text in printed if _balanced(text))

    exact = 0
    edits = 0
    chars = 0
    reference_rows = []
    for compound_id, lines in sorted(REFERENCE_LINES.items()):
        reference = "".join(lines)
        text = by_id.get(str(compound_id), {}).get("smiles") or ""
        distance = edit_distance(reference, text)
        edits += distance
        chars += len(reference)
        exact += reference == text
        reference_rows.append({
            "cmpd_id": compound_id,
            "reference_chars": len(reference),
            "read_chars": len(text),
            "edit_distance": distance,
            "exact": reference == text,
            "parses": _parses(text),
        })

    # The six activity grades must be readable, not guessed.  A grade
    # outside the closed set A-D would be a misread.
    grade_columns = list(profile.extra_assay_fields)
    grade_cells = 0
    grade_invalid = 0
    for row in rows:
        for name in grade_columns:
            value = (row.get(name) or "").strip()
            if not value:
                continue
            grade_cells += 1
            grade_invalid += value not in ("A", "B", "C", "D")

    summary = {
        "patent": PATENT,
        "profile": PATENT,
        "route": route,
        "rows": n,
        "columns": len(columns),
        "extra_columns": grade_columns,
        "printed_smiles": n_printed,
        "rdkit_parseable": n_parse,
        "bracket_balanced": n_balanced,
        "hand_references": {
            "compounds": len(REFERENCE_LINES),
            "exact": exact,
            "edit_distance": edits,
            "chars": chars,
            "cer_percent": round(100.0 * edits / max(chars, 1), 4),
        },
        "activity_grades": {
            "columns": len(grade_columns),
            "non_empty_cells": grade_cells,
            "values_outside_A_D": grade_invalid,
        },
        "decimer": "shelved: the structure columns are empty on purpose",
    }
    (AUDIT_DIR / "patent2_delivery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    with (AUDIT_DIR / "patent2_delivery_references.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(reference_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(reference_rows)

    print()
    print(f"rows                : {n}")
    print(f"columns             : {len(columns)} "
          f"({len(grade_columns)} grade columns added)")
    print(f"printed SMILES      : {n_printed}")
    print(f"RDKit-parseable     : {n_parse}")
    print(f"bracket balanced    : {n_balanced}")
    print(f"hand references     : {exact}/{len(REFERENCE_LINES)} exact, "
          f"{edits} edits, CER {summary['hand_references']['cer_percent']}%")
    print(f"grade cells         : {grade_cells} filled, "
          f"{grade_invalid} outside A-D")
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

