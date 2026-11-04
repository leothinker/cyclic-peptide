"""Authoritative statistics of the delivered CSV.

This is the script to quote from.  It reads the shipped
``data/processed/compounds_combined.csv`` and counts, from the file
itself:

* how many printed-SMILES readings exist,
* how many of them RDKit accepts,
* how many are bracket-balanced,
* how many use a character outside the SMILES alphabet,
* and the same numbers for the archived previous build, so the two
  can be compared without re-running the pipeline.

Every number in the reports comes from here.  The earlier corpus
scripts counted only the *first band* of a printed row, so they saw
truncated molecules; this script counts what was actually delivered.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_csv_stats.py
    .venv\\Scripts\\python.exe experiments/phase9_csv_stats.py \\
        --csv data/processed/archive/compounds_combined_validated_route.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.vote import SMILES_ALPHABET  # noqa: E402

RDLogger.DisableLog("rdApp.*")

LEGAL = set(SMILES_ALPHABET)


def _parses(text: str) -> bool:
    if not text:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _balanced(text: str) -> bool:
    return text.count("(") == text.count(")") and text.count("[") == text.count("]")


def stats(csv_path: Path) -> dict[str, object]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    printed = [r for r in rows if (r.get("smiles") or "").strip()]
    texts = [r["smiles"] for r in printed]
    best = [r for r in rows if (r.get("smiles_best") or "").strip()]
    best_texts = [r["smiles_best"] for r in best]

    illegal = Counter()
    for text in texts:
        for char in set(text) - LEGAL:
            illegal[char] += 1

    return {
        "csv": str(csv_path.relative_to(REPO)) if csv_path.is_relative_to(REPO)
        else str(csv_path),
        "rows": len(rows),
        "columns": len(rows[0]) if rows else 0,
        "unique_cmpd_id": len({r["cmpd_id"] for r in rows}),
        "printed_readings": len(texts),
        "printed_rdkit_parses": sum(1 for t in texts if _parses(t)),
        "printed_bracket_balanced": sum(1 for t in texts if _balanced(t)),
        "printed_with_illegal_char": sum(
            1 for t in texts if set(t) - LEGAL
        ),
        "printed_illegal_characters": dict(illegal.most_common()),
        "printed_c(=0)_cells": sum(1 for t in texts if "C(=0)" in t),
        "best_non_empty": len(best_texts),
        "best_rdkit_parses": sum(1 for t in best_texts if _parses(t)),
        "best_source": dict(
            Counter(r.get("smiles_best_source") or "" for r in best).most_common()
        ),
        "structure_confidence": dict(
            Counter(r.get("structure_confidence") or "" for r in rows).most_common()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "data" / "processed" / "compounds_combined.csv",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=REPO / "data" / "processed" / "archive"
        / "compounds_combined_validated_route.csv",
    )
    args = parser.parse_args()

    for path in (args.csv, args.compare):
        if not path.exists():
            print(f"(missing: {path})")
            continue
        result = stats(path)
        print(f"=== {result['csv']} ===")
        for key, value in result.items():
            if key == "csv":
                continue
            print(f"  {key:32s} {value}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
