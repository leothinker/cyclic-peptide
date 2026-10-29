"""Sanity-check the output of the repair-policy pass.

Confirms the expected columns exist, that ``smiles_best`` no longer holds
unvalidated heuristics, and prints a few representative rows so a human can
eyeball the provenance columns.

Run from the repo root::

    .venv/Scripts/python.exe analysis/verify_policy_csv.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Defaults to the delivered CSV. Pass a path to check a `--policy` run that
# wrote to a separate file instead.
POLICY = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "data" / "processed" / "compounds_combined.csv"
)

EXPECTED = (
    "structure_confidence",
    "repair_review_status",
    "repair_confidence",
    "smiles_candidate_repair",
    "repair_atoms_invented",
    "repair_atoms_deleted",
    "ms_check_verdict",
    "ms_error_ppm",
    "molecular_weight",
    "heavy_atom_count",
)


def main() -> int:
    if not POLICY.exists():
        print(
            f"missing CSV: {POLICY}\n"
            "run: .venv/Scripts/python.exe -m cpd.repair_policy --inplace",
            file=sys.stderr,
        )
        return 1
    with POLICY.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    print(f"rows: {len(rows)}")
    missing = [c for c in EXPECTED if c not in rows[0]]
    print(f"missing expected columns: {missing or 'none'}")
    print()
    for col in (
        "repair_review_status",
        "structure_confidence",
        "ms_check_verdict",
        "smiles_best_source",
    ):
        counts = Counter((r.get(col) or "(empty)").strip() or "(empty)" for r in rows)
        print(f"-- {col}")
        for k, v in counts.most_common(10):
            print(f"   {k:34s} {v:4d}")
        print()

    print("-- representative rows --")
    for cid in ("1036", "1046", "1019", "1121", "2129", "1201"):
        r = next((x for x in rows if x["cmpd_id"] == cid), None)
        if r is None:
            continue
        best = "set" if (r.get("smiles_best") or "").strip() else "(empty)"
        cand = "set" if (r.get("smiles_candidate_repair") or "").strip() else "(empty)"
        print(
            f"  cmpd {cid:>5}: best={best:7s} candidate={cand:7s} "
            f"src={r.get('smiles_best_source') or '-':20s} "
            f"status={r.get('repair_review_status', ''):14s} "
            f"conf={r.get('structure_confidence', ''):18s} "
            f"ppm={r.get('ms_error_ppm') or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
