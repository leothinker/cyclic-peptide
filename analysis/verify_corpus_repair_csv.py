"""Verify the OCR-repair corpus run wrote the expected columns.

After :func:`analysis.run_ocr_repair_corpus.main` runs the search on
every row of compounds_combined.csv, this script confirms the five
new columns are populated (or honestly empty with a non-empty trace).

Run::
    .venv/Scripts/python.exe analysis/verify_corpus_repair_csv.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "compounds_combined.csv"


def main() -> None:
    with CSV.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys())
    new_cols = ["smiles_ocr_repaired", "ocr_repair_trace",
                "ocr_repair_formula", "ocr_repair_largest_ring",
                "ocr_repair_mass_ppm"]
    print(f"rows: {len(rows)}")
    for c in new_cols:
        present = c in cols
        non_empty = sum(1 for r in rows if (r.get(c) or "").strip())
        print(f"  {c:<25} present={present} non_empty={non_empty}")

    by_trace = Counter((r.get("ocr_repair_trace") or "").strip()
                       for r in rows)
    print("\ntrace buckets (sorted by count):")
    for k, v in sorted(by_trace.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k[:60]:<60} {v}")

    ok_rows = [r for r in rows
               if (r.get("ocr_repair_trace") or "").startswith("ok ")]
    print(f"\nok rows: {len(ok_rows)}")
    for r in ok_rows[:3]:
        print(f"  cmpd {r['cmpd_id']}: formula={r['ocr_repair_formula']}"
              f" ring={r['ocr_repair_largest_ring']} "
              f"ppm={r['ocr_repair_mass_ppm']}")
        print(f"    smiles={r['smiles_ocr_repaired'][:80]}...")


if __name__ == "__main__":
    main()
