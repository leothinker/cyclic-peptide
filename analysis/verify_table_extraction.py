"""Spot-check extracted assay values against the patent images.

The non-structural columns (RT, m/z, LC-MS method, polarity, KD) are the
part of the deliverable a reviewer will actually trust, so they deserve a
manual audit trail rather than an assumption that the extractor is right.

This script prints a random-but-fixed sample of extracted rows next to the
source image path and row position, formatted for eyeball comparison against
the JPEG.  Values are shown with the exact string that landed in the CSV so
that OCR noise (a misread decimal, a dropped digit) is visible rather than
rounded away.

Run from the repo root::

    .venv/Scripts/python.exe analysis/verify_table_extraction.py
    .venv/Scripts/python.exe analysis/verify_table_extraction.py 2115-2172
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"
COLUMNS = ("cmpd_id", "rt_min", "ms_mz", "lcms_method", "ms_polarity", "kd_nm")


def _rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    rows = _rows()
    wanted: set[str] | None = None
    if len(sys.argv) > 1:
        lo, _, hi = sys.argv[1].partition("-")
        wanted = {
            r["cmpd_id"].strip()
            for r in rows
            if lo <= r["cmpd_id"].strip() <= (hi or lo)
        }

    subset = [r for r in rows if wanted is None or r["cmpd_id"].strip() in wanted]
    print(f"rows shown: {len(subset)}")
    print()
    print(
        f"{'cmpd':>6} {'rt_min':>9} {'ms_mz':>10} {'lcms_method':>14} "
        f"{'polarity':>12} {'kd_nm':>10}  source_image"
    )
    print("-" * 108)
    for r in subset:
        print(
            f"{r['cmpd_id']:>6} {r['rt_min']:>9} {r['ms_mz']:>10} "
            f"{r['lcms_method']:>14} {r['ms_polarity']:>12} "
            f"{r['kd_nm']:>10}  {r['assay_source_image']}"
        )

    # Completeness summary over the whole table, since the user's acceptance
    # criterion is "no missing data outside SMILES and 2D structures".
    print()
    print("=" * 108)
    print("COMPLETENESS over all", len(rows), "rows (excluding smiles / structure)")
    print("=" * 108)
    for col in COLUMNS[1:] + ("assay_source_image",):
        filled = sum(1 for r in rows if (r.get(col) or "").strip())
        pct = filled / len(rows) * 100
        flag = "OK" if filled == len(rows) else "GAP"
        print(f"  {col:22s} {filled:4d}/{len(rows)}  ({pct:5.1f}%)  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
