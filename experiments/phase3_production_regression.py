"""Regression check: the production route must still reproduce the CSV.

The right-edge fix only touches the validated route, but the shipped
CSV was produced by the production route, so that route has to be
re-checked against the file that is actually in the repository.

For every page the script runs the production SMILES recognition and
compares the result with the ``smiles`` column of
``data/processed/compounds_combined.csv``, keyed by ``cmpd_id``.

The CSV is read only; nothing is written.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import ROUTE_PRODUCTION  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

PAGES = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "pipeline_check"

STEMS = (
    "PCTCN2025075389-ftappb-I100297",
    "PCTCN2025075389-ftappb-I100298",
    "PCTCN2025075389-ftappb-I100299",
    "PCTCN2025075389-ftappb-I100300",
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest().upper()
    print(f"official CSV sha256: {digest}")

    published: dict[str, str] = {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            published[row["cmpd_id"]] = row.get("smiles") or ""

    extractor = RapidOcrTableExtractor(smiles_route=ROUTE_PRODUCTION)
    rows = []
    same = 0
    different = 0

    for stem in STEMS:
        path = PAGES / f"{stem}.jpg"
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        columns = extractor._detect_columns(img)
        row_bounds = extractor._detect_row_boundaries(img)
        smiles_col = next((c for c in columns if c.name == "smiles"), None)
        cid_col = next((c for c in columns if c.name == "cmpd_id"), None)
        if smiles_col is None or cid_col is None:
            continue
        for y1, y2 in row_bounds:
            cid = extractor._read_cmpd_id(
                img[y1:y2, cid_col.x1 : cid_col.x2]
            )
            if cid is None:
                continue
            cell = img[y1:y2, smiles_col.x1 : smiles_col.x2]
            text = extractor._ocr_smiles_with_stats(cell)[0] or ""
            expected = published.get(cid, "")
            # The CSV stores the auto-repaired string, so compare the
            # same way the pipeline does.
            from cpd.chem import auto_repair_smiles

            repaired = auto_repair_smiles(text)
            match = repaired == expected
            same += int(match)
            different += int(not match)
            rows.append(
                {
                    "source_image": path.name,
                    "cmpd_id": cid,
                    "csv_smiles": expected,
                    "recomputed": repaired,
                    "match": match,
                }
            )
            flag = "OK " if match else "DIFF"
            print(f"  {flag} {cid}: csv_len={len(expected):4d} "
                  f"recomputed_len={len(repaired):4d}")

    with (OUT / "production_regression.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nproduction route: {same} identical, {different} different")
    digest_after = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest().upper()
    print(f"official CSV sha256 after: {digest_after}")
    print(f"unchanged: {digest == digest_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
