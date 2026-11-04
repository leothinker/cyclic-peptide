"""Dump every reading of every activity cell of a patent.

The activity reader has four sources: the whole value region read as
one unit (``crop``), the printed line read band by band (``line``),
the Chinese detector + recogniser (``ch``) and the English detector +
recogniser (``en``).  Which of them is used is a policy, and a policy
can only be judged against the archived extraction.

Reading all 103 pages takes about twenty-five minutes, so the
readings are written to a CSV once and the policy is then evaluated
offline, in seconds, by ``phase7_activity_policy.py``.  The dump holds
every candidate of every cell, so a new policy never needs another
pass over the images.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_activity_dump.py \
        --profile WO2025162428
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import main as pipeline  # noqa: E402
from cpd.ocr.activity import ActivityCellReader  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402

ACTIVITY_FIELDS = ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="WO2025162428")
    parser.add_argument("--route", default=None)
    parser.add_argument(
        "--out", default="data/audit/phase7/activity_dump.csv"
    )
    parser.add_argument(
        "--cache", default="data/audit/phase7/activity_readings.csv"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.profile)
    route = args.route or profile.smiles_route
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=route, profile=profile,
    )
    reader = ActivityCellReader(
        english_engine=extractor.rec_engine,
        chinese_engine=extractor.engine,
        line_engine=extractor.engine,
    )

    image_dir = REPO / "data" / "raw" / args.profile / "golden_tables"
    pages = sorted(image_dir.glob("*.jpg"))
    print(f"profile {args.profile}  route {route}  pages {len(pages)}")

    rows: list[dict[str, str]] = []
    for index, path in enumerate(pages, start=1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        columns = extractor._detect_columns(img, [])
        rows_bounds = extractor._detect_row_boundaries(img)
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        # Some pages open with a row that belongs to a *different*
        # table (I100376 starts with a 3-column SMILES row above the
        # dense activity table).  The extractor resolves that with
        # ``_detect_first_row_columns``; the dump has to use the same
        # columns, or the cells of that row are read at the wrong
        # positions.
        first_row_columns = (
            extractor._detect_first_row_columns(img, rows_bounds)
            if len(columns) == 6
            else None
        )
        for slice_index, (y1, y2) in enumerate(rows_bounds):
            slice_columns = (
                first_row_columns
                if slice_index == 0 and first_row_columns is not None
                else columns
            )
            cid_col_local = next(
                (c for c in slice_columns if c.name == "cmpd_id"), cid_col
            )
            cid = extractor._read_cmpd_id(
                img[y1:y2, cid_col_local.x1:cid_col_local.x2]
            )
            if not cid:
                continue
            for col in slice_columns:
                if col.name not in ACTIVITY_FIELDS:
                    continue
                cell = img[y1:y2, col.x1:col.x2]
                if cell.size == 0:
                    continue
                trimmed = extractor._trim_to_content(cell)
                if trimmed is None or trimmed.size == 0:
                    continue
                spec = profile.assay_field(col.name)
                numeric = bool(spec and spec.kind == "number")
                result = reader.read(trimmed, numeric=numeric)
                row = {
                    "cmpd_id": cid,
                    "field": col.name,
                    "numeric": "1" if numeric else "0",
                    "page": path.name,
                    "chosen": result.reading.text or "",
                    "chosen_source": result.reading.source,
                }
                for candidate in result.candidates:
                    row[f"{candidate.source}_text"] = candidate.text or ""
                    row[f"{candidate.source}_boxes"] = str(candidate.n_boxes)
                rows.append(row)
        print(
            f"  [{index:3d}/{len(pages)}] {path.name} "
            f"-> {len(rows)} cells",
            flush=True,
        )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cmpd_id", "field", "numeric", "page",
        "crop_text", "crop_boxes",
        "line_text", "line_boxes",
        "ch_text", "ch_boxes",
        "en_text", "en_boxes",
        "chosen", "chosen_source",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\n{len(rows)} cells -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
