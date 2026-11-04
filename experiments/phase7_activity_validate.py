"""Validate the activity-cell reader against the archived extraction.

``data/processed/archive/compounds_combined_production_route.csv`` is
the extraction that was shipped before the printed-SMILES route
changed.  Its activity columns are the reference: they were checked
against the patent images by hand.

This script reads every activity cell of every page of a patent with
the *current* reader and compares it, field by field, with that
reference.  It prints the number of cells that match exactly, the
number that differ, and the ones that were lost, and it writes the
disagreements to a CSV so that each one can be looked at.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_activity_validate.py \
        --profile WO2025162428 \
        --archive data/processed/archive/compounds_combined_production_route.csv
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
from cpd.ocr.activity import ActivityCellReader, parse_number  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402

ACTIVITY_FIELDS = ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="WO2025162428")
    parser.add_argument("--route", default=None)
    parser.add_argument(
        "--archive",
        default=(
            "data/processed/archive/"
            "compounds_combined_production_route.csv"
        ),
    )
    parser.add_argument(
        "--out", default="data/audit/phase7/activity_validation.csv"
    )
    return parser.parse_args(argv)


def _comparable(field: str, value: str | None) -> str:
    """The comparison key of a field value."""
    if value is None:
        return ""
    if field in ("kd_nm", "rt_min", "ms_mz"):
        number = parse_number(value)
        return "" if number is None else repr(number)
    return " ".join(value.split())


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

    archive_path = REPO / args.archive
    with archive_path.open(encoding="utf-8-sig", newline="") as handle:
        archive = {r["cmpd_id"]: r for r in csv.DictReader(handle)}

    image_dir = REPO / "data" / "raw" / args.profile / "golden_tables"
    pages = sorted(image_dir.glob("*.jpg"))
    print(f"profile {args.profile}  route {route}  pages {len(pages)}")

    read: dict[str, dict[str, str]] = {}
    for index, path in enumerate(pages, start=1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        columns = extractor._detect_columns(img, [])
        rows = extractor._detect_row_boundaries(img)
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        for y1, y2 in rows:
            cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
            if not cid:
                continue
            values = read.setdefault(cid, {})
            for col in columns:
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
                values[col.name] = result.reading.text or ""
        print(
            f"  [{index:3d}/{len(pages)}] {path.name} "
            f"-> {len(read)} compounds",
            flush=True,
        )

    rows_out: list[dict[str, str]] = []
    summary = {field: {"match": 0, "differ": 0, "lost": 0} for field in ACTIVITY_FIELDS}
    for cid, values in sorted(read.items()):
        reference = archive.get(cid)
        if reference is None:
            continue
        for field in ACTIVITY_FIELDS:
            want = _comparable(field, reference.get(field))
            got = _comparable(field, values.get(field))
            if want == got:
                summary[field]["match"] += 1
                continue
            if not got and want:
                summary[field]["lost"] += 1
            else:
                summary[field]["differ"] += 1
            rows_out.append({
                "cmpd_id": cid,
                "field": field,
                "archived": reference.get(field) or "",
                "read": values.get(field) or "",
                "page": reference.get("source_image") or "",
            })

    print()
    print(f"{'field':<14} {'match':>7} {'differ':>7} {'lost':>6}")
    for field in ACTIVITY_FIELDS:
        row = summary[field]
        print(
            f"{field:<14} {row['match']:>7} {row['differ']:>7} "
            f"{row['lost']:>6}"
        )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cmpd_id", "field", "archived", "read", "page"],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n{len(rows_out)} disagreements -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
