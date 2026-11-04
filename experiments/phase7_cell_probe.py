"""Probe what the activity-cell reader actually sees on one page.

The rebuilt CSV disagreed with the archived extraction on a number of
``rt_min`` / ``kd_nm`` values (``1.72`` became ``172.0``).  This script
prints the raw OCR output for the affected cells so the disagreement can
be traced to a specific stage: the crop, the detector boxes, the text
order, or the float conversion.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_cell_probe.py \
        --page PCTCN2025075389-ftappb-I100280.jpg \
        --cmpd 1002 1003 1004 1005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import main as pipeline  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--cmpd", nargs="*", default=[])
    parser.add_argument("--route", default="validated")
    parser.add_argument("--patent", default="WO2025162428")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.patent)
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=args.route, profile=profile,
    )
    image_path = (
        REPO / "data" / "raw" / args.patent / "golden_tables" / args.page
    )
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"cannot read {image_path}")
        return 1
    print(f"image {args.page}  size {img.shape[1]}x{img.shape[0]}")

    columns = extractor._detect_columns(img, [])
    rows = extractor._detect_row_boundaries(img)
    print(f"columns: {[(c.name, c.x1, c.x2) for c in columns]}")
    print(f"row bands: {len(rows)}")

    wanted = set(args.cmpd)
    for y1, y2 in rows:
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid is None:
            continue
        if wanted and cid not in wanted:
            continue
        print(f"\n=== cmpd {cid}  rows {y1}..{y2} ===")
        for col in columns:
            if col.name in ("cmpd_id", "structure", "smiles"):
                continue
            cell = img[y1:y2, col.x1:col.x2]
            raw = extractor._upscale(extractor._trim_to_content(cell))
            result = extractor.rec_engine(raw)
            boxes = extractor._boxes(result)
            txts = extractor._txts(result)
            print(f"  {col.name:<12} raw={txts}")
            if boxes is not None:
                order = sorted(
                    zip(boxes, txts),
                    key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2),
                )
                print(f"  {'':<12} sorted={[t for _, t in order]}")
                print(
                    f"  {'':<12} midpoints="
                    f"{[round((b[0][1] + b[2][1]) / 2, 1) for b in boxes]}"
                )
            final = extractor._ocr_cell(cell, col.name)
            print(f"  {'':<12} FINAL={final!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
