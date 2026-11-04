"""Show what the line reader sees inside one activity cell.

Two symptoms are being chased: the ionisation mode loses its opening
bracket (``[M+H]+`` comes back as ``M+H]+``) and the retention time
loses its decimal point (``1.53`` comes back as ``13``).  Both are
properties of the *crop* the line reader makes, so the crop is printed
band by band.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_line_probe.py \
        --page PCTCN2025075389-ftappb-I100377.jpg --cmpd 1080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import main as pipeline  # noqa: E402
from cpd.ocr.activity import ActivityCellReader  # noqa: E402
from cpd.ocr.lines import ink_bands, strip_horizontal_rules, text_span  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--cmpd", nargs="*", default=[])
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument("--route", default="validated")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.patent)
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=args.route, profile=profile,
    )
    reader = ActivityCellReader(
        english_engine=extractor.rec_engine,
        chinese_engine=extractor.engine,
        line_engine=extractor.engine,
    )
    image_path = (
        REPO / "data" / "raw" / args.patent / "golden_tables" / args.page
    )
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"cannot read {image_path}")
        return 1
    columns = extractor._detect_columns(img, [])
    rows = extractor._detect_row_boundaries(img)
    wanted = set(args.cmpd)

    for y1, y2 in rows:
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid is None or (wanted and cid not in wanted):
            continue
        print(f"\n=== cmpd {cid}  rows {y1}..{y2} ===")
        for col in columns:
            if col.name in ("cmpd_id", "structure", "smiles"):
                continue
            cell = img[y1:y2, col.x1:col.x2]
            trimmed = extractor._trim_to_content(cell)
            if trimmed is None or trimmed.size == 0:
                continue
            result = reader.read(trimmed, numeric=False)
            print(f"  {col.name:<12} -> {result.reading.text!r}")
            for candidate in result.candidates:
                print(f"  {'':<12}    {candidate.source:<5} {candidate.text!r}")
            if col.name not in ("rt_min", "ms_polarity"):
                continue
            gray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
            gray = strip_horizontal_rules(gray)
            bands = ink_bands(gray)
            print(f"  {'':<12}    bands {bands}")
            for top, bottom in bands:
                row = trimmed[top:bottom + 1, :]
                row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
                x1, x2 = text_span(row_gray)
                variant = row[:, x1:x2 + 1]
                text = reader._recognise_only(
                    cv2.copyMakeBorder(
                        variant, 0, 0, 6, 6, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                )
                print(
                    f"  {'':<12}      band {top}..{bottom} "
                    f"x {x1}..{x2} -> {text!r}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
