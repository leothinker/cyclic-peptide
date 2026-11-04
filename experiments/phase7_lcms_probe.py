"""Show what the line reader sees in an activity cell, band by band.

The ``lcms_method`` column disagrees with the archived extraction on
28 of 406 rows.  The line reading returns ``42-100-`` where the
printed value is ``40-100- 2min``: the first group is misread and the
second one is missing.  This script prints the cell, its ink bands
and what each band is recognised as, so that the loss can be located.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_lcms_probe.py \
        --page PCTCN2025075389-ftappb-I100283.jpg --cmpd 1013
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
from cpd.ocr.lines import ink_bands, strip_horizontal_rules  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--cmpd", nargs="*", default=[])
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument("--field", default="lcms_method")
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
            if col.name != args.field:
                continue
            cell = img[y1:y2, col.x1:col.x2]
            trimmed = extractor._trim_to_content(cell)
            print(
                f"  cell {cell.shape[1]}x{cell.shape[0]}"
                f"  trimmed {trimmed.shape[1]}x{trimmed.shape[0]}"
            )
            result = reader.read(trimmed, numeric=False)
            print(f"  reader -> {result.reading.text!r}")
            for candidate in result.candidates:
                print(f"     {candidate.source:<5} {candidate.text!r}")

            gray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
            stripped = strip_horizontal_rules(gray, coverage=0.6)
            bands = ink_bands(stripped)
            print(f"     bands {bands}")
            for top, bottom in bands:
                band = stripped[top:bottom + 1, :]
                dark = band < 160
                columns_ink = np.flatnonzero(dark.sum(axis=0) > 0)
                if not len(columns_ink):
                    continue
                x1, x2 = int(columns_ink[0]), int(columns_ink[-1])
                variant = band[:, x1:x2 + 1]
                text = reader._recognise_only(
                    cv2.copyMakeBorder(
                        variant, 0, 0, 6, 6, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                )
                print(
                    f"     band {top}..{bottom} x {x1}..{x2}"
                    f" -> {text!r}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
