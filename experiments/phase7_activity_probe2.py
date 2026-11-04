"""Show every reading of one activity cell, side by side.

The band-based line reader and the whole-crop reader disagree on the
numeric columns: the band reader loses the decimal point of ``1.53``
and reads it as ``123``, while the whole-crop reader keeps it.  This
script prints the trimmed cell, the whole-crop variant, the ink bands
and what each of them is recognised as, so that the difference can be
located in the image rather than in the numbers.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_activity_probe2.py \
        --cmpd 1048 --field rt_min
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
from cpd.ocr.lines import (  # noqa: E402
    ink_bands,
    strip_horizontal_rules,
    strip_vertical_rules,
    text_span,
)
from cpd.parsers.profiles import get_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmpd", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument("--route", default=None)
    parser.add_argument("--save", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = get_profile(args.patent)
    route = args.route or profile.smiles_route
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=route, profile=profile,
    )
    reader = ActivityCellReader(
        english_engine=extractor.rec_engine,
        chinese_engine=extractor.engine,
        line_engine=extractor.engine,
    )

    image_dir = REPO / "data" / "raw" / args.patent / "golden_tables"
    for path in sorted(image_dir.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        columns = extractor._detect_columns(img, [])
        rows = extractor._detect_row_boundaries(img)
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        for y1, y2 in rows:
            cid = extractor._read_cmpd_id(
                img[y1:y2, cid_col.x1:cid_col.x2]
            )
            if cid != args.cmpd:
                continue
            for col in columns:
                if col.name != args.field:
                    continue
                cell = img[y1:y2, col.x1:col.x2]
                trimmed = extractor._trim_to_content(cell)
                print(f"\n=== {args.cmpd} {args.field} on {path.name} ===")
                print(f"cell {cell.shape[1]}x{cell.shape[0]}"
                      f"  trimmed {trimmed.shape[1]}x{trimmed.shape[0]}")

                gray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
                stripped = strip_horizontal_rules(gray, coverage=0.6)
                stripped = strip_vertical_rules(
                    stripped, height=stripped.shape[0]
                )

                crop = reader._value_crop(trimmed)
                if crop is not None:
                    variant = cv2.copyMakeBorder(
                        crop, 0, 0, 6, 6, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    print(f"  whole crop {crop.shape[1]}x{crop.shape[0]}"
                          f" -> {reader._recognise_only(variant)!r}")

                bands = ink_bands(stripped)
                print(f"  bands {bands}")
                for top, bottom in bands:
                    band = stripped[top:bottom + 1, :]
                    x1, x2 = text_span(band, rule_height=stripped.shape[0])
                    piece = band[:, x1:x2 + 1]
                    if piece.size == 0:
                        continue
                    variant = cv2.copyMakeBorder(
                        piece, 0, 0, 6, 6, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    print(f"    band {top}..{bottom} x {x1}..{x2}"
                          f" -> {reader._recognise_only(variant)!r}")

                result = reader.read(trimmed, numeric=True)
                print(f"  reader -> {result.reading.text!r}")
                for candidate in result.candidates:
                    print(f"     {candidate.source:<5} {candidate.text!r}")

                if args.save:
                    out = Path(args.save)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), trimmed)
                    print(f"  saved {out}")
            return 0
    print(f"cmpd {args.cmpd} not found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
