"""Try crop variants of one printed SMILES cell against a reference.

The ``validated`` route reads the printed cell of the activity table.
Phase 3 established that the *cell* has two geometry defects: a
vertical table rule that runs through the whole cell, and a vertical
padding of only a few pixels, which squeezes the glyphs of the
printed line.  The Phase 3 final route fixed both, but it read a
purpose-built crop.  This script measures the same variants on the
cell as the pipeline produces it, so that the fix can be judged on
the real input.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_cell_variants.py \
        --cmpd 1036 1037 1038
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

import main as pipeline  # noqa: E402
from cpd.ocr.lines import (  # noqa: E402
    ink_bands,
    strip_horizontal_rules,
    strip_vertical_rules,
    text_span,
)
from cpd.ocr.printed_smiles import PAD_BOTTOM, PAD_TOP, upscale  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from phase7_route_verify import PAGES, REFERENCES  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmpd", nargs="*", default=sorted(REFERENCES))
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument("--route", default="validated")
    parser.add_argument(
        "--out", default="data/audit/phase7/cell_variants.csv"
    )
    return parser.parse_args(argv)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def read_variant(
    engine, cell: np.ndarray, *, trim_rules: bool, trim_h: bool,
    pad_top: int, pad_bottom: int, scale: int,
) -> str:
    """Read the printed SMILES of a cell with one crop variant."""
    source = cell
    if trim_rules:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        gray = strip_vertical_rules(gray, height=gray.shape[0])
        source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if trim_h:
        gray = strip_horizontal_rules(gray)
    bands = ink_bands(gray)
    parts: list[str] = []
    for top, bottom in bands:
        y1 = max(0, top - pad_top)
        y2 = min(source.shape[0], bottom + pad_bottom)
        row = source[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size == 0:
            continue
        if scale != 1:
            variant = upscale(variant, scale)
        text = _recognise(engine, variant)
        if text:
            parts.append(text)
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from rapidocr import RapidOCR

    engine = RapidOCR(params={})
    profile = get_profile(args.patent)
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=args.route, profile=profile,
    )
    image_dir = REPO / "data" / "raw" / args.patent / "golden_tables"

    variants = [
        ("current", dict(trim_rules=False, trim_h=False,
                         pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM, scale=2)),
        ("trim_rules", dict(trim_rules=True, trim_h=False,
                             pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM, scale=2)),
        ("trim_rules+h", dict(trim_rules=True, trim_h=True,
                               pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM, scale=2)),
        ("trim_rules+h+pad24", dict(trim_rules=True, trim_h=True,
                                    pad_top=24, pad_bottom=24, scale=2)),
    ]

    rows_out = []
    for cmpd in args.cmpd:
        page = image_dir / PAGES[cmpd]
        img = cv2.imread(str(page))
        columns = extractor._detect_columns(img, [])
        bounds = extractor._detect_row_boundaries(img)
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        smiles_col = next(c for c in columns if c.name == "smiles")
        cell = None
        for y1, y2 in bounds:
            cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
            if cid == cmpd:
                cell = img[y1:y2, smiles_col.x1:smiles_col.x2]
                break
        if cell is None:
            print(f"{cmpd}: cell not found")
            continue
        reference = REFERENCES[cmpd]
        print(f"\n=== {cmpd}  reference {len(reference)} chars ===")
        for name, kwargs in variants:
            text = read_variant(engine, cell, **kwargs)
            distance = edit_distance(reference, text)
            cer = distance / len(reference)
            print(
                f"  {name:<20} edit {distance:>3}  cer {cer:.5f}  "
                f"len {len(text):>4}  brackets "
                f"{text.count('(')}/{text.count(')')}"
            )
            rows_out.append({
                "cmpd_id": cmpd,
                "variant": name,
                "edit_distance": distance,
                "cer": f"{cer:.6f}",
                "length": len(text),
                "reference_length": len(reference),
                "bracket_open": text.count("("),
                "bracket_close": text.count(")"),
                "text": text,
            })

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cmpd_id", "variant", "edit_distance", "cer", "length",
                "reference_length", "bracket_open", "bracket_close", "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
