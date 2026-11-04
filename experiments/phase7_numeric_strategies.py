"""Which reader recovers a numeric activity cell such as ``1.72``?

The rebuilt CSV turned ``rt_min`` ``1.72`` into ``172.0``.  The cause
is that the English detector splits the cell into the boxes ``1`` and
``72``; the decimal point is dropped and the two boxes are joined
without a separator, so ``_as_float`` reads ``172``.

This script compares the readers that are available for one cell:

* ``EN det+rec``  -- the current primary reader (PP-OCRv4 English);
* ``CH det+rec``  -- the default pipeline on the same crop;
* ``ink-band v6`` -- our own ink-band segmentation, recognised once
  per line with the upstream default recogniser in recognition-only
  mode (the machinery the validated SMILES route uses);
* ``ink-band EN`` -- the same, with the English recogniser.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_numeric_strategies.py \
        --page PCTCN2025075389-ftappb-I100280.jpg --cmpd 1002 1003 1004
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
from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    default_recogniser,
    english_recogniser_only,
    ink_bands,
    text_span,
)
from cpd.parsers.profiles import get_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--cmpd", nargs="*", default=[])
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument("--route", default="validated")
    return parser.parse_args(argv)


def ink_band_read(engine, cell: np.ndarray) -> str:
    """Read every ink band of a cell with recognition-only mode."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts: list[str] = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size == 0:
            continue
        result = engine(variant, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        if txts:
            parts.append((txts[0] or "").strip())
    return "".join(parts)


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

    v6 = default_recogniser()
    en_only = english_recogniser_only()

    columns = extractor._detect_columns(img, [])
    rows = extractor._detect_row_boundaries(img)
    wanted = set(args.cmpd)

    for y1, y2 in rows:
        cid_col = next(c for c in columns if c.name == "cmpd_id")
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid is None or (wanted and cid not in wanted):
            continue
        print(f"\n=== cmpd {cid} ===")
        for col in columns:
            if col.name in ("cmpd_id", "structure", "smiles"):
                continue
            cell = img[y1:y2, col.x1:col.x2]
            trimmed = extractor._trim_to_content(cell)
            if trimmed.size == 0:
                continue
            up = extractor._upscale(trimmed)
            en = extractor._txts(extractor.rec_engine(up))
            ch = extractor._txts(extractor.engine(up))
            band_v6 = ink_band_read(v6, cell)
            band_en = ink_band_read(en_only, cell)
            print(f"  {col.name:<12} EN={en!r} CH={ch!r}")
            print(f"  {'':<12} band_v6={band_v6!r} band_EN={band_en!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
