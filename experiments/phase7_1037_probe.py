"""Look at the three residual errors of compound 1037.

The ``validated`` route reads 1036 and 1038 character for character.
On 1037 three characters are still wrong: one ``]`` is missing and two
``O`` are read as the digit ``0``.  This script prints the printed
lines of the cell and what each line is recognised as, so that the
three errors can be located in the image.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_1037_probe.py
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

sys.path.insert(0, str(REPO / "experiments"))

import main as pipeline  # noqa: E402
from cpd.ocr.lines import ink_bands, text_span  # noqa: E402
from cpd.ocr.printed_smiles import PAD_BOTTOM, PAD_TOP, upscale  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from phase7_route_verify import REFERENCES  # noqa: E402

# The printed lines of 1037, read by hand from the page
# (data/audit/phase3/reference_status.md).
PRINTED_LINES = [
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=",
    "CC=C2)C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]",
    "4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F",
    ")=C5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(",
    "CCCC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[",
    "C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F",
    ")=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmpd", default="1037")
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument(
        "--page", default="PCTCN2025075389-ftappb-I100297.jpg"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from rapidocr import RapidOCR

    engine = RapidOCR(params={})
    profile = get_profile(args.patent)
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route="validated", profile=profile,
    )
    page = (
        REPO / "data" / "raw" / args.patent / "golden_tables" / args.page
    )
    img = cv2.imread(str(page))
    columns = extractor._detect_columns(img, [])
    bounds = extractor._detect_row_boundaries(img)
    cid_col = next(c for c in columns if c.name == "cmpd_id")
    smiles_col = next(c for c in columns if c.name == "smiles")

    cell = None
    for y1, y2 in bounds:
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid == args.cmpd:
            cell = img[y1:y2, smiles_col.x1:smiles_col.x2]
            break
    if cell is None:
        print(f"{args.cmpd}: cell not found")
        return 1

    out_dir = REPO / "data" / "audit" / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{args.cmpd}_cell.png"), cell)

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bands = ink_bands(gray)
    print(f"cell {cell.shape[1]}x{cell.shape[0]}, {len(bands)} bands")
    for index, (top, bottom) in enumerate(bands):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        big = upscale(variant, 2)
        result = engine(big, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        text = (txts[0] or "").strip() if txts else ""
        printed = (
            PRINTED_LINES[index] if index < len(PRINTED_LINES) else ""
        )
        mark = "" if text == printed else "   <-- differs"
        print(f"  band {index} ({top}..{bottom}){mark}")
        print(f"      printed: {printed!r}")
        print(f"      read   : {text!r}")
        if text != printed:
            cv2.imwrite(
                str(out_dir / f"{args.cmpd}_band{index}.png"), big
            )
    print(f"\nwrote {out_dir}/{args.cmpd}_cell.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
