"""Check the printed-SMILES route against the hand-verified references.

Three compounds have a reference string that was checked character by
character against the patent page: 1036 (transcribed by the
researcher), 1037 (re-read from the page in a second, independent
pass) and 1038 (supplied by the researcher and cross-checked with a
pixel-column measurement).

The references say what the *printed text* is.  They say nothing about
whether the drawn molecule is the intended one, so nothing here is a
statement about chemistry.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_route_verify.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import ROUTES, PrintedSmilesOCR  # noqa: E402

# The three references that are verified character by character.
# Source: data/audit/phase3/reference_status.md
REFERENCES: dict[str, str] = {
    "1036": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
        "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
        "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
        "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)="
        "C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
    ),
    "1037": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=CC=C2)C(=O)N3[C@@H]"
        "(CC3)C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)"
        "=C5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
        "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)="
        "C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
    ),
    "1038": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
        "(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)"
        "C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H]"
        "(C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)="
        "C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C"
    ),
}

# The page each compound is printed on.
PAGES: dict[str, str] = {
    "1036": "PCTCN2025075389-ftappb-I100297.jpg",
    "1037": "PCTCN2025075389-ftappb-I100297.jpg",
    "1038": "PCTCN2025075389-ftappb-I100298.jpg",
}


def edit_distance(a: str, b: str) -> int:
    """The Levenshtein distance of two strings."""
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patent", default="WO2025162428")
    parser.add_argument(
        "--out", default="data/audit/phase7/route_verification.csv"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image_dir = REPO / "data" / "raw" / args.patent / "golden_tables"
    cells_dir = REPO / "data" / "processed" / "cells"

    rows_out = []
    print(
        f"{'route':<12} {'cmpd':<6} {'edit':>5} {'CER':>9} "
        f"{'len':>5} {'ref':>5} {'brackets':>9}"
    )
    print("-" * 60)
    for route in ROUTES:
        reader = PrintedSmilesOCR(route=route)
        for cmpd, reference in REFERENCES.items():
            # The cropped SMILES cells are written by the pipeline.
            matches = sorted(cells_dir.glob(f"*_{cmpd}_smiles.png"))
            if not matches:
                matches = sorted(cells_dir.glob(f"*_{cmpd}*.png"))
            text = None
            source = ""
            for path in matches:
                source = path.name
                if path.stem.endswith("_smiles"):
                    continue
                break
            # Fall back to reading the cell out of the page.
            if text is None:
                page = image_dir / PAGES[cmpd]
                text = _read_from_page(reader, page, cmpd)
                source = PAGES[cmpd]
            if text is None:
                print(f"{route:<12} {cmpd:<6} (cell not found)")
                continue
            distance = edit_distance(reference, text)
            cer = distance / len(reference) if reference else 0.0
            rows_out.append({
                "route": route,
                "cmpd_id": cmpd,
                "edit_distance": distance,
                "cer": f"{cer:.6f}",
                "length": len(text),
                "reference_length": len(reference),
                "bracket_open": text.count("("),
                "bracket_close": text.count(")"),
                "exact": text == reference,
                "source": source,
            })
            print(
                f"{route:<12} {cmpd:<6} {distance:>5} {cer:>9.5f} "
                f"{len(text):>5} {len(reference):>5} "
                f"{text.count('('):>4}/{text.count(')'):<4}"
            )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "route", "cmpd_id", "edit_distance", "cer", "length",
                "reference_length", "bracket_open", "bracket_close",
                "exact", "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n-> {out_path}")
    return 0


def _read_from_page(reader, page: Path, cmpd: str) -> str | None:
    """Read the printed SMILES of one compound straight from the page."""
    import cv2

    sys.path.insert(0, str(REPO))
    import main as pipeline
    from cpd.parsers.profiles import get_profile

    profile = get_profile("WO2025162428")
    extractor = pipeline.RapidOcrTableExtractor(
        rec_top_k=0, smiles_route=reader.route, profile=profile,
    )
    img = cv2.imread(str(page))
    if img is None:
        return None
    columns = extractor._detect_columns(img, [])
    rows = extractor._detect_row_boundaries(img)
    cid_col = next(c for c in columns if c.name == "cmpd_id")
    smiles_col = next((c for c in columns if c.name == "smiles"), None)
    if smiles_col is None:
        return None
    for y1, y2 in rows:
        cid = extractor._read_cmpd_id(img[y1:y2, cid_col.x1:cid_col.x2])
        if cid != cmpd:
            continue
        cell = img[y1:y2, smiles_col.x1:smiles_col.x2]
        return reader.read_cell(cell)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
