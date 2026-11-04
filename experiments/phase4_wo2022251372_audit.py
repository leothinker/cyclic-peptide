"""Audit the WO2022251372 printed-SMILES tables without changing production.

The patent pages are full-page TIFF images, unlike the tightly cropped
JPG tables used by WO2025162428.  This script isolates the two questions:

1. Does the current end-to-end pipeline accept this input layout?
2. After applying the correct table geometry, how well do the existing
   production and validated printed-SMILES OCR routes read the cells?

The audit is deliberately independent of ``src/cpd`` and never writes the
formal combined CSV.  It writes machine-readable results and evidence crops
under ``data/audit/phase4/wo2022251372``.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles  # noqa: E402
from cpd.ocr.printed_smiles import (  # noqa: E402
    PrintedSmilesOCR,
    ink_bands,
)
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

from wo2022251372_references import (  # noqa: E402
    REFERENCE_LINES,
    REFERENCE_PAGES,
    cer,
    edit_distance,
)

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase4" / "wo2022251372"
CELLS_OUT = OUT / "table1a_cells"

# Table 1a is a three-column table: No. | SMILES String | Chemical Structure.
# These coordinates were verified against the 300 dpi source pages.  The
# audit finds the horizontal row rules dynamically but keeps the two
# vertical SMILES boundaries fixed because the structure drawings contain
# many vertical strokes that make generic rule detection ambiguous.
NO_COLUMN = (241, 371)
SMILES_COLUMN = (371, 915)

# The compound number is printed in the narrow first column.  It is more
# reliable to map the known sequence from the source pages than to trust
# the OCR number alone: the first-column rule occasionally joins a digit
# to a nearby stroke and yields values such as 28 or 31.
EXPECTED_IDS = {
    "000116": (1, 2),
    "000117": (3, 4),
    "000118": (5, 6),
    "000119": (7, 8),
    "000120": (9, 10, 11),
    "000121": (12, 13, 14),
    "000122": (15, 16, 17),
    "000123": (18, 19, 20),
    "000124": (21, 22, 23),
    "000125": (24, 25, 26),
    "000126": (27, 28),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the WO2022251372 printed-SMILES tables.",
    )
    parser.add_argument(
        "--layout-probe",
        choices=("sample", "all"),
        default="sample",
        help=(
            "Run the current extractor on two representative pages "
            "(sample, default) or every TIFF page (all)."
        ),
    )
    return parser.parse_args(argv)


def _cluster(values: list[int], gap: int = 10) -> list[int]:
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def _horizontal_rules(image: np.ndarray) -> list[int]:
    """Find full-width table rules in page coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 200).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (700, 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    x1, x2 = 200, 2300
    ys = np.where(lines[:, x1:x2].sum(axis=1) > 1000)[0]
    return _cluster([int(y) for y in ys])


def _table_1a_rows(image: np.ndarray, stem: str) -> tuple[int, list[tuple[int, int]]]:
    """Return the header bottom and data-row ranges for Table 1a."""
    rules = _horizontal_rules(image)
    if stem == "000116":
        # This page contains the end of Table 1, explanatory notes, and the
        # beginning of Table 1a.
        header_bottom = 1349
        data_rules = [y for y in rules if header_bottom < y <= 3042]
    else:
        header_bottom = 425
        data_rules = [y for y in rules if header_bottom < y <= 3075]

    boundaries = [header_bottom, *data_rules]
    rows = [
        (boundaries[index] + 1, boundaries[index + 1] - 1)
        for index in range(len(boundaries) - 1)
    ]
    return header_bottom, rows


def _read_number(engine, cell: np.ndarray) -> str:
    """Read the one- or two-digit row number from the No. column."""
    result = engine(cell)
    texts = getattr(result, "txts", None) or []
    for text in texts:
        for token in re.findall(r"\d+", text or ""):
            if token:
                return token
    return ""


def _parses(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _write_montage(cells: list[dict]) -> None:
    if not cells:
        return
    scale = 2
    label_width = 150
    gap = 18
    rendered = []
    for row in cells:
        image = cv2.imread(str(row["cell_path"]))
        if image is None:
            continue
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        rendered.append((row, image))

    width = label_width + max(image.shape[1] for _, image in rendered) + 30
    height = sum(image.shape[0] + gap for _, image in rendered) + 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (15, 12),
        "WO2022251372 Table 1a - printed SMILES cells",
        font=_font(24, bold=True),
        fill=(20, 20, 20),
    )
    y = 50
    for row, image in rendered:
        draw.text(
            (15, y + 8),
            f"page {row['page']}\ncompound {row['compound_id']}",
            font=_font(18),
            fill=(70, 70, 70),
        )
        canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                     (label_width, y))
        y += image.shape[0] + gap
    canvas.save(OUT / "figure_table1a_smiles_cells.png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS_OUT.mkdir(parents=True, exist_ok=True)

    production = PrintedSmilesOCR(route="production")
    validated = PrintedSmilesOCR(route="validated")
    from rapidocr import RapidOCR

    number_engine = RapidOCR(params={})
    rows: list[dict] = []
    for path in sorted(PAGES.glob("*.tif")):
        if path.stem not in EXPECTED_IDS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        header_bottom, row_ranges = _table_1a_rows(image, path.stem)
        expected_ids = EXPECTED_IDS[path.stem]
        if len(row_ranges) != len(expected_ids):
            raise RuntimeError(
                f"{path.stem}: found {len(row_ranges)} Table 1a rows, "
                f"expected {len(expected_ids)}"
            )
        for row_index, ((y1, y2), compound_id) in enumerate(
            zip(row_ranges, expected_ids)
        ):
            no_cell = image[y1:y2, NO_COLUMN[0]:NO_COLUMN[1]]
            detected_number = _read_number(number_engine, no_cell)

            smiles_cell = image[y1:y2, SMILES_COLUMN[0]:SMILES_COLUMN[1]]
            cell_path = CELLS_OUT / f"{path.stem}_cmpd{compound_id}.png"
            cv2.imwrite(str(cell_path), smiles_cell)

            production_text = production.read_cell(smiles_cell) or ""
            validated_text = validated.read_cell(smiles_cell) or ""
            gray = cv2.cvtColor(smiles_cell, cv2.COLOR_BGR2GRAY)
            band_count = len(ink_bands(gray))

            repaired = auto_repair_smiles(validated_text)
            reference = "".join(REFERENCE_LINES.get(compound_id, ()))
            row = {
                "page": path.stem,
                "compound_id": compound_id,
                "detected_number": detected_number,
                "bbox_x1": SMILES_COLUMN[0],
                "bbox_y1": y1,
                "bbox_x2": SMILES_COLUMN[1],
                "bbox_y2": y2,
                "printed_line_count": band_count,
                "manual_reference_status": (
                    "hand_transcribed" if reference else "not_transcribed"
                ),
                "manual_reference_page": REFERENCE_PAGES.get(
                    compound_id, ""
                ),
                "manual_reference": reference,
                "cell_path": str(cell_path.relative_to(REPO)).replace("\\", "/"),
                "production_text": production_text,
                "production_length": len(production_text),
                "production_balanced": (
                    production_text.count("(") == production_text.count(")")
                ),
                "production_parses": _parses(production_text),
                "validated_text": validated_text,
                "validated_length": len(validated_text),
                "validated_balanced": (
                    validated_text.count("(") == validated_text.count(")")
                ),
                "validated_parses": _parses(validated_text),
                "validated_auto_repaired": repaired,
                "validated_auto_repair_parses": _parses(repaired),
                "route_edit_distance": _edit_distance(
                    production_text, validated_text
                ),
                "production_edit_to_reference": (
                    edit_distance(reference, production_text)
                    if reference
                    else ""
                ),
                "production_cer_to_reference": (
                    round(cer(reference, production_text), 5)
                    if reference
                    else ""
                ),
                "validated_edit_to_reference": (
                    edit_distance(reference, validated_text)
                    if reference
                    else ""
                ),
                "validated_cer_to_reference": (
                    round(cer(reference, validated_text), 5)
                    if reference
                    else ""
                ),
            }
            rows.append(row)
            print(
                f"{path.stem} cmpd {compound_id:>2}: "
                f"production {len(production_text):3} "
                f"valid {len(validated_text):3} "
                f"parse={row['validated_parses']}"
            )

    csv_path = OUT / "table1a_ocr_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Demonstrate that the production entry point cannot discover TIFF input.
    jpg_inputs = list(PAGES.glob("*.jpg"))
    tif_inputs = list(PAGES.glob("*.tif"))

    # Exercise the existing row extractor directly on the TIFFs.  This
    # isolates layout compatibility from the JPG-only directory scanner.
    extractor = RapidOcrTableExtractor(smiles_route="validated")
    previous_columns = []
    direct_compounds = 0
    direct_assays = 0
    direct_pages_with_rows = 0
    probe_pages = (
        [path for path in tif_inputs if path.stem in {"000108", "000116"}]
        if args.layout_probe == "sample"
        else tif_inputs
    )
    for path in probe_pages:
        compounds, assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=previous_columns
        )
        if columns:
            previous_columns = columns
        direct_compounds += len(compounds)
        direct_assays += len(assays)
        if compounds or assays:
            direct_pages_with_rows += 1

    summary = {
        "pages": len(tif_inputs),
        "jpg_inputs_found_by_current_scanner": len(jpg_inputs),
        "tif_inputs_present": len(tif_inputs),
        "table1a_data_rows": len(rows),
        "current_layout_direct_extraction": {
            "pages_probed": len(probe_pages),
            "compounds": direct_compounds,
            "assays": direct_assays,
            "pages_with_rows": direct_pages_with_rows,
        },
        "production_after_correct_crop": {
            "balanced": sum(row["production_balanced"] for row in rows),
            "parses": sum(row["production_parses"] for row in rows),
        },
        "validated_after_correct_crop": {
            "balanced": sum(row["validated_balanced"] for row in rows),
            "parses": sum(row["validated_parses"] for row in rows),
            "auto_repair_parses": sum(
                row["validated_auto_repair_parses"] for row in rows
            ),
        },
        "hand_transcribed_compounds": sorted(REFERENCE_LINES),
        "hand_transcribed_edit_to_reference": {
            "production": [
                row["production_edit_to_reference"]
                for row in rows
                if row["manual_reference"]
            ],
            "validated": [
                row["validated_edit_to_reference"]
                for row in rows
                if row["manual_reference"]
            ],
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_montage(rows)
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
