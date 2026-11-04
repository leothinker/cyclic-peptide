"""Audit the hand transcriptions of WO2022251372 against the pixels.

The route comparison scores the pipeline against the hand
transcriptions in ``wo2022251372_references.py``.  If a transcription
itself is wrong, the pipeline is punished for being right, and every
number derived from it is wrong too.

This script writes each transcribed line next to the printed band it
claims to describe, at a magnification that lets the characters be
read one by one.  The printed text of this patent is a *serif* face,
and its chiral marker is a circled ``a`` -- which is what makes
``[C@@H]`` easy to mistranscribe as ``[C@H]`` or ``[C@@@H]``.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_reference_audit.py
"""

from __future__ import annotations

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

from cpd.ocr.lines import ink_bands  # noqa: E402
from wo2022251372_references import REFERENCE_LINES  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9" / "patent2_reference_audit"


def _captured_cells() -> dict[str, list[np.ndarray]]:
    """Cut the SMILES cell of every compound, as the pipeline does."""
    from cpd.parsers.profiles import get_profile
    from cpd.parsers.table_extractor import (
        OcrStats,
        RapidOcrTableExtractor,
    )

    class _Capturing(RapidOcrTableExtractor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cells: dict[str, list[np.ndarray]] = {}
            self._current_id: str | None = None

        def _read_cmpd_id(self, cell: np.ndarray) -> str | None:
            result = super()._read_cmpd_id(cell)
            if result:
                self._current_id = result
            return result

        def _ocr_smiles_with_stats(self, cell: np.ndarray):
            if self._current_id and cell.size:
                self.cells.setdefault(self._current_id, []).append(
                    cell.copy()
                )
            return None, OcrStats()

    profile = get_profile("WO2022251372")
    extractor = _Capturing(
        rec_top_k=0, smiles_route="padded", profile=profile,
    )
    image_dir = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"
    prev_columns: list = []
    for path in sorted(image_dir.iterdir()):
        image = cv2.imread(str(path))
        if image is None:
            continue
        _records, _assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
    extractor.flush_pending()
    return extractor.cells


def _write_pair(
    compound_id: str,
    index: int,
    printed: np.ndarray,
    reference: str,
    reading: str,
) -> Path:
    """Write the printed band above the two strings that claim it."""
    OUT.mkdir(parents=True, exist_ok=True)
    scale = 4
    band = cv2.resize(
        printed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    height = max(band.shape[0], 60)
    canvas = np.full((height + 90, band.shape[1], 3), 255, np.uint8)
    canvas[0:band.shape[0], :] = band
    cv2.putText(
        canvas, f"printed band {index}", (10, band.shape[0] + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2,
    )
    cv2.putText(
        canvas, f"reference: {reference}", (10, band.shape[0] + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2,
    )
    target = OUT / f"cmpd{compound_id}_line{index:02d}.png"
    cv2.imwrite(str(target), canvas)
    return target


def main() -> int:
    cells = _captured_cells()
    # The pipeline's own reading of each band.
    from cpd.ocr.printed_smiles import PrintedSmilesOCR

    reader = PrintedSmilesOCR(route="padded")
    readings_path = (
        REPO / "data" / "audit" / "phase8" / "patent2_vote_effect.csv"
    )
    with readings_path.open(encoding="utf-8", newline="") as handle:
        joined = {
            row["cmpd_id"]: row["text_after"] for row in csv.DictReader(handle)
        }

    rows: list[dict[str, object]] = []
    for compound_id, lines in REFERENCE_LINES.items():
        bands = cells.get(str(compound_id), [])
        printed_lines: list[str] = []
        printed_images: list[np.ndarray] = []
        for band in bands:
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            for top, bottom in ink_bands(gray):
                crop = band[top:bottom, :]
                printed_images.append(crop)
                printed_lines.append(reader.read_cell(crop) or "")
        reading = joined.get(str(compound_id), "")
        for index, reference in enumerate(lines):
            image = (
                printed_images[index]
                if index < len(printed_images)
                else np.full((10, 10, 3), 255, np.uint8)
            )
            target = _write_pair(
                str(compound_id), index, image, reference,
                printed_lines[index] if index < len(printed_lines) else "",
            )
            rows.append({
                "cmpd_id": compound_id,
                "line_index": index,
                "reference": reference,
                "pipeline_reading": (
                    printed_lines[index]
                    if index < len(printed_lines)
                    else ""
                ),
                "printed_lines_available": len(printed_lines),
                "evidence": target.as_posix(),
            })

    out_csv = OUT / "reference_audit.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_csv}  ({len(rows)} reference lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
