"""Trace one WO2022251372 compound from the pixels to the final string.

The route comparison reports the *final* string, so a difference
between the printed text and the final string cannot be attributed.
This script shows every step for one compound: the raw reading of the
cell, the reading after the repair chain, and the reading after the
second recogniser votes.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_trace.py 28
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles  # noqa: E402
from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import (  # noqa: E402
    OcrStats,
    RapidOcrTableExtractor,
)

IMAGE_DIR = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


class _Capturing(RapidOcrTableExtractor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cells: dict[str, list] = {}
        self._current_id: str | None = None

    def _read_cmpd_id(self, cell):
        result = super()._read_cmpd_id(cell)
        if result:
            self._current_id = result
        return result

    def _ocr_smiles_with_stats(self, cell):
        if self._current_id and cell.size:
            self.cells.setdefault(self._current_id, []).append(cell.copy())
        return None, OcrStats()


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "28"
    profile = get_profile("WO2022251372")
    extractor = _Capturing(
        rec_top_k=0, smiles_route="padded", profile=profile,
    )
    prev_columns: list = []
    for path in sorted(IMAGE_DIR.iterdir()):
        image = cv2.imread(str(path))
        if image is None:
            continue
        _records, _assays, columns = extractor.extract_records(
            path, cells_dir=None, prev_columns=prev_columns,
        )
        if columns:
            prev_columns = columns
    extractor.flush_pending()

    for route in ("padded", "padded_vote"):
        reader = PrintedSmilesOCR(route=route)
        parts = []
        for band in extractor.cells.get(wanted, []):
            parts.append(reader.read_cell(band) or "")
        raw = "".join(parts)
        print(f"\n=== route {route}")
        print(f"  raw      : {raw!r}")
        print(f"  repaired : {auto_repair_smiles(raw)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
