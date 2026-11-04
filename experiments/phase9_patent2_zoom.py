"""Cut the printed SMILES cell of a WO2022251372 compound for reading.

The patent is a pure rasterised scan with no text layer, so the only
way to check a transcription is to look at the pixels.  This script
captures the cell the pipeline actually hands to the recogniser and
writes it as a PNG at 3x, so the characters can be read one by one.

The cell is captured the same way ``phase8_pipeline_cells_corpus.py``
captures it: the extractor is subclassed and the method that hands
the crop to the recogniser is overridden, so the pixels are exactly
the ones the pipeline sees.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_zoom.py 10 28
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import (  # noqa: E402
    OcrStats,
    RapidOcrTableExtractor,
)

OUT = REPO / "data" / "audit" / "phase9" / "patent2_zoom"
IMAGE_DIR = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


class _CellCapturingExtractor(RapidOcrTableExtractor):
    """Records the SMILES cell instead of recognising it."""

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
            self.cells.setdefault(self._current_id, []).append(cell.copy())
        return None, OcrStats()


def main() -> int:
    wanted = set(sys.argv[1:]) or {"10", "28"}
    profile = get_profile("WO2022251372")
    extractor = _CellCapturingExtractor(
        rec_top_k=0, smiles_route="padded", profile=profile,
    )
    OUT.mkdir(parents=True, exist_ok=True)

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

    for compound_id, bands in extractor.cells.items():
        if compound_id not in wanted:
            continue
        for index, band in enumerate(bands):
            big = cv2.resize(
                band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            target = OUT / f"cmpd{compound_id}_band{index}.png"
            cv2.imwrite(str(target), big)
            print(f"-> {target}  {big.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
