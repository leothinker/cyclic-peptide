"""Read the captured WO2022251372 cells line by line.

The reference transcriptions in ``wo2022251372_references.py`` split
the printed text into lines.  This script shows what the pipeline
actually reads for each printed band of one compound, so that a
reference line and a printed line can be put next to each other.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_lines.py 28
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.lines import ink_bands  # noqa: E402
from cpd.ocr.printed_smiles import PrintedSmilesOCR  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9" / "patent2_zoom"


def main() -> int:
    compound = sys.argv[1]
    route = sys.argv[2] if len(sys.argv) > 2 else "padded"
    reader = PrintedSmilesOCR(route=route)
    for band_path in sorted(OUT.glob(f"cmpd{compound}_band*.png")):
        if "_line" in band_path.name or "_zoom" in band_path.name:
            continue
        image = cv2.imread(str(band_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"\n=== {band_path.name}")
        for index, (top, bottom) in enumerate(ink_bands(gray)):
            crop = image[top:bottom, :]
            print(f"  line {index:>2}: {reader.read_cell(crop)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
