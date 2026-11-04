"""Find the most reliable way to read the WO2022251372 compound number.

The page-level OCR of the ``No.`` column failed on several numbers
(4, 9 and 14 were read as nothing), so the compound key cannot rely
on the generic cell reader.  This probe compares the candidate
readers on the real number cells:

* the generic detector path,
* the glyph-isolating path used for the activity grades, and
* the same path with the number restricted to the top of the cell.

Every band of every table is included, and the printed number is
taken from the band order so the readers can be scored.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_cmpd_number_probe.py
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

from cpd.ocr.constrained import isolate_glyph  # noqa: E402
from cpd.ocr.printed_smiles import (  # noqa: E402
    default_recogniser,
    english_recogniser_only,
)
from cpd.parsers.page_tables import (  # noqa: E402
    horizontal_rules,
    locate_page_tables,
)

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


def main() -> int:
    ch = default_recogniser()
    en = english_recogniser_only()
    stats = {
        "det_full": [0, 0],
        "det_top": [0, 0],
        "glyph": [0, 0],
        "glyph_top": [0, 0],
    }
    # Table 1a prints its numbers 1..28 in order, so the expected
    # value of a band can be derived from the running count.
    expected = {"8": 0, "3": 0}

    for path in sorted(PAGES.glob("*.tif")):
        page = cv2.imread(str(path))
        rules = horizontal_rules(page)
        for table in locate_page_tables(page):
            key = str(table.n_columns)
            inner = [y for y in rules if table.y1 <= y <= table.y2]
            x1 = table.x1 + table.column_boundaries[0]
            x2 = table.x1 + table.column_boundaries[1]
            for index, (y1, y2) in enumerate(
                zip(inner, inner[1:], strict=False)
            ):
                if index == 0:
                    continue  # header band
                cell = page[y1:y2, x1:x2]
                # A continuation band has no number; skip it.
                glyph_full = isolate_glyph(cell)
                if glyph_full is None:
                    continue
                expected[key] += 1
                truth = str(expected[key])

                top = cell[:90, :]
                glyph_top = isolate_glyph(top)

                def read_glyph(glyph):
                    if glyph is None:
                        return None
                    padded = cv2.copyMakeBorder(
                        glyph, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    for engine in (ch, en):
                        result = engine(padded, use_det=False, use_rec=True)
                        txts = getattr(result, "txts", None) or []
                        if txts and txts[0].strip():
                            return txts[0].strip()
                    return None

                def read_det(image):
                    if image.size == 0:
                        return None
                    result = ch(image)
                    txts = getattr(result, "txts", None) or []
                    return txts[0].strip() if txts else None

                readings = {
                    "det_full": read_det(cell),
                    "det_top": read_det(top),
                    "glyph": read_glyph(glyph_full),
                    "glyph_top": read_glyph(glyph_top),
                }
                for name, value in readings.items():
                    stats[name][1] += 1
                    if value == truth:
                        stats[name][0] += 1
                bad = {
                    k: v for k, v in readings.items() if v != truth
                }
                if bad:
                    print(
                        f"{path.stem} c{table.n_columns} b{index} "
                        f"truth={truth} {bad}"
                    )

    print()
    for name, (hits, total) in stats.items():
        rate = hits / total if total else 0.0
        print(f"{name:12s} {hits}/{total} = {rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
