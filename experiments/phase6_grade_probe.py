"""Probe how the activity-grade cells (A/B/C/D) can be read.

WO2022251372 reports activity as a letter grade per cell line, not as
a number.  The generic cell reader runs a *detector* first, and a
detector has little to fire on for a single isolated capital letter in
the middle of a wide cell.  This probe compares the detector-based
path with a recognition-only path on the same crops.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_grade_probe.py
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

from cpd.ocr.printed_smiles import (  # noqa: E402
    default_recogniser,
    english_recogniser_only,
)
from cpd.parsers.page_tables import locate_page_tables  # noqa: E402

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"

# The six activity-grade columns of Table 1, as crop-relative pairs.
# Table 1 has 8 columns: No. | SMILES | MDM2 | MDM4 | MOLM-13 | MV4-11
# | WaGa | MKL-1, and the last six hold the A/B/C/D grades.
GRADE_X = (
    (655, 861), (861, 1068), (1068, 1349),
    (1349, 1574), (1574, 1780), (1780, 2005),
)


def trim(cell: np.ndarray) -> np.ndarray:
    """Crop to the single ink run, ignoring table rules and noise.

    A naive "any dark pixel" test keeps every column alive, because a
    scanned rule leaves anti-aliased fringes on neighbouring columns and
    the paper carries speckle.  A column only counts as ink when it has
    a meaningful number of dark pixels *and* is not a full-height rule.
    The letter is then the widest contiguous run of such columns.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    dark = gray < 160
    height, width = dark.shape

    # Mask the horizontal rules: a rule row is dark across nearly the
    # whole cell width.  Left in place they make every column look
    # inked, because each column crosses the rule.
    row_counts = dark.sum(axis=1)
    rule_rows = row_counts >= 0.8 * width
    work = dark.copy()
    work[rule_rows, :] = False

    counts = work.sum(axis=0)
    rule_cols = counts >= 0.8 * height
    ink = (counts >= 3) & (~rule_cols)
    keep = np.flatnonzero(ink)
    if not len(keep):
        return cell
    # Widest contiguous run of ink columns = the glyph.
    runs: list[list[int]] = [[int(keep[0])]]
    for value in keep[1:]:
        if int(value) - runs[-1][-1] <= 3:
            runs[-1].append(int(value))
        else:
            runs.append([int(value)])
    best = max(runs, key=lambda run: run[-1] - run[0])
    rows = np.flatnonzero(
        work[:, best[0]:best[-1] + 1].sum(axis=1) >= 2
    )
    if not len(rows):
        return cell
    return cell[rows[0]:rows[-1] + 1, best[0]:best[-1] + 1]


def main() -> int:
    ch = default_recogniser()
    en = english_recogniser_only()
    page = cv2.imread(str(PAGES / "000107.tif"))
    table = locate_page_tables(page)[0]
    bands = [(2513, 2714), (2714, 3075)]
    for band_index, (y1, y2) in enumerate(bands):
        for x1, x2 in GRADE_X:
            cell = page[y1:y2, table.x1 + x1:table.x1 + x2]
            tight = trim(cell)
            padded = cv2.copyMakeBorder(
                tight, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            det = ch(padded)
            det_txt = (getattr(det, "txts", None) or [None])[0]
            rec_ch = ch(padded, use_det=False, use_rec=True)
            rec_ch_txt = (getattr(rec_ch, "txts", None) or [None])[0]
            rec_en = en(padded, use_det=False, use_rec=True)
            rec_en_txt = (getattr(rec_en, "txts", None) or [None])[0]
            print(
                f"band{band_index} x=[{x1},{x2}] "
                f"tight={tight.shape[1]}x{tight.shape[0]} "
                f"det={det_txt!r} rec_ch={rec_ch_txt!r} "
                f"rec_en={rec_en_txt!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
