r"""Why does the cell-crop path lose the row-start glyph, and the page
path does not?

The audit in ``row_crop_audit`` measured the cell path: 215 of 215
truncation events, every one of them on the left edge, every one of them
shaped like the equals sign that starts a printed row.

That measurement says nothing about the page-level path, which has no
column crop in it. If the page path loses far fewer characters, then the
damage is caused by the crop-and-recognise-per-cell extraction and not by
the printed page itself.

This script measures the same geometric quantity on both paths:
for every detected line, the detector box's left edge is compared with the
leftmost inked column of that line, computed from the ORIGINAL page with
the table's vertical rules removed.

The comparison is fair because the input image differs in exactly one way:
the cell path gets a 831x625 crop of one compound, the page path gets the
whole 1939x1312 page. The recogniser, the preprocessing (2x INTER_CUBIC then
threshold 200) and the ink definition are identical.

Read-only.

Run:
    .venv\Scripts\python.exe phase3_detector_geometry.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase3_page_extract import (  # noqa: E402
    GOLD, PREP, attribute, baseline_engine, load_cells, recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "detector_geometry"
PAGE = "I100297"
INK = 128
PREP_NAME = "up2_th200"


def rule_columns(gray: np.ndarray) -> np.ndarray:
    """Columns inked over most of the page height: the table's vertical rules."""
    h, w = gray.shape
    dark = (gray < INK).sum(axis=0)
    return np.where(dark > 0.6 * h)[0]


def left_ink(gray: np.ndarray, y1: int, y2: int, rules: np.ndarray) -> int | None:
    """Leftmost inked column of a line band, with the vertical rules removed."""
    band = gray[y1:y2].copy()
    if rules.size:
        band[:, rules] = 255
    cols = np.where((band < INK).sum(axis=0) > 0)[0]
    return int(cols.min()) if cols.size else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()[PAGE]
    page = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{PAGE}.jpg"))
    gray_page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    rules = rule_columns(gray_page)
    engine = baseline_engine()
    prep = PREP[PREP_NAME]

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    say(f"page {PAGE} {page.shape}  prep={PREP_NAME}")
    say(f"vertical rule columns on the page: "
        f"{rules[:5].tolist()}...{rules[-5:].tolist()} "
        f"({rules.size} columns)")
    say()

    # ---- page path ----
    say("=== page path: detection on the whole 2x binarised page ===")
    up_page = prep(page)
    rows = recognise_page(engine, up_page)
    buckets = attribute(rows, cells, 0.5)
    page_stats = []
    for c in cells:
        got = buckets.get(c.cmpd_id, [])
        for r in got:
            if len(r.text.strip()) < 8:
                continue
            # the box is in 2x page coordinates
            x1 = float(r.poly[:, 0].min()) / 2.0
            y1 = float(r.poly[:, 1].min()) / 2.0
            y2 = float(r.poly[:, 1].max()) / 2.0
            ink_x1 = left_ink(gray_page, int(y1), int(y2), rules)
            if ink_x1 is None:
                continue
            over = ink_x1 - x1
            page_stats.append((c.cmpd_id, over, r.text[:34]))
    n_over = sum(1 for _, o, _ in page_stats if o > 1)
    say(f"lines measured: {len(page_stats)}")
    say(f"lines whose box starts RIGHT of the first ink: {n_over}")
    for cid, over, t in page_stats:
        if over > 1:
            say(f"   cmpd {cid}: box starts {over:.0f} px right of ink  {t!r}")
    say()

    # ---- cell path ----
    say("=== cell path: detection on the same 2x binarised 1036 crop ===")
    meta = next(c for c in cells if c.cmpd_id == "1036")
    cell = page[int(meta.y1):int(meta.y2), 1088:1919]
    up_cell = prep(cell)
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    rules_cell = rule_columns(gray_cell)
    rows = recognise_page(engine, up_cell)
    cell_stats = []
    for r in rows:
        if len(r.text.strip()) < 8:
            continue
        x1 = float(r.poly[:, 0].min()) / 2.0
        y1 = float(r.poly[:, 1].min()) / 2.0
        y2 = float(r.poly[:, 1].max()) / 2.0
        ink_x1 = left_ink(gray_cell, int(y1), int(y2), rules_cell)
        if ink_x1 is None:
            continue
        over = ink_x1 - x1
        cell_stats.append((over, r.text[:34], over))
    n_over_c = sum(1 for o, _, _ in cell_stats if o > 1)
    say(f"lines measured: {len(cell_stats)}")
    say(f"lines whose box starts RIGHT of the first ink: {n_over_c}")
    for over, t, _ in cell_stats:
        if over > 1:
            say(f"   box starts {over:.0f} px right of ink  {t!r}")
    say()
    say("=== summary ===")
    say(f"page path: {n_over}/{len(page_stats)} lines clip their first glyph")
    say(f"cell path: {n_over_c}/{len(cell_stats)} lines clip their first glyph")

    (OUT / "detector_geometry.txt").write_text("\n".join(lines),
                                               encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
