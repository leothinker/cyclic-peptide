r"""Row-by-row comparison of the cell path against the page path.

The stage-by-stage diagnostic showed that the production cell path gives
cmpd 1036 an edit distance of 82 while the page path with the SAME
preprocessing gives 3, and that the CSV reproduces the cell path exactly.
The detector geometry is identical in both cases (no box clips its first
glyph), so the difference is in what the recogniser is shown.

This script prints the rows of each path next to each other so the
mechanism is visible rather than inferred.

Read-only.

Run:
    .venv\Scripts\python.exe phase3_row_by_row.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, PREP, attribute, baseline_engine, load_cells, recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "row_by_row"
PAGE = "I100297"


def _sorted_texts(result) -> list[str]:
    """Recognition outputs in reading order."""
    boxes = list(result.boxes)
    txts = list(result.txts)
    pairs = sorted(zip(boxes, txts), key=lambda bt: bt[0][0][1])
    return [t.strip() for _, t in pairs if t and t.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()[PAGE]
    page = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{PAGE}.jpg"))
    engine = baseline_engine()
    ex = RapidOcrTableExtractor(rec_top_k=0)
    ref = REFERENCES["1036"]["smiles"]

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    up_page = PREP["up2_th200"](page)
    say(f"page path input: {up_page.shape} (page {page.shape})")
    rows = recognise_page(engine, up_page)
    buckets = attribute(rows, cells, 0.5)
    page_rows = buckets["1036"]
    say(f"page path rows for 1036: {len(page_rows)}")
    for i, r in enumerate(page_rows):
        x1 = r.poly[:, 0].min() / 2.0
        x2 = r.poly[:, 0].max() / 2.0
        y1 = r.poly[:, 1].min() / 2.0
        say(f"  P{i}  x[{x1:.0f},{x2:.0f}] y[{y1:.0f}] {r.text!r}")
    say()

    meta = next(c for c in cells if c.cmpd_id == "1036")
    cell = page[int(meta.y1):int(meta.y2), 1088:1919]
    trimmed = ex._trim_to_content(cell)
    up_cell = ex._upscale(trimmed)
    say(f"cell path input: {up_cell.shape} "
        f"(crop {cell.shape}, trimmed {trimmed.shape})")
    cell_rows = _sorted_texts(engine(up_cell))
    say(f"cell path rows for 1036: {len(cell_rows)}")
    for i, t in enumerate(cell_rows):
        say(f"  C{i}  {t!r}")
    say()

    say("=== row-by-row: page path vs cell path ===")
    n = max(len(page_rows), len(cell_rows))
    for i in range(n):
        p = page_rows[i].text.strip() if i < len(page_rows) else ""
        c = cell_rows[i] if i < len(cell_rows) else ""
        mark = "  " if p == c else "**"
        say(f"{mark} {i}:")
        say(f"     page: {p!r}")
        say(f"     cell: {c!r}")
    say()
    pj = "".join(r.text.strip() for r in page_rows)
    cj = "".join(cell_rows)
    say(f"page path: len={len(pj)} edit={edit_ops(ref, pj)}")
    say(f"cell path: len={len(cj)} edit={edit_ops(ref, cj)}")
    say(f"reference: len={len(ref)}")
    say(f"page path == reference ? {pj == ref}")

    (OUT / "row_by_row.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
