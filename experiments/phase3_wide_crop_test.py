r"""Is the narrow column the reason the cell crop loses a line?

What the data shows so far
--------------------------
Reading the variants on 1036 with the default engine:

    cell_untrimmed   ...NC7(CCCC()()(=((()(.)C()((=)(C)O=...
    cell_padded      ...NC7(CCCC)CS()((=)()[).).)(()=)...
    page             ...NC7(CCCC7)C(=O)N(C)[C@@H](C8...

In both cell variants the damage is at ONE place: the printed line

    5)C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CC

is not read as a line at all. What appears instead is a run of
punctuation, which is what the recogniser outputs for strokes that are
not glyphs.

The hypothesis
--------------
The SMILES column is narrow (about 830 px) and the printed line is
about 38 characters, so the characters are small. The detector has to
find a 38-character line inside a narrow strip. If the detector
clips that line horizontally, the recogniser receives a partial line
and fills the rest with punctuation.

The test
--------
The same cell content, read with the read region progressively widened
to the left and to the right. If widening recovers the missing line,
the detector was clipping it horizontally.

A second, independent test pastes the same cell content into a
page-sized white canvas. If the line is recovered there, the
difference is the surrounding context, which is what the page path
provides.

Read-only. Writes to ``data/audit/phase3/wide_crop_test/``.

Run:
    .venv\Scripts\python.exe phase3_wide_crop_test.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import GOLD, default_engine  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "wide_crop_test"

WIDENINGS = ((0, 0), (30, 0), (60, 0), (0, 30), (0, 60),
             (60, 60), (120, 120), (250, 250))


def _recognise_joined(engine, img):
    result = engine(img)
    if not result:
        return "", []
    items = sorted(zip(list(result.boxes), list(result.txts)),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    parts = [t.strip() for _, t in items if t and t.strip()]
    return "".join(parts), parts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    ex = RapidOcrTableExtractor(rec_top_k=0)
    engine = default_engine()

    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    rows = ex._detect_row_boundaries(img)

    y1, y2 = rows[1]
    ref = REFERENCES["1036"]["smiles"]

    out = io.open(OUT / "wide_crop_test.txt", "w", encoding="utf-8")
    out.write("=== is the narrow column the reason a line is lost? ===\n\n")
    out.write("1036, default engine, the same cell content read with the\n")
    out.write("read region widened to the left and to the right.\n\n")
    out.write(f"{'variant':<28} {'rows':>5} {'len':>5} {'edit':>5} "
              f"{'CER':>8}  note\n")

    rows_out: list[dict] = []

    def _run(name, prepared, note):
        txt, parts = _recognise_joined(engine, prepared)
        ed = edit_ops(ref, txt)
        out.write(f"{name:<28} {len(parts):>5} {len(txt):>5} {ed:>5} "
                  f"{ed / len(ref):>8.5f}  {note}\n")
        rows_out.append({
            "variant": name, "rows": len(parts), "len": len(txt),
            "edit": ed, "cer": round(ed / len(ref), 5), "text": txt,
        })

    for left, right in WIDENINGS:
        x1 = max(0, col.x1 - left)
        x2 = min(img.shape[1], col.x2 + right)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        _run(f"col[left-{left},right+{right}]", crop,
             f"x={x1}..{x2} width={x2 - x1}")

    cell = img[y1:y2, col.x1:col.x2]
    canvas = np.full((img.shape[0], img.shape[1], 3), 255, np.uint8)
    canvas[y1:y2, col.x1:col.x2] = cell
    _run("cell_on_page_canvas", canvas, "same pixels, page-sized canvas")

    for f in (1.5, 2.0, 3.0):
        up = cv2.resize(cell, None, fx=f, fy=f,
                        interpolation=cv2.INTER_CUBIC)
        _run(f"cell_scaled_{f}x", up, f"INTER_CUBIC {f}x, no threshold")

    with (OUT / "wide_crop_test.csv").open("w", newline="",
                                            encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    out.write("\n--- reading the table ---\n")
    best = min(rows_out, key=lambda r: r["edit"])
    out.write(f"best variant: {best['variant']} edit={best['edit']}\n")
    out.write("If widening the read region recovers the line, the detector")
    out.write(" was clipping it horizontally and the fix is a wider read")
    out.write(" region, not a different model.\n")
    out.write("If only the page canvas works, the detector needs the")
    out.write(" surrounding table context, and the production answer is to")
    out.write(" recognise the page and attribute the lines afterwards.\n")

    out.close()
    print("".join(io.open(OUT / "wide_crop_test.txt",
                           encoding="utf-8").read()))


if __name__ == "__main__":
    main()
