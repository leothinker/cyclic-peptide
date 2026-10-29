r"""Which cell-crop variant lets the default engine work?

The puzzle
----------
The default engine reads 1036 perfectly on the whole page and produces
garbage on the cell crop of the same compound:

    page, default/none    CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3...
    cell, default/none    ...C7(CCCC()()(=((()(.)C()((=)(C)O=(()()=((S=)(),()C(=))...

The detector's box geometry does not explain it: the page path has MORE
overlapping boxes and MORE over-tall boxes, and still reads the compound
exactly. So the difference is in the pixels the recogniser is given.

The hypothesis
--------------
``_trim_to_content`` crops the cell to the bounding box of its dark
content, with a 4 px pad. That places the first printed line against
the top edge of the image and the last against the bottom edge, and it
can cut into ascenders and descenders of the outermost lines. A line
that is clipped at the image border is a line the recogniser has to
finish from incomplete strokes, and that is where the '(()()=((' garbage
comes from.

The test
--------
The same cell region, cut five different ways, all read by the same
default engine at the same scale:

    cell_trimmed      production: _trim_to_content, 4 px pad
    cell_untrimmed    the raw column band, no trim
    cell_padded       trimmed, then 20 px of white margin added back
    cell_untrim_pad   raw band, then 20 px of white margin added back
    page              the whole page, as the reference

The margin is added as white padding, which is only used to test
whether the border itself is the problem. It is NOT a proposed fix: a
white margin cannot restore a clipped glyph, so if the trimmed crop
fails and the padded one succeeds, the trim was cutting real ink.

Read-only. Writes to ``data/audit/phase3/cell_crop_variants/``.

Run:
    .venv\Scripts\python.exe phase3_cell_crop_variants.py
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

OUT = REPO / "data" / "audit" / "phase3" / "cell_crop_variants"


def _pad_white(img, pad=20):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


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

    out = io.open(OUT / "cell_crop_variants.txt", "w", encoding="utf-8")
    out.write("=== cell crop variants, default engine, same pixels ===\n\n")
    out.write("1036 = human-verified reference. 1037 = agent transcription, "
              "not human-confirmed.\n")
    out.write("A white margin is a TEST of whether the image border is the")
    out.write("problem, not a proposed fix.\n\n")
    out.write(f"{'variant':<16} {'cmpd':>6} {'rows':>5} {'len':>5} "
              f"{'open':>5} {'close':>6} {'edit':>5} {'CER':>8}\n")

    csv_rows: list[dict] = []
    for band_idx in (1, 2):
        y1, y2 = rows[band_idx]
        raw = img[y1:y2, col.x1:col.x2]
        if raw.size == 0:
            continue
        trimmed = ex._trim_to_content(raw)

        variants = {
            "cell_trimmed": trimmed,
            "cell_untrimmed": raw,
            "cell_padded": _pad_white(trimmed),
            "cell_untrim_pad": _pad_white(raw),
        }
        for name, prepared in variants.items():
            txt, parts = _recognise_joined(engine, prepared)
            cmpd = "1036" if band_idx == 1 else "1037"
            ref = REFERENCES.get(cmpd, {}).get("smiles", "")
            ed = edit_ops(ref, txt) if ref else ""
            cer = round(ed / len(ref), 5) if ref else ""
            out.write(f"{name:<16} {cmpd:>6} {len(parts):>5} {len(txt):>5} "
                      f"{txt.count('('):>5} {txt.count(')'):>6} "
                      f"{str(ed):>5} {str(cer):>8}\n")
            csv_rows.append({
                "variant": name, "cmpd_id": cmpd, "rows": len(parts),
                "len": len(txt), "open": txt.count("("),
                "close": txt.count(")"), "edit": ed, "cer": cer,
                "text": txt,
            })

    out.write("\n--- the page path, for comparison ---\n")
    from phase3_page_extract import Cell, attribute, recognise_page
    page_cells = []
    for band_idx, cmpd in ((1, "1036"), (2, "1037")):
        y1, y2 = rows[band_idx]
        page_cells.append(Cell(cmpd_id=cmpd, page="I100297",
                               x1=col.x1, y1=y1, x2=col.x2, y2=y2,
                               n_lines=0))
    rws = recognise_page(engine, img)
    buckets = attribute(rws, page_cells, 1.0)
    for c in page_cells:
        picked = buckets.get(c.cmpd_id, [])
        txt = "".join(r.text.strip() for r in picked)
        ref = REFERENCES.get(c.cmpd_id, {}).get("smiles", "")
        ed = edit_ops(ref, txt) if ref else ""
        cer = round(ed / len(ref), 5) if ref else ""
        out.write(f"{'page':<16} {c.cmpd_id:>6} {len(picked):>5} "
                  f"{len(txt):>5} {txt.count('('):>5} {txt.count(')'):>6} "
                  f"{str(ed):>5} {str(cer):>8}\n")
        csv_rows.append({
            "variant": "page", "cmpd_id": c.cmpd_id, "rows": len(picked),
            "len": len(txt), "open": txt.count("("), "close": txt.count(")"),
            "edit": ed, "cer": cer, "text": txt,
        })

    with (OUT / "cell_crop_variants.csv").open("w", newline="",
                                                encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    out.write("\n--- reading the table ---\n")
    out.write("If 'cell_trimmed' fails while 'cell_padded' succeeds, the")
    out.write("trim cut real ink and the border was the issue.\n")
    out.write("If both fail while 'page' succeeds, the crop itself is")
    out.write("fundamentally harder for the detector than the page, and the")
    out.write("answer is to recognise the page and attribute the lines")
    out.write("afterwards, rather than to keep tuning the cell crop.\n")

    out.close()
    print("".join(io.open(OUT / "cell_crop_variants.txt",
                           encoding="utf-8").read()))


if __name__ == "__main__":
    main()
