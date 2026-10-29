r"""Locate exactly where the production path loses printed SMILES characters.

The production CSV holds a damaged string for every compound: cmpd 1036
has 18 opening parentheses where the printed cell has 38. Two mechanisms
could each account for that, and they need to be told apart.

    A. crop    the SMILES column is cut at a fraction of the page width, so
               the first glyph of every printed row falls outside the crop
    B. detect  the detector's box starts to the right of the first glyph,
               so the glyph is outside the crop fed to the recogniser

This script walks the production sequence step by step and prints the
parenthesis counts and the edit distance against the human-verified
reference after each stage:

    page  ->  column crop  ->  _trim_to_content  ->  2x upscale
          ->  detector + recogniser

and, separately, the same page recognised at page level, which has no
column crop in it at all.

Read-only.

Run:
    .venv\Scripts\python.exe phase3_prod_loss_diag.py
"""

from __future__ import annotations

import io
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
    GOLD, PREP, attribute, baseline_engine, joined, load_cells,
    recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "prod_loss_diag"
PAGE = "I100297"


def stats(tag: str, text: str, ref: str) -> str:
    return (f"{tag:<34s} len={len(text):>4d} "
            f"open={text.count('('):>3d} close={text.count(')'):>3d} "
            f"eq={text.count('='):>3d} "
            f"edit={edit_ops(ref, text):>4d}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()[PAGE]
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{PAGE}.jpg"))
    ex = RapidOcrTableExtractor(rec_top_k=0)
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    rows = ex._detect_row_boundaries(img)

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    say(f"page {PAGE}  shape={img.shape}")
    say(f"production SMILES column: x1={col.x1} x2={col.x2} "
        f"(fraction 0.58..0.99 of width={img.shape[1]})")
    say(f"row bands: {rows}")
    say()

    # ---------------- mechanism A: the column crop ----------------
    say("=== A. what the column crop itself removes ===")
    for cid in ("1036", "1037"):
        cell_meta = next(c for c in cells if c.cmpd_id == cid)
        # the audit crop starts at the first inked column inside the cell
        audit_x1 = int(cell_meta.x1)
        say(f"cmpd {cid}: audit crop starts at x={audit_x1}, "
            f"production column starts at x={col.x1} "
            f"-> {col.x1 - audit_x1} px of the cell are outside the crop")
        band = img[int(cell_meta.y1):int(cell_meta.y2), audit_x1:col.x1]
        if band.size == 0:
            say("  no pixels excluded on the left")
            continue
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        ink = gray < 128
        cols_with_ink = np.where(ink.sum(axis=0) > 0)[0]
        rows_with_ink = np.where(ink.sum(axis=1) > 0)[0]
        say(f"  ink in the excluded strip: "
            f"{len(cols_with_ink)} columns x "
            f"{len(rows_with_ink)} rows "
            f"(excluded width {col.x1 - audit_x1} px)")
        # how many printed text rows have ink inside the excluded strip
        rowink = ink.sum(axis=1)
        bands = []
        s = None
        for y, v in enumerate(rowink):
            if v > 0 and s is None:
                s = y
            elif v == 0 and s is not None:
                if y - s >= 8:
                    bands.append((s, y))
                s = None
        if s is not None:
            bands.append((s, len(rowink)))
        say(f"  printed rows inside the excluded strip: {len(bands)}")
    # the right-hand edge of the column
    for cid in ("1036", "1037"):
        cell_meta = next(c for c in cells if c.cmpd_id == cid)
        band = img[int(cell_meta.y1):int(cell_meta.y2), col.x2:int(cell_meta.x2)]
        if band.size:
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            ink = gray < 128
            say(f"cmpd {cid}: right edge excludes "
                f"{int(cell_meta.x2) - col.x2} px, "
                f"ink there: {int(ink.sum())} px")
        else:
            say(f"cmpd {cid}: right edge excludes 0 px")
    say()

    # ---------------- mechanism B: the production cell path ----------
    say("=== B. the production cell path, stage by stage (cmpd 1036) ===")
    ref = REFERENCES["1036"]["smiles"]
    meta = next(c for c in cells if c.cmpd_id == "1036")
    y1, y2 = int(meta.y1), int(meta.y2)

    for tag, x1, x2 in (("production col", col.x1, col.x2),
                         ("audit col", int(meta.x1), int(meta.x2))):
        cell = img[y1:y2, x1:x2]
        say(f"-- {tag}: crop x[{x1}:{x2}] shape={cell.shape}")
        for trim in (False, True):
            c = ex._trim_to_content(cell) if trim else cell
            up = ex._upscale(c)
            r = ex.rec_engine(up)
            items = sorted(zip(list(r.boxes), list(r.txts)),
                           key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
            txt = "".join(t.strip() for _, t in items if t and t.strip())
            say("   " + stats(f"trim={trim} 2x det+rec", txt, ref))
        say()

    # ---------------- the same page at page level -------------------
    say("=== C. the same page recognised at page level (no column crop) ===")
    engine = baseline_engine()
    for prep_name in ("none", "up2", "up2_th200"):
        prep = PREP[prep_name]
        scale = 1.0 if prep_name == "none" else 0.5
        rws = recognise_page(engine, prep(img))
        buckets = attribute(rws, cells, scale)
        for cid in ("1036", "1037"):
            r = REFERENCES[cid]["smiles"]
            txt = joined(buckets.get(cid, []))
            say("   " + stats(f"page {prep_name} {cid}", txt, r))
        say()

    say("=== D. the same comparison on the production CSV ===")
    import csv as _csv
    with (REPO / "data" / "processed" / "compounds_combined.csv").open(
            encoding="utf-8-sig") as fh:
        crows = {r["cmpd_id"]: r for r in _csv.DictReader(fh)}
    for cid in ("1036", "1037"):
        r = REFERENCES[cid]["smiles"]
        say("   " + stats(f"CSV {cid}", crows[cid]["smiles"], r))

    (OUT / "prod_loss_diag.txt").write_text("\n".join(lines),
                                            encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
