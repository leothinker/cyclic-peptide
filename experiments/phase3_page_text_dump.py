r"""Dump the actual recognised text for a sample of compounds.

The aggregate numbers in ``page_scale_summary.txt`` are encouraging but
they cannot be inspected. A string can balance its parentheses and still
be wrong: it may have absorbed a neighbouring column, have lost a whole
printed row, or simply be a different string altogether.

This script writes the RECOGNISED TEXT ITSELF for a sample of
compounds, for every arm, together with the shipped CSV string, so that
each one can be read and compared by a human. The sample deliberately
includes:

    1036, 1037      the two compounds with a text reference
    the cells where the page arm and the cell arm disagree most
    the cells the page arm failed to parse
    a few ordinary cells as controls

For each string the check is structural and decidable without a gold
standard:

    chars       every character is legal in a SMILES string
    len         the length is plausible for a printed cell
    open/close  parentheses balance
    rows        how many printed rows were recovered

Read-only. Writes to ``data/audit/phase3/page_text_dump/``.

Run:
    .venv\Scripts\python.exe phase3_page_text_dump.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, Cell, attribute, baseline_engine, default_engine, recognise_page,
)
from phase3_quality_screen import SMILES_CHARS  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "page_text_dump"


def _prep(img, name):
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if name == "none":
        return img, 1.0
    if name == "up2":
        return up, 0.5
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), 0.5


def _illegal(s: str) -> str:
    return "".join(sorted({c for c in s if c not in SMILES_CHARS}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dfc = pd.read_csv(REPO / "data" / "processed" / "compounds_combined.csv",
                      dtype=str)
    csv_map = dict(zip(dfc["cmpd_id"], dfc["smiles"].fillna("")))

    man = pd.read_csv(REPO / "data" / "audit" / "phase2" / "row_crop_audit"
                      / "manifest.csv", encoding="utf-8-sig", dtype=str)
    man = man[(man.is_header == "no") & (man.extraction_status == "ok")]

    scale = pd.read_csv(REPO / "data" / "audit" / "phase3" / "page_scale"
                        / "page_scale.csv", encoding="utf-8-sig")

    sample_ids: list[str] = ["1036", "1037"]
    piv = scale.pivot_table(index="cmpd_id", columns="arm",
                             values="paren_excess")
    if "default/up2" in piv.columns and "baseline/up2_th200" in piv.columns:
        piv["diff"] = piv["baseline/up2_th200"] - piv["default/up2"]
        worst = piv.sort_values("diff", ascending=False).head(8).index
        sample_ids += [c for c in worst if c not in sample_ids]
    unparsed = scale[(scale.arm == "default/up2") & (~scale.parses)]
    sample_ids += [c for c in unparsed.cmpd_id.head(5)
                   if c not in sample_ids]
    ordinary = scale[(scale.arm == "default/up2") & scale.parses]
    sample_ids += [c for c in ordinary.cmpd_id.head(5)
                   if c not in sample_ids]

    sub = man[man.cmpd_id.isin(sample_ids)]
    pages = sorted(sub.page.unique())

    engines = {"default": default_engine(), "baseline": baseline_engine()}
    rows: list[dict] = []

    for page in pages:
        img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
        if img is None:
            continue
        cells_sub = sub[sub.page == page]
        page_cells = []
        for _, r in cells_sub.iterrows():
            page_cells.append(Cell(
                cmpd_id=r.cmpd_id, page=page,
                x1=float(r.bbox_x1), y1=float(r.bbox_y1),
                x2=float(r.bbox_x2), y2=float(r.bbox_y2),
                n_lines=int(r.n_lines)))

        for arm in ("default/none", "default/up2", "default/up2_th200",
                    "baseline/up2_th200"):
            eng_kind, prep_name = arm.split("/")
            prepared, scale_f = _prep(img, prep_name)
            rws = recognise_page(engines[eng_kind], prepared)
            buckets = attribute(rws, page_cells, scale_f)
            for c in page_cells:
                picked = buckets.get(c.cmpd_id, [])
                txt = "".join(r.text.strip() for r in picked)
                csv_txt = csv_map.get(c.cmpd_id, "") or ""
                ref = REFERENCES.get(c.cmpd_id, {}).get("smiles", "")
                rows.append({
                    "arm": arm,
                    "page": page,
                    "cmpd_id": c.cmpd_id,
                    "n_rows": len(picked),
                    "len": len(txt),
                    "open": txt.count("("),
                    "close": txt.count(")"),
                    "illegal_chars": _illegal(txt),
                    "csv_len": len(csv_txt),
                    "csv_illegal": _illegal(csv_txt),
                    "edit_vs_ref": edit_ops(ref, txt) if ref else "",
                    "text": txt,
                })

    rows.sort(key=lambda r: (r["cmpd_id"], r["arm"]))
    with (OUT / "page_text_dump.csv").open("w", newline="",
                                            encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    out = io.open(OUT / "page_text_dump.txt", "w", encoding="utf-8")
    out.write("=== recognised text, sample by sample ===\n\n")
    out.write("Every string below was produced by the arm named above it.\n")
    out.write("Read them directly: a contaminated or truncated string is\n")
    out.write("visible, and no aggregate number can substitute for that.\n\n")

    by_cmpd: dict[str, list[dict]] = {}
    for r in rows:
        by_cmpd.setdefault(r["cmpd_id"], []).append(r)

    for cmpd in sorted(by_cmpd, key=lambda c: (len(c), c)):
        group = by_cmpd[cmpd]
        ref = REFERENCES.get(cmpd, {}).get("smiles", "")
        out.write(f"===== compound {cmpd} (page {group[0]['page']}) "
                  f"=====\n")
        if ref:
            out.write(f"  REFERENCE (text)  len={len(ref)} "
                      f"open={ref.count('(')} close={ref.count(')')}\n")
            out.write(f"    {ref}\n\n")
        for r in group:
            flag = ""
            if r["illegal_chars"]:
                flag = f"  !! illegal chars {r['illegal_chars']!r}"
            if r["n_rows"] != 7:
                flag += f"  !! rows={r['n_rows']}"
            out.write(f"  {r['arm']:<22} len={r['len']:<4} "
                      f"open={r['open']:<3} close={r['close']:<3} "
                      f"rows={r['n_rows']:<3} "
                      f"edit_vs_ref={r['edit_vs_ref']}{flag}\n")
            out.write(f"    {r['text']}\n")
        out.write("\n")

    out.write("=== checks across the whole sample ===\n")
    for arm in sorted({r["arm"] for r in rows}):
        g = [r for r in rows if r["arm"] == arm]
        out.write(f"  {arm:<22} n={len(g):<3} "
                  f"illegal={sum(1 for r in g if r['illegal_chars']):<3} "
                  f"rows!=7={sum(1 for r in g if r['n_rows'] != 7):<3} "
                  f"median_len={sorted(r['len'] for r in g)[len(g) // 2]}\n")
    out.write("\n  csv arm for comparison: "
              f"illegal={sum(1 for r in rows if r['csv_illegal'])} "
              f"(over {len(rows)} samples)\n")

    out.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
