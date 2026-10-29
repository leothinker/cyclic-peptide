r"""The report's core table: four routes against two verified references.

Every claim in the Phase 3 summary about character accuracy comes from
this table, so it is recomputed from scratch here rather than copied
from an earlier run.

Routes
------
    production        the shipped pipeline: cell crop, 2x, English
                      PP-OCRv4 recogniser, no binarisation
    + threshold 200   the same, with the binarisation the pipeline is
                      missing
    whole page        whole page, 2x, default engine, rows attributed
                      inside the production SMILES column
    page + line rec   page detection, then every row cropped out of the
                      2x page and recognised on its own, default engine

References
----------
    1036   human-verified transcription (the user's own, read off the
           patent image)
    1037   human-verified this session: the earlier transcription was
           re-read from the printed cell and agrees character for
           character (see ``reference_status.md``)

The chemical structure is NOT verified by this table and the mass gate
is not consulted.

Read-only. Writes to ``data/audit/phase3/ref_routes/``.

Run:
    .venv\Scripts\python.exe phase3_ref_routes.py
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
from phase3_page_extract import Cell, recognise_page  # noqa: E402
from phase3_page_scale import _engines  # noqa: E402
from phase3_page_scale import _prep  # noqa: E402
from phase3_page_scale import _smiles_column  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "ref_routes"
GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")

REFS = ("1036", "1037")
UP = 2.0


def _manifest():
    out = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({"cmpd_id": r["cmpd_id"], "x1": float(r["bbox_x1"]), "y1": float(r["bbox_y1"]), "x2": float(r["bbox_x2"]), "y2": float(r["bbox_y2"]), "n_lines": int(r["n_lines"])})
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _cell_routes(ex, engine, page, cell):
    """The two cell-crop routes, on the production crop."""
    engine.use_det = True
    engine.use_rec = True
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    # The audit manifest records the compound's own rectangle on the
    # page, which is the same row band the production extractor cuts.
    crop = img[int(cell["y1"]):int(cell["y2"]), col.x1:col.x2]
    if crop.size == 0:
        return
    trimmed = ex._trim_to_content(crop)
    up = ex._upscale(trimmed)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    for name, image in (("production", up), ("+ threshold 200", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))):
        res = engine(image)
        if not res:
            continue
        items = sorted(zip(list(res.boxes), list(res.txts)), key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
        txt = "".join(t.strip() for _, t in items if t and t.strip())
        yield name, txt


def _page_routes(page, cell, col_x1, col_x2):
    """The two page routes, on the same page."""
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    prepared, scale = _prep(img, "up2")
    engine = _engines("default")
    # RapidOCR's update_params writes the flags onto the engine
    # instance and never resets them, so a recognition-only call would
    # silently disable detection for every later call on that object.
    engine.use_det = True
    engine.use_rec = True
    rows = recognise_page(engine, prepared)
    info = []
    for r in rows:
        xs = r.poly[:, 0] * scale
        ys = r.poly[:, 1] * scale
        info.append({"text": r.text.strip(), "x1": float(xs.min()), "y1": float(ys.min()), "x2": float(xs.max()), "y2": float(ys.max()), "cx": float(xs.mean()), "cy": float(ys.mean())})

    picked = [r for r in info if cell["y1"] <= r["cy"] <= cell["y2"] and col_x1 <= r["cx"] <= col_x2]
    picked.sort(key=lambda r: r["cy"])
    yield "whole page", "".join(r["text"] for r in picked if r["text"])

    parts = []
    for r in picked:
        h, w = prepared.shape[:2]
        x1 = max(0, int(np.floor(r["x1"] * UP)) - 3)
        y1 = max(0, int(np.floor(r["y1"] * UP)) - 3)
        x2 = min(w, int(np.ceil(r["x2"] * UP)) + 3)
        y2 = min(h, int(np.ceil(r["y2"] * UP)) + 3)
        if x2 <= x1 or y2 <= y1:
            continue
        res = engine(prepared[y1:y2, x1:x2], use_det=False, use_rec=True)
        txts = getattr(res, "txts", None)
        if txts and txts[0]:
            parts.append(txts[0].strip())
    yield "page + line rec", "".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    ex = RapidOcrTableExtractor(rec_top_k=0)
    engine = _engines("baseline")
    cells_by_page = _manifest()

    rows = []
    for page in sorted(cells_by_page):
        targets = [c for c in cells_by_page[page] if c["cmpd_id"] in REFS]
        if not targets:
            continue
        img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
        col_x1, col_x2 = _smiles_column(img)
        for cell in targets:
            for name, txt in _cell_routes(ex, engine, page, cell):
                rows.append({"cmpd_id": cell["cmpd_id"], "route": name, "text": txt})
            for name, txt in _page_routes(page, cell, col_x1, col_x2):
                rows.append({"cmpd_id": cell["cmpd_id"], "route": name, "text": txt})

    lines = []
    lines.append("=== four routes against two human-verified references ===")
    lines.append("")
    lines.append(f"{'cmpd':<6} {'route':<20} {'edit':>5} {'cer':>8} {'len':>5} {'ref':>5} {'open':>5} {'close':>6}")
    for cid in REFS:
        ref = REFERENCES[cid]["smiles"]
        for name in ("production", "+ threshold 200", "whole page", "page + line rec"):
            hit = [r for r in rows if r["cmpd_id"] == cid and r["route"] == name]
            if not hit:
                continue
            txt = hit[0]["text"]
            edit = edit_ops(ref, txt)
            lines.append(f"{cid:<6} {name:<20} {edit:>5} {edit / len(ref):>8.5f} {len(txt):>5} {len(ref):>5} {txt.count('('):>5} {txt.count(')'):>6}")
    lines.append("")
    lines.append("1036 and 1037 are both human-verified transcriptions of the")
    lines.append("printed cells. Text level only: the chemical structure is")
    lines.append("not verified and the mass gate is not consulted.")

    with (OUT / "ref_routes.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["cmpd_id", "route", "text"])
        w.writeheader()
        w.writerows(rows)
    io.open(OUT / "ref_routes.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
