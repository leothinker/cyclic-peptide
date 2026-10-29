r"""Removing the structure column from the whole-page route.

The problem
-----------
The whole-page route reads the page, so the detector also sees the
structure column next to the SMILES column. A detected box that
straddles the column boundary contributes text that does not belong to
the printed SMILES. On 1036 that is not a theory: all 31 character
errors of the whole-page route sit in one trailing fragment,

    reference  ...C(=O)N1)C(=O)N(C)C
    whole page ...C(=O)N1)(C)(()=)((()=))(()=()()()

which is structure-drawing ink read as characters.

The fix under test
------------------
Attribute a detected row to a compound only when the WHOLE box lies
inside the production SMILES column, not merely when its centre does.
Three arms share one page recognition, so the comparison costs almost
nothing:

    centre        the current rule: centre inside the column
    boxclip       the whole box must be inside the column
    boxclip_recrop  as boxclip, but every surviving row is cropped out
                   of the page and recognised on its own

Read-only. Writes to ``data/audit/phase3/column_clean/``.

Run:
    .venv\Scripts\python.exe phase3_page_column_clean.py --workers 3
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from phase3_quality_screen import screen_joined  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
OUT = REPO / "data" / "audit" / "phase3" / "column_clean"

ARMS = ("centre", "boxclip", "boxclip_recrop")
LEN_MIN, LEN_MAX = 220, 300
MARGIN = 4

_LOCAL = threading.local()
_LOCK = threading.Lock()


def _manifest():
    """Audit cells with the printed row count attached."""
    out = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({"cmpd_id": r["cmpd_id"], "x1": float(r["bbox_x1"]), "y1": float(r["bbox_y1"]), "x2": float(r["bbox_x2"]), "y2": float(r["bbox_y2"]), "n_lines": int(r["n_lines"])})
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _recrop(engine, prepared, row, up):
    """Crop one row out of the upscaled page and recognise it on its own.

    ``row`` holds page coordinates; ``prepared`` is the page at ``up``
    times the original size.
    """
    x1 = row["x1"] * up
    y1 = row["y1"] * up
    x2 = row["x2"] * up
    y2 = row["y2"] * up
    h, w = prepared.shape[:2]
    px1 = max(0, int(np.floor(x1)) - MARGIN)
    py1 = max(0, int(np.floor(y1)) - MARGIN)
    px2 = min(w, int(np.ceil(x2)) + MARGIN)
    py2 = min(h, int(np.ceil(y2)) + MARGIN)
    if px2 <= px1 or py2 <= py1:
        return ""
    crop = prepared[py1:py2, px1:px2]
    if crop.size == 0:
        return ""
    res = engine(crop, use_det=False, use_rec=True)
    txts = getattr(res, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _process_page(page, cells):
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    if img is None:
        return []
    col_x1, col_x2 = _smiles_column(img)
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

    recrop_text = {}
    for arm in ARMS:
        need_recrop = arm == "boxclip_recrop"
        for r in info:
            if need_recrop and r["cx"] < col_x1 - MARGIN:
                continue
            if r["x1"] < col_x1 - MARGIN:
                continue
            if r["x2"] > col_x2 + MARGIN:
                continue
            key = (r["x1"], r["y1"], r["x2"], r["y2"])
            if key not in recrop_text:
                recrop_text[key] = _recrop(engine, prepared, r, 2.0)

    out = []
    for c in cells:
        cx1 = max(c["x1"], col_x1)
        cx2 = min(c["x2"], col_x2)
        for arm in ARMS:
            picked = []
            for r in info:
                if not (c["y1"] <= r["cy"] <= c["y2"]):
                    continue
                if arm == "centre":
                    if not (cx1 <= r["cx"] <= cx2):
                        continue
                    picked.append((r, r["text"]))
                    continue
                if r["x1"] < col_x1 - MARGIN or r["x2"] > col_x2 + MARGIN:
                    continue
                if not (cx1 <= r["cx"] <= cx2):
                    continue
                key = (r["x1"], r["y1"], r["x2"], r["y2"])
                txt = recrop_text.get(key, r["text"]) if arm == "boxclip_recrop" else r["text"]
                picked.append((r, txt))
            picked.sort(key=lambda t: t[0]["cy"])
            txt = "".join(t for _, t in picked if t)
            s = screen_joined(c["cmpd_id"], txt)
            out.append({"page": page, "cmpd_id": c["cmpd_id"], "arm": arm, "n_rows": len(picked), "expected_rows": c["n_lines"], "rows_match": len(picked) == c["n_lines"], "len": len(txt), "open": txt.count("("), "close": txt.count(")"), "balanced": s.joined_balanced, "parses": s.joined_parses, "strict_pass": bool(s.joined_balanced and len(picked) == c["n_lines"] and LEN_MIN <= len(txt) <= LEN_MAX), "text": txt})
    return out


def _worker(args):
    page, cells = args
    try:
        return _process_page(page, cells)
    except Exception as exc:
        return [{"page": page, "cmpd_id": "", "arm": "ERROR", "error": repr(exc)[:200]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _manifest()
    if args.pages:
        cells_by_page = {p: c for p, c in cells_by_page.items() if p in set(args.pages)}
    tasks = [(p, c) for p, c in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} arms={len(ARMS)} workers={args.workers}")

    all_rows = []
    done = 0
    pf = (OUT / "column_clean_partial.csv").open("w", newline="", encoding="utf-8-sig")
    pw = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            try:
                new_rows = fut.result()
            except Exception as exc:
                print(f"  failed: {exc!r}")
                continue
            all_rows.extend(new_rows)
            with _LOCK:
                if pw is None and new_rows:
                    pw = csv.DictWriter(pf, fieldnames=list(new_rows[0]))
                    pw.writeheader()
                if pw is not None:
                    for row in new_rows:
                        pw.writerow(row)
                    pf.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} pages done")
    pf.close()

    errors = [r for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    rows.sort(key=lambda r: (r["arm"], r["page"], r["cmpd_id"]))
    if rows:
        with (OUT / "column_clean.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    lines = []
    lines.append("=== structure-column contamination removed ===")
    lines.append("")
    lines.append(f"cells  : {len({r['cmpd_id'] for r in rows})}")
    lines.append(f"pages  : {len({r['page'] for r in rows})}")
    lines.append(f"errors : {len(errors)}")
    lines.append("")
    lines.append(f"{'arm':<18} {'n':>4} {'bal':>5} {'rowok':>6} {'strict':>7} {'parse':>6} {'edit1036':>9}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        edit = ""
        for r in sub:
            if r["cmpd_id"] == "1036":
                edit = edit_ops(REFERENCES["1036"]["smiles"], r["text"])
                break
        lines.append(f"{arm:<18} {len(sub):>4} {sum(1 for r in sub if r['balanced']):>5} {sum(1 for r in sub if r['rows_match']):>6} {sum(1 for r in sub if r['strict_pass']):>7} {sum(1 for r in sub if r['parses']):>6} {str(edit):>9}")
    lines.append("")
    lines.append("centre        : a row counts when its centre is inside the column")
    lines.append("boxclip       : the whole box must be inside the column")
    lines.append("boxclip_recrop: as boxclip, every row is cropped out and recognised alone")
    lines.append("")
    lines.append("Text level. No structure is verified.")
    io.open(OUT / "column_clean.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
