r"""Can a wider left crop recover the missing opening brackets?

The residual error of the page route with the default recogniser is
now characterised: of 335 cells, 190 come out with balanced
parentheses and 145 do not, and the failures are small. A typical
failing cell has 39 opening and 40 closing brackets, so a single
opening bracket was lost.

That is the same failure mode as the shipped pipeline, only far rarer,
and it points at the crop rather than at the recogniser: the detector
box is cropped exactly, so a bracket that leans outside the box is
cut away before the recogniser ever sees it.

This script widens the LEFT edge of every detected row crop by a
fixed number of pixels and re-recognises it. Four margins are run on
the same detections, so the detection stage is shared and the margin
is the only variable.

    margin 0    the current route
    margin 8    8 px extra on the left
    margin 16   16 px extra on the left
    margin 24   24 px extra on the left

The widened crop can reach into the structure column, so the script
also counts how often that happens and checks the two human-verified
references for degradation.

Read-only. Writes to ``data/audit/phase3/recrop_margin/``.

Run:
    .venv\Scripts\python.exe phase3_recrop_margin.py --workers 3
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
from phase3_page_scale import _engines  # noqa: E402
from phase3_page_scale import _prep  # noqa: E402
from phase3_page_scale import _smiles_column  # noqa: E402
from phase3_page_extract import recognise_page  # noqa: E402
from phase3_quality_screen import screen_joined  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
OUT = REPO / "data" / "audit" / "phase3" / "recrop_margin"

MARGINS = (0, 8, 16, 24)
LEN_MIN, LEN_MAX = 220, 300
UP = 2.0

_LOCAL = threading.local()
_LOCK = threading.Lock()


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


def _crop_recognise(engine, prepared, row, margin):
    """Crop one row with a widened left edge and recognise it alone."""
    h, w = prepared.shape[:2]
    x1 = max(0, int(np.floor(row["x1"] * UP)) - margin)
    y1 = max(0, int(np.floor(row["y1"] * UP)) - 4)
    x2 = min(w, int(np.ceil(row["x2"] * UP)) + 4)
    y2 = min(h, int(np.ceil(row["y2"] * UP)) + 4)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = prepared[y1:y2, x1:x2]
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
    det = recognise_page(engine, prepared)

    rows = []
    for r in det:
        xs = r.poly[:, 0] * scale
        ys = r.poly[:, 1] * scale
        rows.append({"x1": float(xs.min()), "y1": float(ys.min()), "x2": float(xs.max()), "y2": float(ys.max()), "cx": float(xs.mean()), "cy": float(ys.mean())})

    cache = {}
    for m in MARGINS:
        for r in rows:
            key = (m, r["x1"], r["y1"], r["x2"], r["y2"])
            cache[key] = _crop_recognise(engine, prepared, r, m)

    out = []
    for c in cells:
        cx1 = max(c["x1"], col_x1)
        cx2 = min(c["x2"], col_x2)
        for m in MARGINS:
            picked = []
            crosses = 0
            for r in rows:
                if not (c["y1"] <= r["cy"] <= c["y2"]):
                    continue
                if not (cx1 <= r["cx"] <= cx2):
                    continue
                key = (m, r["x1"], r["y1"], r["x2"], r["y2"])
                picked.append((r, cache[key]))
                if r["x1"] - m < col_x1:
                    crosses += 1
            picked.sort(key=lambda t: t[0]["cy"])
            txt = "".join(t for _, t in picked if t)
            s = screen_joined(c["cmpd_id"], txt)
            out.append({"page": page, "cmpd_id": c["cmpd_id"], "margin": m, "n_rows": len(picked), "expected_rows": c["n_lines"], "rows_match": len(picked) == c["n_lines"], "len": len(txt), "open": txt.count("("), "close": txt.count(")"), "balanced": s.joined_balanced, "parses": s.joined_parses, "strict_pass": bool(s.joined_balanced and len(picked) == c["n_lines"] and LEN_MIN <= len(txt) <= LEN_MAX), "rows_crossing_column": crosses, "text": txt})
    return out


def _worker(args):
    page, cells = args
    try:
        return _process_page(page, cells)
    except Exception as exc:
        return [{"page": page, "cmpd_id": "", "margin": "ERROR", "error": repr(exc)[:200]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _manifest()
    tasks = [(p, c) for p, c in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} margins={MARGINS} workers={args.workers}")

    all_rows = []
    done = 0
    pf = (OUT / "recrop_margin_partial.csv").open("w", newline="", encoding="utf-8-sig")
    pw = None
    fields = ["page", "cmpd_id", "margin", "n_rows", "expected_rows", "rows_match", "len", "open", "close", "balanced", "parses", "strict_pass", "rows_crossing_column", "text"]
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
                if pw is None:
                    pw = csv.DictWriter(pf, fieldnames=fields)
                    pw.writeheader()
                if pw is not None:
                    for row in new_rows:
                        pw.writerow({k: row.get(k, "") for k in fields})
                        if "error" in row:
                            print(f"  error on {row.get('page')}: {row.get('error')}")
                    pf.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} pages done")
    pf.close()

    errors = [r for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    rows.sort(key=lambda r: (r["margin"], r["page"], r["cmpd_id"]))
    if rows:
        with (OUT / "recrop_margin.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    lines = []
    lines.append("=== widening the left edge of every detected row crop ===")
    lines.append("")
    lines.append(f"cells  : {len({r['cmpd_id'] for r in rows})}")
    lines.append(f"pages  : {len({r['page'] for r in rows})}")
    lines.append(f"errors : {len(errors)}")
    lines.append("")
    lines.append(f"{'margin':>7} {'bal':>5} {'rowok':>6} {'strict':>7} {'parse':>6} {'cross':>6} {'edit1036':>9} {'edit1037':>9}")
    for m in MARGINS:
        sub = [r for r in rows if r["margin"] == m]
        if not sub:
            continue
        edits = []
        for cid in ("1036", "1037"):
            hit = [r for r in sub if r["cmpd_id"] == cid]
            edits.append(edit_ops(REFERENCES[cid]["smiles"], hit[0]["text"]) if hit else "")
        lines.append(f"{m:>7} {sum(1 for r in sub if r['balanced']):>5} {sum(1 for r in sub if r['rows_match']):>6} {sum(1 for r in sub if r['strict_pass']):>7} {sum(1 for r in sub if r['parses']):>6} {sum(1 for r in sub if r['rows_crossing_column']):>6} {str(edits[0]):>9} {str(edits[1]):>9}")
    lines.append("")
    lines.append("cross = number of row crops whose widened left edge")
    lines.append("reaches past the production SMILES column boundary.")
    lines.append("")
    lines.append("Text level. No structure is verified.")
    io.open(OUT / "recrop_margin.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
