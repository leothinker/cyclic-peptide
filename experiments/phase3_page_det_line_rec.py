r"""Page-scale detection with line-level recognition, at corpus scale.

Why this experiment
-------------------
Two page-scale measurements are already in hand, and they point at two
different layers:

    page + default engine            197/335 pass the strict screen
    cell + baseline engine + th200    40/335 pass the strict screen

The page measurement was made with the default engine (PP-OCRv6 det
small + a Chinese recogniser), so "page level beats cell level" is
currently confounded: the DETECTOR changed and the RECOGNISER changed.
That confound decides whether the fix is "use the page" or "use a
different model", and the two lead to completely different patches.

This script removes the confound. Detection and recognition are run
as separate stages:

    1. DETECT on the whole page. The detector sees the SMILES column
       at page scale, where the printed rows are separated by the
       table rules, and returns one box per printed row.
    2. For every box that falls inside a compound's SMILES cell,
       crop that row out of the page and RECOGNISE the crop on its own.

The crop is what the recogniser sees, so step 2 can use the PRODUCTION
recogniser (PP-OCRv4 mobile, English) while step 1 uses whichever
detector finds the rows best. A configuration that works here can be
described as a patch to the detection stage only.

Arms
----
    pdet_def_rec_base_1x   default detector, 1x page, baseline rec
    pdet_def_rec_base_2x   default detector, 2x page, baseline rec
    pdet_base_rec_base_1x  baseline detector, 1x page, baseline rec
    pdet_def_rec_def_1x    default detector, 1x page, default rec

The last arm reproduces the page route with the crop taken from the
1x page instead of the page the detector ran on, which isolates the
effect of the crop resolution.

Row-drop rate
-------------
The detector can miss a printed row entirely, and a cell that lost a
row can still have balanced parentheses. Every cell therefore reports
the number of rows recovered against the row count the audit manifest
measured on the page, so a lost row cannot hide behind a good balance.

Read-only. Writes to ``data/audit/phase3/page_det_line_rec/``.

Run:
    .venv\Scripts\python.exe phase3_page_det_line_rec.py --workers 3
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

from phase3_quality_screen import screen_joined  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "page_det_line_rec"

LEN_MIN, LEN_MAX = 220, 300
MARGIN = 3

ARMS = {
    "pdet_def_rec_base_1x": ("default", 1.0, "baseline"),
    "pdet_def_rec_base_2x": ("default", 2.0, "baseline"),
    "pdet_base_rec_base_1x": ("baseline", 1.0, "baseline"),
    "pdet_def_rec_def_1x": ("default", 1.0, "default"),
    "pdet_base_rec_base_2x": ("baseline", 2.0, "baseline"),
    "pdet_def_rec_def_2x": ("default", 2.0, "default"),
}

_LOCAL = threading.local()
_LOCK = threading.Lock()


def _engines(kind: str):
    """Per-thread engines. RapidOCR mutates config state in __call__."""
    cache = getattr(_LOCAL, "engines", None)
    if cache is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import (
            EngineType, ModelType, OCRVersion, TaskType,
        )
        baseline = RapidOCR(params={
            "Det.lang_type": "en",
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.task_type": TaskType.DET,
            "Rec.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        })
        cache = {"default": RapidOCR(params={}), "baseline": baseline}
        _LOCAL.engines = cache
    return cache[kind]


def _smiles_column(img):
    """The production SMILES column of a page, in page coordinates."""
    from cpd.parsers.table_extractor import RapidOcrTableExtractor
    ex = RapidOcrTableExtractor(rec_top_k=0)
    cols = ex._detect_columns(img)
    for c in cols:
        if c.name == "smiles":
            return c.x1, c.x2
    w = img.shape[1]
    return int(0.58 * w), int(0.99 * w)


def _load_manifest():
    out: dict[str, list[dict]] = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({
                "cmpd_id": r["cmpd_id"],
                "x1": float(r["bbox_x1"]), "y1": float(r["bbox_y1"]),
                "x2": float(r["bbox_x2"]), "y2": float(r["bbox_y2"]),
                "n_lines": int(r["n_lines"]),
            })
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _upscale(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)


def _detect_rows(det_engine, page_img, det_scale, col_x1, col_x2):
    """Detect on the page, return rows as (text_box_page_coords, poly).

    Detection only: ``use_rec=False`` returns the detector output with
    the boxes mapped back to the coordinates of the image that was
    passed in, so the boxes are in page coordinates divided by
    ``det_scale``.
    """
    det_img = _upscale(page_img, det_scale)
    # Both flags are explicit on every call: RapidOCR's update_params
    # writes the flag onto the engine instance and does not reset it,
    # so a detection-only call would silently disable recognition for
    # every later call on the same object.
    res = det_engine(det_img, use_det=True, use_rec=False)
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    for poly in np.asarray(boxes, dtype=float).reshape(-1, 4, 2):
        xs = poly[:, 0] / det_scale
        ys = poly[:, 1] / det_scale
        cx = float(xs.mean())
        cy = float(ys.mean())
        if not (col_x1 - MARGIN <= cx <= col_x2 + MARGIN):
            continue
        out.append((float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max()), cx, cy))
    return out


def _recognise_row(rec_engine, page_img, img_scale, row):
    """Crop one printed row out of the page and recognise it alone."""
    # ``row`` is in page coordinates; ``page_img`` may be scaled.
    x1 = row[0] * img_scale
    y1 = row[1] * img_scale
    x2 = row[2] * img_scale
    y2 = row[3] * img_scale
    h, w = page_img.shape[:2]
    px1 = max(0, int(np.floor(x1)) - MARGIN)
    py1 = max(0, int(np.floor(y1)) - MARGIN)
    px2 = min(w, int(np.ceil(x2)) + MARGIN)
    py2 = min(h, int(np.ceil(y2)) + MARGIN)
    if px2 <= px1 or py2 <= py1:
        return ""
    crop = page_img[py1:py2, px1:px2]
    if crop.size == 0:
        return ""
    res = rec_engine(crop, use_det=False, use_rec=True)
    txts = getattr(res, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _process_page(page: str, cells: list[dict],
                  arms: tuple[str, ...] = ARMS) -> list[dict]:
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    if img is None:
        return []
    col_x1, col_x2 = _smiles_column(img)

    rows: list[dict] = []
    for arm in arms:
        det_kind, det_scale, rec_kind = ARMS[arm]
        det_engine = _engines(det_kind)
        rec_engine = _engines(rec_kind)
        detected = _detect_rows(det_engine, img, det_scale, col_x1, col_x2)
        crop_img = _upscale(img, det_scale)

        # Recognise every detected row once, then attribute by centre.
        texts = {}
        for row in detected:
            texts[row] = _recognise_row(rec_engine, crop_img, det_scale, row)

        for c in cells:
            picked = [r for r in detected
                      if c["x1"] <= r[4] <= c["x2"]
                      and c["y1"] <= r[5] <= c["y2"]]
            picked.sort(key=lambda r: r[5])
            txt = "".join(texts[r] for r in picked if texts[r])
            s = screen_joined(c["cmpd_id"], txt)
            rows.append({
                "page": page,
                "cmpd_id": c["cmpd_id"],
                "arm": arm,
                "n_rows": len(picked),
                "expected_rows": c["n_lines"],
                "rows_match": len(picked) == c["n_lines"],
                "len": len(txt),
                "open": txt.count("("),
                "close": txt.count(")"),
                "paren_excess": s.paren_excess,
                "balanced": s.joined_balanced,
                "parses": s.joined_parses,
                "illegal_chars": "" if s.charset_ok else "yes",
                "len_plausible": LEN_MIN <= len(txt) <= LEN_MAX,
                "strict_pass": bool(s.joined_balanced
                                    and len(picked) == c["n_lines"]
                                    and LEN_MIN <= len(txt) <= LEN_MAX),
                "text": txt,
            })
    return rows


def _worker(args):
    page, cells = args
    try:
        return _process_page(page, cells)
    except Exception as exc:
        return [{"page": page, "cmpd_id": "", "arm": "ERROR",
                 "error": repr(exc)[:200]}]


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _load_manifest()
    if args.pages:
        cells_by_page = {p: c for p, c in cells_by_page.items()
                         if p in set(args.pages)}

    tasks = [(p, c) for p, c in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} arms={len(ARMS)} workers={args.workers}")

    all_rows: list[dict] = []
    done = 0
    partial = OUT / "page_det_line_rec_partial.csv"
    pf = partial.open("w", newline="", encoding="utf-8-sig")
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
        with (OUT / "page_det_line_rec.csv").open(
                "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out = io.open(OUT / "page_det_line_rec_summary.txt", "w",
                  encoding="utf-8")
    out.write("=== page detection + line recognition, corpus scale ===\n\n")
    out.write(f"cells : {len({r['cmpd_id'] for r in rows})}\n")
    out.write(f"pages : {len({r['page'] for r in rows})}\n")
    out.write(f"errors: {len(errors)}\n\n")
    out.write("strict_pass = balanced AND rows recovered == rows on page "
              "AND length in 220-300.\n")
    out.write("rows recovered is against the row count the audit manifest "
              "measured on the page,\nso a dropped printed row cannot be "
              "hidden by a balanced string.\n\n")
    out.write(f"{'arm':<26} {'n':>4} {'bal':>5} {'rowok':>6} "
              f"{'strict':>6} {'parse':>6} {'dropped':>8} {'medlen':>7}\n")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        dropped = [r for r in sub if not r["rows_match"]]
        out.write(
            f"{arm:<26} {len(sub):>4} "
            f"{sum(1 for r in sub if r['balanced']):>5} "
            f"{sum(1 for r in sub if r['rows_match']):>6} "
            f"{sum(1 for r in sub if r['strict_pass']):>6} "
            f"{sum(1 for r in sub if r['parses']):>6} "
            f"{len(dropped):>8} "
            f"{_median([r['len'] for r in sub]):>7}\n")
    out.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
