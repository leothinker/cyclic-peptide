r"""Cell-level crop x recogniser, at corpus scale.

The missing cell of the design
------------------------------
Four results are already in hand, and they do not yet combine into a
single recommendation:

    cell crop + baseline engine + threshold 200
        never picks up a neighbouring column, and on the two reference
        compounds it is accurate (1036 edit 9, 1037 edit 6), but at
        corpus scale only 40/335 cells pass the strict screen.

    page + default engine, no preprocessing
        179/335 pass the strict screen and reproduces 1036 exactly
        (edit 0), but on 1037 one printed row collapses into garbage.

    page + default engine, 2x upscale
        197/335 pass, reproduces 1037 exactly (edit 0), but on 1036 the
        string gains a contaminated tail.

    page + baseline engine, 2x + threshold 200
        40/335 pass.

The untested combination is the one that would settle the question: the
CELL CROP (which bounds what can be picked up, so no neighbouring column
can be absorbed) with the DEFAULT engine (which reads 1036 perfectly).
This script runs that at corpus scale.

Arms
----
    cell/default/none        cell crop, default engine, no preprocessing
    cell/default/up2         cell crop, default engine, 2x
    cell/default/up2_th200   cell crop, default engine, 2x + threshold
    cell/baseline/up2_th200  cell crop, production engine, 2x + threshold

All four share the identical crop, so the only differences are the
recogniser and the preprocessing.

Scoring
-------
The strict screen from ``phase3_page_arm_verify`` is used, and the
recognised text is kept so it can be read:

    balanced   parentheses balance
    rows       row count matches what the patent prints
    length     plausible for a printed cell

Read-only. Writes to ``data/audit/phase3/cell_engine_scale/``.

Run:
    .venv\Scripts\python.exe phase3_cell_engine_scale.py --workers 3
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

import pandas as pd  # noqa: E402

from phase3_quality_screen import SMILES_CHARS, screen_joined  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "cell_engine_scale"

ARMS = ("cell/default/none", "cell/default/up2", "cell/default/up2_th200",
        "cell/baseline/up2_th200")

LEN_MIN, LEN_MAX = 220, 300

_LOCAL = threading.local()


def _engines(kind: str):
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


def _extractor():
    ex = getattr(_LOCAL, "extractor", None)
    if ex is None:
        from cpd.parsers.table_extractor import RapidOcrTableExtractor
        ex = RapidOcrTableExtractor(rec_top_k=0)
        _LOCAL.extractor = ex
    return ex


def _prep(img, name):
    if name == "none":
        return img
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if name == "up2":
        return up
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _recognise_joined(engine, img):
    result = engine(img)
    if not result:
        return "", []
    boxes = list(result.boxes)
    txts = list(result.txts)
    if not boxes:
        return "", []
    items = sorted(zip(boxes, txts),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    parts = [t.strip() for _, t in items if t and t.strip()]
    return "".join(parts), parts


def _load_manifest():
    out = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no" or r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({
                "cmpd_id": r["cmpd_id"],
                "y1": int(float(r["bbox_y1"])), "y2": int(float(r["bbox_y2"])),
                "n_lines": int(r["n_lines"]),
            })
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _process_page(page, cells, csv_map):
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    if img is None:
        return []
    ex = _extractor()
    cols = ex._detect_columns(img)
    col = next((c for c in cols if c.name == "smiles"), None)
    if col is None:
        w = img.shape[1]
        col_x1, col_x2 = int(0.58 * w), int(0.99 * w)
    else:
        col_x1, col_x2 = col.x1, col.x2

    rows: list[dict] = []
    for c in cells:
        crop = img[c["y1"]:c["y2"], col_x1:col_x2]
        if crop.size == 0:
            continue
        trimmed = ex._trim_to_content(crop)
        for arm in ARMS:
            _, eng_kind, prep_name = arm.split("/")
            prepared = _prep(trimmed, prep_name)
            txt, parts = _recognise_joined(_engines(eng_kind), prepared)
            s = screen_joined(c["cmpd_id"], txt)
            illegal = "".join(sorted({ch for ch in txt
                                      if ch not in SMILES_CHARS}))
            rows.append({
                "page": page,
                "cmpd_id": c["cmpd_id"],
                "arm": arm,
                "n_rows": len(parts),
                "expected_rows": c["n_lines"],
                "rows_match": len(parts) == c["n_lines"],
                "len": len(txt),
                "open": txt.count("("),
                "close": txt.count(")"),
                "paren_excess": s.paren_excess,
                "balanced": s.joined_balanced,
                "parses": s.joined_parses,
                "illegal_chars": illegal,
                "len_plausible": LEN_MIN <= len(txt) <= LEN_MAX,
                "text": txt,
            })
    return rows


def _worker(args):
    page, cells, csv_map = args
    try:
        return _process_page(page, cells, csv_map)
    except Exception as exc:
        return [{"page": page, "cmpd_id": "", "arm": "ERROR",
                 "error": repr(exc)[:200]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _load_manifest()
    dfc = pd.read_csv(CSV_PATH, dtype=str)
    csv_map = dict(zip(dfc["cmpd_id"], dfc["smiles"].fillna("")))
    if args.pages:
        cells_by_page = {p: c for p, c in cells_by_page.items()
                         if p in set(args.pages)}

    tasks = [(p, c, csv_map) for p, c in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} arms={len(ARMS)} workers={args.workers}")

    all_rows: list[dict] = []
    done = 0
    partial = OUT / "cell_engine_partial.csv"
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
        with (OUT / "cell_engine_scale.csv").open("w", newline="",
                                                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out = io.open(OUT / "cell_engine_scale_summary.txt", "w",
                  encoding="utf-8")
    out.write("=== cell crop x recogniser, corpus scale ===\n\n")
    out.write(f"cells {len(rows)}, pages {len({r['page'] for r in rows})}, "
              f"errors {len(errors)}\n")
    out.write("All arms share the identical cell crop, so only the\n")
    out.write("recogniser and the preprocessing differ.\n\n")
    out.write("strict = balanced AND rows match the printed count AND "
              "length in 220-300\n\n")
    out.write(f"{'arm':<28} {'n':>4} {'bal':>5} {'rowsOK':>7} "
              f"{'strict':>7} {'parses':>7} {'illegal':>8} {'medlen':>7}\n")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        bal = sum(1 for r in sub if r["balanced"])
        rk = sum(1 for r in sub if r["rows_match"])
        st = sum(1 for r in sub if r["balanced"] and r["rows_match"]
                 and r["len_plausible"])
        pa = sum(1 for r in sub if r["parses"])
        il = sum(1 for r in sub if r["illegal_chars"])
        med = sorted(r["len"] for r in sub)[len(sub) // 2]
        out.write(f"{arm:<28} {len(sub):>4} {bal:>5} {rk:>7} {st:>7} "
                  f"{pa:>7} {il:>8} {med:>7}\n")

    out.write("\n--- per-arm, the two reference compounds ---\n")
    out.write("Only 1036 and 1037 have a text reference. The rest is "
              "reference-free.\n")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        for cid in ("1036", "1037"):
            hit = [r for r in sub if r["cmpd_id"] == cid]
            if hit:
                r0 = hit[0]
                out.write(f"  {arm:<28} {cid} len={r0['len']:<4} "
                          f"rows={r0['n_rows']}/{r0['expected_rows']} "
                          f"balanced={str(r0['balanced']):<5} "
                          f"parses={str(r0['parses'])}\n")

    out.close()
    print("".join(io.open(OUT / "cell_engine_scale_summary.txt",
                           encoding="utf-8").read()))


if __name__ == "__main__":
    main()
