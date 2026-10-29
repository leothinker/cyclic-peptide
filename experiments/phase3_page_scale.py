r"""Corpus-scale evaluation of the page-level configurations.

Why
---
Two configurations now look strong on the two human-checked compounds:

    default engine + no preprocessing   1036 edit 0,  1037 edit 38
    default engine + 2x upscale        1036 edit 31, 1037 edit 0

Neither wins on both, and n=2 is far too small to choose between them.
This script runs the promising page-level configurations over the whole
corpus and scores them with the same reference-free screen used
everywhere else, so the choice is made on 335 cells rather than two.

What is measured
----------------
For every audited printed-SMILES cell:

    default/none        upstream default engine, whole page, no preproc
    default/up2         upstream default engine, whole page, 2x
    default/up2_th200   upstream default engine, whole page, 2x + th200
    baseline/up2_th200  production engine, 2x + th200

The screen is reference-free, so it cannot be fooled by a configuration
that merely happens to match a gold string. It reports the fingerprint
that matters here: unmatched closing parentheses, which is what the
missing threshold destroys.

Note on page-level recognition
------------------------------
Whole-page recognition lets the detector see the SMILES column without
the cell crop, which is why the artefact got 1036 exactly right. It can
also drop a printed row (the artefact lost rows 19 and 25 for 1037), so
the row count is reported next to the parenthesis balance: a
configuration that balances parentheses by losing rows is not a fix.

Read-only. Writes to ``data/audit/phase3/page_scale/``.

Run:
    .venv\Scripts\python.exe phase3_page_scale.py --workers 4
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

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from phase3_quality_screen import screen_joined  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase3" / "page_scale"

ARMS = ("default/none", "default/up2", "baseline/up2_th200")

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


def _prep(img, name: str):
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if name == "none":
        return img, 1.0
    if name == "up2":
        return up, 0.5
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), 0.5


def _load_manifest() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] != "no":
                continue
            if r["extraction_status"] != "ok":
                continue
            out.setdefault(r["page"], []).append({
                "cmpd_id": r["cmpd_id"],
                "x1": float(r["bbox_x1"]), "y1": float(r["bbox_y1"]),
                "x2": float(r["bbox_x2"]), "y2": float(r["bbox_y2"]),
            })
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _smiles_column(img):
    """The production SMILES column of a page, in page coordinates.

    The audit manifest's bbox runs to the right page edge (x2 = 1939 on
    a 1939 px page), which is 20 px wider than the production column
    (x2 = 1919). Whole-page recognition picks up the table rule and the
    neighbouring structure area in that strip, and the resulting noise
    can still have balanced parentheses. The attribution region is
    therefore clipped to the production column.
    """
    from cpd.parsers.table_extractor import RapidOcrTableExtractor
    ex = RapidOcrTableExtractor(rec_top_k=0)
    cols = ex._detect_columns(img)
    for c in cols:
        if c.name == "smiles":
            return c.x1, c.x2
    w = img.shape[1]
    return int(0.58 * w), int(0.99 * w)


def _process_page(page: str, cells: list[dict],
                  csv_map: dict[str, str]) -> list[dict]:
    from phase3_page_extract import Cell, attribute, recognise_page

    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    if img is None:
        return []

    page_cells = []
    col_x1, col_x2 = _smiles_column(img)
    for c in cells:
        page_cells.append(Cell(
            cmpd_id=c["cmpd_id"], page=page,
            # Clip the attribution band to the production SMILES column.
            # The manifest bbox runs past the column into the table
            # rule, and whole-page recognition picks up noise from that
            # strip which can still have balanced parentheses.
            x1=max(c["x1"], col_x1), y1=c["y1"],
            x2=min(c["x2"], col_x2), y2=c["y2"], n_lines=0,
        ))

    rows: list[dict] = []
    for arm in ARMS:
        eng_kind, prep_name = arm.split("/")
        prepared, scale = _prep(img, prep_name)
        rws = recognise_page(_engines(eng_kind), prepared)
        buckets = attribute(rws, page_cells, scale)
        for c in cells:
            picked = buckets.get(c["cmpd_id"], [])
            txt = "".join(r.text.strip() for r in picked)
            s = screen_joined(c["cmpd_id"], txt)
            csv_txt = csv_map.get(c["cmpd_id"], "") or ""
            rows.append({
                "page": page,
                "cmpd_id": c["cmpd_id"],
                "arm": arm,
                "n_rows": len(picked),
                "len": len(txt),
                "open": txt.count("("),
                "close": txt.count(")"),
                "paren_excess": s.paren_excess,
                "balanced": s.joined_balanced,
                "parses": s.joined_parses,
                "verdict": s.verdict,
                "csv_len": len(csv_txt),
                "csv_open": csv_txt.count("("),
                "csv_close": csv_txt.count(")"),
            })
    return rows


def _worker(args):
    page, cells, csv_map = args
    try:
        return _process_page(page, cells, csv_map)
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _load_manifest()
    df = pd.read_csv(CSV_PATH, dtype=str)
    csv_map = dict(zip(df["cmpd_id"], df["smiles"].fillna("")))
    if args.pages:
        cells_by_page = {p: c for p, c in cells_by_page.items()
                         if p in set(args.pages)}

    tasks = [(p, c, csv_map) for p, c in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} arms={len(ARMS)} workers={args.workers}")

    all_rows: list[dict] = []
    done = 0
    # Written incrementally so a long run that is interrupted still
    # leaves the pages it finished on disk.
    partial = OUT / "page_scale_partial.csv"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial_file = partial.open("w", newline="", encoding="utf-8-sig")
    partial_writer = None
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
                if partial_writer is None and new_rows:
                    partial_writer = csv.DictWriter(
                        partial_file, fieldnames=list(new_rows[0]))
                    partial_writer.writeheader()
                if partial_writer is not None:
                    for row in new_rows:
                        partial_writer.writerow(row)
                    partial_file.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} pages done")
    partial_file.close()

    errors = [r for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    rows.sort(key=lambda r: (r["arm"], r["page"], r["cmpd_id"]))
    if rows:
        with (OUT / "page_scale.csv").open("w", newline="",
                                            encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out = io.open(OUT / "page_scale_summary.txt", "w", encoding="utf-8")
    out.write("=== corpus-scale page-level configuration comparison ===\n\n")
    out.write(f"cells : {len({r['cmpd_id'] for r in rows})}\n")
    out.write(f"pages : {len({r['page'] for r in rows})}\n")
    out.write(f"errors: {len(errors)}\n\n")
    out.write("paren_excess = unmatched ')' over the joined string.\n")
    out.write("A correct SMILES string has 0. A positive value means\n")
    out.write("opening parentheses were lost.\n\n")
    out.write(f"{'arm':<22} {'n':>4} {'ex0':>5} {'ex>0':>5} {'median':>7} "
              f"{'bal':>5} {'parse':>6} {'rows=7':>7} {'medlen':>7}\n")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        ex = [r["paren_excess"] for r in sub]
        out.write(
            f"{arm:<22} {len(sub):>4} {sum(1 for x in ex if x == 0):>5} "
            f"{sum(1 for x in ex if x > 0):>5} {_median(ex):>7} "
            f"{sum(1 for r in sub if r['balanced']):>5} "
            f"{sum(1 for r in sub if r['parses']):>6} "
            f"{sum(1 for r in sub if r['n_rows'] == 7):>7} "
            f"{_median([r['len'] for r in sub]):>7}\n"
        )
    out.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
