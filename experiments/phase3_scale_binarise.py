r"""Corpus-scale A/B: production preprocessing vs production + binarisation.

Why this experiment
-------------------
Phase 2 measured every OCR change against a "binarize_up2_en" baseline
that the pipeline never actually uses. The production SMILES path in
``src/cpd/parsers/table_extractor.py::_ocr_smiles_with_stats`` is

    cell -> _trim_to_content -> _upscale (2x INTER_CUBIC) -> rec_engine

with no threshold anywhere. The 2x2 factorial (``phase3_factorial.py``)
showed on the two reference compounds that this missing threshold is
what costs the opening parentheses: production drops 20 of 38 opening
parens on 1036, while the identical crop with a threshold of 200 keeps
them.

That was n=2. This script runs the same A/B over all 335 audited
printed-SMILES cells across 80 pages, using a reference-free screen, so
the effect can be measured on the whole corpus rather than on two
hand-checked strings.

Arms
----
    production    trim -> 2x -> recognise          (exactly production)
    binarised     trim -> 2x -> threshold 200 -> recognise
    csv           the string shipped in compounds_combined.csv

The only difference between the two OCR arms is the threshold, and the
crop is identical, so the comparison is clean. The ``csv`` arm is scored
with exactly the same screen, which is what makes "the shipped data
carries the same fingerprint" a measurable claim.

Self-validation
---------------
For every cell the production arm is also compared character-for-character
against the shipped CSV string. If the production arm reproduces the CSV,
the experiment is measuring the real pipeline. That agreement is reported
as ``csv_reproduced``, not assumed.

Read-only: no pipeline code, no core module and no official CSV is
modified. All output goes to ``data/audit/phase3/scale_binarise/``.

Run:
    .venv\Scripts\python.exe phase3_scale_binarise.py
    .venv\Scripts\python.exe phase3_scale_binarise.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import io
import os
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
OUT = REPO / "data" / "audit" / "phase3" / "scale_binarise"

THRESHOLD = 200

# Per-thread singletons. RapidOCR mutates shared config state inside
# ``__call__`` via ``update_params``, so one instance cannot serve two
# concurrent calls. Each worker thread therefore owns its own recogniser
# and extractor, built once and reused for every cell it handles.
_LOCAL = threading.local()


def _engine():
    eng = getattr(_LOCAL, "engine", None)
    if eng is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import (
            EngineType, ModelType, OCRVersion, TaskType,
        )
        eng = RapidOCR(params={
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
            # Keep each session from grabbing every core; the thread
            # pool already provides the parallelism.
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        })
        _LOCAL.engine = eng
    return eng


def _extractor():
    ex = getattr(_LOCAL, "extractor", None)
    if ex is None:
        from cpd.parsers.table_extractor import RapidOcrTableExtractor
        ex = RapidOcrTableExtractor(rec_top_k=0)
        _LOCAL.extractor = ex
    return ex


def _threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _recognise_joined(img) -> str:
    """Recognise a prepared image and join the lines in reading order.

    The sort key and the joining rule mirror ``_ocr_smiles_with_stats``
    exactly: boxes are ordered by their vertical midpoint and the texts
    are concatenated with no separator.
    """
    result = _engine()(img)
    if not result:
        return ""
    boxes = list(result.boxes)
    txts = list(result.txts)
    if not boxes:
        return ""
    items = sorted(zip(boxes, txts),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    return "".join(t.strip() for _, t in items if t and t.strip())


def _smiles_column(img):
    """The SMILES column of a page, in the same way production finds it."""
    ex = _extractor()
    cols = ex._detect_columns(img)
    for c in cols:
        if c.name == "smiles":
            return c.x1, c.x2, "detected"
    # Fallback: the documented SMILES fraction of the page width. This
    # only triggers if the header could not be read; the source of the
    # numbers is recorded so the fallback is never silent.
    w = img.shape[1]
    return int(0.58 * w), int(0.99 * w), "fallback_fracs"


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
                "y1": int(float(r["bbox_y1"])),
                "y2": int(float(r["bbox_y2"])),
            })
    for v in out.values():
        v.sort(key=lambda c: c["y1"])
    return out


def _process_page(page: str, cells: list[dict],
                  csv_map: dict[str, str]) -> list[dict]:
    """Run both OCR arms over every cell of one page."""
    img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
    if img is None:
        return []
    x1, x2, col_src = _smiles_column(img)
    ex = _extractor()
    rows: list[dict] = []
    for cell in cells:
        y1, y2 = cell["y1"], cell["y2"]
        if y2 <= y1 or x2 <= x1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        trimmed = ex._trim_to_content(crop)
        upscaled = ex._upscale(trimmed)
        prod_txt = _recognise_joined(upscaled)
        bin_txt = _recognise_joined(_threshold(upscaled))
        csv_txt = csv_map.get(cell["cmpd_id"], "") or ""

        s_prod = screen_joined(cell["cmpd_id"], prod_txt)
        s_bin = screen_joined(cell["cmpd_id"], bin_txt)
        s_csv = screen_joined(cell["cmpd_id"], csv_txt)

        rows.append({
            "page": page,
            "cmpd_id": cell["cmpd_id"],
            "col_source": col_src,
            "crop_px": f"{crop.shape[1]}x{crop.shape[0]}",
            "prod_len": len(prod_txt),
            "prod_open": prod_txt.count("("),
            "prod_close": prod_txt.count(")"),
            "prod_paren_excess": s_prod.paren_excess,
            "prod_balanced": s_prod.joined_balanced,
            "prod_parses": s_prod.joined_parses,
            "prod_verdict": s_prod.verdict,
            "bin_len": len(bin_txt),
            "bin_open": bin_txt.count("("),
            "bin_close": bin_txt.count(")"),
            "bin_paren_excess": s_bin.paren_excess,
            "bin_balanced": s_bin.joined_balanced,
            "bin_parses": s_bin.joined_parses,
            "bin_verdict": s_bin.verdict,
            "csv_len": len(csv_txt),
            "csv_open": csv_txt.count("("),
            "csv_close": csv_txt.count(")"),
            "csv_paren_excess": s_csv.paren_excess,
            "csv_balanced": s_csv.joined_balanced,
            "csv_parses": s_csv.joined_parses,
            "csv_verdict": s_csv.verdict,
            "csv_reproduced": "yes" if prod_txt == csv_txt else "no",
        })
    return rows


def _worker(args):
    page, cells, csv_map = args
    try:
        return _process_page(page, cells, csv_map)
    except Exception as exc:  # keep one bad page from killing the run
        return [{"page": page, "cmpd_id": "", "error": repr(exc)[:200]}]


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _arm_stats(rows: list[dict], prefix: str) -> dict:
    ex = [r[f"{prefix}_paren_excess"] for r in rows
          if f"{prefix}_paren_excess" in r]
    balanced = [r[f"{prefix}_balanced"] for r in rows
                if f"{prefix}_balanced" in r]
    parses = [r[f"{prefix}_parses"] for r in rows
              if f"{prefix}_parses" in r]
    verdicts: dict[str, int] = {}
    for r in rows:
        v = r.get(f"{prefix}_verdict", "")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1
    return {
        "n": len(ex),
        "excess0": sum(1 for x in ex if x == 0),
        "excess_neg": sum(1 for x in ex if x < 0),
        "excess_pos": sum(1 for x in ex if x > 0),
        "excess_median": round(_median(ex), 2),
        "excess_mean": round(_mean(ex), 2),
        "balanced": sum(1 for b in balanced if b),
        "parses": sum(1 for p in parses if p),
        "verdicts": verdicts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pages", nargs="*", default=None,
                    help="restrict to these page ids (debug aid)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = _load_manifest()
    df = pd.read_csv(CSV_PATH, dtype=str)
    csv_map = dict(zip(df["cmpd_id"], df["smiles"].fillna("")))

    if args.pages:
        cells_by_page = {p: c for p, c in cells_by_page.items()
                         if p in set(args.pages)}

    tasks = [(p, cells, csv_map) for p, cells in sorted(cells_by_page.items())]
    print(f"pages={len(tasks)} cells="
          f"{sum(len(c) for _, c, _ in tasks)} workers={args.workers}")

    all_rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            page = futs[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                print(f"  page {page} failed: {exc!r}")
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} pages done")

    errors = [r for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    rows.sort(key=lambda r: (r["page"], r["cmpd_id"]))

    if rows:
        with (OUT / "scale_binarise.csv").open("w", newline="",
                                              encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out = io.open(OUT / "scale_binarise_summary.txt", "w", encoding="utf-8")
    out.write("=== corpus-scale A/B: production vs production+binarise ===\n\n")
    out.write("Production path (src/cpd/parsers/table_extractor.py):\n")
    out.write("  cell -> _trim_to_content -> _upscale(2x INTER_CUBIC) "
              "-> rec_engine\n")
    out.write("  no threshold is applied anywhere in that path.\n\n")
    out.write("The binarised arm differs ONLY by cv2.threshold(gray, "
              f"{THRESHOLD}).\n")
    out.write("Same crop, same recogniser, same joining rule, same screen.\n\n")
    out.write(f"cells processed : {len(rows)}\n")
    out.write(f"pages processed : {len({r['page'] for r in rows})}\n")
    out.write(f"page errors     : {len(errors)}\n")
    if errors:
        for e in errors[:10]:
            out.write(f"  {e.get('page')}: {e.get('error')}\n")
    out.write("\n")

    out.write("--- column source used for the crop ---\n")
    srcs: dict[str, int] = {}
    for r in rows:
        key = r.get("col_source", "?")
        srcs[key] = srcs.get(key, 0) + 1
    for k, v in sorted(srcs.items()):
        out.write(f"  {k:<16} {v}\n")

    out.write("\n--- does the production arm reproduce the shipped CSV? ---\n")
    yes = sum(1 for r in rows if r.get("csv_reproduced") == "yes")
    out.write(f"  identical            {yes}/{len(rows)} "
              f"({100.0 * yes / max(1, len(rows)):.1f}%)\n")
    out.write("  -> where this is high, the A/B is measuring the real "
              "pipeline, not a reconstruction of it\n")

    out.write("\n--- reference-free screen, per arm ---\n")
    out.write(f"{'arm':<12} {'n':>4} {'excess=0':>9} {'excess>0':>9} "
              f"{'median':>8} {'mean':>8} {'balanced':>9} {'parses':>8}\n")
    for label, prefix in (("production", "prod"), ("binarised", "bin"),
                           ("csv", "csv")):
        s = _arm_stats(rows, prefix)
        out.write(
            f"{label:<12} {s['n']:>4} {s['excess0']:>9} {s['excess_pos']:>9} "
            f"{s['excess_median']:>8} {s['excess_mean']:>8} "
            f"{s['balanced']:>9} {s['parses']:>8}\n"
        )

    out.write("\n--- verdict distribution, per arm ---\n")
    for label, prefix in (("production", "prod"), ("binarised", "bin"),
                           ("csv", "csv")):
        s = _arm_stats(rows, prefix)
        out.write(f"  {label:<12} {s['verdicts']}\n")

    out.write("\n--- paired comparison: binarised vs production ---\n")
    better = sum(1 for r in rows
                  if r["bin_paren_excess"] < r["prod_paren_excess"])
    worse = sum(1 for r in rows
                 if r["bin_paren_excess"] > r["prod_paren_excess"])
    same = len(rows) - better - worse
    out.write(f"  fewer unmatched ')' : {better}/{len(rows)}\n")
    out.write(f"  unchanged           : {same}/{len(rows)}\n")
    out.write(f"  more unmatched ')'  : {worse}/{len(rows)}\n")
    out.write("\n--- paired comparison: binarised vs shipped csv ---\n")
    bb = sum(1 for r in rows
             if r["bin_paren_excess"] < r["csv_paren_excess"])
    bw = sum(1 for r in rows
             if r["bin_paren_excess"] > r["csv_paren_excess"])
    out.write(f"  fewer unmatched ')' : {bb}/{len(rows)}\n")
    out.write(f"  more unmatched ')'  : {bw}/{len(rows)}\n")

    out.write("\n--- per-cell detail (production vs binarised) ---\n")
    out.write(f"{'page':<10} {'cmpd':>6} {'prod_ex':>8} {'bin_ex':>7} "
              f"{'prod_len':>8} {'bin_len':>8} {'prod_ok':>8} {'bin_ok':>7}\n")
    for r in rows:
        out.write(
            f"{r['page']:<10} {r['cmpd_id']:>6} "
            f"{r['prod_paren_excess']:>8} {r['bin_paren_excess']:>7} "
            f"{r['prod_len']:>8} {r['bin_len']:>8} "
            f"{str(r['prod_parses']):>8} {str(r['bin_parses']):>7}\n"
        )
    out.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
