r"""Scale test: does a different printed-SMILES extraction configuration help?

*** SUPERSEDED - the numbers this script produced are NOT valid. ***

This script paired the detector's boxes with the recognised text after
sorting the boxes, so wherever the detector emitted its rows in a
different order from the reading order, one row's text was attached to
another row's box. The pairing bug was found and fixed in
``phase3_page_extract.py`` (text is bound to its own box before any
sorting). Use ``phase3_scale_binarise.py`` and
``phase3_page_scale.py`` instead; the output under
``data/audit/phase3/scale/`` is kept only as a record of the buggy run.

The rest of this docstring describes the intent of the original run.

The preprocessing lead from 2026-10-23 was ambiguous: raw-page defaults gave
edit distance 0 on cmpd 1036 but 38 on cmpd 1037, and only two compounds in
the whole patent have a human-verified reference.

This script removes that blocker by scoring with the reference-free screen
in ``phase3_quality_screen`` instead of CER, so every compound can be
compared. Three configurations:

    production    the official CSV's ``smiles`` column, i.e. the cell-crop
                  path with the baseline preprocessing and the top-k fix.
                  Read from the CSV, so it costs no OCR time.
    page_raw      page-level detection on the untouched page, default
                  RapidOCR parameters. This is what the orphaned
                  ``_demo_topk.py`` did, and the only configuration that kept
                  the row-start '=' on cmpd 1036.
    page_up2th200 page-level detection with the baseline preprocessing.
                  Isolates preprocessing from extraction path.

The headline signal is ``paren_excess``: the number of closing parentheses
without a matching opener. Every one of the 335 compounds in the current CSV
has a surplus, and that surplus is the direct footprint of the row-start
clipping, so it measures the damage without needing a gold standard.

Read-only with respect to the pipeline, ``src/cpd`` and the official CSV.
Outputs go to ``data/audit/phase3/scale/``.

Run:
    .venv\Scripts\python.exe phase3_scale_run.py [page ...]
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import BASELINE_THRESHOLD, build_engine  # noqa: E402
from phase3_quality_screen import screen, screen_joined  # noqa: E402
from rapidocr import RapidOCR  # noqa: E402
from row_crop_audit import preprocess  # noqa: E402
from topk_preproc_probe import _attribute, _recognise_all  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase3" / "scale"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")

CONFIGS = ("page_raw", "page_up2th200")


def _cells():
    out = {}
    for r in csv.DictReader(MANIFEST.open(encoding="utf-8-sig")):
        if r["is_header"] != "no":
            continue
        out.setdefault(r["page"], []).append(r)
    return out


def _production():
    """The current production printed-SMILES output, straight from the CSV."""
    out = {}
    for r in csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")):
        t = r.get("smiles") or ""
        if t:
            out[r["cmpd_id"]] = t
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = _cells()
    pages = sys.argv[1:] or sorted(cells)
    page_of = {c["cmpd_id"]: p for p, cs in cells.items() for c in cs}

    plain = RapidOCR(params={})          # raw-page defaults
    baseline = build_engine()            # en PP-OCRv4 mobile

    rows_out = []
    t0 = time.time()
    for n, page in enumerate(pages, 1):
        cs = cells[page]
        img = cv2.imread(str(GOLD / cs[0]["source_image"]))
        if img is None:
            continue
        variants = {
            "page_raw": (plain, img),
            "page_up2th200": (baseline, preprocess(img, BASELINE_THRESHOLD)),
        }
        for cfg, (engine, prep) in variants.items():
            det_rows = _recognise_all(engine, prep)
            cmpds, _, _ = _attribute(det_rows)
            for cid in sorted(cmpds, key=lambda s: int(s)):
                rws = [r["text"] for r in cmpds[cid]]
                res = screen(cid, rws)
                rows_out.append({
                    "config": cfg, "cmpd_id": cid, "page": page,
                    "n_rows": res.n_rows, "joined_len": res.joined_len,
                    "row_len_min": res.row_len_min,
                    "row_len_max": res.row_len_max,
                    "paren_excess": res.paren_excess,
                    "bracket_excess": res.bracket_excess,
                    "joined_parses": int(res.joined_parses),
                    "verdict": res.verdict,
                    "reasons": ";".join(res.reasons),
                })
        print(f"  {n}/{len(pages)} {page} ({time.time() - t0:.0f}s)",
              flush=True)

    for cid, t in _production().items():
        res = screen_joined(cid, t)
        rows_out.append({
            "config": "production", "cmpd_id": cid,
            "page": page_of.get(cid, ""),
            "n_rows": "", "joined_len": len(t), "row_len_min": "",
            "row_len_max": "", "paren_excess": res.paren_excess,
            "bracket_excess": res.bracket_excess,
            "joined_parses": int(res.joined_parses),
            "verdict": res.verdict, "reasons": ";".join(res.reasons),
        })

    with (OUT / "scale_results.csv").open("w", newline="",
                                          encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    _summary(rows_out)
    print(f"\nwrote {OUT}")


def _summary(rows) -> None:
    from collections import Counter
    out = io.open(OUT / "scale_summary.txt", "w", encoding="utf-8")
    out.write("=== Phase 3 scale comparison ===\n\n")
    out.write(f"{'config':<16} {'n':>5} {'pass':>6} {'suspect':>8} "
              f"{'fail':>6} {'parse':>7} {'excess=0':>9} {'med_excess':>11}\n")
    for cfg in ("production",) + CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        c = Counter(r["verdict"] for r in sub)
        ex = sorted(r["paren_excess"] for r in sub
                    if r["paren_excess"] != "")
        med = ex[len(ex) // 2] if ex else 0
        out.write(f"{cfg:<16} {len(sub):>5} {c['pass']:>6} {c['suspect']:>8} "
                  f"{c['fail']:>6} "
                  f"{sum(r['joined_parses'] for r in sub):>7} "
                  f"{sum(1 for x in ex if x == 0):>9} {med:>11}\n")
    out.write("\nparen_excess = closing parentheses without an opener.\n")
    out.write("It is the direct footprint of the row-start clipping, so a\n")
    out.write("smaller median means less clipping damage. It needs no gold\n")
    out.write("standard and therefore applies to every compound.\n")
    out.close()


if __name__ == "__main__":
    main()
