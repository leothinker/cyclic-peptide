r"""Separate the OCR model from the preprocessing, with the pairing fixed.

The question
------------
The earlier probe reported that running the OCR on the RAW page produced
an exact match on cmpd 1036 (edit distance 0) while the baseline
preprocessing produced edit distance 3, and concluded that "preprocessing
is the variable". That comparison is confounded: the raw-page run used
 ``RapidOCR(params={})`` (PP-OCRv6 det_small + ch PP-OCRv6 rec small +
 ch cls) and
the preprocessed run used the baseline engine (PP-OCRv4 mobile, English
detector + English recogniser). Two things changed at once.

This script runs the full 2x2 on one page:

    model      = baseline (PP-OCRv4 mobile en) | default (PP-OCRv6 ch)
    preprocess = none | up2 | up2_th200

The extraction path is held fixed (whole page, per-compound bands from the
audit manifest) and the text/box pairing is correct, so any difference is
attributable to the one axis that moved.

Two references are scored, because one sample cannot separate two effects:

    1036  human-verified transcription (researcher supplied)
    1037  agent transcription, not yet confirmed by a human reader

Both are reported. Neither is treated as chemical truth; the mass-gate
notes in ``phase2_ocr_compare.REFERENCES`` apply.

Read-only. Writes to ``data/audit/phase3/engine_prep_matrix/``.

Run:
    .venv\Scripts\python.exe phase3_engine_prep_matrix.py [page ...]
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, PREP, attribute, baseline_engine, default_engine, joined,
    load_cells, recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "engine_prep_matrix"

MODELS = ("baseline", "default")
PREPS = ("none", "up2", "up2_th200")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    pages = sys.argv[1:] or ["I100297"]

    engines = {"baseline": baseline_engine(), "default": default_engine()}
    results: list[dict] = []
    detail: list[str] = []

    for page in pages:
        cs = cells.get(page, [])
        if not cs:
            continue
        img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
        if img is None:
            continue
        for model in MODELS:
            engine = engines[model]
            for prep_name in PREPS:
                prep = PREP[prep_name]
                scale = 1.0 / 2.0 if prep_name != "none" else 1.0
                rows = recognise_page(engine, prep(img))
                buckets = attribute(rows, cs, scale)
                for c in cs:
                    got = buckets.get(c.cmpd_id, [])
                    txt = joined(got)
                    ref = REFERENCES.get(c.cmpd_id, {}).get("smiles")
                    ed = edit_ops(ref, txt) if ref else None
                    starts = "".join(r.text.strip()[:1] for r in got)
                    results.append({
                        "page": page,
                        "cmpd_id": c.cmpd_id,
                        "model": model,
                        "prep": prep_name,
                        "n_rows": len(got),
                        "expected_rows": c.n_lines,
                        "joined_len": len(txt),
                        "ref_len": len(ref) if ref else "",
                        "edit": ed if ed is not None else "",
                        "cer": round(ed / len(ref), 5) if ref else "",
                        "row_starts": starts,
                    })
                    line = (f"{page} cmpd {c.cmpd_id} {model:8s} "
                            f"{prep_name:9s} rows={len(got)}/{c.n_lines} "
                            f"len={len(txt)}")
                    if ref:
                        line += (f"/{len(ref)} edit={ed} "
                                 f"CER={ed / len(ref):.5f}")
                    else:
                        line += " (no reference)"
                    line += f" starts={starts!r}"
                    detail.append(line)
                print(f"  {page} {model:8s} {prep_name:9s} done", flush=True)

    if results:
        with (OUT / "matrix.csv").open("w", newline="",
                                       encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)

    out = io.open(OUT / "matrix_summary.txt", "w", encoding="utf-8")
    out.write("=== engine x preprocessing matrix "
              "(page level, pairing fixed) ===\n\n")
    out.write("The earlier probe changed model and preprocessing together.\n")
    out.write("This matrix changes one axis at a time.\n\n")
    for line in detail:
        out.write(line + "\n")
    out.write("\n--- edit distance summary, references only ---\n")
    hdr = (f"{'page':<10} {'cmpd':<6} {'model':<9} {'prep':<10} "
           f"{'rows':<8} {'edit':<6} {'CER':<9}")
    out.write(hdr + "\n")
    for r in results:
        if r["edit"] == "":
            continue
        rows_txt = f"{r['n_rows']}/{r['expected_rows']}"
        out.write(f"{r['page']:<10} {r['cmpd_id']:<6} {r['model']:<9} "
                  f"{r['prep']:<10} {rows_txt:<8} "
                  f"{r['edit']:<6} {r['cer']:<9}\n")
    out.close()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
