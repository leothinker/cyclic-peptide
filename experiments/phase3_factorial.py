r"""The clean 2x2: crop-vs-page, binarised-vs-not.

What earlier work assumed
--------------------------
Phase 2 used "binarize_up2_en" as THE baseline and measured every OCR
change against it. But the production pipeline does not binarise at all:
``_ocr_smiles_with_stats`` does ``_trim_to_content`` -> ``_upscale`` (2x
INTER_CUBIC) -> the recogniser, with no threshold anywhere. So the Phase 2
baseline was never the production configuration, and every conclusion drawn
against it describes a configuration the pipeline does not use.

The earlier stage-by-stage diagnostic was itself confounded: it compared
"raw crop, 2x, no threshold" against "whole page, 2x, threshold 200", which
differs in two variables at once. This script separates them:

    input      = cell crop (production column + row band)
               | whole page
    binarise   = no | yes (threshold 200 after the 2x upscale)

and additionally reports the production preprocessing exactly
(trim-to-content, no binarisation) as its own arm.

All arms use the production recogniser (PP-OCRv4 mobile, English) and the
same pairing rules. Scores are edit distance against the human-verified
reference for 1036 and the agent transcription for 1037.

Read-only. Writes to ``data/audit/phase3/factorial/``.

Run:
    .venv\Scripts\python.exe phase3_factorial.py [page ...]
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

from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import (  # noqa: E402
    GOLD, attribute, baseline_engine, load_cells, recognise_page,
)

OUT = REPO / "data" / "audit" / "phase3" / "factorial"


def _up2(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _th200(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _sorted_texts(result):
    boxes = list(result.boxes)
    txts = list(result.txts)
    pairs = sorted(zip(boxes, txts), key=lambda bt: bt[0][0][1])
    return [t.strip() for _, t in pairs if t and t.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_page = load_cells()
    pages = sys.argv[1:] or ["I100297"]
    engine = baseline_engine()
    ex = RapidOcrTableExtractor(rec_top_k=0)

    rows_out: list[dict] = []
    lines: list[str] = []

    for page in pages:
        cells = cells_by_page.get(page, [])
        if not cells:
            continue
        img = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{page}.jpg"))
        if img is None:
            continue
        cols = ex._detect_columns(img)
        col = next(c for c in cols if c.name == "smiles")

        # ---------------- whole-page arms ----------------
        page_arms = {}
        up = _up2(img)
        page_arms["page_up2"] = up
        page_arms["page_up2_th200"] = _th200(up)
        for arm, prepared in page_arms.items():
            rws = recognise_page(engine, prepared)
            buckets = attribute(rws, cells, 0.5)
            for c in cells:
                txt = "".join(r.text.strip() for r in buckets.get(c.cmpd_id, []))
                ref = REFERENCES.get(c.cmpd_id, {}).get("smiles")
                rows_out.append({
                    "page": page, "cmpd_id": c.cmpd_id, "arm": arm,
                    "n_rows": len(buckets.get(c.cmpd_id, [])),
                    "len": len(txt),
                    "open": txt.count("("), "close": txt.count(")"),
                    "edit": edit_ops(ref, txt) if ref else "",
                    "cer": round(edit_ops(ref, txt) / len(ref), 5) if ref else "",
                })

        # ---------------- cell arms ----------------
        for c in cells:
            crop = img[int(c.y1):int(c.y2), col.x1:col.x2]
            trimmed = ex._trim_to_content(crop)
            cell_arms = {
                # exactly the production preprocessing
                "cell_prod_trim_up2": ex._upscale(trimmed),
                # the same, plus binarisation
                "cell_prod_trim_up2_th200": _th200(ex._upscale(trimmed)),
                # no trim, so the crop alone is the variable
                "cell_notrim_up2": _up2(crop),
                "cell_notrim_up2_th200": _th200(_up2(crop)),
            }
            for arm, prepared in cell_arms.items():
                txt = "".join(_sorted_texts(engine(prepared)))
                ref = REFERENCES.get(c.cmpd_id, {}).get("smiles")
                rows_out.append({
                    "page": page, "cmpd_id": c.cmpd_id, "arm": arm,
                    "n_rows": len(_sorted_texts(engine(prepared))),
                    "len": len(txt),
                    "open": txt.count("("), "close": txt.count(")"),
                    "edit": edit_ops(ref, txt) if ref else "",
                    "cer": round(edit_ops(ref, txt) / len(ref), 5) if ref else "",
                })

        lines.append(f"=== page {page} ===")
        for r in rows_out:
            if r["page"] != page:
                continue
            lines.append(
                f"  cmpd {r['cmpd_id']:6s} {r['arm']:<26s} "
                f"rows={r['n_rows']} len={r['len']:>4d} "
                f"open={r['open']:>3d} close={r['close']:>3d} "
                f"edit={r['edit']}"
            )
        lines.append("")

    if rows_out:
        with (OUT / "factorial.csv").open("w", newline="",
                                         encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)

    out = io.open(OUT / "factorial_summary.txt", "w", encoding="utf-8")
    out.write("=== crop-vs-page x binarise-vs-not ===\n\n")
    out.write("The production pipeline does NOT binarise. Phase 2's "
              "'baseline'\n(binarize_up2_en) was never the production "
              "configuration.\n\n")
    out.write("Reference: 1036 = human-verified, 1037 = agent transcription "
              "(not human-confirmed).\n")
    out.write("Neither is treated as chemical truth.\n\n")
    out.write("\n".join(lines))
    out.write("\n--- mean edit distance over the reference compounds ---\n")
    arms = sorted({r["arm"] for r in rows_out})
    out.write(f"{'arm':<28} {'n':>4} {'mean edit':>10} {'mean CER':>10}\n")
    for arm in arms:
        sub = [r for r in rows_out if r["arm"] == arm and r["edit"] != ""]
        if not sub:
            continue
        me = sum(r["edit"] for r in sub) / len(sub)
        mc = sum(r["cer"] for r in sub) / len(sub)
        out.write(f"{arm:<28} {len(sub):>4} {me:>10.2f} {mc:>10.5f}\n")
    out.close()
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
