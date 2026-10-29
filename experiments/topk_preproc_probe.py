r"""Isolate which variable actually preserved the row-start '=' ?

The orphaned ``demo_rapidocr_topk/topk_printed_smiles.json`` was produced by
``_demo_topk.py``, which called ``RapidOCR(params={})`` directly on the RAW
page JPG. It never applied the pipeline baseline's
``2x INTER_CUBIC -> threshold 200`` preprocessing.

So the artifact's intact row-start ``=`` cannot be credited to
"page-level detection": the rebuild in ``topk_page_level.py`` uses the same
page-level detector but DOES apply the baseline preprocessing, and it still
clips the glyph. Preprocessing is the variable that actually differs.

This probe holds the extraction path fixed (page-level detection on all three
variants) and varies only the image handed to the engine:

    raw_default   RapidOCR(params={}) on the untouched page
    up2_only      2x INTER_CUBIC, no binarisation
    up2_th200     the pipeline baseline

For each variant it reports, per compound, the joined SMILES against the
human-verified reference: edit distance, CER, and the first character of
every row so a row-start clip is visible directly.

The per-character top-k field is deliberately not recomputed. The original
script indexed ``top_k_lines`` (one entry per KEPT CTC timestep) by character
position, which is what made its top3 drift out of alignment; the
character-level candidates are covered by ``phase2_recog_candidates.py``.

Read-only with respect to the pipeline and the official CSV.

Run:
    .venv\Scripts\python.exe topk_preproc_probe.py
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
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD, build_engine,
)

from rapidocr import RapidOCR  # noqa: E402
from row_crop_audit import line_boxes, preprocess  # noqa: E402

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase2" / "preproc_probe"
PAGE = "I100297"

HEADERS = ("SMILES", "CMPD #", "CMPD#", "CMPD", "STRUCTURE")


def _columns(rows):
    """SMILES column x-range and the compound label positions, from the page."""
    col = {}
    for r in rows:
        t = r["text"].strip().upper()
        xs = r["box"][0::2]
        if t == "SMILES" and "smi" not in col:
            col["smi"] = (min(xs), max(xs))
        if t.replace(" ", "") == "CMPD#" and "id" not in col:
            col["id"] = (min(xs), max(xs))
    labels = []
    if "id" in col and "smi" in col:
        ilo, ihi = col["id"]
        for r in rows:
            t = r["text"].strip()
            if not t.isdigit():
                continue
            xs = r["box"][0::2]
            ys = r["box"][1::2]
            mid = sum(xs) / len(xs)
            if ilo - 0.3 * max(1.0, ihi - ilo) <= mid <= ihi:
                labels.append((min(ys), t))
    labels.sort()
    return col, labels


def _attribute(rows):
    """Bucket SMILES-column rows into compounds.

    A compound's printed SMILES sits ABOVE its own "Cmpd #" label on these
    pages, so a compound's rows run from the previous label down to its own.
    Getting that backwards silently swaps neighbouring compounds.
    """
    col, labels = _columns(rows)
    if "smi" not in col or not labels:
        return {}, col, labels
    lo, hi = col["smi"]
    span = max(1.0, hi - lo)
    out = {}
    for i, (y_top, cid) in enumerate(labels):
        y_start = labels[i - 1][0] if i > 0 else -1
        for r in rows:
            if r["text"].strip().upper() in HEADERS:
                continue
            xs = r["box"][0::2]
            ys = r["box"][1::2]
            if not (lo - 0.40 * span <= sum(xs) / len(xs)
                    <= hi + 0.40 * span):
                continue
            if not (y_start < sum(ys) / len(ys) <= y_top):
                continue
            out.setdefault(cid, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["line"])
    return out, col, labels


def _recognise_all(engine, img):
    """Full detection + recognition on one prepared image, keeping geometry."""
    det = engine(img)
    boxes = line_boxes(det)
    txts = list(getattr(det, "txts", []) or [])
    rows = []
    for i, poly in enumerate(boxes):
        if i >= len(txts) or not txts[i]:
            continue
        pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
        rows.append({"line": i, "text": txts[i], "box": pts.reshape(-1)})
    return rows


def _up2(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = cv2.imread(str(GOLD / f"PCTCN2025075389-ftappb-{PAGE}.jpg"))

    # exactly what _demo_topk.py built: default params, no preprocessing
    plain = RapidOCR(params={})
    baseline = build_engine()

    variants = {
        "raw_default": (plain, raw),
        "up2_only": (baseline, _up2(raw)),
        "up2_th200": (baseline, preprocess(raw, BASELINE_THRESHOLD)),
    }

    out = io.open(OUT / "preproc_probe.txt", "w", encoding="utf-8")
    out.write(f"=== preprocessing probe, page {PAGE} ===\n")
    out.write("extraction path held fixed (page-level detection);\n")
    out.write("only the image handed to the engine varies.\n")
    out.write("reference = human-verified transcription.\n\n")

    table = []
    for name, (engine, img) in variants.items():
        rows = _recognise_all(engine, img)
        cmpds, _, _ = _attribute(rows)
        out.write(f"--- {name}   image {img.shape[1]}x{img.shape[0]}   "
                  f"{len(rows)} text boxes\n")
        for cid in sorted(cmpds, key=lambda s: int(s)):
            grp = cmpds[cid]
            txt = "".join(r["text"] for r in grp)
            ref = REFERENCES.get(cid, {}).get("smiles")
            starts = "".join(r["text"][:1] for r in grp)
            ed = edit_ops(ref, txt) if ref else None
            out.write(f"    cmpd {cid}: rows={len(grp)} joined={len(txt)} "
                      f"row_starts={starts!r}")
            if ref:
                out.write(f" edit={ed} CER={ed / len(ref):.5f}")
            out.write("\n")
            table.append({
                "variant": name, "cmpd_id": cid, "rows": len(grp),
                "joined_len": len(txt), "ref_len": len(ref) if ref else "",
                "edit": ed if ed is not None else "",
                "cer": round(ed / len(ref), 5) if ref else "",
                "row_starts": starts,
            })
        out.write("\n")
    out.close()

    with (OUT / "preproc_probe.csv").open("w", newline="",
                                         encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
