r"""Task B: audit the geometry of every printed-SMILES text cell.

For each cell this re-detects the text lines and measures, per line, how
much ink sits outside the detector box on each side, then classifies that
overhang by SHAPE rather than by size:

    wide and short   many cols, few rows   a whole glyph such as '=' has
                                            been excluded by the box
    narrow and tall  few cols, many rows   the edge of an ordinary glyph
                                            peeking out, i.e. slack

The second case is NOT a lost character and is not counted as one. Nothing
is called confirmed on pixel counts alone.

Read-only. Writes only into data/audit/phase2/row_crop_audit/.

Run:
    .venv\Scripts\python.exe row_crop_audit_batch.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    build_engine,
)

from row_crop_audit import (  # noqa: E402
    CELLS,
    INK,
    OUT,
    RULE_FRAC,
    RULE_PAD,
    box_rect,
    dilate,
    edge_contact,
    line_boxes,
    preprocess,
)

MIN_GLYPH_COLS = 12
GLYPH_MAX_ROWS = 14


def overhang_profile(gray, poly, excluded, side):
    """Characterise the ink just outside one side of a detector box.

    Returns (px, n_cols, n_rows, label) where px is how far the outermost
    ink column sits beyond the box, and label is one of glyph_excluded,
    glyph_edge, rule or none.
    """
    x1, y1, x2, y2 = box_rect(poly)
    h, w = gray.shape
    y0, y1b = max(0, y1 - 3), min(h, y2 + 3)
    lo, hi = (0, x1) if side == "left" else (x2 + 1, w)
    band = gray[y0:y1b]
    if band.size == 0 or hi <= lo:
        return 0, 0, 0, "none"
    found = np.where((band < INK).any(axis=0))[0]
    outside = [
        int(c) for c in found
        if int(c) not in excluded and lo <= int(c) < hi
    ]
    if not outside:
        return 0, 0, 0, "none"
    c_lo, c_hi = min(outside), max(outside)
    px = (x1 - c_lo) if side == "left" else (c_hi - x2)
    sub = gray[y0:y1b, c_lo:c_hi + 1]
    ncol = int((sub < INK).any(axis=0).sum())
    nrow = int((sub < INK).any(axis=1).sum())
    if ncol >= MIN_GLYPH_COLS and nrow <= GLYPH_MAX_ROWS:
        label = "glyph_excluded"
    elif ncol < MIN_GLYPH_COLS:
        label = "glyph_edge"
    else:
        label = "rule"
    return px, ncol, nrow, label


def blank_record(m):
    return {
        "cmpd_id": m["cmpd_id"],
        "page": m["page"],
        "source_image": m["source_image"],
        "row_index": m["row_index"],
        "layout": m["layout"],
        "extraction_status": m["extraction_status"],
        "is_header": "no" if m["cmpd_id"].strip() else "yes",
        "crop_path": m["crop_path"],
        "crop_px": m["crop_px"],
        "bbox_x1": m["bbox_x1"],
        "bbox_y1": m["bbox_y1"],
        "bbox_x2": m["bbox_x2"],
        "bbox_y2": m["bbox_y2"],
        "n_lines": "",
        "n_rule_cols": "",
        "worst_left_overhang_px": "",
        "worst_right_overhang_px": "",
        "n_lines_glyph_excluded": "",
        "glyph_detail": "",
        "contact_left": "",
        "contact_right": "",
        "contact_top": "",
        "contact_bottom": "",
        "error": "",
    }


def audit_cell(engine, m):
    """Full geometry record for one cell crop."""
    path = REPO / "data" / "audit" / "phase2" / m["crop_path"]
    img = cv2.imread(str(path))
    if img is None:
        rec = blank_record(m)
        rec["error"] = "crop unreadable"
        return rec
    proc = preprocess(img, BASELINE_THRESHOLD)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rules = np.where((gray < INK).sum(axis=0) > h * RULE_FRAC)[0]
    excluded = dilate(rules, RULE_PAD, w)
    rec = blank_record(m)
    rec["n_rule_cols"] = len(rules)
    rec["proc_w"] = w
    rec["proc_h"] = h
    rec.update(edge_contact(gray, excluded))
    try:
        polys = line_boxes(engine(proc))
    except Exception as exc:
        rec["error"] = f"detect failed: {type(exc).__name__}"
        return rec
    rec["n_lines"] = len(polys)
    worst_l = worst_r = 0
    n_glyph = 0
    details = []
    for li, poly in enumerate(polys):
        lp, lc, lr, llabel = overhang_profile(gray, poly, excluded, "left")
        rp, rc, rr, rlabel = overhang_profile(gray, poly, excluded, "right")
        worst_l = max(worst_l, lp)
        worst_r = max(worst_r, rp)
        if "glyph_excluded" in (llabel, rlabel):
            n_glyph += 1
            details.append(
                f"L{li} left={lp}px {llabel} {lc}x{lr}"
                f" right={rp}px {rlabel} {rc}x{rr}"
            )
    rec["worst_left_overhang_px"] = worst_l
    rec["worst_right_overhang_px"] = worst_r
    rec["n_lines_glyph_excluded"] = n_glyph
    rec["glyph_detail"] = " | ".join(details)
    return rec


COLS = [
    "cmpd_id", "page", "source_image", "row_index", "layout",
    "is_header", "extraction_status", "crop_path", "crop_px",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "proc_w", "proc_h", "n_lines", "n_rule_cols",
    "worst_left_overhang_px", "worst_right_overhang_px",
    "n_lines_glyph_excluded", "glyph_detail",
    "contact_left", "contact_right", "contact_top", "contact_bottom",
    "error",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (CELLS / "cells_text_meta.csv").open(
        newline="", encoding="utf-8-sig"
    ) as fh:
        meta = list(csv.DictReader(fh))
    engine = build_engine()
    rows = []
    for i, m in enumerate(meta, 1):
        rec = audit_cell(engine, m)
        if rec is not None:
            rows.append(rec)
        if i % 50 == 0:
            print(f"  ...{i}/{len(meta)}", flush=True)
    with (OUT / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"audited {len(rows)} cells -> {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
