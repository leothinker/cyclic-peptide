"""Recogniser-only comparison on identical crops.

The corpus run (``phase3_validated_corpus.py``) shows what the
validated route achieves.  This script isolates *which* part of that
route does the work, by holding the crop geometry completely fixed
and swapping only the recogniser:

* crop geometry: identical for both arms -- the same ink bands, the
  same vertical table rules trimmed, the same padding, the same 2x
  upscale;
* recogniser A: the production English ``PP-OCRv4_rec_mobile``;
* recogniser B: the upstream default ``PP-OCRv6_rec_small``.

Both arms are fed to the recogniser in recognition-only mode
(``use_det=False``), because the validated route does its own line
segmentation and the detector would otherwise introduce a second
variable.

This isolates the recogniser model from the geometry, which is the
comparison the earlier "up2 vs none" probe confounded.

Read-only.  Writes to ``data/audit/phase3/recogniser_only/``.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    default_recogniser,
    english_recogniser,
    ink_bands,
    text_span,
    upscale,
)

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "recogniser_only"

LEGAL = set("BCNOPSFIHbcnopsf0123456789=#@[]()+-/\\%.")


def read_cell(engine, image: np.ndarray) -> str:
    """The validated route's geometry, with a swappable recogniser."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parts: list[str] = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(image.shape[0], bottom + PAD_BOTTOM)
        row = image[y1:y2, :]
        if row.size == 0:
            continue
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        variant = row[:, x1:x2 + 1]
        if variant.size == 0:
            continue
        result = engine(upscale(variant), use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        if txts and (txts[0] or "").strip():
            parts.append((txts[0] or "").strip())
    return "".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    engines = {
        "english_ppocrv4_mobile": english_recogniser(),
        "default_ppocrv6_small": default_recogniser(),
    }

    rows: list[dict[str, object]] = []
    for path in sorted(CELLS.glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        stem = path.stem
        cmpd_id = stem.split("_")[-2] if "_" in stem else stem
        row: dict[str, object] = {"cell": stem, "cmpd_id": cmpd_id}
        for label, engine in engines.items():
            text = read_cell(engine, image)
            row[f"{label}_length"] = len(text)
            row[f"{label}_balanced"] = text.count("(") == text.count(")")
            row[f"{label}_bracket_open"] = text.count("(")
            row[f"{label}_bracket_close"] = text.count(")")
            row[f"{label}_illegal"] = "".join(
                sorted({c for c in text if c not in LEGAL})
            )
            row[f"{label}_text"] = text
        rows.append(row)

    n = len(rows)
    summary = {"cells": n}
    for label in engines:
        lengths = [r[f"{label}_length"] for r in rows]
        summary[label] = {
            "balanced": sum(1 for r in rows if r[f"{label}_balanced"]),
            "illegal": sum(1 for r in rows if r[f"{label}_illegal"]),
            "median_length": statistics.median(lengths or [0]),
            "total_brackets_open": sum(r[f"{label}_bracket_open"] for r in rows),
            "total_brackets_close": sum(r[f"{label}_bracket_close"] for r in rows),
        }
    print(json.dumps(summary, indent=2))

    out_csv = OUT / "recogniser_only.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "recogniser_only_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
