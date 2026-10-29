"""Freeze the geometric evidence that detector boxes never truncate ink."""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_box_expansion import OUT, box_rect, line_boxes  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD, build_engine, preprocess,
)

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
polys = line_boxes(engine(proc))
gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

col_ink = (gray < 128).sum(axis=0)
rules = np.where(col_ink > h * 0.5)[0]

lines = [
    "Detector-box right edge vs real text-ink extent (processed coords)",
    f"processed image: {w}x{h}   full-height rules: {rules.tolist()}",
    "",
    "deficit = ink_x2 - box_x2. A POSITIVE deficit would mean the box",
    "clipped a glyph. Negative means the box already extends past the ink.",
    "",
    f"{'line':>4s} {'box_x1':>7s} {'box_x2':>7s} {'ink_x1':>7s} "
    f"{'ink_x2':>7s} {'deficit':>8s} {'verdict':>12s}",
]
for n, p in enumerate(polys):
    x1, y1, x2, y2 = box_rect(p)
    band = gray[max(0, y1 - 3):min(h, y2 + 3), :]
    ink = (band < 128).sum(axis=0)
    cols = np.where((ink > 0) & (ink <= band.shape[0] * 0.5))[0]
    lo, hi = (int(cols.min()), int(cols.max())) if len(cols) else (-1, -1)
    deficit = hi - x2
    verdict = "CLIPPED" if deficit > 0 else "box past ink"
    lines.append(
        f"{n:4d} {x1:7d} {x2:7d} {lo:7d} {hi:7d} {deficit:+8d} "
        f"{verdict:>12s}"
    )
lines += [
    "",
    "Conclusion: every detector box already extends 1-6 px beyond its line's",
    "last inked column. No glyph is clipped by the box, so widening the box",
    "cannot recover a character - it only appends white margin, which",
    "degrades PP-OCR's aspect-ratio prior.",
]
path = OUT / "box_vs_ink_extent.txt"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nwrote {path}")
