"""Measure real ink extent per detected line vs the detector box right edge."""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_box_expansion import box_rect, line_boxes  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD, build_engine, preprocess,
)

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
polys = line_boxes(engine(proc))
gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"processed image {w}x{h}")
col_ink = (gray < 128).sum(axis=0)
rules = np.where(col_ink > h * 0.5)[0]
print("full-height vertical rules (page borders):", rules.tolist())
text_cols = np.where(
    (col_ink > 0) & (col_ink <= h * 0.5)
)[0]
print(
    f"{'line':>4s} {'box_x1':>7s} {'box_x2':>7s} {'ink_x2':>7s} "
    f"{'deficit':>8s}"
)
for n, p in enumerate(polys):
    x1, y1, x2, y2 = box_rect(p)
    band = gray[max(0, y1 - 3):min(h, y2 + 3), :]
    ink = (band < 128).sum(axis=0)
    cols = np.where((ink > 0) & (ink <= band.shape[0] * 0.5))[0]
    ink2 = int(cols.max()) if len(cols) else -1
    print(
        f"{n:4d} {x1:7d} {x2:7d} {ink2:7d} {ink2 - x2:+8d}"
    )

print()
print(
    f"global text-ink extent: {text_cols.min()}..{text_cols.max()}"
)
print()
print("--- right-edge ink column profile for line 1 (y=101..175) ---")
band = gray[98:178, :]
prof = (band < 128).sum(axis=0)
for x in range(1620, 1680):
    print(f"  x={x}  ink={int(prof[x])}")
