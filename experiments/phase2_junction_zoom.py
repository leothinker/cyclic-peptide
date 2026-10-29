"""Zoom the exact junction from the FULL processed image.

Row 1 ends at the far right of the cell (x~1642) and row 2 starts at the
far left (x~41), so the missing '=' belongs to the right edge of row 1.
This re-cuts that region from the full processed image with a generous
margin, so nothing depends on the detector's own box.
"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_box_expansion import box_rect, line_boxes  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    load_meta,
    preprocess,
)

OUT = PHASE2 / "recognizer_candidates"
OUT.mkdir(parents=True, exist_ok=True)

meta = load_meta()
img = cv2.imread(str(PHASE2 / meta["1036"]["crop_path"]))
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
full = engine(proc)
polys = line_boxes(full)
gray_full = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

x1, y1, x2, y2 = box_rect(polys[1])
print(f"line1 det box x=[{x1},{x2}] y=[{y1},{y2}]  cell width={proc.shape[1]}")

# generous window: past the detector's right edge, clamped to the cell
for pad in (0, 20, 60):
    xr = min(proc.shape[1] - 1, x2 + pad)
    x0 = max(0, xr - 150)
    band = gray_full[y1 - 6:y2 + 6, x0:xr + 1]
    bw = (band < 128).astype(np.uint8) * 255
    big = cv2.resize(bw, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    path = OUT / f"junction_row1_right_pad{pad}_x8.png"
    cv2.imwrite(str(path), big)
    cols = np.where((band < 128).any(axis=0))[0]
    print(
        f"pad={pad:3d} window x=[{x0},{xr}]  ink cols "
        f"{cols.min() if len(cols) else '-'}..{cols.max() if len(cols) else '-'}"
        f"  (window width {band.shape[1]})"
    )
    print(f"  wrote {path}")

# full rows, to confirm reading order and where each row starts/ends
for lidx in (1, 2):
    bx1, by1, bx2, by2 = box_rect(polys[lidx])
    band = gray_full[by1 - 4:by2 + 4, :]
    small = cv2.resize(
        band,
        (band.shape[1] // 2, band.shape[0] * 3),
        interpolation=cv2.INTER_NEAREST,
    )
    p = OUT / f"row{lidx}_fullrow_x0.5.png"
    cv2.imwrite(str(p), small)
    print(f"wrote {p}")
