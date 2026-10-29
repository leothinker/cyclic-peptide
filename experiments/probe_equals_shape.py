"""Visual check: is the ink left of line 2's box a '=' glyph or a rule?"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    build_engine,
)
from row_crop_audit import (  # noqa: E402
    INK,
    OUT,
    box_rect,
    line_boxes,
    preprocess,
)

crop = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(crop, BASELINE_THRESHOLD)
g = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
polys = line_boxes(build_engine()(proc))

for li in (0, 2, 3):
    x1, y1, x2, y2 = box_rect(polys[li])
    sub = g[max(0, y1 - 3):y2 + 3, 0:150]
    k = 4
    big = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    head = np.full((26, big.shape[1], 3), 255, np.uint8)
    cv2.putText(
        head, f"line {li}  box x1={x1}  (x=0..150 of processed)",
        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.line(big, (x1 * k, 0), (x1 * k, big.shape[0] - 1), (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"probe_line{li}_leftmost_ink.png"),
                np.vstack([head, big]))
    col = (g[max(0, y1 - 3):y2 + 3, 0:150] < INK).sum(axis=0)
    nz = [int(c) for c in np.where(col > 0)[0]]
    print(f"line {li}: box x1={x1}  first 12 inked cols={nz[:12]}")

print("wrote probe_line{0,2,3}_leftmost_ink.png")
