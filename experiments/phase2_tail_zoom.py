"""Is the 'lost' '=' actually inside the rectified line crop?"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_box_expansion import (  # noqa: E402
    OUT, box_rect, extract_line, line_boxes, rec_only,
)
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD, build_engine, preprocess,
)

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
polys = line_boxes(engine(proc))
cell_right = proc.shape[1]

tail_dir = OUT / "tail_zoom"
tail_dir.mkdir(parents=True, exist_ok=True)

for pad in (0, 8, 12, 24):
    line = extract_line(proc, polys[1], pad, cell_right)
    w = line.shape[1]
    crop = line[:, max(0, w - 260):]
    ch, cw = crop.shape[:2]
    big = cv2.resize(crop, (cw * 4, ch * 4), interpolation=cv2.INTER_NEAREST)
    out = tail_dir / f"line1_tail_pad{pad}.png"
    cv2.imwrite(str(out), big)
    ink = int((cv2.cvtColor(line, cv2.COLOR_BGR2GRAY) < 128).sum())
    right_ink = int(
        (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 128).sum()
    )
    print(
        f"pad={pad:3d} crop={line.shape[1]:5d} total_ink={ink:6d} "
        f"tail256_ink={right_ink:5d} -> {out.name}"
    )

for pad in (0, 8, 12, 24):
    lines = [extract_line(proc, p, pad, cell_right) for p in polys]
    txt = rec_only(engine, lines)
    first = txt.split("(")[0]
    print(f"pad={pad:3d} line1 recognised: {lines[0].shape} "
          f"text starts {txt[:40]!r}")
