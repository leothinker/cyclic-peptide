"""Is the missing '=' ink physically present in either line crop?

Line1 ends '...4C(' and line2 begins 'O)...'. The reference has
'C(=O)'. So the '=' falls exactly on the detector's line boundary.
This measures the ink extent of each rectified crop to see whether the
glyph survives cropping at all.
"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

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

lines = [get_rotate_crop_image(proc, np.array(p, dtype=np.float32)) for p in polys]

for i, (ln, poly) in enumerate(zip(lines, polys)):
    gray = cv2.cvtColor(ln, cv2.COLOR_BGR2GRAY)
    bw = gray < 128
    cols = np.where(bw.any(axis=0))[0]
    x1, y1, x2, y2 = box_rect(poly)
    print(
        f"line{i}: crop {ln.shape[1]}x{ln.shape[0]}  "
        f"ink cols {cols.min() if len(cols) else '-'}.."
        f"{cols.max() if len(cols) else '-'}  "
        f"det box x=[{x1},{x2}] y=[{y1},{y2}]"
    )

# tail of line1 and head of line2 at high zoom
for lidx, side in ((1, "tail"), (2, "head")):
    ln = lines[lidx]
    gray = cv2.cvtColor(ln, cv2.COLOR_BGR2GRAY)
    bw = gray < 128
    cols = np.where(bw.any(axis=0))[0]
    lo, hi = int(cols.min()), int(cols.max())
    if side == "tail":
        x0, x1 = max(0, hi - 130), hi + 1
    else:
        x0, x1 = max(0, lo - 1), min(ln.shape[1], lo + 130)
    crop = cv2.cvtColor(gray[:, x0:x1], cv2.COLOR_GRAY2BGR)
    big = cv2.resize(
        crop, (crop.shape[1] * 6, crop.shape[0] * 6), interpolation=cv2.INTER_NEAREST
    )
    path = OUT / f"junction_line{lidx}_{side}_x6.png"
    cv2.imwrite(str(path), big)
    print(f"wrote {path}  (x range {x0}..{x1} of {ln.shape[1]})")

# stack the two junction views into one image for inspection
t = cv2.imread(str(OUT / "junction_line1_tail_x6.png"))
h = cv2.imread(str(OUT / "junction_line2_head_x6.png"))
width = max(t.shape[1], h.shape[1])


def fit(a):
    if a.shape[1] == width:
        return a
    return cv2.copyMakeBorder(
        a, 0, 0, 0, width - a.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


stack = np.concatenate([fit(t), fit(h)], axis=0)
stack = cv2.copyMakeBorder(stack, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
cv2.imwrite(str(OUT / "junction_line1_tail_over_line2_head_x6.png"), stack)
print(f"wrote {OUT/'junction_line1_tail_over_line2_head_x6.png'}")
