"""Fine-grained left-extension test on line 2, printing its own text.

The coarse sweep was confounded: moving a detector box's left edge changes
the perspective homography for the WHOLE line, so the line's content
rescales and spurious characters appear. Here the cell's own vertical rules
are excluded from the ink measurement, and each pad reports line 2's text
so the actual effect on the leading '=' is visible.
"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import box_rect, line_boxes, rec_only  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    load_meta,
    preprocess,
)

meta = load_meta()
img = cv2.imread(str(PHASE2 / meta["1036"]["crop_path"]))
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
full = engine(proc)
polys = line_boxes(full)
gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
W = gray.shape[1]

ref = REFERENCES["1036"]["smiles"]


def ink_span(y1: int, y2: int, guess_x1: int, guess_x2: int, margin: int = 25):
    """Ink columns near one row, ignoring the cell's vertical rules."""
    lo = max(0, guess_x1 - margin)
    hi = min(W, guess_x2 + margin)
    band = gray[y1:y2 + 1, lo:hi]
    cols = np.where((band < 128).any(axis=0))[0]
    if not len(cols):
        return None
    return lo + int(cols.min()), lo + int(cols.max())


print("per-row box vs ink (cell rules excluded by a local window)")
spans = []
for i, poly in enumerate(polys):
    x1, y1, x2, y2 = box_rect(poly)
    sp = ink_span(y1, y2, x1, x2)
    spans.append(sp)
    print(f"line{i}: box x=[{x1},{x2}]  ink x=[{sp[0]},{sp[1]}]  "
          f"left deficit={sp[0]-x1:+d} right deficit={x2-sp[1]:+d}")

print()
print("line 2 detector quad corners:")
for k, pt in enumerate(polys[2]):
    print(f"  p{k} = {pt}")

print()
print("=== line 2 with left edge moved; prints line-2 text only ===")
best = None
for pad in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48):
    out = []
    for i, poly in enumerate(polys):
        p = np.array(poly, dtype=np.float32).copy()
        if i == 2 and pad:
            nl = max(0.0, float(p[0][0]) - pad)
            p[0][0] = nl
            p[3][0] = nl
        out.append(get_rotate_crop_image(proc, p))
    texts = engine.recognize_txt(out).txts
    l2 = list(texts)[2]
    joined = "".join(t for t in texts if t)
    e = edit_ops(ref, joined)
    flag = ""
    if best is None or e < best[1]:
        best = (pad, e)
    if l2.startswith("=O)N"):
        flag = "  <-- '=' RECOVERED"
    print(f"pad={pad:3d} edits={e:3d} len={len(joined):4d} "
          f"line2[0:22]={l2[:22]!r}{flag}")

print(f"\nbest pad={best[0]} edits={best[1]} (baseline edits=2)")
