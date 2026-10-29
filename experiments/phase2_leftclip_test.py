"""Test the new hypothesis: the missing '=' is clipped on line 2's LEFT edge.

Line 2's detector box starts at x=41 while every other row starts at x~10,
and the printed text there is '=O)N(C)('. So the box cuts the leading '='.
This measures per-row ink vs box and re-recognises with a left-extended box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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

ref = REFERENCES["1036"]["smiles"]
lines = [get_rotate_crop_image(proc, np.array(p, dtype=np.float32)) for p in polys]
base = rec_only(engine, lines)
print(f"baseline len={len(base)} edits={edit_ops(ref, base)}")
print(
    f"{'line':>4s} {'box x1':>7s} {'box x2':>7s} {'ink x1':>7s} "
    f"{'ink x2':>7s} {'left def':>8s} {'right def':>9s}"
)
for i, poly in enumerate(polys):
    x1, y1, x2, y2 = box_rect(poly)
    band = gray[y1:y2 + 1, :]
    cols = np.where((band < 128).any(axis=0))[0]
    ix1, ix2 = int(cols.min()), int(cols.max())
    print(
        f"{i:4d} {x1:7d} {x2:7d} {ix1:7d} {ix2:7d} "
        f"{ix1 - x1:8d} {x2 - ix2:9d}"
    )

print()
print("=== re-recognise with line 2's box extended LEFT ===")
for pad in (0, 10, 20, 30, 40):
    out_lines = []
    for i, poly in enumerate(polys):
        p = np.array(poly, dtype=np.float32).copy()
        if i == 2 and pad:
            nl = max(0.0, float(p[0][0]) - pad)
            p[0][0] = nl
            p[3][0] = nl
        out_lines.append(get_rotate_crop_image(proc, p))
    got = rec_only(engine, out_lines)
    e = edit_ops(ref, got)
    print(f"pad_left={pad:3d} line2 crop w={out_lines[2].shape[1]} "
          f"total len={len(got)} edits={e}")
    print(f"              head={got[:46]!r}")
