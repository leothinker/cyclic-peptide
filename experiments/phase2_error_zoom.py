"""Locate the two residual OCR failures and save zoomed crops of them."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_threshold_scan import (  # noqa: E402
    PHASE2, build_engine, preprocess,
)

OUT = PHASE2 / "threshold_scan" / "error_zoom"
OUT.mkdir(parents=True, exist_ok=True)

crop = PHASE2 / "cells_text/I100297_1036_text.png"
img = cv2.imread(str(crop))
proc = preprocess(img, 200)

engine = build_engine()
res = engine(proc)
boxes = res.boxes
txts = res.txts
print("boxes:", None if boxes is None else len(boxes))
for b, t in zip(boxes, txts):
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    print(f"  x={min(xs):7.1f}-{max(xs):7.1f} y={min(ys):7.1f}-"
          f"{max(ys):7.1f}  {t!r}")

if boxes is None:
    sys.exit(0)

# The two failing glyphs sit in the printed lines carrying "C(=O)" (the
# amide carbonyls) and "=C(C(F)=C5)" (the fluorinated aryl ring). Save a
# high-zoom strip of every detected text line so a human can see whether
# the equals signs are thin, broken, or simply mis-segmented.
h, w = proc.shape[:2]
for n, b in enumerate(boxes):
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    x1, x2 = max(0, int(min(xs)) - 6), min(w, int(max(xs)) + 6)
    y1, y2 = max(0, int(min(ys)) - 6), min(h, int(max(ys)) + 6)
    line = proc[y1:y2, x1:x2]
    lh, lw = line.shape[:2]
    big = cv2.resize(line, (int(lw * 3), int(lh * 3)),
                     interpolation=cv2.INTER_NEAREST)
    out = OUT / f"line{n:02d}_x{x1}-{x2}.png"
    cv2.imwrite(str(out), big)
    print(f"wrote {out.name}  text={txts[n]!r}")

gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
ink = int((gray < 128).sum())
print("total ink px in processed image:", ink)
