"""Split a text-cell crop into individual printed lines, upscaled."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
dst.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(src))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Drop the cell's own vertical rules first: they ink every single row and
# would otherwise merge all printed lines into one band.
col_ink = (gray < 128).sum(axis=0)
body_cols = np.where(col_ink < gray.shape[0] * 0.5)[0]
if len(body_cols) >= 2:
    x1, x2 = int(body_cols.min()), int(body_cols.max()) + 1
else:
    x1, x2 = 0, gray.shape[1]
inner = gray[:, x1:x2]
ink = (inner < 128).sum(axis=1) > 1

runs = []
start = None
for i, v in enumerate(ink):
    if v and start is None:
        start = i
    if not v and start is not None:
        if i - start >= 8:
            runs.append((start, i))
        start = None
if start is not None and len(ink) - start >= 8:
    runs.append((start, len(ink)))

print(f"{len(runs)} text lines detected")
for n, (y1, y2) in enumerate(runs):
    pad = 4
    y1 = max(0, y1 - pad)
    y2 = min(img.shape[0], y2 + pad)
    line = img[y1:y2]
    lh, lw = line.shape[:2]
    line = cv2.resize(
        line, (int(lw * scale), int(lh * scale)), interpolation=cv2.INTER_CUBIC
    )
    out = dst / f"{src.stem}_L{n:02d}.png"
    cv2.imwrite(str(out), line)
    print(f"{n:02d} y={y1}-{y2} h={y2-y1} -> {out.name}")
