"""Split a text-cell crop into readable horizontal bands for transcription."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(src))
h, w = img.shape[:2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows = (gray < 128).sum(axis=1)
ink = np.where(rows > 0)[0] if False else None

band_h = h // n
for i in range(n):
    y1 = max(0, i * band_h - 6)
    y2 = min(h, (i + 1) * band_h + 6)
    band = img[y1:y2]
    bh, bw = band.shape[:2]
    band = cv2.resize(
        band, (int(bw * scale), int(bh * scale)), interpolation=cv2.INTER_CUBIC
    )
    out = dst / f"{src.stem}_band{i}.png"
    cv2.imwrite(str(out), band)
    print(out)
