"""Compare the two preprocessing pipelines pixel by pixel."""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_ocr_compare import preprocess as old_pre  # noqa: E402
from phase2_threshold_scan import preprocess as new_pre  # noqa: E402

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
a = old_pre(img, upscale=2, binarize=True, strip=0)
b = new_pre(img, 200)
print("old (upscale->binarize):", a.shape, "ink", int((a < 128).sum()))
print("new (binarize->upscale):", b.shape, "ink", int((b < 128).sum()))
print("identical:", a.shape == b.shape and bool((a == b).all()))
if a.shape == b.shape:
    d = (a != b).sum(axis=2)
    print("differing pixels:", int((d > 0).sum()))
