"""Locate the true right-hand table rule, ignoring the page-border filter."""

from __future__ import annotations

from pathlib import Path

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402

ex = RapidOcrTableExtractor.__new__(RapidOcrTableExtractor)
for f in sorted(glob.glob("data/raw/WO2025162428/golden_tables/*.jpg")):
    b = os.path.basename(f)
    if not ("I100297" in b or "I100298" in b or "I100377" in b):
        continue
    img = cv2.imread(f)
    h, w = img.shape[:2]
    body = ex._body_slice(img)
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(1, int(body.shape[0] * 0.4)))
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    xs = np.where(np.sum(lines, axis=0) > int(body.shape[0] * 0.4))[0]
    clusters = ex._cluster(xs, gap=ex.COLUMN_GAP_PX)
    print(f"{b}  page_w={w}")
    print(f"  all full-height rules = {clusters}")
    print(f"  filtered borders      = {ex._detect_vertical_borders(img)}")
