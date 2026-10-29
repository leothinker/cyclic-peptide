r"""Zoom the row boundaries on page I100298.

Why
---
The printed SMILES is cut at arbitrary points, sometimes in the middle
of a bracket pair, so a row boundary can fall between ``(C`` and ``)``.
Deciding where a row really ends by eye is error prone, and one wrong
boundary silently changes the whole reference.

This script renders, for every boundary of every compound on this
page, the tail of the upper row next to the head of the lower row, so
the split can be read off the pixels.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_junction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase3_i100298_validation import (  # noqa: E402
    CELLS, TARGETS, _segment_rows,
)

OUT = REPO / "data" / "audit" / "phase3" / "i100298_read" / "junctions"

TAIL_CHARS = 26
HEAD_CHARS = 26


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _glyph_cells(band: np.ndarray) -> list[tuple[int, int]]:
    """Column spans of the glyphs in one ink band."""
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    dark = (gray < 160).sum(axis=0)
    active = dark > 0
    spans: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(active):
        if on and start is None:
            start = x
        elif not on and start is not None:
            spans.append((start, x - 1))
            start = None
    if start is not None:
        spans.append((start, len(active) - 1))
    return spans


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cmpd_id in TARGETS:
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)

        for boundary in range(len(bands) - 1):
            top = bands[boundary]
            bottom = bands[boundary + 1]
            upper = image[max(0, top[0] - 6):top[1] + 7, :]
            lower = image[max(0, bottom[0] - 6):bottom[1] + 7, :]

            upper_spans = _glyph_cells(upper)
            lower_spans = _glyph_cells(lower)
            if not upper_spans or not lower_spans:
                continue

            upper_x1 = upper_spans[-TAIL_CHARS][0] if len(
                upper_spans) > TAIL_CHARS else upper_spans[0][0]
            lower_x2 = lower_spans[min(HEAD_CHARS, len(lower_spans)) - 1][1]
            lower_x1 = lower_spans[0][0]

            tail = upper[:, upper_x1:]
            head = lower[:, lower_x1:lower_x2 + 1]
            scale = 6
            tail_big = cv2.resize(tail, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            head_big = cv2.resize(head, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)

            pad = 18
            label_w = 150
            width = label_w + tail_big.shape[1] + pad * 3
            height = (tail_big.shape[0] + head_big.shape[0]
                      + pad * 3 + 56)
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((pad, 12),
                      f"{cmpd_id}  boundary {boundary + 1}/{boundary + 2}",
                      font=_font(22, bold=True), fill=(27, 27, 27))
            draw.text((pad, 40),
                      "tail of the upper printed row, then the head of "
                      "the lower one",
                      font=_font(15), fill=(110, 110, 110))
            draw.text((pad, 68), "row "
                      f"{boundary + 1} ends", font=_font(17),
                      fill=(150, 60, 60))
            canvas.paste(Image.fromarray(
                cv2.cvtColor(tail_big, cv2.COLOR_BGR2RGB)),
                (label_w, pad))
            y = pad + tail_big.shape[0] + 22
            draw.text((pad, y), "row "
                      f"{boundary + 2} starts", font=_font(17),
                      fill=(60, 110, 60))
            canvas.paste(Image.fromarray(
                cv2.cvtColor(head_big, cv2.COLOR_BGR2RGB)),
                (label_w, y + 24))

            canvas.save(OUT / f"{cmpd_id}_boundary{boundary + 1}.png")
        print(f"{cmpd_id}: {len(bands) - 1} boundaries")


if __name__ == "__main__":
    main()
