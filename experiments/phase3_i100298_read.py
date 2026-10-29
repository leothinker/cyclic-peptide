r"""Row-level extraction of the printed SMILES text on page I100298.

Purpose
-------
1039-1041 have no researcher-verified reference yet.  Before any
candidate can be scored at character level, the printed rows have to be
readable by a human.  This script produces that reading surface:

    data/audit/phase3/i100298_read/rows/<cmpd>/row<N>.png   3x per row
    data/audit/phase3/i100298_read/strip_<cmpd>.png         stacked strip

The strip is the artefact the researcher uses to check the OCR output
line by line.  Nothing here interprets the text: it only segments the
printed rows and renders them.

Row segmentation reuses the ink-projection method already validated in
``phase3_i100298_validation.py`` so the row boundaries are identical to
the ones the recognition route uses.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_read.py
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

OUT = REPO / "data" / "audit" / "phase3" / "i100298_read"


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _render_strip(cmpd_id: str, row_images: list[np.ndarray]) -> None:
    """Stack the printed rows into one strip with row labels."""
    scale = 3
    pad = 26
    gap = 12
    label_w = 130
    rendered = []
    for crop in row_images:
        big = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        rendered.append(big)
    width = label_w + max(r.shape[1] for r in rendered) + pad * 2
    height = sum(r.shape[0] + gap for r in rendered) + pad * 2

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12),
              f"{cmpd_id}  printed rows  (page I100298, 3x)",
              font=_font(24, bold=True), fill=(27, 27, 27))

    y = 52
    for index, big in enumerate(rendered, start=1):
        draw.text((pad, y + 4), f"row {index}", font=_font(20),
                  fill=(120, 120, 120))
        canvas.paste(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)),
                     (label_w, y))
        y += big.shape[0] + gap

    canvas.save(OUT / f"strip_{cmpd_id}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cmpd_id in TARGETS:
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)

        row_dir = OUT / "rows" / cmpd_id
        row_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        for index, (top, bottom) in enumerate(bands, start=1):
            y1 = max(0, top - 6)
            y2 = min(image.shape[0], bottom + 7)
            crop = image[y1:y2, :]
            crops.append(crop)
            big = cv2.resize(crop, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(row_dir / f"row{index}.png"), big)
        _render_strip(cmpd_id, crops)
        print(f"{cmpd_id}: {len(bands)} rows -> {OUT}")


if __name__ == "__main__":
    main()
