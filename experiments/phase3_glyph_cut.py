"""Is a *partially cut* trailing glyph enough to break a whole line?

The pipeline's SMILES column ends 6 px before the printed ink does, so
the last glyph of the longest rows is sliced through the middle.  This
script measures that directly:

* where the ink of each row ends, in page coordinates;
* whether the pipeline crop cuts a glyph (ink touching the last
  column of the crop) and how many pixels of that glyph are lost;
* what the recogniser returns for the same row with the glyph
  complete, with the glyph cut, and with the cut row but one extra
  column of white padding.

If a half-cut glyph is the trigger, the fix belongs in the crop
geometry (never cut a glyph), not in the recogniser.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    default_recogniser,
    ink_bands,
    text_span,
    upscale,
)

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
PIPELINE = (
    REPO / "data" / "audit" / "phase3" / "i100298_validation" / "cellcrops"
)
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"

COMPOUNDS = ("1038", "1039", "1040", "1041")


def recognise(engine, image: np.ndarray) -> str:
    result = engine(image, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def band_strips(cell: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append((top, bottom, x1, x2, row[:, x1 : x2 + 1]))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()
    rows = []

    for idx, cmpd in enumerate(COMPOUNDS, start=1):
        phase2 = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
        pipe = cv2.imread(str(PIPELINE / f"pipeline_smiles_cell_{idx:02d}.png"))

        bands_p2 = band_strips(phase2)
        bands_pipe = band_strips(pipe)

        for band, ((t, b, x1, x2, strip_p2), (_, _, q1, q2, strip_pipe)) in enumerate(
            zip(bands_p2, bands_pipe), start=1
        ):
            # Does the pipeline crop cut the ink?
            ink_at_edge = q2 >= pipe.shape[1] - 1
            lost = x2 - q2

            text_p2 = recognise(engine, upscale(strip_p2))
            text_pipe = recognise(engine, upscale(strip_pipe))

            # The cut strip with white padding on the right: does the
            # decoder recover when the glyph is *not* cut but the strip
            # is still narrower?
            padded = cv2.copyMakeBorder(
                strip_pipe,
                0,
                0,
                0,
                max(0, strip_p2.shape[1] - strip_pipe.shape[1]),
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            text_pad = recognise(engine, upscale(padded))

            rows.append(
                {
                    "cmpd_id": cmpd,
                    "band": band,
                    "phase2_width": int(strip_p2.shape[1]),
                    "pipeline_width": int(strip_pipe.shape[1]),
                    "px_lost": int(lost),
                    "ink_touches_crop_edge": bool(ink_at_edge),
                    "text_phase2": text_p2,
                    "text_pipeline": text_pipe,
                    "text_pipeline_padded": text_pad,
                    "same_as_phase2": text_pipe == text_p2,
                    "pad_recovers": text_pad == text_p2,
                }
            )
            print(
                f"{cmpd} band{band}: p2={strip_p2.shape[1]} "
                f"pipe={strip_pipe.shape[1]} lost={lost} "
                f"edge={ink_at_edge} "
                f"same={text_pipe == text_p2} "
                f"pad_recovers={text_pad == text_p2}"
            )
            if text_pipe != text_p2:
                print(f"    p2   : {text_p2}")
                print(f"    pipe : {text_pipe}")
                print(f"    pad  : {text_pad}")

    with (OUT / "glyph_cut.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "glyph_cut.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cut = [r for r in rows if r["px_lost"] > 0]
    broke = [r for r in rows if not r["same_as_phase2"]]
    print(
        f"\nrows with ink cut: {len(cut)}/{len(rows)}; "
        f"rows whose text changed: {len(broke)}/{len(rows)}; "
        f"padding recovered: {sum(1 for r in broke if r['pad_recovers'])}"
        f"/{len(broke)}"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
