"""Two report figures for the phase-3 external validation.

``figure_recogniser_only.png``
    The controlled comparison: identical crop geometry, only the
    recogniser swapped.  This is the answer to "why did the shipped
    CSV lose every left bracket?" -- and the reason the earlier
    "upscaling made it worse" observation was a confound.

``figure_2173.png``
    Compound 2173: the route outputs and the character-level
    difference.  This is the generalisation evidence -- a compound on
    a page that was never used for tuning, and which the pipeline had
    never managed to read.

Writes to ``data/audit/phase3/external/``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "external"

INK = (30, 30, 30)
GREY = (110, 110, 110)
LIGHT = (205, 205, 205)
# OpenCV is BGR, so these are (blue, green, red).
ACCENT = (190, 90, 40)   # blue   -- the validated route
WARN = (40, 120, 200)    # orange -- the production route


def text_width(text: str, scale: float = 0.42) -> int:
    return int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0])


def put(
    image: np.ndarray,
    origin: tuple[int, int],
    text: str,
    scale: float = 0.42,
    color=INK,
    weight: int = 1,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight,
        cv2.LINE_AA,
    )


def figure_recogniser_only() -> Path:
    """Bar chart of the controlled recogniser comparison."""
    summary_path = (
        REPO
        / "data"
        / "audit"
        / "phase3"
        / "recogniser_only"
        / "recogniser_only_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = summary["cells"]
    arms = [
        (
            "English PP-OCRv4 mobile  (production)",
            summary["english_ppocrv4_mobile"],
            WARN,
        ),
        (
            "default PP-OCRv6 small  (validated)",
            summary["default_ppocrv6_small"],
            ACCENT,
        ),
    ]

    width, height = 1180, 560
    image = np.full((height, width, 3), 255, np.uint8)
    put(image, (40, 46), "Same crops, only the recogniser swapped",
        scale=0.62, color=INK, weight=2)
    put(
        image,
        (40, 74),
        "identical ink bands, identical padding, identical 2x upscale, "
        "recognition-only mode",
        0.44,
        GREY,
    )

    x0, y0, bar_h = 60, 124, 50
    for index, (label, stats, color) in enumerate(arms):
        y = y0 + index * (bar_h + 40)
        put(image, (x0, y - 9), label, 0.45, INK)
        count = stats["balanced"]
        bar_w = int((width - 430) * count / cells)
        cv2.rectangle(image, (x0, y), (x0 + bar_w, y + bar_h), color, -1)
        put(image, (x0 + bar_w + 14, y + 36), f"{count} / {cells}",
            0.55, color, 2)
    put(image, (x0, y0 + 2 * (bar_h + 40) + 2),
        "cells whose brackets balance", 0.45, GREY)

    y = y0 + 2 * (bar_h + 40) + 48
    put(image, (x0, y), "total brackets over the corpus", 0.5, INK, 2)
    y += 32
    for index, (label, stats, color) in enumerate(arms):
        put(image, (x0, y + index * 32), label, 0.42, GREY)
        missing = stats["total_brackets_open"] - stats["total_brackets_close"]
        put(
            image,
            (x0 + 470, y + index * 32),
            f"open {stats['total_brackets_open']}    "
            f"close {stats['total_brackets_close']}    "
            f"left brackets lost: {abs(missing)}",
            0.45,
            color,
        )

    box_y = height - 112
    cv2.rectangle(image, (40, box_y), (width - 40, height - 34), LIGHT, -1)
    put(
        image,
        (60, box_y + 32),
        "The English recogniser drops 2429 left brackets across the corpus.",
        0.52,
        INK,
        2,
    )
    put(
        image,
        (60, box_y + 60),
        "That is the direct cause of the 335/335 bracket imbalance in the "
        "shipped CSV - and it has nothing to do with cropping.",
        0.42,
        GREY,
    )

    target = OUT / "figure_recogniser_only.png"
    cv2.imwrite(str(target), image)
    return target


def figure_2173() -> Path:
    """The 2173 comparison table."""
    check = OUT / "external_2173_check.csv"
    with check.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]

    width, height = 1180, 470
    image = np.full((height, width, 3), 255, np.uint8)
    put(image, (40, 46),
        "Compound 2173 - a page never used for tuning",
        scale=0.62, color=INK, weight=2)
    put(
        image,
        (40, 74),
        "I100376 (= PDF page 240).  The shipped CSV has no usable structure "
        "for this compound.",
        0.44,
        GREY,
    )

    y = 122
    for text, x in (
        ("source", 60),
        ("length", 520),
        ("edit vs reference", 650),
        ("brackets", 890),
    ):
        put(image, (x, y), text, 0.45, GREY)
    cv2.line(image, (40, y + 12), (width - 40, y + 12), LIGHT, 2)

    table = [
        ("validated route", "validated_length",
         "validated_edit_vs_reference", "validated_brackets", ACCENT, True),
        ("production route", "production_length",
         "production_edit_vs_reference", "production_brackets", WARN, False),
        ("shipped CSV smiles", "shipped_length",
         "shipped_edit_vs_reference", "shipped_brackets", WARN, False),
    ]
    for index, (label, lk, ek, bk, color, good) in enumerate(table):
        yy = y + 50 + index * 48
        put(image, (60, yy), label, 0.5, color, 2 if good else 1)
        put(image, (520, yy), row[lk], 0.48, INK)
        put(image, (650, yy), row[ek], 0.48, color, 2 if good else 1)
        put(image, (890, yy), row[bk], 0.48, INK)

    box_y = height - 122
    cv2.rectangle(image, (40, box_y), (width - 40, height - 34), LIGHT, -1)
    put(image, (60, box_y + 32),
        "The one remaining error:  C(=O)  read as  C(=0)", 0.52, INK, 2)
    put(image, (60, box_y + 60),
        "letter O mistaken for digit zero.  The 3x crop confirms the printed "
        "glyph is a letter, so it is an OCR error, not a reference error.",
        0.42, GREY)

    target = OUT / "figure_2173.png"
    cv2.imwrite(str(target), image)
    return target


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (figure_recogniser_only(), figure_2173()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
