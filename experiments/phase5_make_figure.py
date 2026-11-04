"""Figure for the WO2022251372 recognition result.

Three panels, all built from the files this phase produced:

1. the crop of one printed line with and without the 6 px margin, and
   the text each one produces -- the mechanism;
2. the recognised SMILES of the six hand-transcribed compounds against
   the transcription, line by line -- the accuracy;
3. the parseability of all 28 Table 1a compounds per route -- the
   headline.

Drawn with Pillow, like the other figures in this repository.

Run:
    .venv\\Scripts\\python.exe experiments/phase5_make_figure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PADDED_MARGIN_PX,
    WHITE,
    english_recogniser_only,
    ink_bands,
    text_span,
)

from wo2022251372_references import CELL_FILES  # noqa: E402

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
RESULTS = (
    REPO
    / "data"
    / "audit"
    / "phase5"
    / "padded_route_verification"
    / "padded_route_results.csv"
)
LINES = (
    REPO
    / "data"
    / "audit"
    / "phase5"
    / "remaining_failures"
    / "line_failures.csv"
)
OUT = REPO / "data" / "audit" / "phase5" / "figures"

PAD_TOP = 14
PAD_BOTTOM = 15

INK = "#1a1a1a"
ACCENT = "#0b6e4f"
WARN = "#b3401f"
MUTED = "#767c85"
RULE = "#d6d9dd"
BAND = "#eef1f0"

W, H = 1560, 810
PAD = 42


def _font(size: int, bold: bool = False):
    try:
        name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _bands(cell: np.ndarray):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append(row[:, x1:x2 + 1])
    return out


def _recognise(engine, band: np.ndarray, margin: int) -> str:
    if margin:
        band = cv2.copyMakeBorder(
            band,
            0,
            0,
            margin,
            margin,
            cv2.BORDER_CONSTANT,
            value=WHITE,
        )
    result = engine(band, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def _fit(image: np.ndarray, width: int, height: int) -> Image.Image:
    """Scale a crop to fit a box, keeping the aspect ratio."""
    ratio = min(width / image.shape[1], height / image.shape[0])
    new = (
        max(1, int(image.shape[1] * ratio)),
        max(1, int(image.shape[0] * ratio)),
    )
    return Image.fromarray(
        cv2.cvtColor(
            cv2.resize(image, new, interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2RGB,
        )
    )


def _panel_mechanism(draw, canvas, engine, x, y, width):
    """The margin on one printed line."""
    title = _font(19, bold=True)
    label = _font(15)
    mono = _font(15)

    draw.text(
        (x, y),
        "The crop ended on the last ink pixel",
        fill=INK,
        font=title,
    )
    draw.text(
        (x, y + 26),
        "so the final glyph of every printed line was read wrong",
        fill=MUTED,
        font=label,
    )

    cell = cv2.imread(str(CELLS / CELL_FILES[2]))
    band = _bands(cell)[1]
    plain = _recognise(engine, band, 0)
    padded = _recognise(engine, band, PADDED_MARGIN_PX)

    box_w = width - 40
    box_h = 92
    top = y + 66

    for index, (margin, reading) in enumerate(
        ((0, plain), (PADDED_MARGIN_PX, padded))
    ):
        strip = band
        if margin:
            strip = cv2.copyMakeBorder(
                band,
                0,
                0,
                margin,
                margin,
                cv2.BORDER_CONSTANT,
                value=WHITE,
            )
        row_y = top + index * (box_h + 60)
        draw.text(
            (x, row_y - 20),
            f"{margin} px margin",
            fill=MUTED,
            font=label,
        )
        draw.rectangle(
            (x, row_y, x + box_w, row_y + box_h),
            fill=BAND,
            outline=RULE,
        )
        fitted = _fit(strip, box_w - 20, box_h - 20)
        canvas_x = x + 10
        canvas_y = row_y + 10
        canvas.paste(fitted, (canvas_x, canvas_y))
        edge_x = canvas_x + fitted.width - 1
        draw.rectangle(
            (edge_x, canvas_y, edge_x + 1, canvas_y + fitted.height),
            fill=WARN,
        )
        draw.text(
            (x, row_y + box_h + 8),
            f"reads: {reading}",
            fill=INK,
            font=mono,
        )


def _panel_lines(draw, x, y, width):
    """The recognised lines against the hand transcription."""
    title = _font(19, bold=True)
    label = _font(15)
    mono = _font(14)

    draw.text(
        (x, y),
        "Accuracy against the hand transcription",
        fill=INK,
        font=title,
    )

    records = list(csv.DictReader(LINES.open(encoding="utf-8")))
    with_ref = [r for r in records if r["has_reference"] == "True"]
    exact = sum(1 for r in with_ref if r["edit_distance"] == "0")
    wrong = [r for r in with_ref if r["edit_distance"] != "0"]

    draw.text(
        (x, y + 30),
        f"{exact} of {len(with_ref)} printed lines are character-exact",
        fill=ACCENT,
        font=label,
    )
    draw.text(
        (x, y + 54),
        f"{len(wrong)} lines differ, each by one lost character",
        fill=MUTED,
        font=label,
    )

    cursor = y + 94
    for row in wrong:
        draw.text(
            (x, cursor),
            f"cmpd {row['compound_id']}, line {row['line_index']}",
            fill=INK,
            font=label,
        )
        draw.text(
            (x, cursor + 22),
            f"printed : {row['reference_line']}",
            fill=MUTED,
            font=mono,
        )
        draw.text(
            (x, cursor + 42),
            f"read    : {row['ocr_line']}",
            fill=WARN,
            font=mono,
        )
        cursor += 78


def _panel_headline(draw, x, y, width):
    """Parseability of all 28 compounds per route."""
    title = _font(19, bold=True)
    label = _font(15)
    big = _font(30, bold=True)

    records = list(csv.DictReader(RESULTS.open(encoding="utf-8")))

    routes = [
        ("production", "production\n(detector)"),
        ("validated", "validated\n(PP-OCRv6)"),
        ("padded", "padded\n(PP-OCRv4 + 6 px)"),
    ]
    raw = []
    fixed = []
    for route, _ in routes:
        subset = [
            r
            for r in records
            if r["patent"] == "new" and r["route"] == route
        ]
        raw.append(sum(1 for r in subset if r["raw_parses"] == "True"))
        fixed.append(
            sum(1 for r in subset if r["after_repair_parses"] == "True")
        )

    draw.text(
        (x, y),
        "WO2022251372 Table 1a: all 28 compounds, RDKit-parseable",
        fill=INK,
        font=title,
    )

    # Legend, top right of the panel.
    legend_y = y + 4
    draw.rectangle(
        (x + width - 330, legend_y + 2, x + width - 314, legend_y + 14),
        fill="#c9d3cf",
    )
    draw.text(
        (x + width - 306, legend_y), "as recognised", fill=MUTED, font=label
    )
    draw.rectangle(
        (x + width - 160, legend_y + 2, x + width - 144, legend_y + 14),
        fill=ACCENT,
    )
    draw.text(
        (x + width - 136, legend_y),
        "after repair",
        fill=ACCENT,
        font=label,
    )

    top = y + 44
    height = 118
    slot = width // 3
    bar_w = 78
    gap = 24
    base = top + height

    for index, ((route, name), r_raw, r_fixed) in enumerate(
        zip(routes, raw, fixed)
    ):
        cx = x + index * slot + 30

        draw.rectangle((cx, base - r_raw, cx + bar_w, base), fill="#c9d3cf")
        draw.rectangle(
            (cx + bar_w + gap, base - r_fixed, cx + 2 * bar_w + gap, base),
            fill=ACCENT,
        )
        draw.text((cx, base - r_raw - 34), str(r_raw), fill=MUTED, font=big)
        draw.text(
            (cx + bar_w + gap, base - r_fixed - 34),
            str(r_fixed),
            fill=ACCENT,
            font=big,
        )
        draw.text((cx, base + 12), name, fill=INK, font=label)

    draw.line((x, base + 4, x + width, base + 4), fill=RULE, width=1)


def main() -> int:
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)
    engine = english_recogniser_only()

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    title = _font(30, bold=True)
    lead = _font(17)

    draw.text(
        (PAD, 24),
        "Reading the printed SMILES of WO2022251372",
        fill=INK,
        font=title,
    )
    draw.text(
        (PAD, 66),
        "The recogniser was not the problem. The crop ended on the last "
        "ink pixel, so a 6 px white margin around each printed line is "
        "what recovers the text.",
        fill=MUTED,
        font=lead,
    )

    column = (W - 3 * PAD) // 2
    _panel_mechanism(draw, canvas, engine, PAD, 120, column)
    _panel_lines(draw, PAD + column + 60, 120, column)
    _panel_headline(draw, PAD, 566, W - 2 * PAD)

    target = OUT / "wo2022251372_recognition.png"
    canvas.save(target)
    print(f"wrote {target} {canvas.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
