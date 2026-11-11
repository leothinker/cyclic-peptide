"""Draw the presentation figure for the WO2022251372 Table 1a delivery.

The deliverable is a CSV, and a CSV is not something a reader can look
at in a slide.  This script draws one figure that puts the printed
patent cell next to the SMILES the pipeline delivered for the same
compound, so the OCR can be judged without running anything.

Four compounds are shown: three that were transcribed by hand and
match the pipeline output character for character, and compound 16,
the single row of the 28 that RDKit cannot parse.  The failure is
shown on purpose -- a reviewer should be able to see that the one bad
row is visible rather than hidden.

Only Pillow and the standard library are used, so the figure can be
regenerated on a machine without matplotlib.

Usage::

    .venv\\Scripts\\python.exe experiments/phase10_patent2_figure.py
"""

from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "data" / "processed" / "WO2022251372" / "compounds_combined.csv"
CELL_DIR = REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
OUT_PATH = REPO / "data" / "audit" / "phase10" / "figure_patent2_delivery.png"

sys.path.insert(0, str(REPO / "experiments"))

from wo2022251372_references import edit_distance, reference_text  # noqa: E402

GRADES = (
    ("mdm2", "MDM2"),
    ("mdm4", "MDM4"),
    ("molm13", "MOLM13"),
    ("mv4_11", "MV4.11"),
    ("waga", "WAGA"),
    ("mkl1", "MKL1"),
)

SHOW = (1, 10, 16, 28)

INK = (22, 32, 44)
MUTED = (103, 114, 127)
OK = (27, 122, 75)
BAD = (176, 36, 24)
RULE = (221, 227, 233)
CARD = (250, 251, 252)
WHITE = (255, 255, 255)

WIDTH = 2200
HEIGHT = 1520
MARGIN = 44
GAP = 34
HEADER = 152
FOOTER = 62
PAD = 22

_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(family: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a cached TrueType font from the Windows font directory."""
    key = (family, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(
            str(Path("C:/Windows/Fonts") / f"{family}.ttf"), size
        )
    return _FONTS[key]


def _sans(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return _font("segoeuib" if bold else "segoeui", size)


def _mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return _font("consolab" if bold else "consola", size)


def _wrap_words(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    """Wrap on spaces, for prose."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _wrap_mono(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int
) -> list[tuple[int, str]]:
    """Wrap on character count, for SMILES, keeping the source offsets."""
    lines: list[tuple[int, str]] = []
    start = 0
    current = ""
    for index, char in enumerate(text):
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > width:
            lines.append((start, current))
            start = index
            current = char
        else:
            current = candidate
    if current:
        lines.append((start, current))
    return lines


def _draw_mono_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
    line_h: int,
    highlights: tuple[tuple[int, int], ...] = (),
) -> int:
    """Draw a monospace block and underline the given character spans."""
    lines = _wrap_mono(draw, text, font, width)
    for index, (_, chunk) in enumerate(lines):
        draw.text((x, y + index * line_h), chunk, font=font, fill=INK)
    for start, end in highlights:
        # A span can straddle a line break, so underline it piece by piece.
        for index, (offset, chunk) in enumerate(lines):
            if end <= offset or start >= offset + len(chunk):
                continue
            shift = max(start, offset) - offset
            stop = min(end, offset + len(chunk)) - offset
            left = x + draw.textlength(chunk[:shift], font=font)
            right = left + draw.textlength(chunk[shift:stop], font=font)
            under = y + index * line_h + line_h - 7
            draw.line([(left, under), (right, under)], fill=BAD, width=3)
    return y + len(lines) * line_h


def _draw_card(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    compound_id: int,
    row: dict[str, str],
    hand: str | None,
    highlights: tuple[tuple[int, int], ...],
    defect: str | None,
) -> None:
    """Draw one compound: printed cell on the left, reading on the right."""
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill=CARD, outline=RULE, width=2)

    inner_x = left + PAD
    inner_w = (right - left) - 2 * PAD
    image_w = 330
    text_x = inner_x + image_w + 26
    text_w = inner_w - image_w - 26

    draw.text(
        (inner_x, top + PAD - 4),
        f"compound {compound_id}",
        font=_sans(26, bold=True),
        fill=INK,
    )

    smiles = row.get("smiles") or ""
    parses = row.get("is_valid") == "True"

    if hand is None:
        badge = "no hand transcription for this row"
        badge_fill = MUTED
    else:
        distance = edit_distance(hand, smiles)
        badge = (
            f"hand transcription: {'exact' if distance == 0 else f'{distance} edits'}"
            f"  (CER {distance / max(len(hand), 1):.2%})"
        )
        badge_fill = OK if distance == 0 else BAD
    draw.text(
        (inner_x + 215, top + PAD + 3),
        badge,
        font=_sans(17),
        fill=badge_fill,
    )

    rule_y = top + PAD + 42
    draw.line([(inner_x, rule_y), (right - PAD, rule_y)], fill=RULE, width=2)

    caption_y = rule_y + 12
    cell = _cell_path(compound_id)
    draw.text(
        (inner_x, caption_y),
        f"printed cell (patent page image, {cell.stem.split('_')[0]})",
        font=_sans(15),
        fill=MUTED,
    )

    crop = Image.open(cell).convert("RGB")
    scale = min(image_w / crop.width, (bottom - PAD - (caption_y + 26 - top)) / crop.height)
    crop = crop.resize(
        (max(int(crop.width * scale), 1), max(int(crop.height * scale), 1)),
        Image.LANCZOS,
    )
    crop_y = caption_y + 26
    image.paste(crop, (inner_x, crop_y))
    draw.rectangle(
        [inner_x, crop_y, inner_x + crop.width - 1, crop_y + crop.height - 1],
        outline=RULE,
        width=2,
    )

    text_y = caption_y + 26
    draw.text(
        (text_x, text_y - 26),
        "SMILES delivered in the CSV",
        font=_sans(15),
        fill=MUTED,
    )
    after = _draw_mono_block(
        draw,
        text_x,
        text_y,
        smiles,
        _mono(19),
        text_w,
        27,
        highlights,
    )

    note_y = after + 12
    if defect:
        note = (
            "The OCR dropped the '[' before C@@H] and the '(' before CCOC.\n"
            "The brackets no longer balance, so RDKit cannot parse this row.\n"
            "It is flagged in the CSV rather than silently repaired."
        )
        for index, line in enumerate(note.splitlines()):
            draw.text(
                (text_x, note_y + index * 22),
                line,
                font=_sans(16),
                fill=BAD,
            )
    else:
        note = (
            "RDKit parses this string and the brackets balance."
            if parses
            else "RDKit cannot parse this string."
        )
        draw.text((text_x, note_y), note, font=_sans(16), fill=OK if parses else BAD)

    grade_rule = bottom - PAD - 74
    draw.line([(text_x, grade_rule), (right - PAD, grade_rule)], fill=RULE, width=2)
    draw.text(
        (text_x, grade_rule + 10),
        "Table 1a activity grades",
        font=_sans(15),
        fill=MUTED,
    )
    grades = "  ".join(f"{label} {row.get(key) or '.'}" for key, label in GRADES)
    draw.text(
        (text_x, grade_rule + 36),
        grades,
        font=_mono(19),
        fill=INK,
    )


def _load_rows() -> dict[str, dict[str, str]]:
    """Return the delivery rows of WO2022251372 keyed by compound id."""
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["cmpd_id"]: row for row in rows}


def _cell_path(compound_id: int) -> Path:
    matches = sorted(CELL_DIR.glob(f"*_cmpd{compound_id}.png"))
    if not matches:
        raise FileNotFoundError(f"no cell crop for compound {compound_id}")
    return matches[0]


# Character spans of the delivered compound 16 string that lost their
# opening bracket.  Found by comparing the printed cell with the
# delivered string while preparing the report; the missing bracket
# belongs immediately before each span.
DEFECT_SPANS = ((61, 66), (116, 126))


def main() -> int:
    rows = _load_rows()
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)

    draw.text(
        (MARGIN, MARGIN),
        "WO2022251372, Table 1a: printed SMILES to CSV",
        font=_sans(40, bold=True),
        fill=INK,
    )
    draw.text(
        (MARGIN, MARGIN + 56),
        "Route padded_vote   |   28 rows, 54 columns   |   27/28 strings parse in RDKit   |   "
        "6/6 hand-transcribed references match character for character",
        font=_sans(20),
        fill=MUTED,
    )
    draw.text(
        (MARGIN, MARGIN + 88),
        "All 168 activity-grade cells filled, none outside A-D   |   "
        "compound 16 is the single failing row and is shown below",
        font=_sans(20),
        fill=MUTED,
    )

    card_w = (WIDTH - 2 * MARGIN - GAP) // 2
    card_h = (HEIGHT - (MARGIN + HEADER) - FOOTER - GAP) // 2
    top0 = MARGIN + HEADER

    for index, compound_id in enumerate(SHOW):
        row = rows[str(compound_id)]
        column = index % 2
        line = index // 2
        left = MARGIN + column * (card_w + GAP)
        top = top0 + line * (card_h + GAP)
        box = (left, top, left + card_w, top + card_h)

        hand = None
        try:
            hand = reference_text(compound_id)
        except KeyError:
            hand = None

        defect = None
        highlights: tuple[tuple[int, int], ...] = ()
        if compound_id == 16:
            defect = "defect"
            highlights = DEFECT_SPANS

        _draw_card(draw, image, box, compound_id, row, hand, highlights, defect)

    footer_y = HEIGHT - 44
    draw.text(
        (MARGIN, footer_y),
        "Left of every panel: the printed patent cell as it appears in the page image.   "
        "Right: the string the pipeline delivered in data/processed/WO2022251372/compounds_combined.csv.",
        font=_sans(16),
        fill=MUTED,
    )
    draw.text(
        (WIDTH - MARGIN, footer_y),
        "Text recognition only.  RDKit-parseable is not structure-verified; "
        "mass gate and 2D-structure work are shelved.",
        font=_sans(16),
        fill=MUTED,
        anchor="ra",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
