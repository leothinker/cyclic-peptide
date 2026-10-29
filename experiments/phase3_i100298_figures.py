r"""Report figures for the I100298 printed-SMILES route.

Produces three review-ready images:

    figure_i100298_final.png     the headline: all four compounds
                                  character-exact against their reference
    figure_i100298_geometry.png  why the route works: the table rule
                                  and the padding, before and after
    figure_i100298_1040.png       1040 in detail, the compound whose
                                  cell-crop route used to collapse

Consumes existing experiment outputs only; it runs no OCR and changes
no data.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_figures.py
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase3_i100298_sweep import PRINTED_ROWS, TRANSCRIPTION  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "i100298_final"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
FINAL_CSV = OUT / "i100298_final.csv"

REF_1038 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
    "(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N[C@@H]"
    "(CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)"
    "C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H]"
    "(CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)"
    "C(=O)N1)C(=O)N(C)C"
)

TARGETS = ("1038", "1039", "1040", "1041")


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _mono(size: int):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


def _load_final() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with FINAL_CSV.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["cmpd_id"]] = row
    return out


def _flags(reference: str, observed: str) -> list[str]:
    matcher = difflib.SequenceMatcher(a=reference, b=observed)
    flags = [" "] * len(reference)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for index in range(i1, min(i2, len(flags))):
            flags[index] = "x"
    return flags


def _draw_aligned(draw, x, y, reference, observed, flags, mono, cw, ch):
    for index, _ref_char in enumerate(reference):
        cx = x + index * cw
        got = observed[index] if index < len(observed) else " "
        if flags is not None and index < len(flags) and flags[index] == "x":
            draw.rectangle((cx, y - 3, cx + cw, y + ch), fill=(247, 198, 214))
        elif got != " ":
            draw.rectangle((cx, y - 3, cx + cw, y + ch), fill=(207, 232, 207))
        draw.text((cx, y), got, font=mono, fill=(27, 27, 27))


def _make_final_figure(final: dict[str, dict]) -> None:
    block = 46
    cw, ch = 11, 24
    pad, label_width = 34, 96
    mono = _mono(17)
    sans = _font(19)
    title = _font(25, bold=True)
    small = _font(16)

    longest = max(len(REF_1038), *(len(r["ocr_text"]) for r in final.values()))
    n_blocks = (longest + block - 1) // block
    text_height = n_blocks * ch
    row_height = 34 + (text_height + 8) * 2 + 26
    width = max(1180, pad * 2 + label_width + block * cw + 40)
    height = 116 + row_height * len(TARGETS) + 96

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((pad, 20),
              "Printed SMILES on patent page I100298: "
              "final route reaches character-exact output",
              font=title, fill=(27, 27, 27))
    draw.text((pad, 54),
              "pink = differs from the reference; green = recovered. "
              "1038 was held out of the configuration search.",
              font=small, fill=(100, 100, 100))

    y = 90
    for cmpd_id in TARGETS:
        row = final[cmpd_id]
        reference = (REF_1038 if cmpd_id == "1038"
                     else TRANSCRIPTION[cmpd_id])
        observed = row["ocr_text"]
        flags = _flags(reference, observed)

        row_top = y
        draw.text((pad, y), cmpd_id, font=_font(22, bold=True),
                  fill=(27, 27, 27))
        draw.text((pad + 76, y + 4),
                  f"edit {row['edit_distance']}   CER {row['cer']}   "
                  f"length {row['ocr_length']}   "
                  f"brackets {row['open']}/{row['close']}   "
                  f"RDKit parse: {row['parses']}",
                  font=sans, fill=(45, 110, 60))
        y += 32

        draw.text((pad, y + 2), "REF", font=mono, fill=(110, 110, 110))
        for block_index in range(n_blocks):
            start = block_index * block
            for char_index, char in enumerate(
                    reference[start:start + block]):
                draw.text((pad + label_width + char_index * cw,
                           y + block_index * ch), char, font=mono,
                          fill=(27, 27, 27))
        y += n_blocks * ch + 6

        draw.text((pad, y + 2), "OCR", font=mono, fill=(110, 110, 110))
        for block_index in range(n_blocks):
            start = block_index * block
            chunk_ref = reference[start:start + block]
            chunk_obs = observed[start:start + block].ljust(block)
            chunk_flags = flags[start:start + block]
            _draw_aligned(draw, pad + label_width, y + block_index * ch,
                          chunk_ref, chunk_obs, chunk_flags, mono, cw, ch)
        y += text_height + 8
        # Separator between compounds.
        draw.line((pad, y + 8, width - pad, y + 8), fill=(225, 225, 230))
        y = row_top + row_height

    draw.text((pad, height - 58),
              "Route: trim vertical table rules -> 2x INTER_CUBIC -> "
              "no binarisation -> 14/15 px vertical padding.",
              font=sans, fill=(27, 27, 27))
    draw.text((pad, height - 30),
              "Text recognition evidence only. Chemical structure is not "
              "verified and the mass gate is not consulted.",
              font=small, fill=(100, 100, 100))
    image.save(OUT / "figure_i100298_final.png")


def _make_geometry_figure() -> None:
    """The two crop defects, shown on the same printed row of 1040.

    Left: the old cell crop -- the full-height vertical table rule stays
    inside the OCR input and the ink band has only 4/5 px of vertical
    padding, so neighbouring rows squeeze in.

    Right: the final route -- the rule columns are trimmed away and the
    ink band gets 14/15 px of vertical padding.

    Both panels show the same printed characters at the same zoom, so
    the difference is the crop geometry and nothing else.
    """
    from phase3_i100298_final import _ink_bands

    image = cv2.imread(str(CELLS / "I100298_1040_text.png"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bands = _ink_bands(gray)
    top, bottom = bands[3]

    zoom = 6
    window = 96                       # original pixels across
    rule_x2 = 2                       # columns 0-1 are the table rule

    before = image[max(0, top - 4):bottom + 5, 0:window]
    after = image[max(0, top - 14):bottom + 15,
                  rule_x2:rule_x2 + window]

    before_big = cv2.resize(before, None, fx=zoom, fy=zoom,
                            interpolation=cv2.INTER_CUBIC)
    after_big = cv2.resize(after, None, fx=zoom, fy=zoom,
                            interpolation=cv2.INTER_CUBIC)

    pad = 26
    gap = 40
    label_h = 78
    panel_w = max(before_big.shape[1], after_big.shape[1])
    width = pad * 2 + panel_w * 2 + gap
    height = 128 + label_h + max(before_big.shape[0], after_big.shape[0]) + 150

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18),
              "Why the route changed: crop geometry on 1040 row 4",
              font=_font(24, bold=True), fill=(27, 27, 27))
    draw.text((pad, 52),
              "same printed characters, same zoom; only the crop "
              "geometry differs",
              font=_font(16), fill=(100, 100, 100))

    y = 116
    draw.text((pad, y), "before  (old cell crop)",
              font=_font(20, bold=True), fill=(160, 60, 60))
    draw.text((pad + 250, y + 4), "4/5 px padding, table rule kept",
              font=_font(16), fill=(120, 120, 120))
    draw.text((pad + panel_w + gap, y), "after  (final route)",
              font=_font(20, bold=True), fill=(50, 120, 60))
    draw.text((pad + panel_w + gap + 190, y + 4),
              "14/15 px padding, table rule trimmed",
              font=_font(16), fill=(120, 120, 120))

    img_y = y + label_h
    canvas.paste(Image.fromarray(
        cv2.cvtColor(before_big, cv2.COLOR_BGR2RGB)), (pad, img_y))
    canvas.paste(Image.fromarray(
        cv2.cvtColor(after_big, cv2.COLOR_BGR2RGB)),
        (pad + panel_w + gap, img_y))

    # Mark the vertical table rule in the "before" panel.
    rule_w = rule_x2 * zoom
    draw.rectangle((pad, img_y, pad + rule_w, img_y + before_big.shape[0]),
                   outline=(200, 40, 40), width=3)
    draw.text((pad + rule_w + 10, img_y + before_big.shape[0] // 2 - 12),
              "table rule", font=_font(16, bold=True), fill=(200, 40, 40))

    draw.text((pad, height - 100),
              "The rule is a full-height black column, so it reaches the "
              "recogniser in every row.",
              font=_font(17), fill=(60, 60, 60))
    draw.text((pad, height - 74),
              "Trimming it and padding the ink band is the whole "
              "difference; no per-row tuning is involved.",
              font=_font(17), fill=(60, 60, 60))
    draw.text((pad, height - 46),
              "The wider padding does pick up a few pixels of the "
              "neighbouring rows; the row itself is still read exactly.",
              font=_font(15), fill=(90, 90, 90))
    draw.text((pad, height - 18),
              "Crop-geometry evidence only. Chemical structure is not "
              "verified and the mass gate is not consulted.",
              font=_font(14), fill=(120, 120, 120))

    canvas.save(OUT / "figure_i100298_geometry.png")


def _make_1040_figure(final: dict[str, dict]) -> None:
    """1040 in detail: the printed rows above the recovered string."""
    from phase3_i100298_final import _ink_bands

    image = cv2.imread(str(CELLS / "I100298_1040_text.png"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bands = _ink_bands(gray)

    pad = 26
    label_w = 82
    scale = 1.5
    strips = []
    for top, bottom in bands:
        # Tight crop of the printed row: the neighbouring rows stay out
        # of the picture, the padding is only used for recognition.
        crop = image[max(0, top - 3):bottom + 4, :]
        strips.append(cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC))
    strip_w = max(s.shape[1] for s in strips)
    strip_h = sum(s.shape[0] + 8 for s in strips)

    width = label_w + strip_w + pad * 2
    height = 118 + strip_h + 230
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18),
              "1040: printed rows and the recovered string",
              font=_font(24, bold=True), fill=(27, 27, 27))
    draw.text((pad, 52),
              "printed rows at 2x, then the string the final route "
              "recognised from them",
              font=_font(16), fill=(100, 100, 100))

    y = 92
    for index, strip in enumerate(strips, start=1):
        draw.text((pad, y + 4), f"row {index}", font=_font(15),
                  fill=(130, 130, 130))
        canvas.paste(Image.fromarray(
            cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)), (label_w, y))
        y += strip.shape[0] + 8

    row = final["1040"]
    y += 22
    draw.text((pad, y), "recovered", font=_font(19, bold=True),
              fill=(50, 120, 60))
    draw.text((pad + 110, y + 4),
              f"edit {row['edit_distance']}   CER {row['cer']}   "
              f"length {row['ocr_length']}   "
              f"brackets {row['open']}/{row['close']}   "
              f"RDKit parse: {row['parses']}",
              font=_font(17), fill=(45, 110, 60))
    y += 30
    mono = _mono(15)
    text = row["ocr_text"]
    per_line = 96
    for start in range(0, len(text), per_line):
        draw.text((pad, y), text[start:start + per_line], font=mono,
                  fill=(27, 27, 27))
        y += 21

    canvas.save(OUT / "figure_i100298_1040.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = _load_final()
    _make_final_figure(final)
    _make_geometry_figure()
    _make_1040_figure(final)
    for name in ("figure_i100298_final.png",
                 "figure_i100298_geometry.png",
                 "figure_i100298_1040.png"):
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
