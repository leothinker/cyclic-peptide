r"""Manual check sheet for the I100298 printed-SMILES transcriptions.

The strings for 1039-1041 in ``phase3_i100298_sweep.TRANSCRIPTION`` were
transcribed by the agent from the 3x printed-row crops.  They are *not*
human-checked yet.  This script renders the exact spots where a decision
was made, at 6x, so a human can confirm or correct them in one pass.

It renders:

    data/audit/phase3/i100298_read/check_sheet.png
    data/audit/phase3/i100298_read/check_sheet.txt

Nothing is recognised here and nothing is written outside
``data/audit/phase3/``.  The formal CSV and ``src/cpd`` are untouched.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_checklist.py
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

from phase3_i100298_sweep import PRINTED_ROWS  # noqa: E402

ROWS_DIR = REPO / "data" / "audit" / "phase3" / "i100298_read" / "rows"
OUT_DIR = REPO / "data" / "audit" / "phase3" / "i100298_read"

# Each entry: compound, row number, the substring to centre on, what the
# decision is, and the competing readings that have to be told apart.
ITEMS = [
    ("1039", 1, "C(=O)", "terminal C(=O) of row 1",
     "the last 'O' is a letter O, not a zero; 1039 row 1 is "
     "pixel-identical to the human-checked 1038 row 1",
     "O  vs  0"),
    ("1039", 7, "C(=O)N9CCCC9", "tail of 1039",
     "this is where 1039 differs from 1038: a cyclopentyl 'N9CCCC9' "
     "instead of 'N(C)C'",
     "N9CCCC9  vs  N(C)C"),
    ("1040", 1, "NC3(CC3)", "ring size on 1040 row 1",
     "cyclopropyl (two C inside the ring); 1041 row 1 reads "
     "NC3(CCCC3), cyclopentyl",
     "CC3  vs  CCCC3"),
    ("1041", 1, "NC3(CCCC3)", "ring size on 1041 row 1",
     "cyclopentyl (four C inside the ring); 1040 row 1 reads "
     "NC3(CC3), cyclopropyl",
     "CCCC3  vs  CC3"),
    ("1040", 2, "C4CCCC4", "ring size on 1040 row 2",
     "cyclopentyl (four C inside the ring). The earlier phase-2 "
     "transcription of 1040 read C4CCCCC4 (cyclohexyl)",
     "C4CCCC4  vs  C4CCCCC4"),
    ("1041", 2, "C4CCCC4", "ring size on 1041 row 2",
     "cyclopentyl (four C inside the ring); identical motif to 1040 "
     "row 2",
     "C4CCCC4  vs  C4CCCCC4"),
    ("1040", 3, "[C@@H]([C@@H](C)C", "end of 1040 row 3",
     "the C count before the bracket closes; the row continues on the "
     "next printed line",
     "…(C)CC)  vs  …(C)CCC)"),
    ("1041", 3, "[C@@H]([C@@H](C", "end of 1041 row 3",
     "the C count before the bracket closes; 1041 must agree with "
     "1040 here",
     "…(C)CC)  vs  …(C)CCC)"),
]


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _mono(size: int):
    return ImageFont.truetype(f"C:/Windows/Fonts/consola.ttf", size)


def _row_image(cmpd_id: str, row: int) -> np.ndarray:
    path = ROWS_DIR / cmpd_id / f"row{row}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"missing row image {path}")
    return image


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _char_span(image: np.ndarray, text: str, index: int,
               length: int = 1) -> tuple[int, int]:
    """Approximate x range of ``text[index]`` in a 3x printed row image.

    The print is a proportional serif face, so the mapping is estimated
    from the ink extent of the whole row.  The crops below are padded
    generously, so a one-character error in the estimate cannot move the
    target out of the picture.
    """
    ink = image < 160
    cols = np.flatnonzero(ink.sum(axis=0) > 0)
    x0, x1 = int(cols[0]), int(cols[-1])
    count = len(text)
    advance = (x1 - x0) / max(count - 1, 1)
    left = int(x0 + index * advance - advance * 0.35)
    right = int(x0 + (index + length) * advance + advance * 0.35)
    return max(0, left), min(image.shape[1], right)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tiles = []
    lines = []
    for cmpd_id, row, needle, title, detail, contrast in ITEMS:
        printed = PRINTED_ROWS[cmpd_id][row - 1]
        index = printed.find(needle)
        if index < 0:
            raise RuntimeError(f"{needle!r} not found in {cmpd_id} row {row}")

        image = _row_image(cmpd_id, row)
        left, right = _char_span(image, printed, index, len(needle))

        crop = image[:, left:right]
        big = cv2.resize(crop, None, fx=2, fy=2,
                         interpolation=cv2.INTER_CUBIC)
        tiles.append((cmpd_id, row, title, detail, contrast, big,
                      printed[index:index + len(needle) + 10]))
        lines.append(
            f"{cmpd_id} row {row}: {title}\n"
            f"  confirm   : {needle}\n"
            f"  context   : ...{printed[max(0, index - 8):index]}"
            f"[{needle}]"
            f"{printed[index + len(needle):index + len(needle) + 8]}...\n"
            f"  look for  : {contrast}\n"
            f"  why       : {detail}\n")

    pad = 26
    label_w = 520
    max_w = max(t[5].shape[1] for t in tiles)
    tile_h = max(t[5].shape[0] for t in tiles) + 44
    width = pad * 2 + label_w + max_w
    height = 120 + tile_h * len(tiles) + 40

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18),
              "Manual check sheet: printed SMILES on patent page I100298",
              font=_font(24, bold=True), fill=(27, 27, 27))
    draw.text((pad, 52),
              "1039-1041 references are agent transcriptions, not yet "
              "human-checked. These are the exact spots to confirm.",
              font=_font(16), fill=(100, 100, 100))
    draw.text((pad, 76),
              "Crops are 6x the printed size and come from "
              "data/audit/phase3/i100298_read/rows/.",
              font=_font(16), fill=(100, 100, 100))

    y = 118
    for cmpd_id, row, title, detail, contrast, big, context in tiles:
        draw.text((pad, y), f"{cmpd_id} row {row}  -  {title}",
                  font=_font(19, bold=True), fill=(27, 27, 27))
        draw.text((pad, y + 26), f"look for: {contrast}",
                  font=_font(16), fill=(170, 60, 60))
        for line_index, line in enumerate(_wrap(detail, 54)):
            draw.text((pad, y + 48 + line_index * 18), line,
                      font=_font(14), fill=(90, 90, 90))
        canvas.paste(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)),
                     (pad + label_w, y))
        y += tile_h

    canvas.save(OUT_DIR / "check_sheet.png")
    (OUT_DIR / "check_sheet.txt").write_text(
        "I100298 manual check sheet\n"
        "=========================\n\n"
        "1039-1041 references in phase3_i100298_sweep.TRANSCRIPTION are "
        "agent transcriptions\n"
        "from the 3x printed-row crops.  They are NOT human-checked.  "
        "The entries below are\n"
        "the spots where a decision was made and where a competing "
        "reading exists.\n\n"
        + "\n".join(lines)
        + "\nFiles\n"
        "  data/audit/phase3/i100298_read/check_sheet.png\n"
        "  data/audit/phase3/i100298_read/strip_<cmpd>.png  "
        "(all seven rows per compound)\n",
        encoding="utf-8")
    print(f"wrote {OUT_DIR / 'check_sheet.png'}")
    print(f"wrote {OUT_DIR / 'check_sheet.txt'}")


if __name__ == "__main__":
    main()
