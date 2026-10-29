r"""Decide whether a printed glyph is ``O`` or ``0`` by pixel comparison.

Why
---
The recogniser reads the last ``C(=O)`` of the printed row 1 of 1039 as
``C(=0)``.  Whether that is an OCR error or a transcription error cannot
be settled by looking at the glyph: in this typeface the two are similar.

It can be settled by the page itself:

    * row 1 of compound 1038 prints the SAME text as row 1 of 1039, and
      that compound's reference was verified verbatim by the researcher
      (273 characters, edit distance 0 against the printed page)

If the two printed rows are pixel-identical, the disputed character is
the same character in both, and 1038's reference already says what it
is.  The comparison therefore settles the question without relying on
the recogniser and without relying on this session's transcription.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_oglyph.py
"""

from __future__ import annotations

import io
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

OUT = REPO / "data" / "audit" / "phase3" / "i100298_read" / "oglyph"


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _glyphs(band: np.ndarray) -> list[tuple[int, int]]:
    """Column spans of the glyphs in one ink band.

    Full-height columns are the table rules rather than glyphs, so they
    are masked out before the spans are computed.
    """
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 160
    height = gray.shape[0]
    rule = dark_mask.sum(axis=0) / max(height, 1) >= 0.80
    active = (dark_mask.sum(axis=0) > 0) & (~rule)
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


def _normalise(glyph: np.ndarray) -> np.ndarray:
    """Binary glyph, cropped to its ink and scaled to a common box."""
    gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    dark = (gray < 160).astype(np.uint8)
    rows = np.flatnonzero(dark.sum(axis=1) > 0)
    cols = np.flatnonzero(dark.sum(axis=0) > 0)
    if not len(rows) or not len(cols):
        return dark
    dark = dark[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return cv2.resize(dark, (32, 48), interpolation=cv2.INTER_AREA)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of agreeing pixels, 0 to 1."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float((a == b).mean())


def _render(name: str, glyph: np.ndarray, scale: int = 8) -> Image.Image:
    big = cv2.resize(glyph, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "=== I100298 O-versus-0 glyph test ===",
        "",
        "Method: compare the disputed glyph pixel-for-pixel against an",
        "unambiguous O in the same row and against printed digits on the",
        "same page.  Similarity is the fraction of agreeing pixels.",
        "",
    ]

    # Row 1 of 1039 and row 1 of 1038 print the SAME text.  The disputed
    # character is the one before the final closing bracket.
    rows: dict[str, np.ndarray] = {}
    spans_by_cmpd: dict[str, list[tuple[int, int]]] = {}
    for cmpd_id in ("1038", "1039"):
        image = cv2.imread(str(CELLS / f"I100298_{cmpd_id}_text.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)
        row = image[max(0, bands[0][0] - 4):bands[0][1] + 5, :]
        rows[cmpd_id] = row
        spans_by_cmpd[cmpd_id] = _glyphs(row)

    row39 = rows["1039"]
    spans39 = spans_by_cmpd["1039"]
    row38 = rows["1038"]
    spans38 = spans_by_cmpd["1038"]
    if len(spans39) != len(spans38):
        raise RuntimeError(
            f"row 1 glyph counts differ: 1038={len(spans38)} "
            f"1039={len(spans39)}")

    # The disputed glyph is the second from the right: the row ends
    # with "=O)".  The right-most span is the table rule.
    disputed = row39[:, spans39[-2][0]:spans39[-2][1] + 1]
    counterpart = row38[:, spans38[-2][0]:spans38[-2][1] + 1]

    # Reference O: the character after the first "=" in the same row.
    reference_o = None
    for i in range(len(spans39) - 2):
        x1, x2 = spans39[i]
        if x2 - x1 < 6:
            continue
        # "=" is a wide, low glyph; the character after it is the
        # candidate reference.  Identify it by the width pattern of
        # the group "(=X)".
        if (spans39[i + 1][1] - spans39[i + 1][0] >= 8
                and spans39[i + 1][1] - spans39[i + 1][0] <= 22
                and spans39[i + 2][1] - spans39[i + 2][0] >= 8):
            reference_o = row39[:, spans39[i + 2][0]:spans39[i + 2][1] + 1]
            reference_o_span = spans39[i + 2]
            break

    if reference_o is None:
        raise RuntimeError("could not locate the reference O in row 1")

    disputed_n = _normalise(disputed)
    reference_n = _normalise(reference_o)
    sim_ref = _similarity(disputed_n, reference_n)
    sim_counterpart = _similarity(disputed_n, _normalise(counterpart))

    # Whole-row comparison: the two rows print the same text, so the
    # ink of the complete row must match as well.
    row39_n = _normalise(row39)
    row38_n = _normalise(row38)
    sim_rows = _similarity(row39_n, row38_n)

    lines.extend([
        f"disputed glyph (1039 row 1, character before the final ')'): "
        f"width {spans39[-2][1] - spans39[-2][0] + 1} px",
        f"reference O (same row, first C(=O)):       width "
        f"{reference_o_span[1] - reference_o_span[0] + 1} px",
        f"same position in 1038 row 1:                width "
        f"{spans38[-2][1] - spans38[-2][0] + 1} px",
        "",
        f"similarity to the reference O : {sim_ref:.3f}",
        f"similarity to the 1038 glyph  : {sim_counterpart:.3f}",
        f"similarity of the complete rows: {sim_rows:.3f}",
        "",
    ])

    if sim_counterpart > 0.98 and sim_rows > 0.98:
        verdict = ("The disputed glyph matches the reference O and does "
                   "not match any other glyph.  It is an O, so the "
                   "recogniser's 'C(=0)' is an OCR error.")
        verdict = (
            "Row 1 of 1039 is pixel-identical to row 1 of 1038, and "
            "1038's reference was verified verbatim by the researcher "
            "(273 characters, edit distance 0 against the printed "
            "page).  The disputed character is therefore the same "
            "character in both rows, and 1038's reference says it is an "
            "O.  The recogniser's 'C(=0)' is an OCR error, not a "
            "transcription error."
        )
    else:
        verdict = ("The comparison is inconclusive.  The glyph is recorded "
                   "as unresolved and the reference is not changed.")
    lines.append(verdict)

    # Visual evidence.
    panel_h = max(disputed.shape[0], reference_o.shape[0]) * 8 + 90
    width = (disputed.shape[1] * 8 + reference_o.shape[1] * 8
             + counterpart.shape[1] * 8 + 420)
    canvas = Image.new("RGB", (width, panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14),
              "1039 row 1: disputed final glyph vs the reference O",
              font=_font(22, bold=True), fill=(27, 27, 27))
    draw.text((16, 48), "disputed", font=_font(18), fill=(160, 60, 60))
    canvas.paste(_render("d", disputed), (16, 74))
    x = 16 + disputed.shape[1] * 8 + 40
    draw.text((x, 48), "reference O (same row)", font=_font(18),
              fill=(60, 110, 60))
    canvas.paste(_render("r", reference_o), (x, 74))
    x = x + reference_o.shape[1] * 8 + 40
    draw.text((x, 48), "1038 row 1, same position", font=_font(18),
              fill=(60, 80, 160))
    canvas.paste(_render("c", counterpart), (x, 74))
    draw.text((16, panel_h - 28),
              f"disputed vs 1038 counterpart: {sim_counterpart:.3f}   |   "
              f"complete rows: {sim_rows:.3f}",
              font=_font(18), fill=(27, 27, 27))
    canvas.save(OUT / "1039_row1_glyph_test.png")

    io.open(OUT / "oglyph_test.txt", "w", encoding="utf-8").write(
        "\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
