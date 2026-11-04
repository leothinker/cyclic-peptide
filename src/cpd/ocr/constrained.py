"""Read a cell whose value must come from a small closed set.

WO2022251372 reports every activity value as a letter grade
(``A``/``B``/``C``/``D``) instead of a number.  A grade is a single
isolated capital letter in an otherwise empty cell, and the text
*detector* used by the generic cell reader has almost nothing to fire
on: measured on the real pages it returned nothing for most grade
cells, and junk such as ``L`` or ``y`` for the rest.

This module therefore does not rely on the detector to find the text.
It locates the glyph by ink analysis -- masking the horizontal and
vertical table rules first, then taking the widest contiguous run of
ink columns -- and recognises that isolated crop directly.

The answer is only accepted when it is a member of the allowed set, so
a misread glyph becomes "no value" instead of a wrong value.  An
unrecognised grade is not an error; the patent simply leaves that
cell blank.

The *recogniser* is shared with the rest of the pipeline; only the
accepted-value set is patent-specific, and it comes from the profile.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# Ink thresholds used to isolate a glyph inside a ruled table cell.
INK_THRESHOLD = 160
RULE_COVERAGE = 0.8
MIN_INK_PIXELS = 3
MAX_RUN_GAP = 3
PAD_PX = 12


def isolate_glyph(cell: np.ndarray) -> np.ndarray | None:
    """Return the tight crop of the single glyph in ``cell``.

    ``None`` is returned when the cell holds no ink at all.  The
    horizontal and vertical table rules are removed first: a rule that
    is not removed makes every column of the cell look inked, and the
    crop then contains the whole cell instead of the letter.
    """
    if cell is None or cell.size == 0:
        return None
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    dark = gray < INK_THRESHOLD
    height, width = dark.shape

    # Mask the horizontal rules: a rule row is dark across nearly the
    # whole cell width.
    work = dark.copy()
    work[dark.sum(axis=1) >= RULE_COVERAGE * width, :] = False

    # Then the vertical rules, measured on what is left.
    counts = work.sum(axis=0)
    ink = (counts >= MIN_INK_PIXELS) & (counts < RULE_COVERAGE * height)
    keep = np.flatnonzero(ink)
    if not keep.size:
        return None

    # The glyph is the widest contiguous run of ink columns; the
    # leftovers are speckle and rule fringes.
    runs: list[list[int]] = [[int(keep[0])]]
    for value in keep[1:]:
        if int(value) - runs[-1][-1] <= MAX_RUN_GAP:
            runs[-1].append(int(value))
        else:
            runs.append([int(value)])
    best = max(runs, key=lambda run: run[-1] - run[0])

    rows = np.flatnonzero(work[:, best[0]:best[-1] + 1].sum(axis=1) >= 2)
    if not rows.size:
        return None
    return cell[int(rows[0]):int(rows[-1]) + 1, best[0]:best[-1] + 1]


class ConstrainedCellOCR:
    """Recognise a cell whose value belongs to a closed set."""

    def __init__(
        self,
        allowed: Sequence[str],
        *,
        detector_engine=None,
        english_engine=None,
    ) -> None:
        self.allowed = {value.strip().upper() for value in allowed}
        self._detector_engine = detector_engine
        self._english_engine = english_engine

    def _recognise_candidates(self, padded: np.ndarray) -> list[str]:
        """Return candidate strings, best strategy first."""
        candidates: list[str] = []

        engine = self._detector_engine
        if engine is not None:
            result = engine(padded)
            for text in getattr(result, "txts", None) or []:
                if text:
                    candidates.append(text)

        if engine is not None:
            result = engine(padded, use_det=False, use_rec=True)
            for text in getattr(result, "txts", None) or []:
                if text:
                    candidates.append(text)

        if self._english_engine is not None:
            result = self._english_engine(padded, use_det=False, use_rec=True)
            for text in getattr(result, "txts", None) or []:
                if text:
                    candidates.append(text)
        return candidates

    def read(self, cell: np.ndarray) -> str | None:
        """Return the accepted value, or ``None`` if none qualifies."""
        glyph = isolate_glyph(cell)
        if glyph is None or glyph.size == 0:
            return None
        padded = cv2.copyMakeBorder(
            glyph, PAD_PX, PAD_PX, PAD_PX, PAD_PX,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        for text in self._recognise_candidates(padded):
            value = text.strip().upper()
            if value in self.allowed:
                return value
        return None


__all__ = ["ConstrainedCellOCR", "isolate_glyph"]
