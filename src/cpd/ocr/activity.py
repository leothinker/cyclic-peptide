"""Reading the activity cells of a patent characterisation table.

An activity cell holds one short printed value: a retention time, an
m/z, a potency, an LC-MS gradient, an ionisation mode.  The reader
used to rely on the detector to decide how the value is cut into
pieces, and that produced two different kinds of corruption.

**A number split into several boxes.**  The English detector cut the
printed ``1.72`` into the boxes ``1`` and ``72``.  The boxes were
joined without a separator, the decimal point was lost, and the
retention time came out as ``172``.  On a dense page the same thing
happened to ``1.48`` (``7`` + ``48`` + ``-``), which no longer parsed
as a number at all and was dropped from the CSV.

**The boxes were put back in the wrong order.**  The boxes were
sorted by their vertical midpoint, which only works while every box
on a line has the same height.  A line such as ``[M+H]+`` mixes tall
brackets with shorter letters, so the midpoint order returned
``H] + [M+] 1``.

Both are fixed here.  The printed line is segmented by our own ink
bands and recognised as one unit, so a value is never split by the
detector, and the detector readings are kept as corroboration.  When
the readings agree, the detector's text is used, which keeps the
existing output format; when they disagree, the line reading wins
and the disagreement is recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from cpd.ocr.lines import (
    INK_THRESHOLD,
    RULE_HEIGHT_RATIO,
    ink_bands,
    order_boxes,
    strip_horizontal_rules,
    strip_vertical_rules,
    text_span,
)

# White margin around the printed line.  The pixels are unchanged; the
# margin only keeps the first and the last glyph off the crop border,
# which is the same trick the ``padded`` SMILES route uses.
MARGIN_PX = 6

# A row of the cell is a table rule when this much of its width is
# ink.  The rule of a patent table is a solid line, so the threshold
# can be generous without touching the text.
RULE_COVERAGE = 0.6

# A value that is a number must parse as one after the thousands
# separators are removed.
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")


def parse_number(text: str | None) -> float | None:
    """Return the number printed in ``text``, or ``None``."""
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise(text: str | None) -> str:
    """The comparison key of a reading: no spaces, no case."""
    if not text:
        return ""
    return re.sub(r"\s+", "", text).upper()


@dataclass(frozen=True)
class ActivityReading:
    """One reading of one activity cell."""

    text: str | None
    source: str
    numeric: float | None = None
    n_boxes: int = 0


@dataclass(frozen=True)
class ActivityResult:
    """The reading that was used, plus the ones that disagreed."""

    reading: ActivityReading
    candidates: tuple[ActivityReading, ...]


class ActivityCellReader:
    """Read the activity cells of a patent characterisation table.

    ``english_engine`` and ``chinese_engine`` are full detector +
    recogniser engines; ``line_engine`` is the same kind of engine but
    is called with ``use_det=False``, because the printed line is
    segmented by :func:`cpd.ocr.lines.ink_bands` instead of by the
    detector.
    """

    def __init__(
        self,
        *,
        english_engine,
        chinese_engine,
        line_engine=None,
        margin_px: int = MARGIN_PX,
    ) -> None:
        self.english_engine = english_engine
        self.chinese_engine = chinese_engine
        self.line_engine = line_engine if line_engine is not None else chinese_engine
        self.margin_px = margin_px

    # ---- the individual readings -------------------------------------

    def read_lines(self, cell: np.ndarray) -> ActivityReading:
        """Read the printed line as one unit (no detector)."""
        if cell is None or cell.size == 0:
            return ActivityReading(None, "line", 0)
        parts = self._read_value_lines(cell)
        if not parts:
            return ActivityReading(None, "line", 0)
        # A cell that carries the printed value on two lines (the
        # gradient ``40-100-`` above ``2min``) is one value, so the
        # parts are joined with a space and the reading counts as a
        # single token.
        text = " ".join(parts)
        return ActivityReading(text, "line", parse_number(text), 1)

    def read_crop(self, cell: np.ndarray) -> ActivityReading:
        """Read the whole value region of the cell as one unit.

        This is the reading that treats the cell as *one* printed
        value: the rules are taken out, the remaining ink is measured,
        and the resulting region is recognised in one go.

        It complements :meth:`read_lines`.  The band reading
        measures the extent of every band separately, which is what
        a cell with two printed lines needs -- but on a cell with a
        single line it can reach past the value into the column
        rules and pick up their residue (``T1 O1 [M+II]+`` instead of
        ``[M+H]+``).  The whole-crop reading has no such blind spot,
        but it merges the two lines of a two-line cell into one
        string (``42-100-`` instead of ``40-100- 2min``).  Both are
        therefore read, and the policy decides.
        """
        if cell is None or cell.size == 0:
            return ActivityReading(None, "crop", 0)
        crop = self._value_crop(cell)
        if crop is None or crop.size == 0:
            return ActivityReading(None, "crop", 0)
        if self.margin_px:
            crop = cv2.copyMakeBorder(
                crop,
                0,
                0,
                self.margin_px,
                self.margin_px,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        text = self._recognise_only(crop)
        if not text:
            return ActivityReading(None, "crop", 0)
        return ActivityReading(text, "crop", parse_number(text), 1)

    @staticmethod
    def _value_crop(cell: np.ndarray) -> np.ndarray | None:
        """The crop of the one printed value of an activity cell.

        An activity cell holds a single value, so the crop is simply
        everything that is ink in the cell, with the table rules taken
        out.  Two details matter.

        *The horizontal rules* run across the whole cell and are
        removed before the extent is measured, otherwise they widen
        the crop to the full column.

        *The vertical rules* are the columns that are dark over the
        whole height of the cell.  The test has to be made against the
        height of the **cell** and not against the height of the text:
        the opening bracket of ``[M+H]+`` is as tall as the text, so
        measured against the text it looks exactly like a rule and is
        cut off, while measured against the cell it is clearly not
        one.
        """
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        gray = strip_horizontal_rules(gray, coverage=RULE_COVERAGE)
        dark = gray < INK_THRESHOLD
        height = gray.shape[0]
        if height == 0:
            return None

        column_ink = dark.sum(axis=0)
        rule = column_ink >= RULE_HEIGHT_RATIO * height
        for x in np.flatnonzero(rule).tolist():
            for dx in (-1, 1):
                if 0 <= x + dx < gray.shape[1]:
                    if column_ink[x + dx] >= 0.5 * height:
                        rule[x + dx] = True
        content = dark & ~rule[None, :]

        rows = np.flatnonzero(content.sum(axis=1) > 0)
        if not len(rows):
            return None
        top, bottom = int(rows[0]), int(rows[-1])
        columns = np.flatnonzero(content[top:bottom + 1].sum(axis=0) > 0)
        if not len(columns):
            return None
        x1, x2 = int(columns[0]), int(columns[-1])
        crop = cell[top:bottom + 1, x1:x2 + 1]
        if crop.size == 0:
            return None
        return crop

    def _read_value_lines(self, cell: np.ndarray) -> list[str]:
        """Recognise every printed line of the cell.

        A cell does not always hold one line of text.  The
        ``lcms_method`` column prints the gradient on two lines --
        ``40-100-`` above ``2min`` -- and reading the two lines as one
        image makes the recogniser return ``42-100-``: the second line
        is in the middle of the picture and the first line is cut
        short.  The lines are therefore found with the same ink-band
        segmentation the SMILES reader uses and recognised one by one.
        """
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        gray = strip_horizontal_rules(gray, coverage=RULE_COVERAGE)
        # The vertical rules are removed from the image, measured
        # against the height of the whole cell: measured against a
        # single band, the opening bracket of ``[M+H]+`` is as tall as
        # the text and would be taken for a rule.
        gray = strip_vertical_rules(gray, height=gray.shape[0])
        bands = ink_bands(gray)
        if not bands:
            return []

        parts: list[str] = []
        for top, bottom in bands:
            band = gray[top:bottom + 1, :]
            x1, x2 = text_span(band, rule_height=gray.shape[0])
            variant = band[:, x1:x2 + 1]
            if variant.size == 0:
                continue
            if self.margin_px:
                variant = cv2.copyMakeBorder(
                    variant,
                    0,
                    0,
                    self.margin_px,
                    self.margin_px,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            text = self._recognise_only(variant)
            if text:
                parts.append(text)
        return parts

    def read_detected(
        self, cell: np.ndarray, engine, label: str, upscale: int = 2
    ) -> ActivityReading:
        """Read a cell with a detector + recogniser engine."""
        if cell is None or cell.size == 0:
            return ActivityReading(None, label, 0)
        image = cell
        if upscale and upscale != 1:
            height, width = image.shape[:2]
            image = cv2.resize(
                image,
                (width * upscale, height * upscale),
                interpolation=cv2.INTER_CUBIC,
            )
        result = engine(image)
        if not result:
            return ActivityReading(None, label, 0)
        txts = getattr(result, "txts", None)
        if txts is None:
            txts = [item[1] for item in result]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            try:
                boxes = [item[0] for item in result]
            except (TypeError, IndexError):
                boxes = None
        if boxes is not None and len(boxes) == len(txts):
            parts = order_boxes(boxes, txts)
        else:
            parts = [str(t).strip() for t in txts if t and str(t).strip()]
        if not parts:
            return ActivityReading(None, label, 0)
        text = " ".join(parts)
        return ActivityReading(
            text, label, parse_number(text), len(parts)
        )

    # ---- the decision ------------------------------------------------

    def read(
        self, cell: np.ndarray, *, numeric: bool = False
    ) -> ActivityResult:
        """Read one activity cell and choose between the readings."""
        candidates = [
            self.read_crop(cell),
            self.read_lines(cell),
            self.read_detected(cell, self.chinese_engine, "ch"),
            self.read_detected(cell, self.english_engine, "en"),
        ]
        candidates = [c for c in candidates if c.text]
        if not candidates:
            return ActivityResult(ActivityReading(None, "none"), ())

        if numeric:
            chosen = self._choose_number(candidates)
        else:
            chosen = self._choose_text(candidates)
        return ActivityResult(chosen, tuple(candidates))

    def _choose_number(
        self, candidates: list[ActivityReading]
    ) -> ActivityReading:
        """Choose between the readings of a numeric cell.

        A printed measurement is one token.  A reading that needed
        several detector boxes has already lost information, so the
        number it yields is only used when the other readings agree
        with it.
        """
        return self._choose(candidates, numeric=True)

    def _choose_text(
        self, candidates: list[ActivityReading]
    ) -> ActivityReading:
        """Choose between the readings of a text cell."""
        return self._choose(candidates, numeric=False)

    def _choose(
        self, candidates: list[ActivityReading], *, numeric: bool
    ) -> ActivityReading:
        """Pick the reading that a printed value is most likely to be.

        The rule has two parts.

        *A reading that needed several detector boxes has already lost
        information.*  The detector split ``1.72`` into ``1`` and
        ``72``, so the value it yields (``172``) is not a different
        way of writing ``1.72`` -- it is a different number.  Such a
        reading therefore cannot outvote a reading that was
        recognised as one token.

        *Readings that agree keep the detector's text.*  When the
        readings only differ in spacing -- the detector reads
        ``10-80- 2min`` and the line reading reads ``10-80-2min`` --
        they describe the same value, and the column keeps the format
        it already had.
        """
        groups: dict[object, list[ActivityReading]] = {}
        for candidate in candidates:
            key = self._key(candidate, numeric)
            groups.setdefault(key, []).append(candidate)

        # Only the groups that hold a reading of the whole token may
        # win.  If no reading is a single token, every group is
        # allowed and the best of them is taken.
        eligible = [
            group for group in groups.values()
            if any(candidate.n_boxes <= 1 for candidate in group)
        ]
        if not eligible:
            eligible = list(groups.values())

        best = max(
            eligible,
            key=lambda group: (
                len(group),
                -min(candidate.n_boxes for candidate in group),
            ),
        )
        if len(best) > 1:
            # The readings agree; keep the detector's text so the
            # existing formatting of the column does not change.
            for source in ("ch", "en"):
                for candidate in best:
                    if candidate.source == source:
                        return candidate
        return self._fewest_boxes(best)

    @staticmethod
    def _key(candidate: ActivityReading, numeric: bool) -> object:
        """The value a reading stands for."""
        if numeric:
            if candidate.numeric is not None:
                return ("number", candidate.numeric)
            return ("text", _normalise(candidate.text))
        return ("text", _normalise(candidate.text))

    @staticmethod
    def _fewest_boxes(
        candidates: list[ActivityReading]
    ) -> ActivityReading:
        """The reading that needed the fewest detector boxes."""
        if not candidates:
            return ActivityReading(None, "none")
        return min(
            candidates,
            key=lambda c: (
                c.n_boxes,
                {"crop": 0, "line": 1, "ch": 2, "en": 3}.get(c.source, 4),
            ),
        )

    # ---- helpers ------------------------------------------------------

    def _recognise_only(self, crop: np.ndarray) -> str:
        result = self.line_engine(crop, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None)
        if not txts:
            return ""
        return (txts[0] or "").strip()


__all__ = [
    "MARGIN_PX",
    "ActivityCellReader",
    "ActivityReading",
    "ActivityResult",
    "parse_number",
]
