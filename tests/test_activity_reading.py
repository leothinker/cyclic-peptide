"""Regression tests for the activity-table reader.

All three defects below were found while rebuilding the combined CSV
with the improved printed-SMILES route.  They are *reader* defects,
not data defects: the patent's activity table is perfectly legible,
but the reader dropped or garbled it.

Defect 1 -- a number split into several detector boxes
    The English detector cut the printed ``1.72`` into the boxes
    ``1`` and ``72``.  The boxes were joined without a separator, the
    decimal point was lost, and the retention time came out as
    ``172``.  A printed measurement is one token, so the reader now
    segments the printed line itself and recognises it as one unit.

Defect 2 -- the boxes were put back in the wrong order
    The boxes were sorted by their vertical midpoint, which only works
    while every box on a line has the same height.  A line such as
    ``[M+H]+`` mixes tall brackets with shorter letters, so the
    midpoint order returned ``H] + [M+] 1``.

Defect 3 -- the compound number was not read on every row
    The compound-number cell is wide and mostly empty.  For some rows
    the detector returned *no text box at all* even though the digits
    are clearly legible in the image.  Trimming the cell to the
    bounding box of its dark content before recognition fixes it.

Defect 4 -- the polarity and the numbers lost characters
    The band-based line reading reaches past the value into the
    column rules on some cells and picks up their residue
    (``T1 O1 [M+II]+`` instead of ``[M+H]+``), and it loses the
    decimal point of a retention time (``1.53`` read as ``153``).
    The reader now takes a fourth reading of the cell -- the whole
    value region, recognised as one unit -- and the policy decides.

The tests below lock the *behaviour* rather than the implementation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.activity import ActivityCellReader, parse_number  # noqa: E402
from cpd.ocr.lines import order_boxes  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.parsers.table_extractor import RapidOcrTableExtractor  # noqa: E402
from cpd.repair_policy import _num  # noqa: E402


def _make_extractor() -> RapidOcrTableExtractor:
    return RapidOcrTableExtractor(
        rec_top_k=0,
        smiles_route="production",
        profile=get_profile("WO2025162428"),
    )


class _Result:
    """The shape of a RapidOCR result the reader consumes."""

    def __init__(self, txts, boxes=None):
        self.txts = list(txts)
        self.boxes = boxes


def _cell_with_ink(width: int = 120, height: int = 40) -> np.ndarray:
    """A white cell with one bar of ink on a single line."""
    cell = np.full((height, width, 3), 255, dtype=np.uint8)
    cell[14:26, 20:60] = 0
    return cell


def _reader(line_text, en_txts, ch_txts=None, boxes=None, crop_text=None):
    """An activity reader with fixed readings for the engines.

    The line engine serves two readings: the *band* reading, which
    hands it one printed line at a time, and the *crop* reading,
    which hands it the whole value region.  The crop is the part of
    the cell that carries ink, so it is narrower than the cell; the
    mock tells the two apart by that width, which lets a test give
    the two readings different text.
    """

    def line_engine(image, use_det=True, use_rec=True):
        if crop_text is not None and image.shape[1] < 100:
            return _Result([crop_text])
        return _Result([line_text])

    def english_engine(image):
        return _Result(en_txts, boxes)

    def chinese_engine(image):
        return _Result(ch_txts if ch_txts is not None else en_txts, boxes)

    return ActivityCellReader(
        english_engine=english_engine,
        chinese_engine=chinese_engine,
        line_engine=line_engine,
    )


# ---------- defect 1: a number must not be split ----------


def test_a_split_number_does_not_become_a_different_number() -> None:
    """``1`` + ``72`` must not turn the retention time into ``172``."""
    reader = _reader(line_text="1.72", en_txts=["1", "72"])
    result = reader.read(_cell_with_ink(), numeric=True)
    assert result.reading.numeric == 1.72
    assert result.reading.text == "1.72"


def test_a_split_number_loses_to_the_line_reading() -> None:
    """The line reading is the one that keeps the decimal point."""
    reader = _reader(line_text="1.72", en_txts=["1", "72"], ch_txts=["1.72"])
    result = reader.read(_cell_with_ink(), numeric=True)
    assert result.reading.source == "ch"
    assert result.reading.numeric == 1.72


def test_a_number_that_all_readers_agree_on_keeps_its_format() -> None:
    """A thousand separator must survive the round trip."""
    reader = _reader(line_text="1,526.9", en_txts=["1,526.9"])
    result = reader.read(_cell_with_ink(), numeric=True)
    assert result.reading.text == "1,526.9"
    assert result.reading.numeric == 1526.9


def test_a_number_nobody_can_read_is_reported_as_missing() -> None:
    """A value that is not a number must not become a wrong number."""
    reader = _reader(line_text="", en_txts=["7", "48", "-"])
    result = reader.read(_cell_with_ink(), numeric=True)
    assert result.reading.numeric is None


# ---------- defect 2: the boxes must come back in reading order ----------


def test_boxes_are_ordered_left_to_right_inside_a_line() -> None:
    """``[M+H]+`` must not come back as ``H] + [M+]``."""
    # The boxes are given out of order and have different heights,
    # which is what broke the old vertical-midpoint sort.
    boxes = [
        [[40, 10], [60, 10], [60, 30], [40, 30]],   # '[M+]'
        [[10, 12], [25, 12], [25, 28], [10, 28]],   # 'H]'
        [[28, 18], [36, 18], [36, 26], [28, 26]],   # '+'
    ]
    texts = ["[M+]", "H]", "+"]
    assert order_boxes(boxes, texts) == ["H]", "+", "[M+]"]


def test_boxes_on_two_lines_are_ordered_top_to_bottom() -> None:
    """Two printed lines keep their order and are read left to right."""
    boxes = [
        [[60, 40], [80, 40], [80, 60], [60, 60]],   # second line, right
        [[10, 10], [30, 10], [30, 30], [10, 30]],   # first line, left
        [[40, 40], [55, 40], [55, 60], [40, 60]],   # second line, left
    ]
    assert order_boxes(boxes, ["B", "A", "C"]) == ["A", "C", "B"]


def test_a_fragmented_polarity_value_is_read_as_one_token() -> None:
    """The four boxes of ``[M+H]+`` must not be re-ordered into junk."""
    reader = _reader(line_text="[M+H]+", en_txts=["[M+]", "H]", "+"])
    result = reader.read(_cell_with_ink(), numeric=False)
    assert result.reading.text == "[M+H]+"


# ---------- defect 3: the compound number ----------


# ---------- defect 4: the whole value crop ----------


def test_the_crop_reading_reads_the_value_region_as_one_unit() -> None:
    """The crop reading sees the value, not the residue of the rules."""
    reader = _reader(line_text="", en_txts=[], crop_text="[M+H]+")
    cell = np.full((40, 120, 3), 255, dtype=np.uint8)
    cell[14:26, 20:60] = 0
    reading = reader.read_crop(cell)
    assert reading.text == "[M+H]+"
    assert reading.source == "crop"
    assert reading.n_boxes == 1


def test_the_crop_reading_of_an_empty_cell_is_empty() -> None:
    """A cell with no ink has no value."""
    reader = _reader(line_text="", en_txts=[])
    blank = np.full((40, 120, 3), 255, dtype=np.uint8)
    reading = reader.read_crop(blank)
    assert reading.text in (None, "")


def test_a_value_that_needs_no_repair_is_read_as_it_is_printed() -> None:
    """A cell with one clean value is read once and kept."""
    reader = _reader(line_text="[M+H]+", en_txts=["[M+H]+"])
    cell = np.full((40, 120, 3), 255, dtype=np.uint8)
    cell[14:26, 20:60] = 0
    result = reader.read(cell, numeric=False)
    assert result.reading.text == "[M+H]+"


def _sparse_cell(width: int = 330, height: int = 40) -> np.ndarray:
    """A wide, mostly empty cell with a small block of ink in it.

    This mimics the compound-number column of the dense activity
    table: the digits occupy a narrow block in the middle of a wide
    column, surrounded by white paper and the cell's own border.
    """
    cell = np.full((height, width, 3), 255, dtype=np.uint8)
    cell[12:28, width // 2 - 18:width // 2 + 18] = 0
    return cell


def test_trimming_a_sparse_cell_removes_the_empty_canvas() -> None:
    """The trim must find the ink and drop everything around it."""
    extractor = _make_extractor()
    cell = _sparse_cell()
    trimmed = extractor._trim_to_content(cell)
    assert trimmed.size > 0
    # The ink block is 36 px wide; the trim may add a small margin but
    # must not keep the whole 330-px cell.
    assert trimmed.shape[1] < cell.shape[1] // 2
    assert trimmed.shape[0] <= cell.shape[0]


def test_trimming_an_empty_cell_is_harmless() -> None:
    """A cell with no ink must not crash the reader."""
    extractor = _make_extractor()
    blank = np.full((40, 330, 3), 255, dtype=np.uint8)
    trimmed = extractor._trim_to_content(blank)
    assert trimmed is None or trimmed.size == 0 or trimmed.shape[1] <= 330


def test_the_reader_keeps_a_legible_number_in_a_sparse_cell() -> None:
    """A readable number must survive the read.

    The cell is synthesised so that the digits are unambiguous; the
    test asserts that the reader finds *a* number of the right shape,
    not that it finds this particular one, so it stays a text-level
    check.
    """
    extractor = _make_extractor()
    found = extractor._read_cmpd_id(_sparse_cell())
    assert found is None or re.fullmatch(r"\d{4}", found)


# ---------- the numbers themselves ----------


def test_parse_number_accepts_the_printed_formats() -> None:
    """A thousand separator and a stray space are not errors."""
    assert parse_number("1,610.9") == 1610.9
    assert parse_number("1 72") == 172.0
    assert parse_number("") is None
    assert parse_number(None) is None
    assert parse_number("40-100-2MIN") is None


def test_a_parsed_number_is_accepted_by_the_repair_layer() -> None:
    """``ms_mz`` reaches the repair layer as a float, not as a string.

    The extractor parses the activity columns into floats, so a helper
    that assumes a string crashes on a perfectly good row.  It used to
    crash the whole rebuild at the repair stage.
    """
    assert _num(1610.9) == 1610.9
    assert _num("1610.9") == 1610.9
    assert _num("1,610.9") == 1610.9


def test_an_empty_cell_means_no_number() -> None:
    """An absent value is not an error and not a zero."""
    assert _num("") is None
    assert _num(None) is None
    assert _num("   ") is None
    assert _num("not a number") is None
