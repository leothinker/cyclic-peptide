"""Shared geometry for reading printed lines out of a table cell.

Several readers need the same three things:

* find the rows of pixels that carry printed text (``ink_bands``);
* find the horizontal extent of that text, with the vertical table
  rules removed (``text_span``);
* put the detected boxes back into reading order (``order_boxes``).

They live here rather than inside the SMILES reader so that the
activity-cell reader can use exactly the same geometry.  A printed
value is read the same way whether it is a SMILES line or a retention
time, and the two readers must not drift apart.

``order_boxes`` is the important one.  Sorting boxes by their vertical
midpoint is only correct while every box on a line has the same
height.  A line such as ``[M+H]+`` mixes tall brackets, a tall ``+``
and short letters, and sorting by midpoint then returns them in the
wrong order.  The boxes are therefore grouped into lines by vertical
overlap and sorted left to right inside each line.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- ink-band geometry ---------------------------------------------------
INK_THRESHOLD = 160
MIN_BAND_HEIGHT = 8
MIN_BAND_INK = 5
RULE_HEIGHT_RATIO = 0.80

# --- upscale -------------------------------------------------------------
UPSCALE = 2
INTERPOLATION = cv2.INTER_CUBIC

WHITE = (255, 255, 255)


def ink_bands(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    min_height: int = MIN_BAND_HEIGHT,
    min_ink: int = MIN_BAND_INK,
) -> list[tuple[int, int]]:
    """Vertical bands of printed text, as ``(top, bottom)`` pixel rows.

    The left and right edges of the cell are ignored so that the
    vertical table rules cannot keep a band alive on their own.
    """
    height, width = gray.shape
    left = min(10, max(1, width // 100))
    right = max(left + 1, width - 10)
    ink = (gray[:, left:right] < threshold).astype(np.uint8)
    sums = ink.sum(axis=1)
    active = np.flatnonzero(sums > min_ink)
    if not len(active):
        return []

    groups: list[list[int]] = []
    start = previous = int(active[0])
    for value in active[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append([start, previous])
            start = current
        previous = current
    groups.append([start, previous])
    return [(a, b) for a, b in groups if b - a >= min_height]


def text_span(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    rule_height_ratio: float = RULE_HEIGHT_RATIO,
    rule_height: int | None = None,
) -> tuple[int, int]:
    """Horizontal ink span of a row, with the vertical table rules removed.

    A patent table rule is a column that is dark over (almost) the whole
    cell height.  Such columns are excluded, and so are the one-pixel
    neighbours that belong to the same rule.

    ``rule_height`` is the height the rule test is measured against.
    It defaults to the height of ``gray`` itself, which is right when
    the row *is* the cell.  When a cell is cut into printed bands, the
    test has to be made against the height of the **cell**: measured
    against the height of a band, the opening bracket of ``[M+H]+`` is
    as tall as the text and looks exactly like a rule, so it would be
    cut off.
    """
    dark = gray < threshold
    height = gray.shape[0] if rule_height is None else rule_height
    rule = dark.sum(axis=0) / max(height, 1) >= rule_height_ratio
    for x in np.flatnonzero(rule).tolist():
        for dx in (-1, 1):
            if 0 <= x + dx < gray.shape[1]:
                if dark[:, x + dx].sum() / max(height, 1) >= 0.5:
                    rule[x + dx] = True
    columns = np.flatnonzero((dark.sum(axis=0) > 0) & (~rule))
    if not len(columns):
        return 0, gray.shape[1] - 1
    return int(columns[0]), int(columns[-1])


def strip_vertical_rules(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    coverage: float = RULE_HEIGHT_RATIO,
    height: int | None = None,
) -> np.ndarray:
    """Blank out the columns that are the vertical table rules.

    A rule is a column that is dark over (almost) the whole height of
    the cell.  The test is made against ``height``, which is the height
    of the cell, not of a single printed band: measured against a band,
    a tall glyph such as the opening bracket of ``[M+H]+`` looks like a
    rule and would be removed.

    Removing the rules from the image itself, rather than only from the
    measured span, keeps them out of the pixels that are handed to the
    recogniser.
    """
    if gray.size == 0:
        return gray
    dark = gray < threshold
    reference = gray.shape[0] if height is None else height
    rule = dark.sum(axis=0) / max(reference, 1) >= coverage
    for x in np.flatnonzero(rule).tolist():
        for dx in (-1, 1):
            if 0 <= x + dx < gray.shape[1]:
                if dark[:, x + dx].sum() / max(reference, 1) >= 0.5:
                    rule[x + dx] = True
    out = gray.copy()
    out[:, rule] = 255
    return out


def strip_horizontal_rules(
    gray: np.ndarray,
    *,
    threshold: int = INK_THRESHOLD,
    coverage: float = 0.75,
) -> np.ndarray:
    """Blank out the rows that are a horizontal table rule.

    A rule is dark across nearly the whole width of the cell.  It has
    to go before the ink bands are measured, because a rule would
    otherwise keep the whole cell alive as one band and the text
    would be recognised together with the rule.
    """
    dark = gray < threshold
    width = gray.shape[1]
    if width == 0:
        return gray
    rows = dark.sum(axis=1) / width
    out = gray.copy()
    out[rows >= coverage] = 255
    return out


def upscale(image: np.ndarray, scale: int = UPSCALE) -> np.ndarray:
    """Upscale without binarisation (binarisation was measured to hurt)."""
    return cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=INTERPOLATION
    )


def _box_geometry(box) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` of a detection box."""
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def order_boxes(boxes, txts) -> list[str]:
    """Return the recognised texts in reading order.

    The boxes are grouped into lines by their vertical overlap and
    sorted left to right inside each line.  Sorting by the vertical
    midpoint alone mixes up any line whose boxes have different
    heights, which is exactly what happens to ``[M+H]+``.
    """
    items: list[dict] = []
    box_list = [] if boxes is None else list(boxes)
    text_list = [] if txts is None else list(txts)
    for box, text in zip(box_list, text_list):
        if not text or not str(text).strip():
            continue
        if box is None:
            items.append({
                "text": str(text).strip(), "x1": 0.0, "x2": 0.0,
                "y1": 0.0, "y2": 0.0,
            })
            continue
        x1, y1, x2, y2 = _box_geometry(box)
        items.append({
            "text": str(text).strip(), "x1": x1, "x2": x2,
            "y1": y1, "y2": y2,
        })
    if not items:
        return []

    items.sort(key=lambda it: (it["y1"], it["x1"]))
    lines: list[list[dict]] = []
    for item in items:
        if lines and _vertical_overlap(item, lines[-1]) >= 0.5:
            lines[-1].append(item)
        else:
            lines.append([item])

    ordered: list[str] = []
    for line in lines:
        line.sort(key=lambda it: (it["x1"], it["y1"]))
        ordered.extend(item["text"] for item in line)
    return ordered


def _vertical_overlap(item: dict, line: list[dict]) -> float:
    """How much of the item's height overlaps the line it is compared to."""
    top = min(other["y1"] for other in line)
    bottom = max(other["y2"] for other in line)
    overlap = min(item["y2"], bottom) - max(item["y1"], top)
    height = max(item["y2"] - item["y1"], 1.0)
    return max(overlap, 0.0) / height


__all__ = [
    "INK_THRESHOLD",
    "INTERPOLATION",
    "MIN_BAND_HEIGHT",
    "MIN_BAND_INK",
    "RULE_HEIGHT_RATIO",
    "UPSCALE",
    "WHITE",
    "ink_bands",
    "order_boxes",
    "strip_horizontal_rules",
    "strip_vertical_rules",
    "text_span",
    "upscale",
]
