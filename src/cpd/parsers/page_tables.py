"""Locate patent tables inside full-page raster documents.

Some patent inputs are already cropped to the table.  Others are full
300-dpi page scans that contain prose, headers, footnotes, and sometimes
more than one table.  The downstream cell extractor only needs a table
image, so this module performs the patent-independent page adaptation:

1. find long horizontal table rules;
2. inspect every band between adjacent rules;
3. keep the vertical rules that span most of that band;
4. group consecutive bands with the same column boundaries.

The result is a crop and the column boundaries measured in that crop.
No chemical or OCR decisions are made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedTable:
    """One table region, its column boundaries and its row bands.

    ``row_boundaries`` holds the y positions of the horizontal rules
    that delimit the table's rows, relative to the top of the table
    crop.  The first and last entries are the outer edges of the
    table, so consecutive pairs are the row bands.  Reusing the rules
    that were already measured for the column geometry is what makes
    the row split reliable on a table that holds a single data row:
    the old crop-level morphology fell back to "one big block"
    whenever it found fewer than two internal lines.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    column_boundaries: tuple[int, ...]
    row_boundaries: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def n_columns(self) -> int:
        return max(0, len(self.column_boundaries) - 1)

    def row_bands(self) -> list[tuple[int, int]]:
        """Return the ``(top, bottom)`` row bands of the table."""
        return list(
            zip(
                self.row_boundaries,
                self.row_boundaries[1:],
                strict=False,
            )
        )


def _cluster(values: Iterable[int], *, gap: int) -> list[int]:
    """Cluster sorted-ish integer positions separated by at most ``gap``."""
    items = sorted(int(value) for value in values)
    if not items:
        return []
    groups: list[list[int]] = [[items[0]]]
    for value in items[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def horizontal_rules(
    image: np.ndarray,
    *,
    min_run_px: int = 700,
    min_line_px: int = 1000,
) -> list[int]:
    """Return y positions of long horizontal rules in page coordinates."""
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 200).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, min_run_px), 1)
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    height, width = lines.shape
    # The long rule must occupy a useful part of the page.  Looking at
    # the central 84% also avoids treating a short underline in the page
    # header as a table rule.
    x1 = int(width * 0.08)
    x2 = int(width * 0.92)
    counts = lines[:, x1:x2].sum(axis=1)
    ys = np.where(counts >= min_line_px)[0]
    return _cluster((int(value) for value in ys), gap=10)


def vertical_boundaries(
    band: np.ndarray,
    *,
    min_height_ratio: float = 0.55,
) -> list[int]:
    """Return vertical rules that span most of one horizontal band."""
    if band is None or band.size == 0:
        return []
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    binary = (gray < 200).astype(np.uint8)
    height = binary.shape[0]
    kernel_height = max(8, int(height * min_height_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_height)
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    counts = lines.sum(axis=0)
    xs = np.where(counts >= int(height * 0.45))[0]
    return _cluster((int(value) for value in xs), gap=8)


def _same_boundaries(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    tolerance: int = 12,
) -> bool:
    return (
        len(left) == len(right)
        and all(
            abs(a - b) <= tolerance
            for a, b in zip(left, right, strict=True)
        )
    )


def locate_page_tables(
    image: np.ndarray,
    *,
    min_bands: int = 2,
) -> list[DetectedTable]:
    """Locate one or more ruled tables in a full-page patent scan.

    A table is emitted only when at least two consecutive horizontal
    bands share the same column boundaries.  That requirement filters
    out isolated underlines and prose while allowing a header plus one
    data row.
    """
    rules = horizontal_rules(image)
    if len(rules) < 2:
        return []

    bands: list[tuple[int, int, tuple[int, ...]]] = []
    for y1, y2 in zip(rules, rules[1:], strict=False):
        if y2 - y1 < 20:
            continue
        boundaries = tuple(vertical_boundaries(image[y1:y2, :]))
        if len(boundaries) >= 2:
            bands.append((y1, y2, boundaries))

    groups: list[list[tuple[int, int, tuple[int, ...]]]] = []
    for band in bands:
        if (
            groups
            and _same_boundaries(groups[-1][-1][2], band[2])
        ):
            groups[-1].append(band)
        else:
            groups.append([band])

    tables: list[DetectedTable] = []
    for group in groups:
        if len(group) < min_bands:
            continue
        boundaries = group[0][2]
        x1 = boundaries[0]
        x2 = boundaries[-1]
        top = group[0][0]
        bottom = group[-1][1]
        tables.append(
            DetectedTable(
                x1=x1,
                y1=top,
                x2=x2,
                y2=bottom,
                column_boundaries=tuple(
                    boundary - x1 for boundary in boundaries
                ),
                # The horizontal rules of the group, relative to the
                # table crop.  ``group`` is a run of consecutive
                # bands, so the rule positions are the band tops plus
                # the bottom edge of the last band.
                row_boundaries=tuple(
                    [band[0] - top for band in group]
                    + [bottom - top]
                ),
            )
        )
    return tables


__all__ = [
    "DetectedTable",
    "horizontal_rules",
    "locate_page_tables",
    "vertical_boundaries",
]
