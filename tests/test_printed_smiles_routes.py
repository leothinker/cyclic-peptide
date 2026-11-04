"""Tests for the printed-SMILES OCR routes.

The routes differ in the recogniser and in the crop geometry:

``production``
    Detector + English PP-OCRv4 on a trimmed, 2x upscaled crop.
``validated``
    Ink-band segmentation + the upstream PP-OCRv6 recogniser.
``padded``
    Ink-band segmentation + the English PP-OCRv4 recogniser and a
    6 px white margin around every printed line.
``validated_vote`` / ``padded_vote``
    The geometry of ``validated`` / ``padded``, but every printed
    line is read by a second recogniser as well and the two readings
    are combined by :func:`cpd.ocr.vote.vote_lines`.

The margin was measured on patent WO2022251372 and must not be
applied to WO2025162428, whose tightly packed rows make the margin
reach into the neighbouring lines.  These tests pin the route table
and the margin constant so the two cannot be confused later on.
"""

import numpy as np
import pytest

from cpd.ocr import (
    ROUTE_PADDED,
    ROUTE_PADDED_VOTE,
    ROUTE_PRODUCTION,
    ROUTE_VALIDATED,
    ROUTE_VALIDATED_VOTE,
    ROUTES,
    PrintedSmilesOCR,
)
from cpd.ocr.printed_smiles import (
    INK_BAND_ROUTES,
    PADDED_MARGIN_PX,
    PADDED_SCALE,
    VOTE_ROUTES,
)


def test_route_table_lists_every_route():
    assert ROUTES == (
        ROUTE_PRODUCTION,
        ROUTE_VALIDATED,
        ROUTE_PADDED,
        ROUTE_VALIDATED_VOTE,
        ROUTE_PADDED_VOTE,
    )


def test_the_vote_routes_are_ink_band_routes():
    # A vote route changes the *reading*, not the geometry: it reads
    # the same ink bands as the route it was derived from.
    assert VOTE_ROUTES == (ROUTE_VALIDATED_VOTE, ROUTE_PADDED_VOTE)
    for route in VOTE_ROUTES:
        assert route in INK_BAND_ROUTES


def test_padded_route_is_an_ink_band_route():
    # ``padded`` shares the crop geometry with ``validated``; only the
    # recogniser and the margin differ.
    assert ROUTE_PADDED in INK_BAND_ROUTES
    assert ROUTE_VALIDATED in INK_BAND_ROUTES
    assert ROUTE_PRODUCTION not in INK_BAND_ROUTES


def test_padded_margin_is_six_pixels():
    # Measured optimum on WO2022251372: 0 px -> 31 line edits,
    # 4 px -> 14, 6 px -> 5, 8 px -> 9.  6 px is the candidate value.
    assert PADDED_MARGIN_PX == 6


def test_padded_route_does_not_upscale():
    # Measured on WO2022251372 (28 compounds): 1x gives 21/28 bracket
    # balanced and 17/28 RDKit-parseable, 2x gives 20/28 and 15/28.
    # The character accuracy against the hand transcriptions is the
    # same either way, so the difference is in bracket balance.
    assert PADDED_SCALE == 1


@pytest.mark.parametrize("route", ROUTES)
def test_each_route_can_be_constructed(route):
    ocr = PrintedSmilesOCR(route=route)
    assert ocr.route == route


def test_unknown_route_is_rejected():
    with pytest.raises(ValueError):
        PrintedSmilesOCR(route="not-a-route")


def _synthetic_cell() -> np.ndarray:
    """A white cell with one short bar of ink on a single line."""
    cell = np.full((40, 60, 3), 255, dtype=np.uint8)
    cell[14:26, 20:40] = 0
    return cell


def _capture(route: str, monkeypatch, *, upscale: bool = True) -> np.ndarray:
    """Run one route on the synthetic cell and return what it recognised."""
    captured = {}

    def fake_recognise(image, engine=None):
        captured["image"] = image
        return "X"

    if not upscale:
        # Capture the crop before the 2x resize, so the pixel values
        # are still the ones that came out of the cell image.
        monkeypatch.setattr(
            "cpd.ocr.printed_smiles.upscale", lambda image, scale=2: image
        )

    ocr = PrintedSmilesOCR(route=route)
    ocr._recognise_only = fake_recognise
    assert ocr.read_cell(_synthetic_cell()) == "X"
    return captured["image"]


def test_padded_margin_adds_white_border_without_touching_pixels(monkeypatch):
    # The margin is padding only: the original glyph pixels must be
    # unchanged and merely surrounded by paper white.  The check runs
    # before the 2x upscale, where the pixels are still original.
    plain = _capture(ROUTE_VALIDATED, monkeypatch, upscale=False)
    padded = _capture(ROUTE_PADDED, monkeypatch, upscale=False)

    # Same height: the margin is horizontal only.
    assert padded.shape[0] == plain.shape[0]
    # 6 px of margin on each side.
    assert padded.shape[1] == plain.shape[1] + 2 * PADDED_MARGIN_PX
    # The glyph pixels are byte-for-byte the ones the validated route
    # sees; the margin only moved them sideways.
    assert np.array_equal(
        padded[:, PADDED_MARGIN_PX : PADDED_MARGIN_PX + plain.shape[1]],
        plain,
    )
    # The new border is paper white.
    assert padded[:, : PADDED_MARGIN_PX].min() == 255
    assert padded[:, -PADDED_MARGIN_PX :].min() == 255


def test_validated_route_adds_no_margin(monkeypatch):
    # The ``validated`` route must stay exactly as it was: the same
    # crop geometry, no border.  This is the regression guard for
    # the old patent.
    plain = _capture(ROUTE_VALIDATED, monkeypatch)

    # The ink span of the synthetic cell is 20 columns wide; the
    # validated route must not widen it.
    assert plain.shape[1] == 20 * 2
