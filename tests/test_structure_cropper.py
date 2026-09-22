"""Tests for the structure-figure cropper."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cpd.extractors.structure_cropper import StructureCropper, parse_pages

PDF = Path(__file__).resolve().parent.parent / "data" / "raw" / "WO2025162428.pdf"


def test_parse_pages_handles_common_specs() -> None:
    """`--pages` parsing should accept ranges, lists, 'all', and reversed ranges."""
    full = list(range(1, 11))
    assert parse_pages(None, 10) == full
    assert parse_pages("", 10) == full
    assert parse_pages("all", 10) == full
    assert parse_pages("1-3", 10) == [1, 2, 3]
    assert parse_pages("1,3,5", 10) == [1, 3, 5]
    assert parse_pages("1-3,7", 10) == [1, 2, 3, 7]
    assert parse_pages("10-1", 10) == full
    assert parse_pages("0,100", 10) == []           # out-of-range dropped
    assert parse_pages("abc", 10) == []            # unparseable dropped
    assert parse_pages(" 1 - 3 ", 10) == [1, 2, 3]  # whitespace tolerated


@pytest.mark.skipif(not PDF.exists(),
                    reason="WO2025162428.pdf not present in data/raw/")
def test_crop_cover_page_produces_structure_crops(tmp_path: Path) -> None:
    """Page 1 of WO2025162428 shows Markush formulas (A) and (I).

    After cropping we should get at least one non-trivial PNG that is far
    from the page edges (barcode / WIPO logo were filtered).
    """
    pymupdf = pytest.importorskip("pymupdf")
    cropper = StructureCropper(tmp_path, patent_id="WO2025162428")
    crops = list(cropper.crop_pages(PDF, pages=[1]))
    assert crops, "expected at least one structure crop on page 1"

    page = pymupdf.open(PDF)[0]
    pix = page.get_pixmap(dpi=cropper.dpi)
    margin_x = pix.width * StructureCropper.EDGE_MARGIN_FRAC
    margin_y = pix.height * StructureCropper.EDGE_MARGIN_FRAC

    for c in crops:
        assert c.path.exists()
        assert c.path.stat().st_size > 5_000, f"crop too small: {c.path}"
        with Image.open(c.path) as im:
            assert im.width >= 100 and im.height >= 100, (
                f"crop dimensions too small: {c.path} ({im.size})"
            )
        x0, y0, x1, y1 = c.bbox
        # The edge-margin filter should keep crops well clear of the page border.
        assert x0 >= margin_x, f"crop touches left edge: {c.path}"
        assert y0 >= margin_y, f"crop touches top edge: {c.path}"
        assert x1 <= pix.width - margin_x, f"crop touches right edge: {c.path}"
        assert y1 <= pix.height - margin_y, f"crop touches bottom edge: {c.path}"


@pytest.mark.skipif(not PDF.exists(),
                    reason="WO2025162428.pdf not present in data/raw/")
def test_crop_skips_pure_table_page(tmp_path: Path) -> None:
    """PDF page 94 of WO2025162428 is the HPLC-purity table (document page 93).

    It is full-page text in a grid -- no structure drawings. The cropper
    should produce no crops; if any survive, none should be full-page-sized.
    """
    pymupdf = pytest.importorskip("pymupdf")
    cropper = StructureCropper(tmp_path, patent_id="WO2025162428")
    crops = list(cropper.crop_pages(PDF, pages=[94]))

    page = pymupdf.open(PDF)[93]
    pix = page.get_pixmap(dpi=cropper.dpi)
    full_area = pix.width * pix.height

    # Anything that survived must be much smaller than the page.
    for c in crops:
        with Image.open(c.path) as im:
            assert im.width * im.height < full_area * 0.5, (
                f"page-94 crop too large (likely a table that escaped filtering): "
                f"{c.path}"
            )
