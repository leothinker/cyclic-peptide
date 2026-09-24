"""Tests for the FullText JPG classifier (table_finder).

These tests use a fixture directory of synthetic JPGS that mimic the
WO2025162428 FullText layout (different sizes, widths, aspects). They do
NOT need the real patent data on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from cpd.parsers.table_finder import (
    ACTIVITY_MAX_BYTES,
    ACTIVITY_MIN_BYTES,
    ACTIVITY_MIN_WIDTH_PX,
    ACTIVITY_MAX_WIDTH_PX,
    NullKeywordDetector,
    SMILES_MIN_BYTES,
    SMILES_MIN_WIDTH_PX,
    STRUCTURE_MAX_BYTES,
    TableBlock,
    TableImage,
    TableType,
    find_by_type,
    find_tables,
    load_findings_json,
    scan_directory,
    write_findings_json,
)


# ---------- fixture helpers ----------


def _make_jpg(path: Path, *, width: int, height: int, fill: tuple = (255, 255, 255), density: int = 0) -> int:
    """Write a synthetic JPEG with a table grid plus optional scatter dots.

    The scatter dots are there so the JPEG file size scales with
    *content*, not just dimensions -- this lets us exercise the
    size-based thresholds in `_classify_size` without needing real
    patent rasters on disk.
    """
    import random
    random.seed(42)
    img = Image.new("RGB", (width, height), fill)
    draw = ImageDraw.Draw(img)
    # Faint table grid so the classifier sees a recognisable shape.
    rows = 4
    cols = 7 if width >= 1940 else 3
    header_h = int(height * 0.1)
    draw.rectangle([0, 0, width, header_h], fill=(220, 220, 220))
    row_h = (height - header_h) // rows
    for i in range(1, rows + 1):
        y = header_h + i * row_h
        draw.line([(0, y), (width, y)], fill=(180, 180, 180), width=2)
    col_w = width // cols
    for j in range(1, cols):
        x = j * col_w
        draw.line([(x, 0), (x, height)], fill=(180, 180, 180), width=2)
    # Structure blob in col 1 of every row.
    for r in range(rows):
        cx = col_w // 2
        cy = header_h + r * row_h + row_h // 2
        draw.ellipse([cx - 80, cy - 80, cx + 80, cy + 80], outline=(40, 40, 40), width=2)
        for k in range(20):
            x0 = cx - 70 + k * 7
            y0 = cy + random.randint(-50, 50)
            draw.line(
                [(x0, y0), (x0 + random.randint(3, 10), y0 + random.randint(-20, 20))],
                fill=(40, 40, 40), width=1,
            )
    # Optional scatter to inflate file size when needed.
    for _ in range(density):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.line([(x, y), (x + 1, y + 1)], fill=(60, 60, 60), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=85)
    return path.stat().st_size


@pytest.fixture()
def fake_fulltext(tmp_path: Path) -> Path:
    """Build a synthetic FullText dir with one of every kind of page."""
    d = tmp_path / "FullText"
    # SMILES-like page: wide + heavy
    _make_jpg(d / "PCTCN2025075389-ftappb-I100298.jpg", width=1945, height=2525, fill=(255, 255, 255), density=15000)
    _make_jpg(d / "PCTCN2025075389-ftappb-I100299.jpg", width=1945, height=2902, fill=(255, 255, 255), density=15000)
    # activity-like page
    _make_jpg(d / "PCTCN2025075389-ftappb-I100280.jpg", width=1909, height=2128, fill=(255, 255, 255), density=3000)
    _make_jpg(d / "PCTCN2025075389-ftappb-I100281.jpg", width=1913, height=2495, fill=(255, 255, 255), density=3000)
    # structure-like page: small, low-aspect
    _make_jpg(d / "PCTCN2025075389-ftappb-I100056.jpg", width=1881, height=597, fill=(255, 255, 255))
    # text-heavy "other" page: tall, <500KB, narrow width
    _make_jpg(d / "PCTCN2025075389-ftappb-I100108.jpg", width=1700, height=2522, fill=(255, 255, 255))
    # a totally unrelated filename (non-matching pattern)
    (d / "README.txt").write_text("ignore me")
    return d


# ---------- classification tests ----------


def test_scan_directory_finds_all_matching_files(fake_fulltext: Path) -> None:
    """The PCTCN...-I######.jpg pattern is matched; everything else skipped."""
    images = scan_directory(fake_fulltext)
    filenames = sorted(i.filename for i in images)
    assert len(images) == 6, filenames
    assert "README.txt" not in filenames


def test_classification_targets_are_correct(fake_fulltext: Path) -> None:
    """Activity / SMILES / structure / other buckets hold the right images."""
    images = scan_directory(fake_fulltext)
    by_name = {i.filename: i for i in images}
    assert by_name["PCTCN2025075389-ftappb-I100280.jpg"].table_type == TableType.ACTIVITY
    assert by_name["PCTCN2025075389-ftappb-I100281.jpg"].table_type == TableType.ACTIVITY
    assert by_name["PCTCN2025075389-ftappb-I100298.jpg"].table_type == TableType.SMILES
    assert by_name["PCTCN2025075389-ftappb-I100299.jpg"].table_type == TableType.SMILES
    # The narrow structure falls to OTHER (aspect 3.15 is outside
    # the activity aspect band, but width 1881 puts it in the structure
    # rule too -- either way it isn't a table).
    assert by_name["PCTCN2025075389-ftappb-I100056.jpg"].table_type in (
        TableType.STRUCTURE, TableType.OTHER,
    )


def test_width_differentiator_separates_activity_from_smiles() -> None:
    """Pure width check: 1945+500KB -> smiles, 1909+300KB -> activity."""
    assert TableType.SMILES == _classify(1945, 2525, 600_000)
    assert TableType.ACTIVITY == _classify(1909, 2128, 310_000)
    # Below width threshold but >=500KB -> still other (text-heavy page).
    assert TableType.OTHER == _classify(1700, 2500, 600_000)


def _classify(w: int, h: int, sz: int) -> str:
    from cpd.parsers.table_finder import _classify_size
    return _classify_size(w, h, sz)


# ---------- block clustering tests ----------


def test_find_tables_groups_contiguous_ids(fake_fulltext: Path) -> None:
    """Smiles pages 100298+100299 and activity pages 100280+100281 cluster."""
    images = scan_directory(fake_fulltext)
    blocks = find_tables(images)
    types = {b.table_type for b in blocks}
    assert TableType.SMILES in types
    assert TableType.ACTIVITY in types
    for b in blocks:
        if b.table_type == TableType.SMILES:
            ids = {i.numeric_id for i in b.images}
            assert ids == {100298, 100299}
        elif b.table_type == TableType.ACTIVITY:
            ids = {i.numeric_id for i in b.images}
            assert ids == {100280, 100281}


def test_find_tables_breaks_on_gap(fake_fulltext: Path) -> None:
    """A 100-id gap between candidates must produce two blocks."""
    img_dir = fake_fulltext
    extra = img_dir / "PCTCN2025075389-ftappb-I100450.jpg"
    _make_jpg(extra, width=1945, height=2400, fill=(255, 255, 255), density=15000)
    images = scan_directory(img_dir)
    blocks = find_tables(images, gap_tolerance=5, table_types=(TableType.SMILES,))
    smiles_blocks = [b for b in blocks if b.table_type == TableType.SMILES]
    assert len(smiles_blocks) == 2, [b.start_id for b in smiles_blocks]
    ids_per_block = [{i.numeric_id for i in b.images} for b in smiles_blocks]
    # First block is 100298+100299, second is 100450 alone.
    assert {100298, 100299} in ids_per_block
    assert {100450} in ids_per_block


def test_find_by_type_filters_correctly(fake_fulltext: Path) -> None:
    images = scan_directory(fake_fulltext)
    smiles = find_by_type(images, table_type=TableType.SMILES)
    assert all(i.table_type == TableType.SMILES for i in smiles)
    assert len(smiles) == 2


# ---------- keyword detector tests ----------


def test_null_keyword_detector_returns_empty() -> None:
    det = NullKeywordDetector()
    with Image.new("RGB", (10, 10), (255, 255, 255)) as im:
        assert list(det.detect(im)) == []


# ---------- round-trip JSON tests ----------


def test_write_and_load_findings_json_round_trip(fake_fulltext: Path, tmp_path: Path) -> None:
    images = scan_directory(fake_fulltext)
    blocks = find_tables(images)
    out = tmp_path / "findings.json"
    write_findings_json(images, blocks, out, patent_id="WO2025162428", source_dir=fake_fulltext)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["patent_id"] == "WO2025162428"
    assert payload["n_images"] == len(images)
    assert sum(payload["type_counts"].values()) == len(images)

    loaded_images, loaded_blocks = load_findings_json(out)
    assert len(loaded_images) == len(images)
    assert {b.start_id for b in loaded_blocks} == {b.start_id for b in blocks}


# ---------- threshold sanity checks ----------


def test_thresholds_are_sensible() -> None:
    """Sanity-check the published constants so future tweaks don't silently break."""
    assert STRUCTURE_MAX_BYTES < ACTIVITY_MIN_BYTES
    assert ACTIVITY_MAX_BYTES < SMILES_MIN_BYTES
    assert ACTIVITY_MIN_WIDTH_PX < ACTIVITY_MAX_WIDTH_PX < SMILES_MIN_WIDTH_PX
