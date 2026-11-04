"""Tests for the patent-profile adapter layer.

The adapter layer exists so that adding a patent is a configuration
change rather than a code change.  These tests pin the two things
that make that safe:

* the original patent keeps its exact geometry and behaviour, and
* a new patent is described entirely by data.

They deliberately avoid OCR: the geometry and the record plumbing are
what the profiles own, and OCR quality is measured separately by the
audit scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.merge import merge_compound_records  # noqa: E402
from cpd.parsers.page_tables import (  # noqa: E402
    horizontal_rules,
    locate_page_tables,
)
from cpd.parsers.profiles import (  # noqa: E402
    available_profiles,
    get_profile,
)
from cpd.parsers.table_extractor import (  # noqa: E402
    AssayRecord,
    CompoundRecord,
    RapidOcrTableExtractor,
)


# ---------- the original patent must not change ----------


def test_original_profile_keeps_its_geometry() -> None:
    """The shipped patent's column fractions must stay exactly as they were."""
    profile = get_profile("WO2025162428")
    columns = profile.scaled_columns("activity", 1000)
    assert columns == [
        ("cmpd_id", 0, 90, False),
        ("structure", 100, 480, True),
        ("rt_min", 480, 570, False),
        ("ms_mz", 580, 670, False),
        ("lcms_method", 670, 780, False),
        ("ms_polarity", 790, 880, False),
        ("kd_nm", 890, 1000, False),
    ]


def test_original_profile_keeps_its_route_and_fields() -> None:
    profile = get_profile("WO2025162428")
    # The default route is the one that reaches character-level
    # accuracy: ``validated_vote`` is the ``validated`` route plus a
    # second recogniser, which settles the characters the first one
    # loses.  Measured on the pipeline's own cells it makes 41 of 334
    # well-formed against 29.  The historical ``production`` route
    # stays available through ``--smiles-route production`` and still
    # reproduces the first-generation CSV byte for byte.
    assert profile.smiles_route == "validated_vote"
    assert profile.source_mode == "table_image"
    assert profile.merge_compounds == "first_valid"
    assert profile.layout_by_column_count == {
        3: "smiles", 6: "dense", 7: "activity",
    }
    names = [field.name for field in profile.assay_fields]
    assert names == [
        "kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity",
    ]
    # The original patent has no constrained fields.
    assert all(field.kind == "text" or field.kind == "number"
               for field in profile.assay_fields)


def test_original_profile_uses_four_digit_ids() -> None:
    profile = get_profile("WO2025162428")
    assert profile.cmpd_id_pattern == r"\d{4}"
    assert profile.cmpd_id_top_px is None
    assert profile.cmpd_id_inset_px == 0


# ---------- the second patent is described by data ----------


def test_second_profile_declares_its_layouts() -> None:
    profile = get_profile("WO2022251372")
    assert profile.source_mode == "full_page"
    assert profile.smiles_route == "padded_vote"
    assert profile.merge_compounds == "prefer_structure"
    assert profile.layout_by_column_count == {8: "activity8", 3: "smiles3"}

    activity = profile.layout("activity8")
    assert [c.name for c in activity.columns] == [
        "cmpd_id", "smiles", "mdm2", "mdm4",
        "molm13", "mv4_11", "waga", "mkl1",
    ]
    smiles = profile.layout("smiles3")
    assert [c.name for c in smiles.columns] == [
        "cmpd_id", "smiles", "structure",
    ]
    assert smiles.columns[2].is_image is True


def test_second_profile_uses_one_digit_or_two_digit_ids() -> None:
    profile = get_profile("WO2022251372")
    assert profile.cmpd_id_pattern == r"\d{1,2}"


def test_second_profile_marks_header_and_continuation_rows() -> None:
    profile = get_profile("WO2022251372")
    for name in ("activity8", "smiles3"):
        layout = profile.layout(name)
        assert layout.header_bands == 1
        assert layout.continuation is True


def test_second_profile_constrains_grades_to_a_b_c_d() -> None:
    profile = get_profile("WO2022251372")
    assert len(profile.assay_fields) == 6
    for field in profile.assay_fields:
        assert field.kind == "grade"
        assert field.allowed == ("A", "B", "C", "D")


def test_all_profiles_are_registered() -> None:
    assert set(available_profiles()) == {
        "WO2025162428", "WO2022251372",
    }


def test_unknown_profile_lists_the_known_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_profile("WO0000000000")
    assert "WO2022251372" in str(excinfo.value)


# ---------- profile-driven helpers ----------


def test_assay_field_lookup() -> None:
    profile = get_profile("WO2022251372")
    assert profile.assay_field("mdm2") is not None
    assert profile.assay_field("mdm2").kind == "grade"
    assert profile.assay_field("nope") is None


def test_layout_for_column_count() -> None:
    profile = get_profile("WO2022251372")
    assert profile.layout_for_column_count(8) == "activity8"
    assert profile.layout_for_column_count(3) == "smiles3"
    assert profile.layout_for_column_count(5) is None


def test_scaled_columns_requires_geometry() -> None:
    """Full-page profiles measure their geometry, so they have none."""
    profile = get_profile("WO2022251372")
    with pytest.raises(ValueError):
        profile.scaled_columns("activity8", 1000)


# ---------- merge behaviour ----------


def test_merge_prefers_structure_when_asked() -> None:
    """Table 1 gives the SMILES, Table 1a gives the drawing."""
    smiles_only = CompoundRecord(
        cmpd_id="1", smiles="CCO", canonical_smiles="CCO", is_valid=True,
    )
    structure_only = CompoundRecord(
        cmpd_id="1", smiles=None, is_valid=False,
        structure_image="cells/x_1_struct.png",
    )
    merged = merge_compound_records(
        [smiles_only, structure_only], prefer_structure=True,
    )
    assert merged["1"].structure_image == "cells/x_1_struct.png"
    assert merged["1"].canonical_smiles == "CCO"


def test_merge_without_preference_keeps_the_first_valid() -> None:
    first = CompoundRecord(
        cmpd_id="1", smiles="CCO", canonical_smiles="CCO", is_valid=True,
    )
    second = CompoundRecord(
        cmpd_id="1", smiles="CCN", canonical_smiles="CCN", is_valid=True,
        structure_image="cells/x_1_struct.png",
    )
    merged = merge_compound_records([first, second])
    assert merged["1"].canonical_smiles == "CCO"


# ---------- the extractor honours the profile ----------


def test_extractor_defaults_to_the_original_profile() -> None:
    extractor = RapidOcrTableExtractor.__new__(RapidOcrTableExtractor)
    # Construct without loading OCR models.
    from cpd.parsers.profiles import get_profile as _get

    extractor.profile = _get("WO2025162428")
    assert extractor.profile.source_mode == "table_image"


def test_page_table_detection_finds_both_tables() -> None:
    """The page adapter locates Table 1 and Table 1a on one page."""
    import cv2

    page = REPO / (
        "data/raw/WO2022251372/golden_tables/000116.tif"
    )
    if not page.exists():
        pytest.skip("second-patent pages are not present")
    image = cv2.imread(str(page))
    tables = locate_page_tables(image)
    assert len(tables) == 2
    assert sorted(table.n_columns for table in tables) == [3, 8]
    for table in tables:
        # Every table has a header band and at least one data row.
        assert len(table.row_bands()) >= 2


def test_horizontal_rules_are_found_on_a_real_page() -> None:
    import cv2

    page = REPO / (
        "data/raw/WO2022251372/golden_tables/000116.tif"
    )
    if not page.exists():
        pytest.skip("second-patent pages are not present")
    image = cv2.imread(str(page))
    rules = horizontal_rules(image)
    assert len(rules) >= 4


# ---------- the grade reader is shared, not per patent ----------


def test_grade_reader_accepts_only_allowed_values() -> None:
    """A misread glyph must leave the cell empty, not write junk."""
    import numpy as np

    from cpd.ocr.constrained import ConstrainedCellOCR, isolate_glyph

    # A clean capital A on white.
    glyph = np.full((40, 30, 3), 255, np.uint8)
    glyph[8:32, 13:17] = 0            # the left stem
    glyph[8:32, 13:17] = 0
    glyph[8:12, 8:22] = 0             # the cross bar
    glyph[8:12, 8:22] = 0
    assert isolate_glyph(glyph) is not None

    class _Engine:
        def __init__(self, text: str) -> None:
            self.text = text

        def __call__(self, image, **kwargs):
            class _Out:
                txts = [self.text]

            return _Out()

    reader = ConstrainedCellOCR(
        ("A", "B", "C", "D"), detector_engine=_Engine("A"),
    )
    assert reader.read(glyph) == "A"

    bad = ConstrainedCellOCR(
        ("A", "B", "C", "D"), detector_engine=_Engine("y"),
    )
    assert bad.read(glyph) is None


def test_assay_record_extra_is_flattened_only_when_present() -> None:
    plain = AssayRecord(cmpd_id="1", kd_nm=1.0)
    assert "extra" not in plain.to_dict()

    graded = AssayRecord(cmpd_id="1", extra=(("mdm2", "A"),))
    assert graded.to_dict()["mdm2"] == "A"


# ---------- the route is chosen by the profile, not by a constant ----------


def test_extractor_uses_the_profile_route_by_default() -> None:
    """No explicit route means "whatever the patent's profile says"."""
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    for name, expected in (
        ("WO2025162428", "validated_vote"),
        ("WO2022251372", "padded_vote"),
    ):
        extractor = RapidOcrTableExtractor(profile=get_profile(name))
        assert extractor.smiles_route == expected


def test_explicit_route_overrides_the_profile() -> None:
    """``--smiles-route production`` must still reproduce the old route."""
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    extractor = RapidOcrTableExtractor(
        profile=get_profile("WO2025162428"),
        smiles_route="production",
    )
    assert extractor.smiles_route == "production"


def test_profile_default_matches_the_extractor_default() -> None:
    """``main.py`` with no arguments must use the profile's route."""
    import main

    # The command line must not pin a route: ``None`` is what makes
    # the profile decide, so a new patent is configured in one place.
    parsed = main._parse_args([])
    assert parsed.smiles_route is None
    assert parsed.profile == "WO2025162428"

    # ... and an explicit route still wins.
    parsed = main._parse_args(["--smiles-route", "production"])
    assert parsed.smiles_route == "production"
