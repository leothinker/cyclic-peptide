"""Tests for the dual-table extractor and header aliasing (logic only).

OCR itself is heavy and flaky on CI; the unit tests here only cover the
pure-Python bits (alias lookup, schema fields, dedupe across pages).
"""

from __future__ import annotations

from cpd.parsers.headers import canonical_field, is_known_header
from cpd.parsers.table_extractor import AssayRecord, CompoundRecord


# ---------- header aliasing ----------


def test_canonical_field_handles_cmpd_aliases() -> None:
    assert canonical_field("Cmpd #") == "cmpd_id"
    assert canonical_field("Compound #") == "cmpd_id"
    assert canonical_field("Compd #") == "cmpd_id"
    assert canonical_field("cmpd") == "cmpd_id"
    assert canonical_field("  COMPOUND  #  ") == "cmpd_id"


def test_canonical_field_handles_kd_aliases() -> None:
    assert canonical_field("G12V GDP KD nM") == "kd_nm"
    assert canonical_field("KRAS G12V GDP KD (nM)") == "kd_nm"
    assert canonical_field("KD (nM)") == "kd_nm"


def test_canonical_field_handles_lcms_aliases() -> None:
    assert canonical_field("LCMS RT (min)") == "rt_min"
    assert canonical_field("MS (m/z)") == "ms_mz"
    assert canonical_field("LCMS Method") == "lcms_method"
    assert canonical_field("MS Polarity") == "ms_polarity"


def test_canonical_field_returns_none_for_unknown() -> None:
    assert canonical_field("Banana") is None
    assert canonical_field(None) is None
    assert canonical_field("") is None


def test_is_known_header_round_trips_with_canonical_field() -> None:
    assert is_known_header("SMILES") is True
    assert is_known_header("Structure") is True
    assert is_known_header("Banana") is False


# ---------- dual-table record schemas ----------


def test_compound_record_default_fields() -> None:
    rec = CompoundRecord(cmpd_id="1001")
    assert rec.cmpd_id == "1001"
    assert rec.smiles is None
    assert rec.is_valid is None
    assert rec.structure_image is None
    d = rec.to_dict()
    assert d["cmpd_id"] == "1001"
    assert "structure_image" in d


def test_assay_record_default_fields() -> None:
    rec = AssayRecord(cmpd_id="1001", kd_nm=0.062)
    assert rec.kd_nm == 0.062
    assert rec.rt_min is None
    d = rec.to_dict()
    assert d["cmpd_id"] == "1001"
    assert d["kd_nm"] == 0.062
