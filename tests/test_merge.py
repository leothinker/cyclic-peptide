"""Tests for cpd.merge: cmpd_id normalisation + dual-table dedupe."""
from __future__ import annotations

import pytest

from cpd.merge import (
    dedupe_assays,
    dedupe_compounds,
    normalise_cmpd_id,
)
from cpd.parsers.table_extractor import AssayRecord, CompoundRecord


# ---------- normalise_cmpd_id ----------


@pytest.mark.parametrize("raw,expected", [
    ("1001", "1001"),
    (" 1001 ", "1001"),
    ("Compound 1001", "1001"),
    ("compound 1041", "1041"),
    ("Cmpd. 1038", "1038"),
    ("Compd #42", "42"),
    ("01001", "1001"),            # leading zeros collapse
    ("1038:", "1038"),            # trailing punctuation dropped
    ("abc", "abc"),               # unparseable -> lowercased
])
def test_normalise_cmpd_id(raw: str, expected: str) -> None:
    assert normalise_cmpd_id(raw) == expected


def test_normalise_cmpd_id_empty_returns_none() -> None:
    assert normalise_cmpd_id("") is None
    assert normalise_cmpd_id("   ") is None
    assert normalise_cmpd_id(None) is None


# ---------- dedupe_compounds ----------


def test_dedupe_compounds_keeps_first_valid() -> None:
    rows = [
        CompoundRecord(cmpd_id="1001", smiles="bad", source_image="x.jpg"),
        CompoundRecord(cmpd_id="1001", smiles="CCO", canonical_smiles="CCO",
                       is_valid=True, source_image="y.jpg"),
        CompoundRecord(cmpd_id="1001", smiles="CCN", canonical_smiles="CCN",
                       is_valid=True, source_image="z.jpg"),
    ]
    out = dedupe_compounds(rows)
    assert list(out.keys()) == ["1001"]
    assert out["1001"].smiles == "CCO"
    assert out["1001"].source_image == "y.jpg"


def test_dedupe_compounds_falls_back_to_invalid() -> None:
    """When every row for a cmpd_id is invalid, keep the first one seen."""
    rows = [
        CompoundRecord(cmpd_id="1001", smiles="bad1", source_image="x.jpg"),
        CompoundRecord(cmpd_id="1001", smiles="bad2", source_image="y.jpg"),
    ]
    out = dedupe_compounds(rows)
    assert out["1001"].source_image == "x.jpg"


# ---------- dedupe_assays ----------


def test_dedupe_assays_first_wins() -> None:
    rows = [
        AssayRecord(cmpd_id="1001", kd_nm=0.062, source_image="x.jpg"),
        AssayRecord(cmpd_id="1001", kd_nm=0.999, source_image="y.jpg"),
    ]
    out = dedupe_assays(rows)
    assert out["1001"].kd_nm == 0.062


def test_dedupe_assays_handles_cmpd_prefix_variants() -> None:
    """`Compound 1001` from one source must dedupe against `1001`."""
    rows = [
        AssayRecord(cmpd_id="Compound 1001", kd_nm=0.062, source_image="a.jpg"),
        AssayRecord(cmpd_id="1001", kd_nm=0.999, source_image="b.jpg"),
    ]
    out = dedupe_assays(rows)
    assert list(out.keys()) == ["1001"]
    assert out["1001"].kd_nm == 0.062
    assert out["1001"].source_image == "a.jpg"
