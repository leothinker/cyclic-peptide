"""Tests for the derived columns of the combined CSV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.postprocess import (  # noqa: E402
    apply_derived_columns,
    bracket_deficit,
    difficulty_bucket,
)


def test_bracket_deficit_counts_lost_brackets():
    assert bracket_deficit("CC(=O)N") == 0
    assert bracket_deficit("CC(=ON") == 1
    assert bracket_deficit("CC(=O)N[C@H") == 1
    assert bracket_deficit("") == 0


def test_difficulty_buckets():
    assert difficulty_bucket(0) == "balanced"
    assert difficulty_bucket(2) == "light"
    assert difficulty_bucket(5) == "medium"
    assert difficulty_bucket(10) == "heavy"
    assert difficulty_bucket(11) == "extreme"


def test_structure_columns_use_the_canonical_decimer_string():
    rows = apply_derived_columns([
        {
            "cmpd_id": "1",
            "canonical_smiles_decimer": "C1CCCCC1",
            "smiles": "",
        }
    ])
    row = rows[0]
    assert row["decimer_num_rings"] == 1
    assert row["decimer_largest_ring_size"] == 6
    # A six-membered ring is not a macrocycle.
    assert row["decimer_macrocycle_intact"] == "False"
    assert row["data_quality"] == "macrocycle_broken"


def test_numeric_out_of_range_is_flagged_and_summarised():
    rows = apply_derived_columns([
        {
            "cmpd_id": "1",
            "canonical_smiles_decimer": "",
            "smiles": "",
            "kd_nm": "201444",
            "rt_min": "2.0",
            "ms_mz": "1000",
        }
    ])
    row = rows[0]
    assert "kd_nm above range" in row["data_quality_flags"]
    assert row["data_quality"] == "partial"


def test_a_clean_row_is_ok():
    rows = apply_derived_columns([
        {
            "cmpd_id": "1",
            "canonical_smiles_decimer": "C1CCCCCCCCCCC1",
            "smiles": "C1CCCCCCCCCCC1",
            "kd_nm": "12.5",
            "rt_min": "2.0",
            "ms_mz": "1000",
        }
    ])
    row = rows[0]
    assert row["data_quality"] == "ok"
    assert row["data_quality_flags"] == ""
    assert row["ocr_bracket_deficit"] == "0"
    assert row["ocr_difficulty"] == "balanced"


def test_missing_ocr_text_is_marked_as_such():
    rows = apply_derived_columns([{"cmpd_id": "1", "smiles": ""}])
    assert rows[0]["ocr_difficulty"] == "no_ocr"
    assert rows[0]["ocr_bracket_deficit"] == ""

