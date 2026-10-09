"""Tests for the combined-table builder and DECIMER fallback (logic only).

The OCSR backend is optional; we verify the surrounding plumbing rather
than the model itself.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from cpd.decimer import is_available, predict_smiles
from cpd.parsers.table_extractor import AssayRecord, CompoundRecord


# ---------- DECIMER graceful fallback ----------


def test_is_available_returns_bool() -> None:
    """is_available() must return a bool and never raise."""
    assert isinstance(is_available(), bool)


def test_predict_smiles_returns_none_when_not_available() -> None:
    """When DECIMER is not installed, predict_smiles should be a no-op."""
    if is_available():
        return  # skip when DECIMER is installed
    assert predict_smiles(Path("/tmp/does-not-matter.png")) is None


# ---------- combined-row schema ----------


def _combined_columns() -> tuple[str, ...]:
    """Read COMBINED_COLUMNS out of main without running the pipeline."""
    import main  # local import keeps this test independent

    return main.COMBINED_COLUMNS


def test_combined_columns_include_both_smiles_sources() -> None:
    cols = _combined_columns()
    assert "smiles" in cols
    assert "smiles_decimer" in cols
    assert "canonical_smiles" in cols
    assert "canonical_smiles_decimer" in cols
    assert "smiles_agree" in cols
    assert "kd_nm" in cols
    assert "structure_image" in cols


def test_build_combined_rows_basic() -> None:
    """Build combined rows from a small set of compounds + assays."""
    import main  # local import keeps tests independent

    compounds = [
        CompoundRecord(
            cmpd_id="1001",
            smiles="CCO",
            canonical_smiles="CCO",
            molecular_weight=46.07,
            heavy_atom_count=2,
            is_valid=True,
            structure_image="cells/1001_struct.png",
            source_image="a.jpg",
        ),
        CompoundRecord(
            cmpd_id="1002",
            smiles="CCN",
            canonical_smiles="CCN",
            molecular_weight=45.09,
            heavy_atom_count=2,
            is_valid=True,
            source_image="a.jpg",
        ),
    ]
    assays: dict = {
        "1001": AssayRecord(cmpd_id="1001", kd_nm=0.062, rt_min=1.7,
                            ms_mz=1512.8, lcms_method="10-80- 2min",
                            ms_polarity="[M+H]+", source_image="b.jpg"),
        "1003": AssayRecord(cmpd_id="1003", kd_nm=0.999, source_image="b.jpg"),
    }
    decimer_raw = {"1001": "CCO", "1002": "CCN"}
    decimer_canon = {"1001": "CCO", "1002": "CCN"}

    rows = main._build_combined_rows(compounds, assays, decimer_raw, decimer_canon)
    by_id = {r["cmpd_id"]: r for r in rows}

    assert "1001" in by_id and "1002" in by_id and "1003" in by_id
    assert by_id["1001"]["smiles"] == "CCO"
    assert by_id["1001"]["smiles_decimer"] == "CCO"
    assert by_id["1001"]["smiles_agree"] == "True"
    assert by_id["1001"]["kd_nm"] == 0.062
    assert by_id["1001"]["assay_source_image"] == "b.jpg"

    # cmpd 1002 has no assay -> empty activity columns
    assert by_id["1002"]["kd_nm"] == ""
    assert by_id["1002"]["smiles_agree"] == "True"

    # cmpd 1003 only appears in the assay stream
    assert by_id["1003"]["smiles"] == ""
    assert by_id["1003"]["kd_nm"] == 0.999
    assert by_id["1003"]["smiles_agree"] == "False"


def test_build_combined_rows_agrees_only_when_canonicals_match() -> None:
    """smiles_agree should be True iff both canonical SMILES are non-empty and equal."""
    import main

    compounds = [CompoundRecord(cmpd_id="1", smiles="CCO",
                                canonical_smiles="CCO", is_valid=True)]
    rows_a = main._build_combined_rows(compounds, {}, {"1": "CCO"}, {"1": "CCO"})
    rows_b = main._build_combined_rows(compounds, {}, {"1": "CCN"}, {"1": "CCN"})
    rows_c = main._build_combined_rows(compounds, {}, {"1": "junk"}, {})
    assert rows_a[0]["smiles_agree"] == "True"
    assert rows_b[0]["smiles_agree"] == "False"
    assert rows_c[0]["smiles_agree"] == "False"


def test_combined_csv_round_trip(tmp_path: Path) -> None:
    """_write_csv produces a valid CSV that re-loads with the same row count."""
    import main

    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    out = tmp_path / "out.csv"
    n = main._write_csv(rows, out, ("a", "b"))
    assert n == 2
    with out.open(encoding="utf-8-sig", newline="") as fh:
        loaded = list(csv.DictReader(fh))
    assert loaded == rows
