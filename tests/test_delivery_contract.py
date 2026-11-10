"""Tests for the two contracts the delivery relies on.

1. Every route the pipeline implements can be selected from the
   command line.  The two patents ship with the ``*_vote`` routes, so
   a supervisor who wants to reproduce a run by hand must be able to
   name them.
2. The combined CSV carries the patent's own activity fields.  The
   first patent declares only the five canonical fields and must keep
   its exact shipped columns; the second patent's six letter grades
   must survive into its own CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.parsers.profiles import get_profile  # noqa: E402
from cpd.schema import COMBINED_COLUMNS, combined_columns  # noqa: E402


# ---------- the CLI can select every route ----------


def test_cli_accepts_every_route() -> None:
    """``--smiles-route`` must accept every route the pipeline has."""
    import main

    from cpd.ocr.printed_smiles import ROUTES

    for route in ROUTES:
        args = main._parse_args(["--smiles-route", route])
        assert args.smiles_route == route


def test_cli_rejects_an_unknown_route() -> None:
    """An unknown route is still a usage error, not a silent default."""
    import pytest

    import main

    with pytest.raises(SystemExit):
        main._parse_args(["--smiles-route", "not-a-route"])


def test_both_shipped_routes_are_selectable() -> None:
    """The routes the two profiles declare must be nameable on the CLI."""
    import main

    for name in ("WO2025162428", "WO2022251372"):
        route = get_profile(name).smiles_route
        assert main._parse_args(["--smiles-route", route]).smiles_route == route


# ---------- the column set follows the patent ----------


def test_first_patent_keeps_the_shipped_columns() -> None:
    """Patent 1 declares no extra field, so its column set is unchanged."""
    profile = get_profile("WO2025162428")
    assert profile.extra_assay_fields == ()
    assert combined_columns(profile.extra_assay_fields) == COMBINED_COLUMNS
    assert len(COMBINED_COLUMNS) == 48


def test_second_patent_adds_its_grade_columns() -> None:
    """Patent 2's six letter grades become extra columns, after the
    canonical block, in the profile's declaration order."""
    profile = get_profile("WO2022251372")
    assert profile.extra_assay_fields == (
        "mdm2", "mdm4", "molm13", "mv4_11", "waga", "mkl1",
    )
    columns = combined_columns(profile.extra_assay_fields)
    assert len(columns) == len(COMBINED_COLUMNS) + 6
    assert columns[:len(COMBINED_COLUMNS)] == COMBINED_COLUMNS
    assert columns[len(COMBINED_COLUMNS):] == profile.extra_assay_fields


def test_extra_columns_carry_the_assay_values() -> None:
    """The values on ``AssayRecord.extra`` reach the combined row."""
    import main

    from cpd.parsers.table_extractor import AssayRecord, CompoundRecord

    profile = get_profile("WO2022251372")
    compounds = [
        CompoundRecord(cmpd_id="1", smiles="CCO", canonical_smiles="CCO",
                       is_valid=True),
    ]
    assays = {
        "1": AssayRecord(
            cmpd_id="1",
            kd_nm=None, rt_min=None, ms_mz=None,
            lcms_method=None, ms_polarity=None,
            extra=(("mdm2", "A"), ("mdm4", "B"), ("molm13", "C"),
                   ("mv4_11", "D"), ("waga", "A"), ("mkl1", "B")),
        ),
    }
    rows = main._build_combined_rows(
        compounds, assays, {}, {},
        extra_columns=profile.extra_assay_fields,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["mdm2"] == "A"
    assert row["mkl1"] == "B"
    # The canonical fields stay empty: this patent has no numbers.
    assert row["kd_nm"] == ""


def test_missing_extra_values_become_empty_cells() -> None:
    """A grade the reader could not read is empty, never a wrong letter."""
    import main

    from cpd.parsers.table_extractor import AssayRecord, CompoundRecord

    profile = get_profile("WO2022251372")
    compounds = [CompoundRecord(cmpd_id="1", smiles="CCO")]
    assays = {
        "1": AssayRecord(cmpd_id="1", extra=(("mdm2", "A"),)),
    }
    rows = main._build_combined_rows(
        compounds, assays, {}, {},
        extra_columns=profile.extra_assay_fields,
    )
    assert rows[0]["mdm2"] == "A"
    assert rows[0]["mdm4"] == ""
    assert rows[0]["mkl1"] == ""


def test_first_patent_rows_carry_no_extra_columns() -> None:
    """Patent 1's rows are unchanged by the extra-column mechanism."""
    import main

    from cpd.parsers.table_extractor import AssayRecord, CompoundRecord

    profile = get_profile("WO2025162428")
    compounds = [
        CompoundRecord(cmpd_id="1036", smiles="CCO", canonical_smiles="CCO",
                       is_valid=True),
    ]
    assays = {
        "1036": AssayRecord(cmpd_id="1036", kd_nm=1.0, rt_min=2.0,
                            ms_mz=3.0, lcms_method="LCMS", ms_polarity="+"),
    }
    rows = main._build_combined_rows(
        compounds, assays, {}, {},
        extra_columns=profile.extra_assay_fields,
    )
    # The row builder emits the pipeline block; the repair and derived
    # layers add the rest of the schema afterwards.  What matters here
    # is that no patent-specific column leaked in.
    assert set(rows[0]) <= set(COMBINED_COLUMNS)
    assert not (set(rows[0]) & set(profile.extra_assay_fields))


def test_written_csv_has_the_profile_columns() -> None:
    """The writer uses the profile's column set, extras included."""
    import csv

    import main

    profile = get_profile("WO2022251372")
    columns = combined_columns(profile.extra_assay_fields)
    out = REPO / "data" / "audit" / "_tmp_delivery_contract.csv"
    try:
        main._write_csv(
            [{"cmpd_id": "1", "mdm2": "A", "mdm4": "B"}], out, columns,
        )
        with out.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns)
    finally:
        if out.exists():
            out.unlink()


def test_writable_target_accepts_a_missing_file() -> None:
    """A patent's first run into a fresh directory must not crash.

    The lock check used to open the target CSV for reading to see
    whether the editor held it, which raised FileNotFoundError when
    the file did not exist yet -- that is, on a patent's first run
    into a fresh directory.
    """
    import main

    missing = REPO / "data" / "audit" / "_tmp_no_such_dir" / "compounds_combined.csv"
    assert main._writable_target(missing) == missing


def test_writable_target_accepts_an_unlocked_file() -> None:
    """An ordinary, unlocked file is written to directly."""
    import main

    target = REPO / "data" / "audit" / "_tmp_unlocked.csv"
    try:
        target.write_text("cmpd_id\n1\n", encoding="utf-8")
        assert main._writable_target(target) == target
    finally:
        if target.exists():
            target.unlink()



