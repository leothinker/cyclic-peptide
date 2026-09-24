"""Tests for cpd.merge: cmpd_id normalisation, inner join, dedupe, CSV/JSONL."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cpd.merge import (
    COLUMNS,
    JoinedRow,
    coverage_report,
    dedupe_activity,
    dedupe_smiles,
    join,
    join_outer,
    normalise_cmpd_id,
    to_csv,
    to_jsonl,
)
from cpd.parsers.table_extractor import ActivityRow, SmilesRow


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


# ---------- dedupe ----------


def test_dedupe_smiles_keeps_first_valid() -> None:
    rows = [
        SmilesRow(cmpd_id="1001", smiles="bad", source_image="x.jpg"),
        SmilesRow(cmpd_id="1001", smiles="CCO", canonical_smiles="CCO",
                  is_valid=True, source_image="y.jpg"),
        SmilesRow(cmpd_id="1001", smiles="CCN", canonical_smiles="CCN",
                  is_valid=True, source_image="z.jpg"),
    ]
    out = dedupe_smiles(rows)
    assert list(out.keys()) == ["1001"]
    assert out["1001"].smiles == "CCO"
    assert out["1001"].source_image == "y.jpg"


def test_dedupe_activity_first_wins() -> None:
    rows = [
        ActivityRow(cmpd_id="1001", kd_nm=0.062, source_image="x.jpg"),
        ActivityRow(cmpd_id="1001", kd_nm=0.999, source_image="y.jpg"),
    ]
    out = dedupe_activity(rows)
    assert out["1001"].kd_nm == 0.062


# ---------- join ----------


def _activity(rows: list[tuple[str, float]]) -> list[ActivityRow]:
    return [
        ActivityRow(cmpd_id=cid, kd_nm=kd, rt_min=1.7, ms_mz=1512.8, source_image="act.jpg")
        for cid, kd in rows
    ]


def _smiles(rows: list[tuple[str, str]]) -> list[SmilesRow]:
    return [
        SmilesRow(cmpd_id=cid, smiles=smi, canonical_smiles=smi,
                  is_valid=True, source_image="smi.jpg")
        for cid, smi in rows
    ]


def test_join_inner_merges_on_cmpd_id() -> None:
    a = _activity([("1001", 0.062), ("1002", 0.064), ("1003", 0.114)])
    s = _smiles([("1001", "CCO"), ("1002", "CCN"), ("9999", "CCC")])
    out = join(a, s)
    assert len(out) == 2
    by_id = {r.cmpd_id: r for r in out}
    assert by_id["1001"].canonical_smiles == "CCO"
    assert by_id["1001"].kd_nm == 0.062
    # 9999 had smiles but no activity -> dropped in inner join.
    assert "9999" not in by_id


def test_join_outer_keeps_smiles_only_rows() -> None:
    a = _activity([("1001", 0.062)])
    s = _smiles([("1001", "CCO"), ("9999", "CCC")])
    out = join_outer(a, s)
    by_id = {r.cmpd_id: r for r in out}
    assert "9999" in by_id
    assert by_id["9999"].canonical_smiles == "CCC"
    assert by_id["9999"].kd_nm is None


def test_join_handles_cmpd_id_prefix_variants() -> None:
    """`Compound 1001` from one source must join `1001` from another."""
    a = [ActivityRow(cmpd_id="Compound 1001", kd_nm=0.062, source_image="a.jpg")]
    s = [SmilesRow(cmpd_id="1001", smiles="CCO", canonical_smiles="CCO",
                   is_valid=True, source_image="s.jpg")]
    out = join(a, s)
    assert len(out) == 1
    assert out[0].cmpd_id == "1001"
    assert out[0].kd_nm == 0.062
    assert out[0].canonical_smiles == "CCO"


def test_coverage_report_counts() -> None:
    a = _activity([("1001", 0.062), ("1002", 0.064)])
    s = _smiles([("1001", "CCO"), ("9999", "CCC")])
    rep = coverage_report(a, s)
    assert rep["n_smiles"] == 2
    assert rep["n_activity"] == 2
    assert rep["n_both"] == 1
    assert rep["n_only_smiles"] == 1
    assert rep["n_only_activity"] == 1
    assert rep["only_smiles_ids"] == ["9999"]
    assert rep["only_activity_ids"] == ["1002"]


# ---------- CSV / JSONL writers ----------


def test_to_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        JoinedRow(
            cmpd_id="1001", canonical_smiles="CCO", smiles_raw="CCO",
            smiles_is_valid=True, kd_nm=0.062, rt_min=1.7, ms_mz=1512.8,
            lcms_method="10-80-2min", ms_polarity="[M+H]+",
            source_smiles_image="s.jpg", source_activity_image="a.jpg",
        ),
    ]
    p = tmp_path / "out.csv"
    n = to_csv(rows, p)
    assert n == 1
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(COLUMNS)
        first = next(reader)
    assert first["cmpd_id"] == "1001"
    assert first["canonical_smiles"] == "CCO"
    assert first["kd_nm"] == "0.062"


def test_to_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [
        JoinedRow(
            cmpd_id="1001", canonical_smiles="CCO", smiles_raw="CCO",
            smiles_is_valid=True, kd_nm=0.062, rt_min=1.7, ms_mz=1512.8,
            lcms_method="10-80-2min", ms_polarity="[M+H]+",
            source_smiles_image="s.jpg", source_activity_image="a.jpg",
        ),
    ]
    p = tmp_path / "out.jsonl"
    n = to_jsonl(rows, p)
    assert n == 1
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["cmpd_id"] == "1001"
    assert payload["kd_nm"] == 0.062
