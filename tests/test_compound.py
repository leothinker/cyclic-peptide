"""Tests for Compound + Bioactivity serialization."""
from __future__ import annotations

from pathlib import Path

import pytest

from cpd.models.compound import (
    Bioactivity,
    Compound,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def test_bioactivity_accepts_known_units() -> None:
    b = Bioactivity(target="KRAS G12V-GDP", measurement="Kd", value=2.3, unit="nM")
    assert b.value == pytest.approx(2.3)


def test_bioactivity_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        Bioactivity(target="X", measurement="Kd", value=1.0, unit="mol/L")  # type: ignore[arg-type]


def test_compound_roundtrip_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "compounds.jsonl"
    rows = [
        Compound(patent_id="WO2025162428", compound_id="I-42",
                 smiles="CC(=O)N[C@@H]1C(=O)N2",
                 bioactivity=[Bioactivity(target="KRAS G12V-GDP", measurement="Kd",
                                          value=2.3, unit="nM")]),
        Compound(patent_id="WO2025162428", compound_id="I-43",
                 smiles="CCO", sequence="cyclo(-Ala-Pro-Phe-)"),
    ]
    write_jsonl(rows, out)
    assert out.exists()
    loaded = read_jsonl(out)
    assert len(loaded) == 2
    assert loaded[0].bioactivity[0].value == pytest.approx(2.3)
    assert loaded[1].sequence == "cyclo(-Ala-Pro-Phe-)"


def test_csv_flattens_bioactivity(tmp_path: Path) -> None:
    out = tmp_path / "compounds.csv"
    rows = [
        Compound(patent_id="WO1", compound_id="A",
                 bioactivity=[Bioactivity(target="t", measurement="Kd", value=1.0, unit="nM"),
                             Bioactivity(target="t", measurement="IC50", value=10.0, unit="nM")]),
        Compound(patent_id="WO1", compound_id="B"),  # no bioactivity
    ]
    write_csv(rows, out)
    text = out.read_text(encoding="utf-8")
    # 1 header + 2 rows for A + 1 row for B = 4 lines
    assert len(text.strip().splitlines()) == 4
    assert text.count("Kd") == 1
    assert text.count("IC50") == 1
    assert "compound_id" in text.splitlines()[0]
