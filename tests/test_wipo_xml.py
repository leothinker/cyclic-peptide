"""Tests for the WIPO body XML parser."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpd.parsers.wipo_xml import (
    load_map_json,
    parse_body_xml,
    write_map_json,
)

BODY_XML = (
    Path(__file__).resolve().parent.parent
    / "WO2025162428"
    / "FullText"
    / "wo-published-application-body.xml"
)
pytestmark = pytest.mark.skipif(
    not BODY_XML.exists(),
    reason="WO2025162428 FullText XML not present at expected path",
)


def test_total_image_count_is_reasonable() -> None:
    """The body XML ships with hundreds of figures; we should see them all."""
    records = parse_body_xml(BODY_XML)
    assert len(records) >= 100, (
        f"expected at least 100 <img> tags, got {len(records)} -- "
        "is the body XML truncated?"
    )


def test_first_image_is_formula_a() -> None:
    """The cover-page Markush (Formula A) is the first <img> in the body."""
    records = parse_body_xml(BODY_XML)
    assert records, "no <img> tags found in body XML"
    first = records[0]
    assert first.category == "formula", (
        f"first record should be a formula, got category={first.category!r}"
    )
    assert first.formula_label == "(A)"
    assert first.img_id == "idf0001"
    assert first.img_file == "PCTCN2025075389-ftappb-I100001.jpg"


def test_early_markush_formulas_extracted() -> None:
    """Formula (A), (I), (II) should be in the first 5 figures (the abstract)."""
    records = parse_body_xml(BODY_XML)
    formulas = [
        (i, r) for i, r in enumerate(records[:5])
        if r.category == "formula"
    ]
    labels = [r.formula_label for _, r in formulas]
    assert "(A)" in labels, f"Formula (A) missing from first 5: {labels}"
    assert "(I)" in labels, f"Formula (I) missing from first 5: {labels}"
    assert "(II)" in labels, f"Formula (II) missing from first 5: {labels}"


def test_compound_3_is_extracted() -> None:
    """Compound 3 is the FIRST compound with an inline figure.

    Compounds 1 and 2 are linear peptide precursors described only in
    synthesis prose (peptide sequence abbreviations, no drawn structure),
    so they never have an adjacent <img>. The first capturable compound
    is Compound 3, which gets an inline <img> in the synthesis paragraph.
    The case-insensitive regex catches both ``compound 3`` and ``Compound 3``.
    """
    records = parse_body_xml(BODY_XML)
    matches = [r for r in records if r.compound_id == "Compound 3"]
    assert matches, "Compound 3 not extracted"
    assert matches[0].category == "compound"
    assert matches[0].img_path and matches[0].img_path.endswith(matches[0].img_file)


def test_categories_are_all_known() -> None:
    """Every record must fall into one of the known categories."""
    records = parse_body_xml(BODY_XML)
    known = {"formula", "scheme", "example", "compound", "figure", "unknown"}
    bad = [r for r in records if r.category not in known]
    assert not bad, f"unexpected categories: {[r.category for r in bad[:5]]}"


def test_round_trip_json(tmp_path: Path) -> None:
    """write_map_json + load_map_json must preserve every record field."""
    records = parse_body_xml(BODY_XML)
    out = tmp_path / "image_compound_map.json"
    write_map_json(records, out, patent_id="WO2025162428", source_xml=BODY_XML)
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["patent_id"] == "WO2025162428"
    assert payload["total_records"] == len(records)
    assert sum(payload["category_counts"].values()) == len(records)

    loaded = load_map_json(out)
    assert len(loaded) == len(records)
    for original, restored in zip(records, loaded):
        assert restored.img_id == original.img_id
        assert restored.img_file == original.img_file
        assert restored.category == original.category
        assert restored.compound_id == original.compound_id
        assert restored.formula_label == original.formula_label
