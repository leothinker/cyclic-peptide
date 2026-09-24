"""Tests for the table content extractor (stub / regex / vision-LLM APIs)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cpd.parsers.table_extractor import (
    ActivityRow,
    RegexTableExtractor,
    SmilesRow,
    StubTableExtractor,
    VisionLlmTableExtractor,
    enrich_smiles_with_rdkit,
)


# ---------- StubTableExtractor ----------


def test_stub_returns_empty_rows(tmp_path: Path) -> None:
    stub = StubTableExtractor()
    img = tmp_path / "fake.jpg"
    assert stub.extract_activity(img) == []
    assert stub.extract_smiles(img) == []


# ---------- RegexTableExtractor (text path) ----------


@pytest.fixture()
def regex() -> RegexTableExtractor:
    return RegexTableExtractor()


def test_regex_parses_activity_table_text(regex: RegexTableExtractor) -> None:
    """Three floats + a polarity marker must yield one ActivityRow per line."""
    text = (
        "1001 1.70 1512.8 10-80-2min [M+H]+ 0.062\n"
        "1002 1.72 1526.9 10-80-2min [M+H]+ 0.064\n"
        "1003 1.68 1498.8 10-80-2min [M+H]+ 0.114\n"
    )
    rows = regex.parse_activity_text(text, source_image="I100280.jpg")
    assert len(rows) == 3, [r.cmpd_id for r in rows]
    by_id = {r.cmpd_id: r for r in rows}
    assert by_id["1001"].rt_min == 1.70
    assert by_id["1001"].ms_mz == 1512.8
    assert by_id["1001"].kd_nm == 0.062
    assert by_id["1001"].ms_polarity == "[M+H]+"


def test_regex_skips_lines_without_three_floats(regex: RegexTableExtractor) -> None:
    text = (
        "1001 1.70\n"  # only one float -> skipped
        "1002 1.72 1526.9 10-80-2min 0.064\n"  # ok
    )
    rows = regex.parse_activity_text(text)
    assert [r.cmpd_id for r in rows] == ["1002"]


def test_regex_parses_smiles_table_text(regex: RegexTableExtractor) -> None:
    """Lines starting with a SMILES-shaped string are kept."""
    text = (
        "1038 CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2...\n"
        "1039 CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2...\n"
        "header line without smiles\n"
    )
    rows = regex.parse_smiles_text(text, source_image="I100298.jpg")
    assert len(rows) == 2
    assert {r.cmpd_id for r in rows} == {"1038", "1039"}
    for r in rows:
        assert r.smiles is not None
        assert r.smiles.startswith("CC[")


def test_regex_extract_methods_raise_for_image_paths() -> None:
    """The regex backend is text-only -- image extraction must say so."""
    extractor = RegexTableExtractor()
    with pytest.raises(NotImplementedError):
        extractor.extract_activity(Path("doesnt-matter.jpg"))
    with pytest.raises(NotImplementedError):
        extractor.extract_smiles(Path("doesnt-matter.jpg"))


# ---------- VisionLlmTableExtractor ----------


def test_vision_llm_extractor_requires_call() -> None:
    """Without an llm_call set, the extractor must raise a clear error."""
    ext = VisionLlmTableExtractor()
    with pytest.raises(RuntimeError, match="llm_call"):
        ext.extract_activity(Path("x.jpg"))
    with pytest.raises(RuntimeError, match="llm_call"):
        ext.extract_smiles(Path("x.jpg"))


def test_vision_llm_extractor_parses_activity_json() -> None:
    """When wired, the extractor turns a JSON response into ActivityRows."""
    captured: dict = {}

    def fake_call(image_path, prompt):
        captured["image_path"] = str(image_path)
        captured["prompt"] = prompt
        return json_dumps_two_rows()

    ext = VisionLlmTableExtractor(llm_call=fake_call)
    rows = ext.extract_activity(Path("I100280.jpg"))
    assert len(rows) == 2
    by_id = {r.cmpd_id: r for r in rows}
    assert by_id["1001"].kd_nm == 0.062
    assert by_id["1001"].rt_min == 1.70
    assert by_id["1001"].ms_polarity == "[M+H]+"
    assert captured["image_path"].endswith("I100280.jpg")


def test_vision_llm_extractor_parses_smiles_json() -> None:
    """SMILES rows get parsed out and source_image gets stamped on."""
    def fake_call(image_path, prompt):
        return json_dumps_smiles_rows()

    ext = VisionLlmTableExtractor(llm_call=fake_call)
    rows = ext.extract_smiles(Path("I100298.jpg"))
    assert len(rows) == 2
    assert {r.cmpd_id for r in rows} == {"1038", "1041"}
    assert all(r.source_image == "I100298.jpg" for r in rows)


def test_vision_llm_extractor_tolerates_fenced_json() -> None:
    """Models often wrap the JSON in ```json fences; strip them."""
    def fake_call(image_path, prompt):
        return "```json\n" + json_dumps_two_rows() + "\n```"

    ext = VisionLlmTableExtractor(llm_call=fake_call)
    rows = ext.extract_activity(Path("x.jpg"))
    assert len(rows) == 2


def test_vision_llm_extractor_handles_garbage() -> None:
    """Bad JSON -> empty rows, no crash."""
    def fake_call(image_path, prompt):
        return "not json at all"

    ext = VisionLlmTableExtractor(llm_call=fake_call)
    assert ext.extract_activity(Path("x.jpg")) == []
    assert ext.extract_smiles(Path("x.jpg")) == []


# ---------- enrich_smiles_with_rdkit ----------


def test_enrich_smiles_with_rdkit_no_rdkit_returns_input() -> None:
    """Without RDKit, the rows pass through unchanged."""
    rows = [SmilesRow(cmpd_id="1038", smiles="CCO", source_image="x.jpg")]
    out = enrich_smiles_with_rdkit(rows)
    assert out is not rows or out[0].is_valid is None  # may mutate or return new
    # canonical_smiles stays None because RDKit isn't importable here.
    assert out[0].canonical_smiles is None
    assert out[0].is_valid is None


# ---------- test fixtures (JSON payloads) ----------


def json_dumps_two_rows() -> str:
    import json
    return json.dumps([
        {
            "cmpd_id": "1001",
            "kd_nm": 0.062,
            "rt_min": 1.70,
            "ms_mz": 1512.8,
            "lcms_method": "10-80-2min",
            "ms_polarity": "[M+H]+",
        },
        {
            "cmpd_id": "1002",
            "kd_nm": 0.064,
            "rt_min": 1.72,
            "ms_mz": 1526.9,
            "lcms_method": "10-80-2min",
            "ms_polarity": "[M+H]+",
        },
    ])


def json_dumps_smiles_rows() -> str:
    import json
    return json.dumps([
        {"cmpd_id": "1038", "smiles": "CC[C@H](C)C(=O)N(C)C"},
        {"cmpd_id": "1041", "smiles": "CCO"},
    ])
