"""Tests for the WIPO Patentscope HTML parser."""
from __future__ import annotations

from cpd.scrapers.wipo import parse_patent_html

SAMPLE_HTML = """
<html>
  <head><title>WO2025162428 - Patent Scope</title></head>
  <body>
    <h1 class="ps-title">WO2025162428 - Cyclic peptide KRAS inhibitors</h1>
    <section class="abstract">A class of cyclic peptides that bind KRAS-G12V.</section>
    <div>Applicant: ACME Pharma, Inc.; Bio Holdings LLC</div>
    <div>Publication date: 2025-08-14</div>
    <a href="//patentscope.wipo.int/patent/WO2025162428/pdf">Download PDF</a>
  </body>
</html>
"""


def test_parse_extracts_title_applicants_abstract() -> None:
    doc = parse_patent_html(SAMPLE_HTML, patent_id="WO2025162428",
                            source_url="https://patentscope.wipo.int/search/en/WO2025162428")
    assert doc.patent_id == "WO2025162428"
    assert "KRAS inhibitors" in doc.title
    assert "cyclic peptides" in doc.abstract.lower()
    assert doc.applicants == ["ACME Pharma, Inc.", "Bio Holdings LLC"]
    assert doc.publication_date is not None
    assert doc.publication_date.isoformat() == "2025-08-14"
    assert doc.pdf_url is not None
    assert doc.pdf_url.startswith("https:")


def test_parse_handles_missing_fields_gracefully() -> None:
    html = "<html><body><h1>WO1 - Something</h1></body></html>"
    doc = parse_patent_html(html, patent_id="WO1", source_url="https://x")
    assert doc.title == "Something"
    assert doc.applicants == []
    assert doc.publication_date is None
    assert doc.pdf_url is None
