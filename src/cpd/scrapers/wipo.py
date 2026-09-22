"""WIPO Patentscope scraper.

Pure stdlib + httpx + BeautifulSoup. Falls back gracefully when fields are
missing — WIPO's DOM is not stable, so we try multiple selectors per field.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from cpd.models.patent import PatentDocument

# WIPO Patentscope occasionally tweaks class names; keep a few fallbacks.
_TITLE_SELECTORS = ("h1.ps-title", "h1.title", "title")
_ABSTRACT_SELECTORS = ("section.abstract", "div.abstract", "section#abstract", "div#abstract")
_DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_PDF_HREF_RE = re.compile(r"href=\"([^\"]+\.pdf)\"", re.IGNORECASE)


class WipoFetcher:
    """Fetch patent HTML from patentscope.wipo.int."""

    BASE_URL = "https://patentscope2.wipo.int/search/en"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.DEFAULT_TIMEOUT,
            headers={"User-Agent": "cyclic-peptide/0.1 (+research)"},
            follow_redirects=True,
        )
        self._cache_dir = cache_dir

    def __enter__(self) -> "WipoFetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(self, patent_id: str, *, save_html: bool = True) -> PatentDocument:
        """Download the Patentscope HTML page and return a parsed document."""
        url = f"{self.BASE_URL}/{patent_id}"
        resp = self._client.get(url)
        resp.raise_for_status()
        html = resp.text

        local_html: Path | None = None
        if save_html and self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            local_html = self._cache_dir / f"{patent_id}.html"
            local_html.write_text(html, encoding="utf-8")

        doc = parse_patent_html(html, patent_id=patent_id, source_url=url)
        return doc.model_copy(update={"local_html": local_html})


def parse_patent_html(html: str, *, patent_id: str, source_url: str) -> PatentDocument:
    """Parse Patentscope HTML into a PatentDocument. Public for testing."""
    soup = BeautifulSoup(html, "html.parser")

    title = _first_text(soup, _TITLE_SELECTORS)
    title = _clean_title(title, patent_id)

    abstract = _first_text(soup, _ABSTRACT_SELECTORS)

    applicants = _extract_applicants(soup)

    pub_date = _extract_date(html, soup)

    pdf_url = _extract_pdf_url(soup)

    return PatentDocument(
        patent_id=patent_id,
        title=title,
        abstract=abstract,
        applicants=applicants,
        publication_date=pub_date,
        source_url=source_url,  # type: ignore[arg-type]
        pdf_url=pdf_url,  # type: ignore[arg-type]
    )


def _first_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str:
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return node.get_text(" ", strip=True)
    return ""


def _clean_title(title: str, patent_id: str) -> str:
    # Patentscope pages often store "<id> - Patent Scope" or just the id.
    return title.replace(f"{patent_id} -", "").replace(patent_id, "").strip(" -")


def _extract_applicants(soup: BeautifulSoup) -> list[str]:
    # Look for any element whose label mentions "Applicant".
    for label in soup.find_all(string=re.compile(r"^[\s]*Applicant", re.IGNORECASE)):
        # The value is usually the next sibling or parent's next sibling.
        nxt = getattr(label.parent, "next_sibling", None) or label.find_next(string=True)
        if nxt and str(nxt).strip():
            return [s.strip() for s in str(nxt).split(";") if s.strip()]
    return []


def _extract_date(html: str, soup: BeautifulSoup) -> date | None:
    m = _DATE_RE.search(html)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _extract_pdf_url(soup: BeautifulSoup) -> str | None:
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            href = a["href"]
            return href if href.startswith("http") else f"https:{href}" if href.startswith("//") else href
    return None
