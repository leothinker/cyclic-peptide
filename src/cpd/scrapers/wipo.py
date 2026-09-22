"""WIPO Patentscope scraper.

Two paths:
  - `parse_patent_html(html, ...)` is the pure HTML parser used by tests.
  - `WipoFetcher.fetch(...)` actually retrieves a patent. WIPO serves a JSF
    shell that JS renders client-side, so we drive a headless Chromium via
    Playwright, then parse the rendered DOM with BeautifulSoup. From the
    Documents tab we pick the PAMPH (application pamphlet) PDF and download
    it via the page context (so the JSF session cookie is attached).

Heavy deps (playwright + chromium) are imported lazily inside `fetch` so the
rest of the project can install without them.
"""
from __future__ import annotations

import base64
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx  # noqa: F401  (re-exported for callers using module-level imports)
from bs4 import BeautifulSoup

from cpd.models.patent import PatentDocument

# WIPO Patentscope occasionally tweaks class names; keep a few fallbacks.
_TITLE_SELECTORS = ("h1.ps-title", "h1.title", "h1", "title")
_ABSTRACT_SELECTORS = ("section.abstract", "div.abstract", "section#abstract", "div#abstract")
_DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
# Strip a leading "Applicant:" or "Applicant -" label.
_APPLICANT_LABEL_RE = re.compile(r"^[\s]*Applicant\s*[:\-]\s*", re.IGNORECASE)
# Strip "1. " / "12. " numbering from the rendered PCT Biblio title row.
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\.\s+")
# Country code suffix WIPO attaches: "[US]/[US]" or "[CN]/[CN](BW)".
_COUNTRY_SUFFIX_RE = re.compile(r"^\s*\[[A-Z]{2}\]/\[[A-Z]{2}\](?:\([^)]+\))?\s*$")

# Section labels in the rendered PCT Biblio page.
_BIBLIO_LABELS = {
    "publication number": "publication_number",
    "publication date": "publication_date_raw",
    "international application no": "intl_application_no",
    "international filing date": "intl_filing_date_raw",
    "applicants": "applicants",
    "inventors": "inventors",
    "agents": "agents",
    "title": "title",
    "abstract": "abstract",
    "ipc": "ipc",
    "cpc": "cpc",
    "priority data": "priority",
    "designated states": "designated_states",
}
_EURO_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
# Markers that appear in the rendered page footer after the abstract; we stop
# collecting the abstract buffer at the first match so it does not eat the
# navigation menu / footer text.
_ABSTRACT_STOP_PHRASES = (
    "related patent documents",
    "latest bibliographic data",
    "processing",
    "simple",
    "ai assisted search",
    "advanced search",
    "field combination",
    "cross lingual expansion",
    "browse by week",
    "gazette archive",
    "sequence listing",
    "national phase entries",
    "authority file",
    "wipo translate",
    "wipo pearl",
    "ipc green",
    "support",
    "covid-19 efforts",
    "sustainable development",
    "portal to patent registers",
)
_LANG_LABEL_RE = re.compile(r"^\(\s*[A-Z]{2}\s*\)$")

# Detail page URL (note: not patentscope2, that one is search-only).
_DETAIL_URL = "https://patentscope.wipo.int/search/en/detail.jsf?docId={patent_id}"


class WipoFetcher:
    """Fetch a WIPO patent: rendered HTML + PAMPH PDF."""

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        headless: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._headless = headless

    def __enter__(self) -> "WipoFetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:  # noqa: D401
        return None

    def fetch(self, patent_id: str, *, save_pdf: bool = True) -> PatentDocument:
        from playwright.sync_api import sync_playwright  # heavy dep

        url = _DETAIL_URL.format(patent_id=patent_id)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self._headless)
            try:
                ctx = browser.new_context(
                    user_agent="cyclic-peptide/0.1 (+research)",
                    accept_downloads=True,
                )
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                _wait_for_render(page)

                biblio_html = page.content()
                pdf_url = _open_documents_and_grab_pdf_url(page)

                local_html = _save_text(biblio_html, self._cache_dir, f"{patent_id}.html")
                local_pdf: Path | None = None
                if save_pdf and pdf_url is not None:
                    local_pdf = _download_pdf(page, pdf_url, self._cache_dir, patent_id)
            finally:
                browser.close()

        doc = parse_patent_html(
            biblio_html, patent_id=patent_id, source_url=url, pdf_url=pdf_url,
        )
        return doc.model_copy(update={"local_html": local_html, "local_pdf": local_pdf})


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _wait_for_render(page: Any) -> None:
    """Wait until the body has the actual patent content (not the JSF shell)."""
    page.wait_for_selector("body", timeout=30000)
    page.wait_for_load_state("networkidle", timeout=30000)
    for _ in range(40):
        text = page.eval_on_selector("body", "el => el.innerText.trim()")
        if "CYCLIC" in text or "PEPTIDE" in text or "INHIBITOR" in text or len(text) > 1500:
            return
        page.wait_for_timeout(500)


def _open_documents_and_grab_pdf_url(page: Any) -> str | None:
    """Click the Documents tab and return the PAMPH PDF href (absolute URL)."""
    try:
        page.click("a:has-text('Documents')", timeout=10000)
    except Exception:
        return None
    page.wait_for_load_state("networkidle", timeout=30000)
    page.wait_for_timeout(2500)
    href = page.eval_on_selector(
        "a[href*='PAMPH'][href*='.pdf']",
        "el => el ? el.getAttribute('href') : null",
    )
    if not href:
        href = page.eval_on_selector(
            "a[href*='/pct/'][href*='/pdf/']",
            "el => el ? el.getAttribute('href') : null",
        )
    if not href:
        return None
    return _absolutize(href)


def _absolutize(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://patentscope.wipo.int{href}"
    return href


def _download_pdf(page: Any, pdf_url: str, cache_dir: Path | None, patent_id: str) -> Path | None:
    """Fetch the PDF inside the page context so the JSF session cookie is used."""
    b64 = page.evaluate(
        "async (url) => {"
        "  const r = await fetch(url, { credentials: 'include' });"
        "  if (!r.ok) return null;"
        "  const buf = await r.arrayBuffer();"
        "  let bin = '';"
        "  const u8 = new Uint8Array(buf);"
        "  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);"
        "  return btoa(bin);"
        "}",
        pdf_url,
    )
    if not b64:
        return None
    body = base64.b64decode(b64)
    if len(body) < 1024 or not body.startswith(b"%PDF"):
        # Got HTML instead (likely a session-expired redirect).
        return None
    if cache_dir is None:
        return None
    target = cache_dir / f"{patent_id}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return target


def _save_text(content: str, cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    target = cache_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Pure-HTML parser (testable without playwright)
# ---------------------------------------------------------------------------

def parse_patent_html(
    html: str,
    *,
    patent_id: str,
    source_url: str,
    pdf_url: str | None = None,
) -> PatentDocument:
    """Parse WIPO HTML into a PatentDocument. Public for testing.

    Two layouts are supported:
      (a) rendered PCT Biblio (label/value grid) - what playwright produces
      (b) hand-crafted samples with `.ps-title` / `section.abstract` etc.

    Pass `pdf_url` when the caller already extracted it (e.g. from the
    Documents tab); otherwise the parser will try to find one in the HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    biblio = _parse_biblio(soup)
    title = biblio.get("title") or _clean_title(_first_text(soup, _TITLE_SELECTORS), patent_id)
    abstract = biblio.get("abstract") or _first_text(soup, _ABSTRACT_SELECTORS)
    applicants = biblio.get("applicants") or _extract_applicants(soup)
    pub_date = _parse_iso_date(biblio.get("publication_date_raw")) or _extract_date(html)
    final_pdf_url = pdf_url or _extract_pdf_url(soup)

    return PatentDocument(
        patent_id=patent_id,
        title=title or "",
        abstract=abstract or "",
        applicants=applicants,
        publication_date=pub_date,
        source_url=source_url,  # type: ignore[arg-type]
        pdf_url=final_pdf_url,  # type: ignore[arg-type]
    )


def _parse_biblio(soup: BeautifulSoup) -> dict[str, Any]:
    """Walk the rendered PCT Biblio grid and pull labelled fields."""
    body_text = soup.get_text("\n", strip=True)
    lines = [ln for ln in body_text.split("\n") if ln.strip()]
    out: dict[str, Any] = {}
    current_label: str | None = None
    buffer: list[str] = []
    known_labels = set(_BIBLIO_LABELS) | {
        "applicant", "inventor", "agent",
        "international application number",
        "priority number", "designated state",
    }
    for ln in lines:
        key = ln.lower().rstrip(":")
        if key in known_labels:
            if current_label is not None:
                out[_BIBLIO_LABELS.get(current_label, current_label)] = _post_process(current_label, buffer)
            current_label = key
            buffer = []
        else:
            buffer.append(ln)
    if current_label is not None:
        out[_BIBLIO_LABELS.get(current_label, current_label)] = _post_process(current_label, buffer)
    return out


def _post_process(label: str, values: list[str]) -> Any:
    if label.startswith("applicant"):
        return _merge_country_suffixes(values)
    if label.startswith("inventor"):
        return [v.strip() for v in values if v.strip()]
    if label.startswith("title") or label.startswith("abstract"):
        return _extract_lang_section(values, stop_phrases=_ABSTRACT_STOP_PHRASES)
    return " ".join(values).strip()


def _extract_lang_section(values: list[str], *, stop_phrases: tuple[str, ...]) -> str:
    """Pick the English block out of a `(EN) ... (FR) ...` rendering.

    Also stops collecting at the first footer-menu phrase so the abstract
    does not eat the rest of the page.
    """
    stop_set = {p.lower() for p in stop_phrases}
    kept: list[str] = []
    started = False
    for v in values:
        low = v.lower().strip()
        if not started:
            if low == "(en)":
                started = True
            continue
        if low in stop_set or any(low.startswith(p) for p in stop_set if len(p) >= 4):
            break
        if _LANG_LABEL_RE.match(v.strip()):
            break
        kept.append(v)
    joined = " ".join(kept).strip()
    joined = _LEADING_NUMBER_RE.sub("", joined)
    return joined


def _merge_country_suffixes(lines: list[str]) -> list[str]:
    """Fold lines that are just country-code suffixes into the previous line."""
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if _COUNTRY_SUFFIX_RE.match(ln) and out:
            out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    return out


def _parse_iso_date(raw: str | None) -> date | None:
    if not raw:
        return None
    m = _EURO_DATE_RE.search(raw)
    if m:
        try:
            d_, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mth, d_)
        except ValueError:
            return None
    return _extract_date(raw)


def _first_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str:
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return node.get_text(" ", strip=True)
    return ""


def _clean_title(title: str, patent_id: str) -> str:
    cleaned = _LEADING_NUMBER_RE.sub("", title)
    return cleaned.replace(f"{patent_id} -", "").replace(patent_id, "").strip(" -")


def _extract_applicants(soup: BeautifulSoup) -> list[str]:
    for label in soup.find_all(string=re.compile(r"^[\s]*Applicant", re.IGNORECASE)):
        text = label.parent.get_text(" ", strip=True)
        cleaned = _APPLICANT_LABEL_RE.sub("", text, count=1)
        if cleaned:
            return [s.strip() for s in cleaned.split(";") if s.strip()]
        sibling_el = label.parent.find_next_sibling()
        if sibling_el is not None:
            sib_text = sibling_el.get_text(" ", strip=True)
            if sib_text:
                return [s.strip() for s in sib_text.split(";") if s.strip()]
    return []


def _extract_date(html: str) -> date | None:
    m = _DATE_RE.search(html)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _extract_pdf_url(soup: BeautifulSoup) -> str | None:
    """Pick the best PDF link from the rendered HTML.

    Skips ftp:// links (WIPO exposes FTP bulk-download links that are not
    the patent itself). Prefers explicit .pdf hrefs.
    """
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or not (href.lower().endswith(".pdf") or "/pdf" in href.lower()):
            continue
        if not href.startswith(("http://", "https://", "//", "/")):
            continue
        if href.startswith("ftp://"):
            continue
        candidates.append(href)
    if not candidates:
        return None
    for href in candidates:
        if href.lower().endswith(".pdf"):
            return _absolutize(href)
    return _absolutize(candidates[0])


__all__ = ["WipoFetcher", "parse_patent_html"]
