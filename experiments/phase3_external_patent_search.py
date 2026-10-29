"""Look for a *second* patent to test the OCR route on.

The golden tables all come from one patent, so every number measured
so far is a within-patent number.  A generalisation claim needs a
different document.

The International Search Report of WO2025/162428 lists the applicant's
own earlier applications (WO2022/234852, WO2022/234853, WO2024/101402,
...), which are the most likely candidates: same applicant, same
medicinal-chemistry table layout, same printed-SMILES convention.

This script queries the public APIs that are reachable from the build
environment and reports what it can get.  It is a *discovery* tool:
whether a download succeeds decides the next step, and every failure
is recorded rather than hidden.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "external"

USER_AGENT = "Mozilla/5.0 (compatible; cyclic-peptide/1.0)"

# Applications cited in the search report of WO2025/162428.
CANDIDATES = [
    ("WO2022234852A1", "2022234852"),
    ("WO2022234853A1", "2022234853"),
    ("WO2024101402A1", "2024101402"),
    ("WO2024101386A1", "2024101386"),
    ("WO2023214576A1", "2023214576"),
]

PDF_MIRRORS = [
    "https://patentimages.storage.googleapis.com/pdfs/{num}.pdf",
    "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{num}",
]


def fetch(url: str, timeout: int = 30) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def try_patentview(number: str) -> dict | None:
    query = urllib.parse.quote(
        json.dumps({"patent_number": number}), safe=""
    )
    url = f"https://api.patentsview.org/patents/query?q={query}"
    try:
        status, body = fetch(url)
    except Exception as exc:  # network failures are expected here
        print(f"    patentsview error: {exc}")
        return None
    if status != 200:
        print(f"    patentsview status {status}")
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        print("    patentsview returned non-JSON")
        return None


def try_pdf(number: str) -> tuple[str | None, int]:
    """Try the known public PDF mirrors for a publication number."""
    digits = number.replace("WO", "")
    for template in PDF_MIRRORS:
        url = template.format(num=digits)
        try:
            status, body = fetch(url, timeout=60)
        except Exception as exc:
            print(f"    {url} -> {exc}")
            continue
        if status == 200 and body[:4] == b"%PDF":
            return url, len(body)
        print(f"    {url} -> status {status}, {len(body)} bytes")
    return None, 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []

    for pub, digits in CANDIDATES:
        print(f"== {pub} ==")
        meta = try_patentview(digits)
        url, size = try_pdf(digits)
        report.append(
            {
                "publication": pub,
                "patentsview_ok": meta is not None,
                "pdf_url": url,
                "pdf_bytes": size,
            }
        )
        if meta is not None:
            (OUT / f"patentsview_{digits}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    (OUT / "external_patent_search.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'external_patent_search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
