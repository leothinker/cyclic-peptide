"""Try to obtain a *second* patent document for a generalisation test.

Every number so far comes from one patent, so "the pipeline works" is
currently a within-patent claim.  This script tries the public patent
mirrors that are reachable from the build environment and reports
which ones actually deliver a PDF.

It is deliberately a *probe*, not a downloader: it records the HTTP
status and the first bytes of every candidate URL so the outcome can
be read off the log, and it only writes a file when the payload
really is a PDF.

Usage:
    .venv\\Scripts\\python.exe experiments/phase3_fetch_second_patent.py WO2022234852
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "raw" / "_external_validation"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def url_templates(publication: str) -> list[str]:
    """Candidate PDF URLs for a WO publication number."""
    digits = publication.replace("WO", "")
    return [
        # EPO publication server -- serves the full WO document.
        "https://data.epo.org/publication-server/rest/v1.2/patents/"
        f"{publication}NWA1/document.pdf",
        "https://data.epo.org/publication-server/rest/v1.2/patents/"
        f"{publication}NWA1/document.pdf?download=true",
        # WIPO Patentscope document service.
        "https://patentscope.wipo.int/search/docs2/pct/ISR/"
        f"{publication}/pdf/{publication}.pdf",
        # CNIPA / national office copies.
        f"https://patentimages.storage.googleapis.com/pdfs/{digits}.pdf",
        # Espacenet / EPO OPS are JSON-only, but the publication server
        # above sometimes redirects here.
        "https://worldwide.espacenet.com/publicationDetails/originalDocument"
        f"?CC=WO&NR={publication}A1&FT=D&locale=en_EP",
    ]


def probe(url: str, timeout: int = 45) -> tuple[str, bytes]:
    """Return ``(status, body)``; never raises."""
    context = ssl.create_default_context()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,text/html,*/*",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            return str(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}", b""
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}", b""


def main(argv: list[str]) -> int:
    publication = (argv[0] if argv else "WO2022234852").upper()
    OUT.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for url in url_templates(publication):
        print(f"probing {url}")
        status, body = probe(url)
        is_pdf = body[:4] == b"%PDF"
        print(f"    -> {status}, {len(body)} bytes, pdf={is_pdf}")
        results.append(
            {
                "url": url,
                "status": status,
                "bytes": len(body),
                "is_pdf": is_pdf,
            }
        )
        if is_pdf:
            target = OUT / f"{publication}.pdf"
            target.write_bytes(body)
            print(f"    -> saved {target}")
            break

    (OUT / f"fetch_{publication}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / f'fetch_{publication}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
