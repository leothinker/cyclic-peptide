"""Render candidate table pages out of the source PDF.

The golden tables are full-width crops of the PDF pages, so the PDF
itself is a source of *additional* table pages that the pipeline has
never seen -- a generalisation test that needs no new patent and no
new reference strings.

Usage:
    python experiments/phase3_render_pdf_pages.py 240 241 242
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "external" / "rendered"

PDF = (
    REPO
    / "data"
    / "raw"
    / "WO2025162428"
    / "WO2025162428-PAMPH-20250807-0047.pdf"
)

# The stored golden tables are 1913 px wide.
TARGET_WIDTH = 1913


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: phase3_render_pdf_pages.py <page> [page ...]")
        return 1

    import pypdfium2 as pdfium

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF))

    for raw in argv:
        number = int(raw)
        page = pdf[number - 1]
        width, height = page.get_size()
        scale = TARGET_WIDTH / width
        image = page.render(scale=scale).to_pil().convert("RGB")
        target = OUT / f"pdf_page_{number:04d}.png"
        image.save(target)
        print(f"page {number}: {image.size} -> {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
