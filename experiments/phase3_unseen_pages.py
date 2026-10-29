"""Find patent pages with characterisation tables that are *not* in the
golden-table set.

The 103 golden tables under ``data/raw/WO2025162428/golden_tables``
are the pages the pipeline has been tuned on.  The source PDF has 270
pages and no text layer (it is a scan), so the search is done on the
rendered pixels: a characterisation table is a page with long full-width
horizontal rules and long vertical rules.

Pages whose render matches the golden tables closely are reported
separately from the rest, so the candidates for a generalisation test
are easy to spot.  The golden images are numbered I100280..I100382 in
source order, which lets the script align them against the PDF.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "external"

PDF = (
    REPO
    / "data"
    / "raw"
    / "WO2025162428"
    / "WO2025162428-PAMPH-20250807-0047.pdf"
)
GOLDEN = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"


def table_score(gray: np.ndarray) -> dict[str, float]:
    """Measure how table-like a page is."""
    binary = (gray < 200).astype(np.uint8)
    h, w = gray.shape

    # Long horizontal rules.
    horiz = binary.sum(axis=1)
    long_h = int((horiz > w * 0.4).sum())

    # Long vertical rules.
    vert = binary.sum(axis=0)
    long_v = int((vert > h * 0.25).sum())

    # A table also has a lot of ink arranged in rows.
    ink = float(binary.mean())

    return {
        "long_horizontal": float(long_h),
        "long_vertical": float(long_v),
        "ink_fraction": round(ink, 5),
    }


def main() -> int:
    import pypdfium2 as pdfium

    OUT.mkdir(parents=True, exist_ok=True)

    golden_ids = sorted(
        int(m.group(1))
        for p in GOLDEN.glob("*.jpg")
        if (m := re.search(r"I(\d+)", p.name))
    )
    print(
        f"golden tables: {len(golden_ids)} images, "
        f"I{min(golden_ids)}..I{max(golden_ids)}"
    )

    pdf = pdfium.PdfDocument(str(PDF))
    rows = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.0)
        image = bitmap.to_pil().convert("L")
        gray = np.asarray(image)
        score = table_score(gray)
        rows.append({"pdf_page": index + 1, **score})

    # A table page has several long rules; prose pages have none.
    table_pages = [
        r
        for r in rows
        if r["long_horizontal"] >= 3 and r["long_vertical"] >= 2
    ]
    print(f"\npages that look like tables: {len(table_pages)}")
    for r in table_pages:
        print(
            f"  page {r['pdf_page']:4d}: "
            f"h={r['long_horizontal']:.0f} v={r['long_vertical']:.0f} "
            f"ink={r['ink_fraction']:.4f}"
        )

    (OUT / "pdf_table_pages.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "pdf_table_pages_only.json").write_text(
        json.dumps(table_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
