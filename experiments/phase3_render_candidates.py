"""Render the PDF's table pages and match them against the golden set.

The 103 golden tables were cropped out of this PDF, so every table page
in the PDF should match one of them.  Matching by image content tells
us which pages the golden set covers and, more usefully, which table
pages were *left out* -- those are the free generalisation test for
the OCR route.

Matching is done on a downscaled grayscale version with normalised
cross-correlation, which is robust to the small crops and scaling
differences between the PDF render and the stored JPEGs.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
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


def signature(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Small grayscale signature of a page, for fast matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.float32
    )


def main() -> int:
    import pypdfium2 as pdfium

    OUT.mkdir(parents=True, exist_ok=True)

    golden = []
    for path in sorted(GOLDEN.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        match = re.search(r"I(\d+)", path.name)
        golden.append(
            {
                "name": path.name,
                "id": int(match.group(1)) if match else -1,
                "sig": signature(image, (64, 64)),
                "shape": image.shape[:2],
            }
        )
    print(f"golden tables loaded: {len(golden)}")

    pdf = pdfium.PdfDocument(str(PDF))
    rows = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.5)
        image = np.asarray(bitmap.to_pil().convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sig = signature(image, (64, 64))

        best = None
        for entry in golden:
            score = float(
                np.mean(sig * entry["sig"])
                / (np.linalg.norm(sig) * np.linalg.norm(entry["sig"]) + 1e-9)
            )
            if best is None or score > best[1]:
                best = (entry, score)

        rows.append(
            {
                "pdf_page": index + 1,
                "best_golden": best[0]["name"],
                "best_golden_id": best[0]["id"],
                "score": round(best[1], 4),
                "page_shape": list(image.shape[:2]),
            }
        )
        print(
            f"  page {index + 1:4d}: best={best[0]['name']:45s} "
            f"score={best[1]:.4f}"
        )

    (OUT / "pdf_to_golden_match.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'pdf_to_golden_match.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
