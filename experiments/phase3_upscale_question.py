"""Why can upscaling make the recogniser *worse*?

This answers a question that came up when the phase-3 numbers were
shown: "if 2x upscaling helps, why does the same recogniser do better
on the un-upscaled crop?"

The answer is in the recogniser's own resize step.  ``RapidOCR``'s
``PP-OCRv6_rec_small`` is configured with ``rec_img_shape: [3, 48,
320]``, and ``ch_ppocr_rec/main.py::resize_norm_img`` does:

    img_height  = 48
    resized_w   = ceil(img_height * w / h)
    cv2.resize(img, (resized_w, 48))

So the recogniser *always* normalises the strip to a height of 48
pixels, no matter what height it is handed.

* A 30 px printed line handed over directly is **upscaled** by the
  recogniser from 30 -> 48.  One interpolation, from the original
  pixels.
* The same line upscaled 2x by us (30 -> 60) is **downscaled** by the
  recogniser from 60 -> 48.  Two interpolations, and the second one
  happens at the recogniser's own sampling grid, which is where thin
  strokes such as the two bars of ``=`` and the ring of ``@`` can be
  merged or dropped.

Upscaling therefore only helps when it changes *what the recogniser
sees*; if the recogniser would have scaled the strip up anyway, the
extra step can only add aliasing.

This script measures the effect on the validated route itself, with
everything else held fixed: the same ink-band crop, the same
recogniser, only the pre-upscale factor changes.  It also records the
recogniser's internal resize target for each strip so the mechanism
is visible in the numbers rather than only asserted.

Read-only.  Writes to ``data/audit/phase3/upscale_question/``.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    default_recogniser,
    ink_bands,
    text_span,
)

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "upscale_question"

REC_HEIGHT = 48  # rec_img_shape[1] from rapidocr/config.yaml

REFERENCES = {
    "1036": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)C(=O)N(C)"
        "[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)C(F)(F)F)C(=O)"
        "N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)"
        "C(=O)N(C)C"
    ),
    "1037": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=CC=C2)C(=O)N3[C@@H](CC3)C(=O)"
        "N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)C(F)(F)F)C(=O)"
        "N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)"
        "C(=O)N(C)C"
    ),
}

TARGETS = {
    "1036": "I100297_1036_text",
    "1037": "I100297_1037_text",
    "1038": "I100298_1038_text",
    "2173": "I100376_2173_text",
}

SCALES = (1, 2, 3)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, iterative two-row form."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def recognise(engine, strip: np.ndarray) -> str:
    result = engine(strip, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()
    rows: list[dict[str, object]] = []

    for cmpd_id, stem in TARGETS.items():
        path = CELLS / f"{stem}.png"
        image = cv2.imread(str(path))
        if image is None:
            print(f"{cmpd_id}: no crop at {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = ink_bands(gray)
        reference = REFERENCES.get(cmpd_id)

        for scale in SCALES:
            parts: list[str] = []
            strip_heights: list[int] = []
            for top, bottom in bands:
                y1 = max(0, top - PAD_TOP)
                y2 = min(image.shape[0], bottom + PAD_BOTTOM)
                row = image[y1:y2, :]
                row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
                x1, x2 = text_span(row_gray)
                variant = row[:, x1:x2 + 1]
                if variant.size == 0:
                    continue
                if scale != 1:
                    variant = cv2.resize(
                        variant,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                strip_heights.append(variant.shape[0])
                text = recognise(engine, variant)
                if text:
                    parts.append(text)
            joined = "".join(parts)

            native_heights = sorted(set(strip_heights))

            row = {
                "cmpd_id": cmpd_id,
                "scale": scale,
                "n_rows": len(parts),
                "strip_heights": "/".join(str(h) for h in native_heights),
                "recogniser_scales_down": "yes"
                if native_heights
                and all(h > REC_HEIGHT for h in native_heights)
                else ("no" if native_heights else ""),
                "length": len(joined),
                "bracket_open": joined.count("("),
                "bracket_close": joined.count(")"),
                "edit_vs_reference": edit_distance(joined, reference)
                if reference
                else "",
                "text": joined,
            }
            rows.append(row)
            print(
                f"{cmpd_id} scale={scale}: rows={len(parts)} "
                f"heights={row['strip_heights']} len={len(joined)} "
                f"edit={row['edit_vs_reference']} "
                f"brackets={row['bracket_open']}/{row['bracket_close']}"
            )
        print()

    out_csv = OUT / "upscale_question.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")

    summary = OUT / "SUMMARY.txt"
    lines = [
        "=== Why upscaling can hurt (phase 3) ===",
        "",
        "RapidOCR PP-OCRv6_rec_small: rec_img_shape = [3, 48, 320]",
        "ch_ppocr_rec/main.py::resize_norm_img -> cv2.resize(img, (ceil(48*w/h), 48))",
        "",
        "So the recogniser always normalises the strip to height 48.",
        "A strip we upscale to 60 px is then downscaled back to 48 by the",
        "recogniser; a strip we hand over at 30 px is upscaled to 48 by it.",
        "The first path adds an interpolation step at the recogniser's own",
        "sampling grid, which is where '=' and '@' strokes can be lost.",
        "",
        "--- measured ---",
    ]
    for row in rows:
        lines.append(
            f"{row['cmpd_id']:>5}  scale={row['scale']}  "
            f"strip_h={row['strip_heights']:<12} "
            f"rows={row['n_rows']}  len={row['length']:<4} "
            f"brackets={row['bracket_open']}/{row['bracket_close']}  "
            f"edit={row['edit_vs_reference']}"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
