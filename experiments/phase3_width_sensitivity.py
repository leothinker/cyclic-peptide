"""Why do 1040/1041 fail while 1038/1039 succeed on the same page?

The validated OCR route segments each printed row with ``ink_bands``,
trims the vertical table rules with ``text_span`` and recognises the
remaining strip with the upstream recogniser in recognition-only mode.

On patent page I100298 that works perfectly for 1038 and 1039 but
mis-reads 1040 and 1041.  The row geometry is identical in all four
cases; only the *width* of the strip that reaches the recogniser
differs, because the pipeline crop and the phase-2 crop disagree on
the right-hand boundary by a handful of pixels.

This script reproduces the difference and tests three candidate
mechanisms:

1. the recogniser normalises its input to a fixed height
   (``resize_norm_img`` in rapidocr), so a few pixels of width change
   the effective glyph size;
2. the angle classifier (``use_cls``) may be rotating the strip;
3. the CTC decoder may be running out of time steps for very wide
   strips, because the width after normalisation grows with the strip
   aspect ratio.

Nothing here selects a candidate by mass gate, DECIMER or RDKit
parseability -- the only reference is the researcher-supplied 1038
string, and for 1040/1041 we only inspect the printed characters.
"""

from __future__ import annotations

import csv
import json
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
    upscale,
)

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"

COMPOUNDS = ("1038", "1039", "1040", "1041")

# Researcher-supplied verbatim SMILES for 1038 (used as the only
# golden reference in this script).
REFERENCE_1038 = (
    r"CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
    r"[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
    r"(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
    r"[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C"
)


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def cer(observed: str, reference: str) -> float:
    if not reference:
        return float("nan")
    return edit_distance(observed, reference) / len(reference)


def rows_of(cell: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Return (top, bottom, x1, x2, strip) for every printed row."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    result = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        result.append((top, bottom, x1, x2, row[:, x1 : x2 + 1]))
    return result


def recognise(engine, strip: np.ndarray, *, use_cls: bool = True) -> str:
    result = engine(
        upscale(strip), use_det=False, use_rec=True, use_cls=use_cls
    )
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()

    rows = []
    for cmpd in COMPOUNDS:
        path = CELLS / f"I100298_{cmpd}_text.png"
        cell = cv2.imread(str(path))
        if cell is None:
            print(f"missing {path}")
            continue
        bands = rows_of(cell)
        print(
            f"{cmpd}: cell={cell.shape[1]}x{cell.shape[0]} "
            f"bands={len(bands)} "
            + " ".join(
                f"[{t}-{b} x{x1}-{x2} w{x2 - x1 + 1}]" for t, b, x1, x2, _ in bands
            )
        )

        joined_plain = "".join(
            recognise(engine, strip) for _, _, _, _, strip in bands
        )
        joined_nocls = "".join(
            recognise(engine, strip, use_cls=False)
            for _, _, _, _, strip in bands
        )

        # Width sweep on the first band: how does the recognised text
        # react when the strip is trimmed by a few pixels?
        sweep = []
        top, bottom, x1, x2, strip = bands[0]
        for trim in range(0, 13):
            cut = strip[:, : max(1, strip.shape[1] - trim)]
            text = recognise(engine, cut)
            sweep.append(
                {
                    "cmpd_id": cmpd,
                    "band": 1,
                    "trim_px": trim,
                    "strip_width": cut.shape[1],
                    "text": text,
                }
            )

        row = {
            "cmpd_id": cmpd,
            "cell_width": cell.shape[1],
            "cell_height": cell.shape[0],
            "n_bands": len(bands),
            "band_geometry": ";".join(
                f"{t}-{b}:{x1}-{x2}" for t, b, x1, x2, _ in bands
            ),
            "joined_cls": joined_plain,
            "joined_nocls": joined_nocls,
            "edit_1038_cls": (
                edit_distance(joined_plain, REFERENCE_1038)
                if cmpd == "1038"
                else ""
            ),
            "cer_1038_cls": (
                cer(joined_plain, REFERENCE_1038) if cmpd == "1038" else ""
            ),
            "edit_1038_nocls": (
                edit_distance(joined_nocls, REFERENCE_1038)
                if cmpd == "1038"
                else ""
            ),
        }
        rows.append(row)

        for entry in sweep:
            entry["cmpd_id"] = cmpd
        with (OUT / "width_sweep.csv").open(
            "a", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(sweep[0].keys())
            )
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(sweep)

    with (OUT / "cmpd_rows.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "cmpd_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
