"""Make the printed-SMILES route independent of the crop width.

Findings so far (I100298):

* the phase-2 text crop and the pipeline's own crop are pixel-identical
  over the overlapping columns; the pipeline crop is simply 20 px
  narrower, which clips the tail of the longest rows;
* the recogniser is *not* stable with respect to the strip width: the
  same ink, padded to different widths, decodes to different strings,
  and trimming six pixels from a correct strip can turn it into
  garbage while trimming seven pixels is fine again.

The second point is the important one.  ``rapidocr`` normalises its
input with ``resize_norm_img``: the height is forced to 48 and the
width is recomputed from the aspect ratio, so a strip that is a few
pixels wider ends up with slightly different glyph sizes and the CTC
decoder can fall off its peak.

This script tests the obvious fix: **normalise the geometry ourselves
before handing the strip to the recogniser**.  The strip is resized so
that its height is exactly the recogniser's target height, and the
width is rounded to a multiple of a fixed quantum.  The recogniser
then receives an input whose geometry does not depend on how wide the
original crop happened to be.

Arms:

``baseline``    the current route (2x upscale, recogniser normalises)
``height48``    resize to height 48 ourselves, then recognise
``height48_q8`` resize to height 48 and round the width to a multiple
                of 8, then recognise
``chunked``     split the strip into fixed-width chunks, recognise each
                independently and concatenate

Evaluation is character level against the researcher-supplied 1038
reference and against the phase-3 transcriptions for 1039-1041.  No
mass gate, no DECIMER, no RDKit parseability is used to pick a winner.
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
OUT = REPO / "data" / "audit" / "phase3" / "geometry_fix"

TARGET_HEIGHT = 48
QUANTUM = 8
CHUNK_WIDTH = 320

REFERENCES: dict[str, tuple[str, str]] = {
    "1038": (
        "researcher_supplied_held_out",
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C",
    ),
    "1039": (
        "phase3_transcription_awaiting_human_check",
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N9CCCC9",
    ),
    "1040": (
        "phase3_transcription_awaiting_human_check",
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ),
    "1041": (
        "phase3_transcription_awaiting_human_check",
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CCCC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ),
}


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


def strips_of(cell: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append(row[:, x1 : x2 + 1])
    return out


def recognise(engine, image: np.ndarray) -> str:
    result = engine(image, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def normalise_height(strip: np.ndarray, height: int = TARGET_HEIGHT) -> np.ndarray:
    """Resize a strip so its height is exactly ``height``."""
    h, w = strip.shape[:2]
    if h == height:
        return strip
    width = max(1, int(round(w * height / h)))
    return cv2.resize(
        strip, (width, height), interpolation=cv2.INTER_CUBIC
    )


def quantise_width(strip: np.ndarray, quantum: int = QUANTUM) -> np.ndarray:
    """Round the width down to a multiple of ``quantum``."""
    h, w = strip.shape[:2]
    width = max(quantum, (w // quantum) * quantum)
    if width == w:
        return strip
    return cv2.resize(
        strip, (width, h), interpolation=cv2.INTER_CUBIC
    )


def recognise_chunked(engine, strip: np.ndarray) -> str:
    """Recognise fixed-width chunks and concatenate the results."""
    h, w = strip.shape[:2]
    parts = []
    for start in range(0, w, CHUNK_WIDTH):
        chunk = strip[:, start : start + CHUNK_WIDTH]
        if chunk.shape[1] == 0:
            continue
        text = recognise(engine, normalise_height(chunk))
        if text:
            parts.append(text)
    return "".join(parts)


ARMS = {
    "baseline": lambda engine, strip: recognise(
        engine, upscale(strip)
    ),
    "height48": lambda engine, strip: recognise(
        engine, normalise_height(upscale(strip))
    ),
    "height48_q8": lambda engine, strip: recognise(
        engine, quantise_width(normalise_height(upscale(strip)))
    ),
    "chunked": lambda engine, strip: recognise_chunked(engine, strip),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()

    rows = []
    for cmpd, (status, reference) in REFERENCES.items():
        cell = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
        strips = strips_of(cell)
        for arm, fn in ARMS.items():
            joined = "".join(fn(engine, strip) for strip in strips)
            edits = edit_distance(joined, reference)
            rows.append(
                {
                    "cmpd_id": cmpd,
                    "arm": arm,
                    "reference_status": status,
                    "ref_len": len(reference),
                    "ocr_len": len(joined),
                    "edit": edits,
                    "cer": round(edits / len(reference), 6),
                    "text": joined,
                }
            )
            print(
                f"{cmpd} {arm:14s} edit={edits:3d} "
                f"cer={edits / len(reference):.5f} len={len(joined)}"
            )

    with (OUT / "arms.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "arms.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- stability under crop perturbation -------------------------
    # The decisive test: trim the strip by 0..12 px and check whether
    # the arm's output changes.  A robust arm must not flip to garbage.
    stability = []
    for cmpd in REFERENCES:
        cell = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
        strips = strips_of(cell)
        for arm, fn in ARMS.items():
            for trim in range(0, 13):
                joined = ""
                for strip in strips:
                    cut = strip[:, : max(1, strip.shape[1] - trim)]
                    joined += fn(engine, cut)
                stability.append(
                    {
                        "cmpd_id": cmpd,
                        "arm": arm,
                        "trim_px": trim,
                        "text": joined,
                    }
                )
    with (OUT / "stability.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(stability[0].keys())
        )
        writer.writeheader()
        writer.writerows(stability)

    print("\n=== stability: number of distinct outputs over trims 0..12 ===")
    for cmpd in REFERENCES:
        line = []
        for arm in ARMS:
            sel = [
                r["text"]
                for r in stability
                if r["cmpd_id"] == cmpd and r["arm"] == arm
            ]
            line.append(f"{arm}={len(set(sel))}")
        print(f"  {cmpd}: " + "  ".join(line))

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
