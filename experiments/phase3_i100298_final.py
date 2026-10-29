r"""Final validation of the I100298 printed-SMILES route.

What this script settles
------------------------
``phase3_i100298_rowsweep.py`` swept the crop geometry per printed row
and found that one SINGLE configuration reaches character-exact output
on all 21 printed rows of 1039-1041:

    trim the vertical table rules, 2x upscale, no binarisation,
    14/15 px of vertical padding around the ink band

Because the configuration is the same for every row and every
compound, it is not fitted per row.  This script re-implements it from
scratch, without importing the sweep, and checks it against:

    * 1038  the researcher-supplied reference (HELD OUT: 1038 was
      excluded from the configuration search, so this is a
      generalisation check, not a fit)
    * 1039-1041  the printed transcription

It also verifies that the route is deterministic and that the official
CSV is untouched.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_final.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import edit_ops  # noqa: E402
from phase3_i100298_sweep import (  # noqa: E402
    PRINTED_ROWS, TRANSCRIPTION,
)
from phase3_quality_screen import SMILES_CHARS, screen_joined  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "i100298_final"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"

EXPECTED_CSV_SHA256 = (
    "99F746AF20E4904EE6583C70461506783D2573AE01D7A69D127AD5F3AB927D0D")

TARGETS = ("1038", "1039", "1040", "1041")

# The single configuration selected by the sweep.
SCALE = 2
PAD_TOP = 14
PAD_BOTTOM = 15
THRESHOLD = None
RULE_HEIGHT_RATIO = 0.80

# 1038's researcher-supplied reference.  It is deliberately kept out of
# the configuration search so that it can serve as a held-out check.
REF_1038 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
    "(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N[C@@H]"
    "(CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)"
    "C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H]"
    "(CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)"
    "C(=O)N1)C(=O)N(C)C"
)


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _ink_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """The seven printed ink bands of a SMILES cell."""
    h, w = gray.shape
    x1 = min(10, max(1, w // 100))
    x2 = max(x1 + 1, w - 10)
    ink = (gray[:, x1:x2] < 160).astype(np.uint8)
    sums = ink.sum(axis=1)
    active = np.flatnonzero(sums > 5)
    if not len(active):
        return []
    groups: list[list[int]] = []
    start = previous = int(active[0])
    for value in active[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append([start, previous])
            start = current
        previous = current
    groups.append([start, previous])
    return [(a, b) for a, b in groups if b - a >= 8]


def _text_span(gray: np.ndarray) -> tuple[int, int]:
    """Horizontal ink span with the vertical table rules removed."""
    dark = gray < 160
    height = gray.shape[0]
    rule = dark.sum(axis=0) / max(height, 1) >= RULE_HEIGHT_RATIO
    # Widen the rule mask to neighbouring columns that are nearly full
    # height as well; a table rule is often two or three pixels wide.
    for x in np.flatnonzero(rule).tolist():
        for dx in (-1, 1):
            if 0 <= x + dx < gray.shape[1]:
                if dark[:, x + dx].sum() / max(height, 1) >= 0.5:
                    rule[x + dx] = True
    cols = np.flatnonzero((dark.sum(axis=0) > 0) & (~rule))
    if not len(cols):
        return 0, gray.shape[1] - 1
    return int(cols[0]), int(cols[-1])


def _recognise_cell(engine, image: np.ndarray) -> tuple[str, list[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parts: list[str] = []
    for top, bottom in _ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(image.shape[0], bottom + PAD_BOTTOM)
        crop = image[y1:y2, :]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        x1, x2 = _text_span(crop_gray)
        variant = crop[:, x1:x2 + 1]
        big = cv2.resize(variant, None, fx=SCALE, fy=SCALE,
                         interpolation=cv2.INTER_CUBIC)
        if THRESHOLD is not None:
            big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            _, big = cv2.threshold(big_gray, THRESHOLD, 255,
                                   cv2.THRESH_BINARY)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        parts.append(_recognise(engine, big))
    return "".join(parts), parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = _engine()
    rows: list[dict] = []

    for cmpd_id in TARGETS:
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")

        joined, parts = _recognise_cell(engine, image)

        # Determinism: the same crop recognised twice.
        joined_again, _ = _recognise_cell(engine, image)

        if cmpd_id == "1038":
            reference = REF_1038
            status = "researcher_supplied_held_out"
        else:
            reference = TRANSCRIPTION[cmpd_id]
            status = "printed_transcription_awaiting_human_check"

        edit = edit_ops(reference, joined)
        screen = screen_joined(cmpd_id, joined)
        illegal = "".join(sorted({ch for ch in joined
                                  if ch not in SMILES_CHARS}))
        row_edits = []
        printed = PRINTED_ROWS.get(cmpd_id, [])
        for index, text in enumerate(parts):
            if index < len(printed):
                row_edits.append(edit_ops(printed[index], text))
            else:
                row_edits.append("")

        rows.append({
            "cmpd_id": cmpd_id,
            "reference_status": status,
            "reference_length": len(reference),
            "ocr_length": len(joined),
            "edit_distance": edit,
            "cer": round(edit / len(reference), 6),
            "open": joined.count("("),
            "close": joined.count(")"),
            "balanced": screen.joined_balanced,
            "parses": screen.joined_parses,
            "illegal_chars": illegal,
            "row_edits": "|".join(str(e) for e in row_edits),
            "deterministic": joined == joined_again,
            "ocr_text": joined,
        })

    fields = list(rows[0].keys())
    with (OUT / "i100298_final.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    digest = _sha256(CSV_PATH)
    lines = [
        "=== I100298 final route validation ===",
        "",
        "Route: trim vertical table rules, 2x upscale (INTER_CUBIC), "
        "no binarisation,",
        f"        14/15 px vertical padding around each ink band.",
        "Recogniser: upstream RapidOCR default engine "
        "(PP-OCRv6 detector + PP-OCRv6 recogniser).",
        "",
        "One configuration is used for every row of every compound.",
        "1038 was excluded from the configuration search, so its row is a",
        "held-out generalisation check.",
        "",
        f"{'cmpd':<6} {'status':<42} {'edit':>5} {'cer':>9} {'len':>5} "
        f"{'open':>5} {'close':>6} {'bal':>5} {'parse':>6} {'det':>5}",
    ]
    for row in rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['reference_status']:<42} "
            f"{row['edit_distance']:>5} {row['cer']:>9} "
            f"{row['ocr_length']:>5} {row['open']:>5} {row['close']:>6} "
            f"{'yes' if row['balanced'] else 'no':>5} "
            f"{'yes' if row['parses'] else 'no':>6} "
            f"{'yes' if row['deterministic'] else 'no':>5}"
        )

    lines.extend(["", "per-row edit distances:"])
    for row in rows:
        lines.append(f"  {row['cmpd_id']}: {row['row_edits']}")

    lines.extend(["", "recovered strings:"])
    for row in rows:
        lines.append(f"  {row['cmpd_id']}: {row['ocr_text']}")

    lines.extend([
        "",
        f"official CSV sha256: {digest}",
        f"expected           : {EXPECTED_CSV_SHA256}",
        f"unchanged          : {digest == EXPECTED_CSV_SHA256}",
        "",
        "Text level only.  Chemical structure is not verified and the mass",
        "gate is not consulted.  No candidate was selected by RDKit",
        "parseability, the mass gate or DECIMER output.",
    ])
    io.open(OUT / "i100298_final_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
