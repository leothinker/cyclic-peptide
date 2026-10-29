r"""Route sweep for 1039-1041 on page I100298.

Goal
----
1038 is now exact against the researcher-supplied reference.  1039-1041
still have no verified reference, so this script does two things:

1. It re-runs the recognised routes on a set of fixed geometry and
   preprocessing variants, keeping the recogniser fixed, so that the
   printed rows of 1039-1041 can be read as well as 1038 is.
2. It reports every route with bracket balance, length, RDKit
   parseability and a character-level comparison against the reference
   strings transcribed from the printed page.

Reference policy
----------------
The 1039-1041 reference strings below were transcribed by reading the
printed rows at 3x from the original page image (the artefacts in
``data/audit/phase3/i100298_read/``).  They are marked
``printed_transcription_awaiting_human_check``: they are a better
instrument than an unverified OCR candidate, but they are NOT a
human-verified reference and are not treated as one.  Every downstream
number is therefore reported as "agreement with the printed
transcription", not "accuracy".

Selection rule
--------------
No candidate is selected by RDKit parseability, by the mass gate, or by
DECIMER output.  Routes are ranked by character agreement with the
printed transcription, and the ranking is reported so that a disagreement
is visible rather than hidden.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_sweep.py
"""

from __future__ import annotations

import csv
import difflib
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
from phase3_i100298_validation import (  # noqa: E402
    CELLS, TARGETS, _segment_rows,
)
from phase3_quality_screen import SMILES_CHARS, screen_joined  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "i100298_sweep"
RECROP = (REPO / "data" / "audit" / "phase3" / "recrop_margin"
          / "recrop_margin.csv")
PAGE_ROUTE = (REPO / "data" / "audit" / "phase3" / "page_det_line_rec"
              / "page_det_line_rec.csv")

# Reference transcribed from the printed rows at 3x.  See the module
# docstring: these are pending human check, not verified references.
TRANSCRIPTION = {
    "1039": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)"
        "N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)"
        "N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)"
        "C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H]"
        "(CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)"
        "C(=O)N1)C(=O)N9CCCC9"
    ),
    "1040": (
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=O)N(C)[C@@H]"
        "(C4CCCC4)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]"
        "([C@@H](C)CC)C(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)C(=O)"
        "N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)C(=O)N(C)"
        "CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)"
        "N(C)C"
    ),
    "1041": (
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CCCC3)C(=O)N(C)[C@@H]"
        "(C4CCCC4)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]"
        "([C@@H](C)CC)C(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)C(=O)"
        "N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)C(=O)N(C)"
        "CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)"
        "N(C)C"
    ),
}

REFERENCE_STATUS = "printed_transcription_awaiting_human_check"

# The printed rows, read one printed line at a time from the 3x row
# images in ``data/audit/phase3/i100298_read/rows/``.  Recording the
# line breaks matters: the patent cuts the string at arbitrary points,
# so the row boundaries cannot be recovered from the string itself.
# Joining these rows reproduces TRANSCRIPTION exactly, which the module
# asserts at run time.
PRINTED_ROWS = {
    "1039": [
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)",
        "N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)[C@H]3C(=",
        "O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4",
        ")C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCC",
        "C6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@",
        "@H](CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)",
        "=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N9CCCC9",
    ],
    "1040": [
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=",
        "O)N(C)[C@@H](C4CCCC4)C(=O)N(C)[C@@H](CC",
        "(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)C",
        "C)C(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)C(",
        "=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F",
        ")=C7F)C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=",
        "C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ],
    "1041": [
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CCCC3)C",
        "(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)[C@@H](",
        "CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C",
        ")CC)C(=O)N(C)[C@@H](C)C(=O)N6[C@@H](CC6)",
        "C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C",
        "(F)=C7F)C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F",
        ")=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C",
    ],
}

for _cmpd, _rows in PRINTED_ROWS.items():
    assert "".join(_rows) == TRANSCRIPTION[_cmpd], (
        f"{_cmpd}: printed rows do not join to the transcription")

# Recogniser fixed at the upstream default engine (PP-OCRv6 detector +
# PP-OCRv6 recogniser), which is the configuration that reproduced 1036,
# 1037 and 1038 character for character.  Only the crop geometry and the
# binarisation threshold vary.
PREPROCESS = {
    "1x_none": (1, None),
    "2x_none": (2, None),
    "2x_th200": (2, 200),
    "2x_th180": (2, 180),
    "2x_th220": (2, 220),
    "3x_none": (3, None),
    "3x_th200": (3, 200),
}


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _preprocess(crop: np.ndarray, arm: str) -> np.ndarray:
    scale, threshold = PREPROCESS[arm]
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    if threshold is None:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _metrics(cmpd_id: str, text: str) -> dict:
    screen = screen_joined(cmpd_id, text)
    illegal = "".join(sorted({ch for ch in text if ch not in SMILES_CHARS}))
    return {
        "length": len(text),
        "open": text.count("("),
        "close": text.count(")"),
        "balanced": screen.joined_balanced,
        "parses": screen.joined_parses,
        "illegal_chars": illegal,
    }


def _diff(reference: str, observed: str) -> list[str]:
    """Per-character difference list against the printed transcription."""
    out: list[str] = []
    matcher = difflib.SequenceMatcher(a=reference, b=observed)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out.append(
            f"{tag} ref[{i1}:{i2}]={reference[i1:i2]!r} "
            f"obs[{j1}:{j2}]={observed[j1:j2]!r}"
        )
    return out


def _existing_page_routes() -> list[dict]:
    """Candidates produced by earlier page-scale routes, re-scored here.

    These strings already exist; this script does not recompute them.
    They are included because the cell-crop route fails on 1040 while
    the page route succeeds, and the comparison has to be made in one
    table to show that.
    """
    out: list[dict] = []
    if RECROP.exists():
        with RECROP.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["cmpd_id"] not in TRANSCRIPTION:
                    continue
                out.append({
                    "cmpd_id": row["cmpd_id"],
                    "arm": f"page_recrop_margin{row['margin']}",
                    "text": row["text"],
                })
    if PAGE_ROUTE.exists():
        with PAGE_ROUTE.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["cmpd_id"] not in TRANSCRIPTION:
                    continue
                out.append({
                    "cmpd_id": row["cmpd_id"],
                    "arm": f"page_{row['arm']}",
                    "text": row["text"],
                })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = _engine()
    rows: list[dict] = []

    for cmpd_id in TARGETS:
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)

        reference = TRANSCRIPTION.get(cmpd_id, "")
        for arm in PREPROCESS:
            parts: list[str] = []
            for top, bottom in bands:
                y1 = max(0, top - 4)
                y2 = min(image.shape[0], bottom + 5)
                parts.append(_recognise(engine,
                                        _preprocess(image[y1:y2, :], arm)))
            joined = "".join(parts)
            edit = edit_ops(reference, joined) if reference else ""
            rows.append({
                "cmpd_id": cmpd_id,
                "row_type": "result",
                "arm": arm,
                "recogniser": "upstream_default_ppocrv6",
                "reference_status": (REFERENCE_STATUS if reference
                                     else "researcher_supplied"),
                "reference_length": len(reference),
                "edit_distance": edit,
                "cer": (round(edit / len(reference), 6) if reference else ""),
                **_metrics(cmpd_id, joined),
                "ocr_text": joined,
            })
            if reference:
                for line in _diff(reference, joined):
                    rows.append({
                        "cmpd_id": cmpd_id,
                        "row_type": "diff",
                        "arm": arm,
                        "recogniser": "diff",
                        "reference_status": REFERENCE_STATUS,
                        "reference_length": len(reference),
                        "edit_distance": "",
                        "cer": "",
                        "length": len(joined),
                        "open": joined.count("("),
                        "close": joined.count(")"),
                        "balanced": "",
                        "parses": "",
                        "illegal_chars": "",
                        "ocr_text": line,
                    })

    # Re-score the already-generated page-scale candidates in the same
    # table so that a route change is visible against the same yardstick.
    for item in _existing_page_routes():
        reference = TRANSCRIPTION[item["cmpd_id"]]
        joined = item["text"]
        edit = edit_ops(reference, joined)
        rows.append({
            "cmpd_id": item["cmpd_id"],
            "row_type": "result",
            "arm": item["arm"],
            "recogniser": "upstream_default_ppocrv6",
            "reference_status": REFERENCE_STATUS,
            "reference_length": len(reference),
            "edit_distance": edit,
            "cer": round(edit / len(reference), 6),
            **_metrics(item["cmpd_id"], joined),
            "ocr_text": joined,
        })
        for line in _diff(reference, joined):
            rows.append({
                "cmpd_id": item["cmpd_id"],
                "row_type": "diff",
                "arm": item["arm"],
                "recogniser": "diff",
                "reference_status": REFERENCE_STATUS,
                "reference_length": len(reference),
                "edit_distance": "",
                "cer": "",
                "length": len(joined),
                "open": joined.count("("),
                "close": joined.count(")"),
                "balanced": "",
                "parses": "",
                "illegal_chars": "",
                "ocr_text": line,
            })

    fields = [
        "cmpd_id", "row_type", "arm", "recogniser", "reference_status",
        "reference_length", "edit_distance", "cer", "length", "open",
        "close", "balanced", "parses", "illegal_chars", "ocr_text",
    ]
    with (OUT / "i100298_sweep.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    joined_rows = [r for r in rows if r.get("row_type") == "result"]
    lines = [
        "=== I100298 route sweep (1038-1041) ===",
        "",
        "Recogniser: upstream RapidOCR default engine "
        "(PP-OCRv6 detector + PP-OCRv6 recogniser), fixed for every arm.",
        "Only crop scale and binarisation threshold vary.",
        "",
        "1038: reference supplied verbatim by the researcher.",
        "1039-1041: reference transcribed from the printed rows at 3x, "
        "status '" + REFERENCE_STATUS + "'. Agreement with these strings is "
        "NOT an accuracy figure.",
        "",
        f"{'cmpd':<6} {'arm':<26} {'edit':>5} {'cer':>9} {'len':>5} "
        f"{'open':>5} {'close':>6} {'bal':>5} {'parse':>6} {'illegal':>8}",
    ]
    for row in joined_rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['arm']:<26} "
            f"{str(row['edit_distance']):>5} {str(row['cer']):>9} "
            f"{row['length']:>5} {row['open']:>5} {row['close']:>6} "
            f"{'yes' if row['balanced'] else 'no':>5} "
            f"{'yes' if row['parses'] else 'no':>6} "
            f"{row['illegal_chars'] or '-':>8}"
        )

    lines.extend(["", "character differences against the printed "
                  "transcription:"])
    for row in rows:
        if row.get("row_type") == "diff":
            lines.append(f"  {row['cmpd_id']:<6} {row['arm']:<26} "
                         f"{row['ocr_text']}")

    lines.extend([
        "",
        "No candidate was selected by RDKit parseability, the mass gate, "
        "or DECIMER output.",
        "Text level only; chemical structure is not verified.",
    ])
    io.open(OUT / "i100298_sweep_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
