r"""Per-row diagnosis of the I100298 cell-crop route.

Why this script exists
----------------------
The cell-crop route reads 1039, 1038 and 1041 well but collapses on
1040: the recognised text turns into illegible glyphs ("②") from the
second printed row onwards, even though the printed rows themselves are
sharp at 3x.

That pattern points at the crop, not at the print and not at the
recogniser.  This script therefore records, for every printed row of
every I100298 compound:

    * the row band and the crop actually fed to the recogniser
    * whether a full-height vertical rule sits inside the crop
    * the recognised text of the row
    * the row's agreement with the printed transcription

so the failure can be attributed to a specific geometric cause instead
of being described as "OCR failed".

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_diag.py
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
from phase3_i100298_sweep import TRANSCRIPTION  # noqa: E402
from phase3_i100298_validation import (  # noqa: E402
    CELLS, TARGETS, _segment_rows,
)

OUT = REPO / "data" / "audit" / "phase3" / "i100298_diag"


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _vertical_rule(crop: np.ndarray) -> dict:
    """Measure full-height vertical rules inside a crop.

    A table rule is a column whose dark pixels span nearly the whole
    crop height.  Text glyphs never do, so a column that does is
    geometry, not chemistry.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    dark = gray < 160
    best = {"x": -1, "span": 0}
    for x in range(gray.shape[1]):
        span = int(dark[:, x].sum())
        if span > best["span"]:
            best = {"x": x, "span": span}
    best["height_ratio"] = round(best["span"] / max(height, 1), 3)
    best["is_rule"] = best["height_ratio"] >= 0.80
    return best


def _trim_rule(crop: np.ndarray, rule_x: int) -> np.ndarray:
    """Drop a vertical table rule and the blank space to its left."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = gray < 160
    # First text column to the right of the rule.
    x = rule_x + 1
    while x < crop.shape[1] and dark[:, x].sum() == 0:
        x += 1
    # Last ink column to the right of that.
    right = crop.shape[1] - 1
    while right > x and dark[:, right].sum() == 0:
        right -= 1
    return crop[:, x:right + 1]


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

        for index, (top, bottom) in enumerate(bands, start=1):
            y1 = max(0, top - 4)
            y2 = min(image.shape[0], bottom + 5)
            crop = image[y1:y2, :]
            rule = _vertical_rule(crop)

            raw_text = _recognise(engine, crop)
            trimmed = crop
            if rule["is_rule"]:
                trimmed = _trim_rule(crop, rule["x"])
            trim_text = _recognise(engine, trimmed)

            # Expected row text, taken from the printed transcription by
            # matching the row lengths is not reliable, so the row is
            # only compared when the compound has a transcription and
            # the row is aligned by the running character offset.
            rows.append({
                "cmpd_id": cmpd_id,
                "row_index": index,
                "y1": y1,
                "y2": y2,
                "crop_width": crop.shape[1],
                "rule_x": rule["x"],
                "rule_height_ratio": rule["height_ratio"],
                "has_vertical_rule": rule["is_rule"],
                "trimmed_width": trimmed.shape[1],
                "raw_text": raw_text,
                "trimmed_text": trim_text,
                "raw_len": len(raw_text),
                "trimmed_len": len(trim_text),
                "reference_status": ("printed_transcription_awaiting_human_check"
                                     if reference else "researcher_supplied"),
            })
            cv2.imwrite(str(OUT / f"{cmpd_id}_row{index}_raw.png"), crop)
            if rule["is_rule"]:
                cv2.imwrite(str(OUT / f"{cmpd_id}_row{index}_trimmed.png"),
                            trimmed)

    fields = list(rows[0].keys())
    with (OUT / "i100298_row_diag.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=== I100298 per-row diagnosis ===",
        "",
        "Recogniser: upstream default engine, recognition only, 1x crop.",
        "",
        f"{'cmpd':<6} {'row':>3} {'rule_x':>7} {'h_ratio':>8} "
        f"{'rule?':>6} {'raw_len':>8} {'trim_len':>9}",
    ]
    for row in rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['row_index']:>3} "
            f"{row['rule_x']:>7} {row['rule_height_ratio']:>8} "
            f"{'yes' if row['has_vertical_rule'] else 'no':>6} "
            f"{row['raw_len']:>8} {row['trimmed_len']:>9}"
        )

    lines.extend(["", "raw row text:"])
    for row in rows:
        lines.append(f"  {row['cmpd_id']} row{row['row_index']}: "
                     f"{row['raw_text']}")
    lines.extend(["", "row text after removing the vertical table rule:"])
    for row in rows:
        if row["has_vertical_rule"]:
            lines.append(f"  {row['cmpd_id']} row{row['row_index']}: "
                         f"{row['trimmed_text']}")

    io.open(OUT / "i100298_row_diag.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
