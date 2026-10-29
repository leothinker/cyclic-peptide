r"""Row-level geometry sweep for the printed SMILES on page I100298.

Purpose
-------
``phase3_i100298_sweep.py`` showed the remaining character errors on
1039 and 1040 are single characters in individual printed rows:

    1039  O -> 0   in one C(=O)
    1040  a lost ')'

The compound-level routes cannot fix a single row without changing
every other row, so this script varies the crop of each printed row on
its own and reports the best configuration per row.

Every variant keeps the recogniser fixed (upstream default engine) and
only changes: which source image the row is cropped from, whether the
vertical table rule is excluded, the upscale factor, and the
binarisation threshold.

Selection rule
--------------
Rows are ranked by character agreement with the printed transcription
in ``phase3_i100298_sweep.py``.  RDKit parseability, the mass gate and
DECIMER output are NOT consulted.  The transcription is itself awaiting
human check, so the numbers are agreement counts, not accuracy.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_rowsweep.py
"""

from __future__ import annotations

import csv
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
from phase3_i100298_sweep import PRINTED_ROWS, TRANSCRIPTION  # noqa: E402
from phase3_i100298_validation import (  # noqa: E402
    CELLS, TARGETS, _segment_rows,
)

OUT = REPO / "data" / "audit" / "phase3" / "i100298_rowsweep"

SCALES = (1, 2, 3)
THRESHOLDS = (None, 180, 200, 220, 240)
PADS = ((0, 0), (2, 3), (4, 5), (8, 9), (14, 15))


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _ink_columns(gray: np.ndarray) -> tuple[int, int]:
    """First and last column that contain any ink."""
    dark = gray < 160
    cols = np.flatnonzero(dark.sum(axis=0) > 0)
    if not len(cols):
        return 0, gray.shape[1] - 1
    return int(cols[0]), int(cols[-1])


def _rule_columns(gray: np.ndarray, threshold: float = 0.80) -> list[int]:
    """Columns whose dark pixels span nearly the whole crop height."""
    dark = gray < 160
    height = gray.shape[0]
    out = []
    for x in range(gray.shape[1]):
        if dark[:, x].sum() / max(height, 1) >= threshold:
            out.append(x)
    return out


def _text_span(gray: np.ndarray, rules: list[int]) -> tuple[int, int]:
    """Ink span that excludes vertical table rules.

    A rule is a full-height column.  Text glyphs are much shorter, so
    the text span is measured from the first non-rule ink to the last
    non-rule ink.
    """
    dark = gray < 160
    height = gray.shape[0]
    is_rule = np.zeros(gray.shape[1], dtype=bool)
    for x in rules:
        is_rule[x] = True

    # A rule may be 1-3 px wide, so widen the mask to neighbouring
    # columns that are also nearly full height.
    for x in rules:
        for dx in (-1, 1):
            if 0 <= x + dx < gray.shape[1]:
                if dark[:, x + dx].sum() / max(height, 1) >= 0.5:
                    is_rule[x + dx] = True

    text_cols = np.flatnonzero((dark.sum(axis=0) > 0) & (~is_rule))
    if not len(text_cols):
        return 0, gray.shape[1] - 1
    return int(text_cols[0]), int(text_cols[-1])


def _preprocess(crop: np.ndarray, scale: int, threshold) -> np.ndarray:
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    if threshold is None:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = _engine()
    rows: list[dict] = []

    for cmpd_id in TARGETS:
        reference = TRANSCRIPTION.get(cmpd_id, "")
        printed = PRINTED_ROWS.get(cmpd_id, [])
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray_full)

        raw_crops: dict[int, np.ndarray] = {}
        for index, (top, bottom) in enumerate(bands, start=1):
            raw_crops[index] = (top, bottom)

        # Pass 2: sweep the geometry per row.
        for index, (top, bottom) in enumerate(bands, start=1):
            exp = printed[index - 1] if index <= len(printed) else ""

            for pad_top, pad_bottom in PADS:
                y1 = max(0, top - pad_top)
                y2 = min(image.shape[0], bottom + pad_bottom)
                crop = image[y1:y2, :]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                rules = _rule_columns(gray)
                text_x1, text_x2 = _text_span(gray, rules)

                for trim in ("no", "rules"):
                    if trim == "rules":
                        x1, x2 = text_x1, text_x2
                    else:
                        x1, x2 = _ink_columns(gray)
                    variant = crop[:, x1:x2 + 1]
                    for scale in SCALES:
                        for threshold in THRESHOLDS:
                            text = _recognise(
                                engine,
                                _preprocess(variant, scale, threshold))
                            edit = edit_ops(exp, text) if exp else ""
                            rows.append({
                                "cmpd_id": cmpd_id,
                                "row_index": index,
                                "trim": trim,
                                "scale": scale,
                                "threshold": ("" if threshold is None
                                              else threshold),
                                "pad_top": pad_top,
                                "pad_bottom": pad_bottom,
                                "n_rules": len(rules),
                                "crop_x1": x1,
                                "crop_x2": x2,
                                "expected_len": len(exp),
                                "expected_text": exp,
                                "ocr_len": len(text),
                                "edit_distance": edit,
                                "ocr_text": text,
                            })

    fields = list(rows[0].keys())
    with (OUT / "i100298_rowsweep.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Best row configuration, lowest edit distance, ties broken by the
    # simplest crop.
    best: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row["edit_distance"] == "":
            continue
        key = (row["cmpd_id"], row["row_index"])
        current = best.get(key)
        rank = (row["edit_distance"],
                0 if row["trim"] == "rules" else 1,
                row["scale"],
                row["threshold"] == "" or row["threshold"],
                row["pad_top"])
        if current is None:
            best[key] = {"rank": rank, "row": row}
        else:
            if rank < current["rank"]:
                best[key] = {"rank": rank, "row": row}

    lines = [
        "=== I100298 per-row geometry sweep ===",
        "",
        "Recogniser fixed (upstream default engine); only crop geometry,",
        "upscale and binarisation threshold vary.",
        "",
        f"{'cmpd':<6} {'row':>3} {'best_edit':>10} {'trim':>6} {'scale':>6} "
        f"{'th':>5} {'rules':>6} {'exp_len':>8} {'ocr_len':>8}",
    ]
    total_edit = 0
    matched_rows = 0
    for (cmpd_id, index), item in sorted(best.items()):
        row = item["row"]
        lines.append(
            f"{cmpd_id:<6} {index:>3} {row['edit_distance']:>10} "
            f"{row['trim']:>6} {row['scale']:>6} "
            f"{str(row['threshold']):>5} {row['n_rules']:>6} "
            f"{row['expected_len']:>8} {row['ocr_len']:>8}"
        )
        total_edit += row["edit_distance"]
        matched_rows += 1

    # Reference-free selection: the most frequent recognised string for
    # each printed row across all tested configurations.  This rule
    # never looks at the printed transcription, so the score it achieves
    # is an honest estimate rather than a fitted one.
    vote: dict[tuple[str, int], dict[str, int]] = {}
    for row in rows:
        key = (row["cmpd_id"], row["row_index"])
        text = row["ocr_text"]
        if not text:
            continue
        bucket = vote.setdefault(key, {})
        bucket[text] = bucket.get(text, 0) + 1

    vote_lines = [
        "",
        "reference-free selection: modal OCR string per printed row",
        "(the rule never consults the printed transcription)",
        "",
        f"{'cmpd':<6} {'row':>3} {'votes':>6} {'edit':>6} {'agreement':>10}",
    ]
    vote_total = 0
    vote_rows = 0
    vote_strings: dict[str, str] = {}
    for (cmpd_id, index), bucket in sorted(vote.items()):
        text, count = max(bucket.items(), key=lambda kv: (kv[1], -len(kv[0])))
        exp = ""
        for row in rows:
            if (row["cmpd_id"] == cmpd_id
                    and row["row_index"] == index
                    and row["expected_text"]):
                exp = row["expected_text"]
                break
        edit = edit_ops(exp, text) if exp else ""
        share = count / max(sum(bucket.values()), 1)
        vote_lines.append(
            f"{cmpd_id:<6} {index:>3} {count:>6} {str(edit):>6} "
            f"{share:>9.0%}"
        )
        vote_total += edit if edit != "" else 0
        vote_rows += 1
        vote_strings.setdefault(cmpd_id, "")
        vote_strings[cmpd_id] += text

    for cmpd_id, joined in sorted(vote_strings.items()):
        vote_lines.append(f"  {cmpd_id} joined ({len(joined)} chars): {joined}")
    vote_lines.append(f"  total edit distance of the voted strings: {vote_total}")

    lines.extend([
        "",
        f"rows scored: {matched_rows}",
        f"summed best-row edit distance: {total_edit}",
    ])
    lines.extend(vote_lines)
    lines.extend([
        "",
        "Ranking uses character agreement with the printed transcription.",
        "No RDKit parseability, mass gate or DECIMER output is consulted.",
        "The transcription is itself awaiting human check.",
    ])
    io.open(OUT / "i100298_rowsweep_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
