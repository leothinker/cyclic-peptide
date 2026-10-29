r"""Validate the default OCR route on compounds 1038-1041.

This is a read-only, targeted follow-up to the Phase 3 route comparison.
It keeps the model and preprocessing fixed while changing only how the
printed cell is presented to the recogniser:

    1. segment the seven printed rows from the original-resolution crop
    2. recognise each row independently
    3. join the rows in reading order

The researcher supplied the 1038 reference verbatim on 2026-10-28, so
1038 can be checked at character level.  The 1039-1041 outputs are only
candidates: no human-verified reference exists for them.

No mass gate, DECIMER output, or RDKit parseability is used to select a
candidate.  The formal CSV and ``src/cpd`` are not modified.

Writes to ``data/audit/phase3/i100298_validation/``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_validation.py
"""

from __future__ import annotations

import csv
import difflib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import edit_ops  # noqa: E402
from phase3_quality_screen import SMILES_CHARS, screen_joined  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "i100298_validation"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
REF_ROUTES = (REPO / "data" / "audit" / "phase3" / "ref_routes"
              / "ref_routes.csv")

TARGETS = ("1038", "1039", "1040", "1041")
LINE_ARMS = ("line_1x_none", "line_2x_none", "line_2x_th200")
INK_THRESHOLD = 160
MIN_INK_PIXELS = 5

# Supplied verbatim by the researcher from the printed I100298 cell.
# Markdown escaping of parentheses is not part of the SMILES.
REF_1038 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
    "(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N[C@@H]"
    "(CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)"
    "C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H]"
    "(CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)"
    "C(=O)N1)C(=O)N(C)C"
)


def _font(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _mono(size: int):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


def _segment_rows(gray: np.ndarray) -> list[tuple[int, int]]:
    """Find the seven ink bands, ignoring the table border."""
    h, w = gray.shape
    x1 = min(10, max(1, w // 100))
    x2 = max(x1 + 1, w - 10)
    ink = (gray[:, x1:x2] < INK_THRESHOLD).astype(np.uint8)
    sums = ink.sum(axis=1)
    active = np.flatnonzero(sums > MIN_INK_PIXELS)
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

    # The top table rule is a one-pixel horizontal band.  All printed
    # SMILES rows are substantially taller.
    rows = [(a, b) for a, b in groups if b - a >= 8]
    if len(rows) != 7:
        raise RuntimeError(f"expected 7 printed rows, found {len(rows)}: {rows}")
    return rows


def _preprocess(crop: np.ndarray, arm: str) -> np.ndarray:
    if arm == "line_1x_none":
        return crop
    up = cv2.resize(crop, None, fx=2, fy=2,
                    interpolation=cv2.INTER_CUBIC)
    if arm == "line_2x_none":
        return up
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise_line(engine, crop: np.ndarray) -> str:
    # RapidOCR mutates use_det/use_rec on the engine instance and does
    # not reset them.  Both flags are therefore explicit on every call.
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


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


def _reference_status(cmpd_id: str) -> tuple[str, str]:
    if cmpd_id == "1038":
        return "researcher_supplied", REF_1038
    return "awaiting_human_check", ""


def _run_i100298() -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    engine = _engine()
    rows: list[dict] = []
    by_cmpd: dict[str, dict[str, list[str]]] = {}

    for cmpd_id in TARGETS:
        path = CELLS / f"I100298_{cmpd_id}_text.png"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)
        line_dir = OUT / "line_crops" / cmpd_id
        line_dir.mkdir(parents=True, exist_ok=True)

        by_cmpd[cmpd_id] = {}
        for arm in LINE_ARMS:
            parts: list[str] = []
            for index, (top, bottom) in enumerate(bands, start=1):
                y1 = max(0, top - 4)
                y2 = min(image.shape[0], bottom + 5)
                crop = image[y1:y2, :]
                if arm == "line_2x_th200":
                    cv2.imwrite(str(line_dir / f"line{index}_{arm}.png"),
                                _preprocess(crop, arm))
                text = _recognise_line(engine, _preprocess(crop, arm))
                parts.append(text)
                rows.append({
                    "cmpd_id": cmpd_id,
                    "arm": arm,
                    "line_index": index,
                    "source_crop": str(path.relative_to(REPO)),
                    "source_y1": y1,
                    "source_y2": y2,
                    "ocr_text": text,
                })
            joined = "".join(parts)
            by_cmpd[cmpd_id][arm] = parts
            status, ref = _reference_status(cmpd_id)
            edit = edit_ops(ref, joined) if ref else ""
            rows.append({
                "cmpd_id": cmpd_id,
                "arm": arm,
                "line_index": "joined",
                "source_crop": str(path.relative_to(REPO)),
                "source_y1": "",
                "source_y2": "",
                "ocr_text": joined,
                "reference_status": status,
                "reference_length": len(ref),
                "edit_distance": edit,
                "cer": round(edit / len(ref), 6) if ref else "",
                **_metrics(cmpd_id, joined),
            })
    return rows, by_cmpd


def _read_1037_routes() -> dict[str, str]:
    out: dict[str, str] = {}
    with REF_ROUTES.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["cmpd_id"] == "1037":
                out[row["route"]] = row["text"]
    return out


def _marks(reference: str, observed: str) -> list[str]:
    matcher = difflib.SequenceMatcher(a=reference, b=observed)
    flags = [" "] * len(reference)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for index in range(i1, min(i2, len(flags))):
            flags[index] = "x"
    return flags


def _draw_aligned(draw, x, y, reference, observed, flags, mono, cw, ch):
    for index, ref_char in enumerate(reference):
        cx = x + index * cw
        got = observed[index] if index < len(observed) else " "
        if flags is not None and index < len(flags) and flags[index] == "x":
            draw.rectangle((cx, y - 3, cx + cw, y + ch), fill=(247, 198, 214))
        elif got != " ":
            draw.rectangle((cx, y - 3, cx + cw, y + ch), fill=(207, 232, 207))
        draw.text((cx, y), got, font=mono, fill=(27, 27, 27))


def _make_1037_figure() -> None:
    routes = _read_1037_routes()
    reference = (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](CC2=CC=CC=C2)C(=O)N3[C@@H](CC3)"
        "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
        "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
        "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)="
        "C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
    )
    variants = [
        ("PROD", routes["production"]),
        ("+TH200", routes["+ threshold 200"]),
        ("PAGE", routes["page + line rec"]),
    ]
    block = 44
    cw, ch = 11, 24
    pad, label_width = 34, 78
    mono = _mono(17)
    sans = _font(19)
    title = _font(24, bold=True)
    small = _font(16)
    n_blocks = (len(reference) + block - 1) // block
    text_height = (1 + len(variants)) * n_blocks * ch + 100
    width = max(1120, pad * 2 + label_width + block * cw + 40)
    height = text_height + 260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((pad, 20), "1037 printed SMILES: recognition route comparison",
              font=title, fill=(27, 27, 27))
    draw.text((pad, 54),
              "pink = character differs from the human-checked reference; "
              "green = character recovered",
              font=small, fill=(100, 100, 100))

    y = 90
    draw.text((pad, y + 2), "REF", font=mono, fill=(100, 100, 100))
    for block_index in range(n_blocks):
        start = block_index * block
        chunk = reference[start:start + block]
        for char_index, char in enumerate(chunk):
            draw.text((pad + label_width + char_index * cw,
                       y + block_index * ch), char, font=mono,
                      fill=(27, 27, 27))
    y += n_blocks * ch + 6

    for label, observed in variants:
        draw.text((pad, y + 2), label, font=mono, fill=(100, 100, 100))
        flags = _marks(reference, observed)
        for block_index in range(n_blocks):
            start = block_index * block
            chunk_ref = reference[start:start + block]
            chunk_observed = observed[start:start + block].ljust(block)
            chunk_flags = flags[start:start + block]
            _draw_aligned(draw, pad + label_width, y + block_index * ch,
                          chunk_ref, chunk_observed, chunk_flags, mono, cw, ch)
        y += n_blocks * ch + 6

    y += 16
    draw.text((pad, y), "character errors against the 1037 reference",
              font=sans, fill=(27, 27, 27))
    y += 34
    values = [(label, edit_ops(reference, observed))
              for label, observed in variants]
    maximum = max(value for _, value in values)
    colors = [(160, 160, 160), (74, 127, 181), (47, 133, 90)]
    for index, (label, value) in enumerate(values):
        bar_y = y + index * 58
        draw.text((pad, bar_y), label, font=sans, fill=(27, 27, 27))
        draw.rectangle((pad + 140, bar_y + 5, pad + 900, bar_y + 31),
                       fill=(245, 245, 247))
        width_value = int(760 * value / maximum) if maximum else 0
        draw.rectangle((pad + 140, bar_y + 5, pad + 140 + width_value,
                        bar_y + 31), fill=colors[index])
        draw.text((pad + 920, bar_y + 5),
                  f"edit {value}   CER {value / len(reference):.5f}",
                  font=sans, fill=(27, 27, 27))

    draw.text((pad, height - 30),
              "Text recognition evidence only. Chemical structure is not "
              "verified and the mass gate is not consulted.",
              font=small, fill=(100, 100, 100))
    image.save(OUT / "figure_1037_route.png")


def _make_i100298_figure(by_cmpd: dict[str, dict[str, list[str]]]) -> None:
    """Show the researcher reference and the fixed 2x line route."""
    arm = "line_2x_none"
    width = 1500
    row_height = 245
    header_height = 92
    image = Image.new("RGB", (width, header_height + row_height * 4),
                       "white")
    draw = ImageDraw.Draw(image)
    title = _font(25, bold=True)
    sans = _font(18)
    small = _font(15)
    mono = _mono(16)
    draw.text((30, 20),
              "I100298: default OCR route, independently recognised printed rows",
              font=title, fill=(27, 27, 27))
    draw.text((30, 56),
              "1038 is checked against the researcher-supplied reference; "
              "1039-1041 remain candidates pending human review.",
              font=small, fill=(95, 95, 95))

    for compound_index, cmpd_id in enumerate(TARGETS):
        y0 = header_height + compound_index * row_height
        draw.rectangle((22, y0 + 4, width - 22, y0 + row_height - 8),
                       outline=(215, 215, 220), width=1)
        draw.text((38, y0 + 15), cmpd_id, font=_font(21, bold=True),
                  fill=(27, 27, 27))
        parts = by_cmpd[cmpd_id][arm]
        joined = "".join(parts)
        metric = _metrics(cmpd_id, joined)
        if cmpd_id == "1038":
            edit = edit_ops(REF_1038, joined)
            state = (f"exact match: {edit == 0}; edit {edit}; "
                     f"CER {edit / len(REF_1038):.5f}")
        else:
            state = ("candidate; no human-verified reference; "
                     f"rows {len(parts)}/7; balanced {metric['balanced']}; "
                     f"parses {metric['parses']}")
        draw.text((115, y0 + 18), state, font=sans, fill=(70, 70, 70))
        for line_index, text in enumerate(parts):
            line_y = y0 + 53 + line_index * 25
            draw.text((40, line_y), f"{line_index + 1}", font=small,
                      fill=(130, 130, 130))
            draw.text((70, line_y), text, font=mono, fill=(27, 27, 27))

    image.save(OUT / "figure_1038_1041_candidates.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, by_cmpd = _run_i100298()
    fields = [
        "cmpd_id", "arm", "line_index", "source_crop", "source_y1",
        "source_y2", "ocr_text", "reference_status", "reference_length",
        "edit_distance", "cer", "length", "open", "close", "balanced",
        "parses", "illegal_chars",
    ]
    with (OUT / "i100298_line_results.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    joined_rows = [row for row in rows if row["line_index"] == "joined"]
    lines = [
        "=== I100298 targeted validation (1038-1041) ===",
        "",
        "Model: upstream RapidOCR default engine "
        "(PP-OCRv6 detector + PP-OCRv6 recogniser).",
        "Input: original-resolution text crop, seven printed rows segmented "
        "from ink projection, each row recognised independently.",
        "",
        "1038 reference status: researcher-supplied verbatim.",
        "1039-1041 reference status: awaiting human check; outputs are "
        "candidates only.",
        "",
        f"{'cmpd':<6} {'arm':<16} {'edit':>5} {'cer':>9} {'len':>5} "
        f"{'rows':>6} {'open':>5} {'close':>6} {'bal':>5} {'parse':>6}",
    ]
    for row in joined_rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['arm']:<16} "
            f"{str(row['edit_distance']):>5} {str(row['cer']):>9} "
            f"{row['length']:>5} {'7/7':>6} {row['open']:>5} "
            f"{row['close']:>6} {'yes' if row['balanced'] else 'no':>5} "
            f"{'yes' if row['parses'] else 'no':>6}"
        )
    lines.extend([
        "",
        "No candidate is selected by mass-gate, DECIMER, or RDKit "
        "parseability.",
        "Text level only; chemical structure is not verified.",
    ])
    io.open(OUT / "i100298_validation_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))

    _make_1037_figure()
    _make_i100298_figure(by_cmpd)
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
