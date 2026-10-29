r"""Create review-ready assets for the I100298 Phase 3 validation.

This script consumes existing experiment outputs only.  It does not run
OCR, change the production pipeline, or modify the formal CSV.

Outputs:
    data/audit/phase3/i100298_validation/candidate_routes.csv
    data/audit/phase3/i100298_validation/candidate_routes_summary.txt
    data/audit/phase3/i100298_validation/figure_1038_route.png
    data/audit/phase3/i100298_validation/review_1038.png ... review_1041.png

Reference policy
----------------
1038 has a researcher-supplied reference and receives a character-level
verdict.  1039-1041 have no human-verified reference; their strings are
labelled candidates and are never called correct.
"""

from __future__ import annotations

import csv
import difflib
import io
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase3" / "i100298_validation"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")
SOURCE_PAGE = (REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
               / "PCTCN2025075389-ftappb-I100298.jpg")
PAGE_ROUTE = (REPO / "data" / "audit" / "phase3" / "page_det_line_rec"
              / "page_det_line_rec.csv")
RECROP = (REPO / "data" / "audit" / "phase3" / "recrop_margin"
          / "recrop_margin.csv")
LINE_RESULTS = OUT / "i100298_line_results.csv"

TARGETS = ("1038", "1039", "1040", "1041")
REF_1038 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N"
    "(CCC(F)(F)F)[C@H]3C(=O)N(C)CC(=O)N[C@@H]"
    "(CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)"
    "C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H]"
    "(CC(=O)N(C)[C@@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)"
    "C(=O)N1)C(=O)N(C)C"
)

# The route shown for manual checking is fixed in advance.  It is not
# selected by RDKit parseability, mass, DECIMER, or a human reference.
ROUTE_BY_CMPD = {
    "1038": ("page_detect_line_rec_2x", "pdet_def_rec_def_2x", PAGE_ROUTE),
    "1039": ("page_line_margin8", "8", RECROP),
    "1040": ("page_line_margin8", "8", RECROP),
    "1041": ("targeted_line_2x_th200", "line_2x_th200", LINE_RESULTS),
}


def _mono(size: int):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


def _sans(size: int, *, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _edit(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(a=a, b=b)
    return sum(max(i2 - i1, j2 - j1)
               for tag, i1, i2, j1, j2 in matcher.get_opcodes()
               if tag != "equal")


def _load_candidates() -> list[dict]:
    rows: list[dict] = []
    for cmpd_id, (route, arm, path) in ROUTE_BY_CMPD.items():
        text = ""
        if path == LINE_RESULTS:
            for row in csv.DictReader(path.open(encoding="utf-8-sig")):
                if (row["cmpd_id"] == cmpd_id
                        and row["arm"] == arm
                        and row["line_index"] == "joined"):
                    text = row["ocr_text"]
                    break
        elif path == RECROP:
            for row in csv.DictReader(path.open(encoding="utf-8-sig")):
                if row["cmpd_id"] == cmpd_id and row["margin"] == arm:
                    text = row["text"]
                    break
        else:
            for row in csv.DictReader(path.open(encoding="utf-8-sig")):
                if row["cmpd_id"] == cmpd_id and row["arm"] == arm:
                    text = row["text"]
                    break
        if not text:
            raise RuntimeError(f"missing candidate for {cmpd_id} in {path}")
        reference = REF_1038 if cmpd_id == "1038" else ""
        rows.append({
            "cmpd_id": cmpd_id,
            "source_image": "PCTCN2025075389-ftappb-I100298.jpg",
            "route": route,
            "reference_status": ("researcher_supplied"
                                 if reference else "awaiting_human_check"),
            "reference_length": len(reference),
            "ocr_length": len(text),
            "edit_distance": _edit(reference, text) if reference else "",
            "cer": (round(_edit(reference, text) / len(reference), 6)
                    if reference else ""),
            "open_parentheses": text.count("("),
            "close_parentheses": text.count(")"),
            "ocr_text": text,
        })
    return rows


def _make_1038_figure() -> None:
    candidate = next(row["ocr_text"] for row in _load_candidates()
                     if row["cmpd_id"] == "1038")
    block = 44
    cw, ch = 11, 24
    pad, label_width = 34, 78
    mono = _mono(17)
    sans = _font = _sans(19)
    title = _sans(24, bold=True)
    small = _sans(16)
    n_blocks = (len(REF_1038) + block - 1) // block
    width = max(1120, pad * 2 + label_width + block * cw + 40)
    height = 90 + 2 * n_blocks * ch + 210
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((pad, 20),
              "1038 printed SMILES: researcher reference vs best OCR route",
              font=title, fill=(27, 27, 27))
    draw.text((pad, 54),
              "pink = character differs; green = exact character match",
              font=small, fill=(100, 100, 100))

    matcher = difflib.SequenceMatcher(a=REF_1038, b=candidate)
    flags = [" "] * len(REF_1038)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            for index in range(i1, min(i2, len(flags))):
                flags[index] = "x"

    y = 90
    for label, observed in (("REF", REF_1038), ("PAGE", candidate)):
        draw.text((pad, y + 2), label, font=mono, fill=(100, 100, 100))
        for block_index in range(n_blocks):
            start = block_index * block
            for char_index, ref_char in enumerate(
                    REF_1038[start:start + block]):
                cx = pad + label_width + char_index * cw
                cy = y + block_index * ch
                got = (observed[start:start + block]
                       .ljust(block)[char_index])
                if label == "PAGE":
                    color = ((247, 198, 214)
                             if flags[start + char_index] == "x"
                             else (207, 232, 207))
                    draw.rectangle((cx, cy - 3, cx + cw, cy + ch), fill=color)
                draw.text((cx, cy), got, font=mono, fill=(27, 27, 27))
        y += n_blocks * ch + 8

    y += 22
    draw.text((pad, y), "character errors against the supplied reference",
              font=sans, fill=(27, 27, 27))
    draw.text((pad, y + 42),
              f"page detect + line recognition: edit 0, CER 0.00000",
              font=sans, fill=(27, 133, 90))
    draw.text((pad, height - 30),
              "Text recognition evidence only. Chemical structure is not "
              "verified and the mass gate is not consulted.",
              font=small, fill=(100, 100, 100))
    image.save(OUT / "figure_1038_route.png")


def _make_review_images(rows: list[dict]) -> None:
    cell_bounds: dict[str, tuple[int, int]] = {}
    for row in csv.DictReader(MANIFEST.open(encoding="utf-8-sig")):
        if row["page"] == "I100298" and row["cmpd_id"] in TARGETS:
            cell_bounds[row["cmpd_id"]] = (int(float(row["bbox_y1"])),
                                           int(float(row["bbox_y2"])))
    source_page = Image.open(SOURCE_PAGE).convert("RGB")
    for row in rows:
        cmpd_id = row["cmpd_id"]
        y1, y2 = cell_bounds[cmpd_id]
        # Re-read the SMILES column from the parent page with 61 px of
        # left context.  The audit crop starts at the cell boundary and
        # can visually clip left-leaning glyphs.
        source = source_page.crop((1030, max(0, y1 - 6),
                                   source_page.width, min(source_page.height,
                                                          y2 + 6)))
        width = 1450
        source = source.resize((1350, int(source.height * 1350 / source.width)))
        text = row["ocr_text"]
        wrapped = textwrap.wrap(text, width=92) or [""]
        text_height = 40 + len(wrapped) * 23
        canvas = Image.new("RGB", (width, source.height + text_height + 80),
                           "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((35, 20),
                  f"{cmpd_id} original printed text (I100298)",
                  font=_sans(21, bold=True), fill=(27, 27, 27))
        canvas.paste(source, (35, 55))
        y = source.height + 76
        if row["reference_status"] == "researcher_supplied":
            state = (f"VERIFIED TEXT: edit {row['edit_distance']}, "
                     f"CER {row['cer']:.5f}")
            color = (30, 120, 70)
        else:
            state = ("CANDIDATE ONLY: no human-verified reference; "
                     "please check against the printed text above")
            color = (170, 90, 20)
        draw.text((35, y), state, font=_sans(18, bold=True), fill=color)
        y += 32
        mono = _mono(16)
        for line in wrapped:
            draw.text((35, y), line, font=mono, fill=(27, 27, 27))
            y += 23
        canvas.save(OUT / f"review_{cmpd_id}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _load_candidates()
    with (OUT / "candidate_routes.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=== I100298 review candidates ===",
        "",
        "1038 is the only candidate in this group with a "
        "researcher-supplied reference.",
        "1039-1041 are OCR candidates pending human review.",
        "",
        f"{'cmpd':<6} {'route':<25} {'ref status':<20} {'edit':>5} "
        f"{'cer':>9} {'len':>5} {'open':>5} {'close':>6}",
    ]
    for row in rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['route']:<25} "
            f"{row['reference_status']:<20} {str(row['edit_distance']):>5} "
            f"{str(row['cer']):>9} {row['ocr_length']:>5} "
            f"{row['open_parentheses']:>5} {row['close_parentheses']:>6}"
        )
    lines.extend([
        "",
        "Routes for 1039-1041 were selected from previously generated "
        "outputs without using RDKit parseability, mass, or DECIMER; "
        "they remain provisional and route-dependent.",
        "Text level only; chemical structures are not verified.",
    ])
    io.open(OUT / "candidate_routes_summary.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    _make_1038_figure()
    _make_review_images(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
