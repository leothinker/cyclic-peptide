r"""One clean evidence figure for the 1036 binarisation fix.

Why this figure exists
----------------------
The earlier report figures were montages of many crops and they were
hard to read. This one is a single three-way alignment:

    REF   the human-verified printed string
    PROD  what the production path recognises today (shipped)
    BIN   the same crop after threshold 200

Every column is one character of the reference. A column where a
recognised row disagrees with the reference is highlighted, so the
reader sees exactly which characters are lost and which are recovered.

The image crop is deliberately NOT the main content: threshold 200
changes the pixels very little to the eye, so a before/after picture
would be misleading. What changes is the recognition result, and that
is what is shown.

The text shown is TEXT recognition evidence only. It says nothing about
whether the chemical structure is correct, and it does not affect the
mass gate.

Read-only. Writes to ``data/audit/phase3/figure_1036/``.

Run:
    .venv\Scripts\python.exe phase3_figure_1036.py
"""

from __future__ import annotations

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

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase3_page_extract import GOLD, baseline_engine  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "figure_1036"

INK = (35, 35, 35)
MUTED = (120, 120, 120)
GOOD = (0, 125, 70)
BAD = (185, 45, 45)
HILITE = (255, 231, 231)
OKBG = (226, 246, 234)

COLS_PER_LINE = 62


def _label(img, text, colour=INK, scale=0.5, org=(12, 20), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thick, cv2.LINE_AA)
    return img


def _partner_map(ref: str, other: str) -> dict[int, str]:
    """Map each reference index to the character aligned with it.

    A character that the other string is missing maps to ''.
    """
    sm = difflib.SequenceMatcher(None, ref, other, autojunk=False)
    out: dict[int, str] = {}
    ri = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[ri] = other[j1 + k]
                ri += 1
        elif tag == "delete":
            for k in range(i1, i2):
                out[ri] = ""
                ri += 1
        elif tag == "insert":
            pass
        else:
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                out[ri] = other[j1 + k] if j1 + k < j2 else ""
                ri += 1
    return out


def _recognise(engine, img):
    result = engine(img)
    if not result:
        return ""
    items = sorted(zip(list(result.boxes), list(result.txts)),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    return "".join(t.strip() for _, t in items if t and t.strip())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    ex = RapidOcrTableExtractor(rec_top_k=0)
    engine = baseline_engine()

    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    rows = ex._detect_row_boundaries(img)

    prod_txt = bin_txt = ""
    for y1, y2 in rows:
        crop = img[y1:y2, col.x1:col.x2]
        if crop.size == 0:
            continue
        up = ex._upscale(ex._trim_to_content(crop))
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        p = _recognise(engine, up)
        b = _recognise(engine, cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
        if p.startswith("CC[C@H]") or b.startswith("CC[C@H]"):
            prod_txt, bin_txt = p, b
            break

    if not prod_txt:
        print("1036 band not located")
        return

    ref = REFERENCES["1036"]["smiles"]
    ep = edit_ops(ref, prod_txt)
    eb = edit_ops(ref, bin_txt)

    map_p = _partner_map(ref, prod_txt)
    map_b = _partner_map(ref, bin_txt)

    n_lines = (len(ref) + COLS_PER_LINE - 1) // COLS_PER_LINE
    char_w, row_h = 12, 24
    label_w = 74
    header_h = 104
    legend_h = 30
    foot_h = 30
    W = label_w + COLS_PER_LINE * char_w + 30
    H = header_h + n_lines * (row_h * 3 + 8) + legend_h + foot_h

    canvas = np.full((H, W, 3), 255, np.uint8)

    _label(canvas, "1036 printed SMILES: production vs one-line fix",
           INK, 0.68, (18, 30), 1)
    _label(canvas,
           f"character errors vs human-verified reference:    "
           f"production {ep}  (CER {ep / len(ref):.3f})      ->      "
           f"+ threshold 200   {eb}  (CER {eb / len(ref):.3f})",
           GOOD, 0.5, (18, 60))
    _label(canvas,
           "Each column is one character of the printed string. "
           "Pink = misread, green = recovered by the fix, _ = character "
           "not recognised at all.",
           MUTED, 0.42, (18, 84))

    y = header_h
    for line_no in range(n_lines):
        lo = line_no * COLS_PER_LINE
        hi = min(len(ref), lo + COLS_PER_LINE)
        if lo >= hi:
            break
        _label(canvas, "REF", MUTED, 0.42, (18, y + 14))
        _label(canvas, "PROD", BAD, 0.42, (18, y + 14 + row_h))
        _label(canvas, "BIN", GOOD, 0.42, (18, y + 14 + row_h * 2))

        for i, k in enumerate(range(lo, hi)):
            x = label_w + i * char_w
            rc = ref[k]
            pc = map_p.get(k, "") or ""
            bc = map_b.get(k, "") or ""
            bad_p = pc != rc
            bad_b = bc != rc
            # Reference row: mark the columns production got wrong.
            if bad_p:
                cv2.rectangle(canvas, (x, y + 2),
                              (x + char_w - 1, y + 2 + row_h - 3),
                              HILITE, -1)
            # Production row: red where it disagrees with the reference.
            if bad_p:
                cv2.rectangle(canvas, (x, y + 2 + row_h),
                              (x + char_w - 1, y + 2 + row_h * 2 - 3),
                              HILITE, -1)
            # Fixed row: green when the fix recovered a column that
            # production got wrong, red when it is still wrong.
            if bad_p and not bad_b:
                cv2.rectangle(canvas, (x, y + 2 + row_h * 2),
                              (x + char_w - 1, y + 2 + row_h * 3 - 3),
                              OKBG, -1)
            elif bad_b:
                cv2.rectangle(canvas, (x, y + 2 + row_h * 2),
                              (x + char_w - 1, y + 2 + row_h * 3 - 3),
                              HILITE, -1)
            _label(canvas, rc, INK, 0.45, (x + 3, y + 14))
            _label(canvas, pc if pc else "_",
                   BAD if bad_p else INK, 0.45, (x + 3, y + 14 + row_h))
            _label(canvas, bc if bc else "_",
                   BAD if bad_b else INK, 0.45,
                   (x + 3, y + 14 + row_h * 2))
        y += row_h * 3 + 8

    ly = H - foot_h - 6
    cv2.rectangle(canvas, (18, ly - 12), (38, ly + 2), HILITE, -1)
    _label(canvas, "misread", INK, 0.42, (46, ly))
    cv2.rectangle(canvas, (232, ly - 12), (252, ly + 2), OKBG, -1)
    _label(canvas, "recovered by the fix", INK, 0.42, (260, ly))
    _label(canvas,
           "Text recognition evidence only. It does not verify the "
           "chemical structure and does not affect the mass gate.",
           MUTED, 0.4, (18, H - 14))

    out_path = OUT / "figure_1036_binarisation.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"production edit={ep}  binarised edit={eb}  ref_len={len(ref)}")
    print(f"wrote {out_path}")

    txt = io.open(OUT / "figure_1036_caption.txt", "w", encoding="utf-8")
    txt.write("Figure: 1036 printed SMILES, production vs threshold-200 "
              "binarisation\n")
    txt.write(f"production : edit {ep}, CER {ep / len(ref):.5f}, "
              f"open parens {prod_txt.count('(')} / close "
              f"{prod_txt.count(')')}\n")
    txt.write(f"binarised  : edit {eb}, CER {eb / len(ref):.5f}, "
              f"open parens {bin_txt.count('(')} / close "
              f"{bin_txt.count(')')}\n")
    txt.write("Text recognition evidence only; the chemical structure is "
              "not verified by this figure.\n")
    txt.close()


if __name__ == "__main__":
    main()
