r"""Report figure: the 1036 route comparison.

Panel 1 (text) shows the human-verified 1036 reference next to the
shipped production output, the one-line binarisation fix and the
page-route output, aligned character by character, so a fix that moved
an error instead of removing it stays visible.

Panel 2 shows the character error count of the same routes against
the same reference.

The figure is text level. It does not verify a chemical structure and
does not affect the mass gate.

Read-only. Writes to ``data/audit/phase3/figure_route/``.

Run:
    .venv\Scripts\python.exe phase3_figure_route.py
"""

from __future__ import annotations

import csv
import difflib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "figure_route"
CER1036 = REPO / "data" / "audit" / "phase3" / "cer_1036" / "cer_1036.txt"
ROUTE_CSV = REPO / "data" / "audit" / "phase3" / "page_det_line_rec" / "page_det_line_rec.csv"

CMPD = "1036"
BLOCK = 44
ROUTE_ARM = "pdet_def_rec_def_2x"
ROUTE_LABEL = "page route (page detect + line crop)"

INK = (27, 27, 27)
MUTED = (122, 122, 122)
PINK = (247, 198, 214)
GREEN = (207, 232, 207)
PANEL = (247, 247, 249)
BAR_COLORS = [(158, 158, 158), (74, 127, 181), (47, 133, 90)]


def _font(size):
    return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)


def _sans(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _read_cer_texts():
    """The production and binarised strings recorded by the CER run."""
    prod = ""
    binr = ""
    mode = ""
    for line in CER1036.read_text(encoding="utf-8").splitlines():
        if line.startswith("production rows:"):
            mode = "prod"
            continue
        if line.startswith("binarised rows:"):
            mode = "binr"
            continue
        if mode and line.startswith("  ") and ":" in line:
            idx, _, row = line.strip().partition(":")
            if idx.isdigit():
                if mode == "prod":
                    prod += row.strip()
                else:
                    binr += row.strip()
    return prod, binr


def _read_route_text():
    """The page-route string for 1036, if the corpus run produced it."""
    partial = ROUTE_CSV.parent / "page_det_line_rec_partial.csv"
    for path in (ROUTE_CSV, partial):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            if r["cmpd_id"] == CMPD and r["arm"] == ROUTE_ARM:
                return r["text"]
    return ""


def _marks(ref, got):
    """Mark the reference positions the output got wrong."""
    sm = difflib.SequenceMatcher(a=ref, b=got)
    flags = [" "] * len(ref)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for k in range(i1, min(i2, len(flags))):
            flags[k] = "x"
    return flags


def _draw_text_block(draw, x, y, ref, got, flags, font, cw, ch):
    """Draw one aligned block of characters with per-character boxes."""
    for i, ch_ref in enumerate(ref):
        cx = x + i * cw
        if got is None:
            draw.text((cx, y), ch_ref, font=font, fill=INK)
            continue
        got_ch = got[i] if i < len(got) else " "
        if flags is not None and i < len(flags) and flags[i] == "x":
            draw.rectangle([cx, y - 3, cx + cw, y + ch], fill=PINK)
        elif got_ch != " ":
            draw.rectangle([cx, y - 3, cx + cw, y + ch], fill=GREEN)
        draw.text((cx, y), got_ch, font=font, fill=INK)


def _draw_bar(draw, x, y, w, h, value, vmax, color, label, note, sans):
    draw.text((x, y), label, font=sans, fill=INK)
    bar_y = y + 26
    bar_w = int(w * (value / vmax)) if vmax else 0
    draw.rectangle([x, bar_y, x + w, bar_y + h], fill=PANEL)
    draw.rectangle([x, bar_y, x + bar_w, bar_y + h], fill=color)
    draw.text((x + bar_w + 10, bar_y + 2), note, font=sans, fill=INK)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = REFERENCES[CMPD]["smiles"]
    prod, binr = _read_cer_texts()
    route = _read_route_text()

    variants = [("PROD", prod), ("+TH200", binr)]
    if route:
        variants.append(("PAGE", route))

    font = _font(17)
    label_font = _font(17)
    sans = _sans(19)
    sans_b = _sans(24, bold=True)
    small = _sans(16)

    cw = 11
    ch = 24
    lab_w = 78
    pad = 34

    n_blocks = (len(ref) + BLOCK - 1) // BLOCK
    rows = [("REF", ref)]
    rows.extend(variants)
    n_rows = len(rows) * n_blocks

    text_h = n_rows * ch + 90
    bar_h = 250
    W = max(pad * 2 + lab_w + BLOCK * cw + 40, 1120)
    H = text_h + bar_h + 70

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((pad, 20), "1036 printed SMILES: three recognition routes against the human-verified reference", font=sans_b, fill=INK)
    draw.text((pad, 54), "pink = misread against the reference, green = character recovered by that route", font=small, fill=MUTED)

    y = 88
    for label, txt in rows:
        draw.text((pad, y + 2), label, font=label_font, fill=MUTED)
        flags = _marks(ref, txt) if label != "REF" else None
        for b in range(n_blocks):
            s = b * BLOCK
            block_ref = ref[s:s + BLOCK]
            block_got = None
            block_flags = None
            if label != "REF":
                block_got = txt[s:s + BLOCK].ljust(BLOCK)
                block_flags = flags[s:s + BLOCK]
            _draw_text_block(draw, pad + lab_w, y, block_ref, block_got, block_flags, font, cw, ch)
            y += ch
        y += 6

    y += 16
    draw.text((pad, y), "character errors against the 1036 reference (lower is better)", font=sans, fill=INK)
    y += 34

    entries = []
    for label, txt in [("production  (cell crop, 2x, no binarisation)", prod), ("+ threshold 200  (cell crop)", binr), ("page route  (page detect + line crop)", route)]:
        if txt:
            entries.append((label, edit_ops(ref, txt), edit_ops(ref, txt) / len(ref)))
    vmax = max(e[1] for e in entries) if entries else 1
    for i, (label, edit, cer) in enumerate(entries):
        _draw_bar(draw, pad, y + i * 62, 760, 26, edit, vmax, BAR_COLORS[min(i, len(BAR_COLORS) - 1)], label, f"edit {edit}    CER {cer:.4f}", sans)

    draw.text((pad, H - 30), "Text recognition evidence only. It does not verify the chemical structure and does not affect the mass gate.", font=small, fill=MUTED)

    path = OUT / "figure_1036_route.png"
    img.save(path)

    cap = ["Figure: 1036 printed SMILES, three recognition routes.", ""]
    cap.append(f"production (cell crop, 2x, no binarisation): edit {edit_ops(ref, prod) if prod else 'n/a'}")
    cap.append(f"+ threshold 200 (cell crop)                  : edit {edit_ops(ref, binr) if binr else 'n/a'}")
    cap.append(f"page route (page detect + line crop)         : edit {edit_ops(ref, route) if route else 'n/a'}")
    cap.append("")
    cap.append("The same comparison on 1037 (second human-verified")
    cap.append("reference): production edit 89, + threshold 200 edit 6,")
    cap.append("page route edit 0.")
    cap.append("")
    cap.append("Text recognition evidence only; the chemical structure is not verified by this figure.")
    io.open(OUT / "figure_1036_route_caption.txt", "w", encoding="utf-8").write("\n".join(cap))
    print("\n".join(cap))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
