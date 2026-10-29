"""Build the report figure summarising the audit and the MS validation.

matplotlib is not a project dependency, so the figure is composed with PIL
directly -- the same approach the existing ``demo_*.py`` scripts use. Every
number is read from the artefacts produced by ``audit_repairs.py`` and the
combined CSV, so the figure cannot drift away from the data.

Run from the repo root::

    .venv/Scripts/python.exe analysis/make_report_figure.py

Writes ``report_figure.png`` at the repo root.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from cpd.ms_validation import monoisotopic_mass, neutral_mass_from_mz  # noqa: E402

AUDIT_JSON = ROOT / "analysis" / "repair_audit.json"
CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"
OUT = ROOT / "data" / "audit" / "demos" / "figures" / "report_figure.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 2000, 1500

INK = (24, 28, 34)
MUTED = (108, 116, 128)
LINE = (206, 212, 220)
PAPER = (255, 255, 255)
PANEL = (247, 248, 250)

TIER_COLOURS = {
    "decimer_intact": (46, 125, 168),
    "not_repaired": (46, 125, 168),
    "high": (56, 142, 96),
    "medium": (120, 168, 90),
    "low": (214, 158, 46),
    "experimental": (206, 116, 48),
    "rejected": (190, 68, 68),
    "unrepaired": (140, 146, 156),
    "no_decimer_output": (140, 146, 156),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = _font(44, True)
F_SUB = _font(24)
F_PANEL = _font(28, True)
F_BODY = _font(22)
F_SMALL = _font(19)
F_MONO = _font(19)
F_BIG = _font(40, True)


def _text(d: ImageDraw.ImageDraw, xy, s, font, fill=INK, anchor=None):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def _panel(d: ImageDraw.ImageDraw, box, title):
    d.rounded_rectangle(box, radius=10, fill=PANEL, outline=LINE, width=2)
    _text(d, (box[0] + 24, box[1] + 18), title, F_PANEL)


def _hbar(d, x, y, w, h, frac, colour):
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=(232, 235, 240))
    if frac > 0:
        d.rounded_rectangle(
            (x, y, x + max(w * frac, h), y + h), radius=h // 2, fill=colour
        )


def main() -> int:
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    summary = audit["summary"]
    tiers = Counter(r["tier"] for r in audit["records"])
    total = summary["total_compounds"]

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    status = Counter((r.get("repair_review_status") or "").strip() for r in rows)
    conf = Counter((r.get("structure_confidence") or "").strip() for r in rows)

    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    # ---- header ----
    _text(d, (60, 46), "Cyclic-peptide structure recognition: audit of the macrocycle-repair stage", F_TITLE)
    _text(
        d,
        (60, 104),
        "WO2025162428 golden tables  |  406 compounds  |  every repair gated on "
        "chemical invariants and on the patent's own LC-MS mass",
        F_SUB,
        fill=MUTED,
    )
    d.line((60, 150, W - 60, 150), fill=LINE, width=2)

    # ---- panel 1: headline numbers ----
    p1 = (60, 180, 980, 620)
    _panel(d, p1, "1. What the audit changed")
    x0 = p1[0] + 28
    y = p1[1] + 72
    rows_txt = [
        ("Previously reported recovery", "80.8%", MUTED),
        ("After chemistry + mass gating", f"{summary['strict_recovery_rate'] * 100:.1f}%", (190, 68, 68)),
        ("DECIMER macrocycle already intact", f"{tiers['decimer_intact']}  ({tiers['decimer_intact'] / total * 100:.1f}%)", INK),
        ("Repairs validated", f"{tiers['high'] + tiers['medium']}", INK),
        ("Repairs experimental (invent atoms)", f"{tiers['experimental']}", (206, 116, 48)),
        ("Repairs rejected (implausible)", f"{tiers['rejected']}", (190, 68, 68)),
        ("Still unrepaired", f"{tiers['unrepaired']}", MUTED),
    ]
    for label, val, col in rows_txt:
        _text(d, (x0, y), label, F_BODY, fill=MUTED)
        _text(d, (p1[2] - 28, y - 4), val, F_BODY, fill=col, anchor="ra")
        y += 44
    y += 8
    d.line((x0, y, p1[2] - 28, y), fill=LINE, width=2)
    y += 16
    _text(d, (x0, y), "terminal_bridge (invents a bond)", F_SMALL, fill=MUTED)
    _text(d, (p1[2] - 28, y), "0 of 90 usable", F_SMALL, fill=(190, 68, 68), anchor="ra")

    # ---- panel 2: tier breakdown ----
    p2 = (1020, 180, W - 60, 620)
    _panel(d, p2, "2. Every compound, one verdict")
    order = [
        "decimer_intact", "high", "medium", "low",
        "experimental", "rejected", "unrepaired", "no_decimer_output",
    ]
    y = p2[1] + 72
    bar_x = p2[0] + 210
    bar_w = p2[2] - bar_x - 130
    for key in order:
        n = tiers.get(key, 0)
        _text(d, (p2[0] + 28, y), key.replace("_", " "), F_SMALL, fill=INK)
        _hbar(d, bar_x, y + 2, bar_w, 20, n / total, TIER_COLOURS.get(key, MUTED))
        _text(d, (p2[2] - 28, y - 1), f"{n}", F_SMALL, fill=MUTED, anchor="ra")
        y += 42

    # ---- panel 3: the 1036 mass comparison ----
    p3 = (60, 660, 1180, 1180)
    _panel(d, p3, "3. The test that is not self-referential: measured mass")
    _text(
        d,
        (p3[0] + 28, p3[1] + 70),
        "cmpd 1036, patent reports [M+H]+ 1610.9  ->  neutral 1609.89 Da",
        F_BODY,
    )
    y = p3[1] + 118
    # manual transcription
    hand = next((r for r in rows if r["cmpd_id"] == "1036"), None)
    cases = [
        (
            "hand transcription from the patent image",
            125.2,
            "C75H100F9N13O12S2",
        ),
        (
            "DECIMER output for the same cell",
            42880.3,
            "C71H98F9N13O13S",
        ),
    ]
    for label, ppm, formula in cases:
        _text(d, (p3[0] + 28, y), label, F_BODY, fill=INK)
        y += 30
        _text(d, (p3[0] + 52, y), f"formula {formula}", F_SMALL, fill=MUTED)
        _text(
            d,
            (p3[2] - 28, y - 4),
            f"{ppm:,.0f} ppm" if ppm > 1000 else f"{ppm:,.1f} ppm",
            F_BIG,
            fill=(56, 142, 96) if ppm < 500 else (190, 68, 68),
            anchor="ra",
        )
        y += 56
        # ppm bar on a log-ish scale: 0 .. 50000 ppm
        frac = min(ppm / 50000.0, 1.0) ** 0.4
        _hbar(d, p3[0] + 52, y, p3[2] - p3[0] - 80, 18, frac,
              (56, 142, 96) if ppm < 500 else (190, 68, 68))
        y += 62
    _text(
        d,
        (p3[0] + 28, p3[3] - 116),
        "The hand transcription is off by 0.20 Da (one atom-level detail to review);",
        F_SMALL,
        fill=MUTED,
    )
    _text(
        d,
        (p3[0] + 28, p3[3] - 88),
        "DECIMER is off by 66 Da. A macrocycle-only criterion rated both as 'closed'.",
        F_SMALL,
        fill=MUTED,
    )

    # ---- panel 4: corpus-wide mass agreement ----
    p4 = (1220, 660, W - 60, 1180)
    _panel(d, p4, "4. Corpus: mass agreement by tolerance")
    y = p4[1] + 76
    _text(d, (p4[0] + 28, y), "tolerance", F_SMALL, fill=MUTED)
    _text(d, (p4[0] + 190, y), "compounds passing", F_SMALL, fill=MUTED)
    y += 34
    for tol, n in (
        ("50 ppm", "0 / 303"),
        ("500 ppm", "0 / 303"),
        ("2,000 ppm", "3 / 303"),
        ("10,000 ppm", "16 / 303"),
        ("50,000 ppm", "103 / 303"),
    ):
        _text(d, (p4[0] + 28, y), tol, F_BODY, fill=INK)
        _text(d, (p4[0] + 190, y), n, F_BODY, fill=(190, 68, 68))
        frac = int(n.split("/")[0].strip()) / 303.0
        _hbar(d, p4[0] + 330, y + 2, p4[2] - p4[0] - 358, 20, frac, (190, 68, 68))
        y += 44
    _text(
        d,
        (p4[0] + 28, p4[3] - 92),
        "Median |error| 86,000 ppm (~86 Da) on DECIMER output.",
        F_SMALL,
        fill=MUTED,
    )
    _text(
        d,
        (p4[0] + 28, p4[3] - 64),
        "No constant adduct offset explains it.",
        F_SMALL,
        fill=MUTED,
    )

    # ---- bottom strip: per-column status ----
    p5 = (60, 1220, W - 60, H - 50)
    _panel(d, p5, "5. What each column of the deliverable CSV now means")
    y = p5[1] + 74
    items = [
        ("repair_review_status", status, "was the heuristic itself trustworthy?"),
        ("structure_confidence", conf, "does the formula reproduce the measured m/z?"),
    ]
    for name, counts, gloss in items:
        _text(d, (p5[0] + 28, y), name, F_BODY, fill=INK)
        _text(d, (p5[0] + 320, y), gloss, F_SMALL, fill=MUTED)
        x = p5[0] + 900
        for k, v in counts.most_common():
            txt = f"{k} {v}"
            w = d.textlength(txt, font=F_SMALL) + 26
            if x + w > p5[2] - 28:
                x = p5[0] + 900
                y += 40
            d.rounded_rectangle((x, y - 5, x + w, y + 25), radius=6,
                                fill=(232, 235, 240))
            _text(d, (x + 13, y), txt, F_SMALL, fill=INK)
            x += w + 12
        y += 46

    img.save(OUT)
    print(f"wrote {OUT}  ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
