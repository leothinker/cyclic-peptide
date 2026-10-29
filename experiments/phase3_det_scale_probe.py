r"""The detector's input geometry for the printed-SMILES cell.

What this probe measures
------------------------
``rapidocr/ch_ppocr_det/utils.py`` resolves the detector input size as:

    if limit_type == "min":
        if min(h, w) < limit_side_len:   # UPSCALE the short side
            ratio = limit_side_len / min(h, w)
        else:
            ratio = 1.0                   # native resolution

with ``limit_side_len = 736`` and ``limit_type = "min"`` in the shipped
config, so the rule is not "shrink big pages" but "grow small crops".

The hypothesis this probe was written for was that the cell crop is
upscaled before detection while the page is not, and that this explains
why the page route reads 1036 correctly and the cell route does not.

**The measurement refutes that hypothesis for the production crop.**
The production cell crop is the compound's row band, not a single
printed line: for 1036 it is 831 x 625, and after the pipeline's own
2x upscale it is 1662 x 1250, whose short side is already above 736.
The detector therefore sees it at native resolution, exactly like the
page. The three regimes are measured and printed so the numbers are on
record rather than assumed.

What the probe does show is that the pixel scale of the crop matters:
the same 1036 text read from a 2x crop and from a 1x crop differs by
31 characters. That sensitivity is the reason no single-sample tuning
is trusted; see ``phase3_recrop_margin.py``.

Read-only. Writes to ``data/audit/phase3/det_scale/``.

Run:
    .venv\Scripts\python.exe phase3_det_scale_probe.py
"""

from __future__ import annotations

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

OUT = REPO / "data" / "audit" / "phase3" / "det_scale"


def detector_input_size(h: int, w: int, limit: int = 736) -> tuple[int, int]:
    """Reproduce DetPreProcess.resize for limit_type='min'."""
    if min(h, w) < limit:
        ratio = float(limit) / (h if h < w else w)
    else:
        ratio = 1.0
    rh = int(round(int(h * ratio) / 32) * 32)
    rw = int(round(int(w * ratio) / 32) * 32)
    return rh, rw


def effective_char_scale(h: int, w: int, limit: int = 736) -> float:
    """How much the detector magnifies the printed glyphs."""
    rh, rw = detector_input_size(h, w, limit)
    return (rw / w if w else 1.0)


def _recognise(engine, img):
    result = engine(img)
    if not result:
        return "", []
    boxes = list(result.boxes)
    txts = list(result.txts)
    items = sorted(zip(boxes, txts),
                   key=lambda bt: ((bt[0][0][1] + bt[0][2][1]) / 2))
    parts = [t.strip() for _, t in items if t and t.strip()]
    return "".join(parts), parts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cpd.parsers.table_extractor import RapidOcrTableExtractor

    ex = RapidOcrTableExtractor(rec_top_k=0)
    engine = baseline_engine()

    img = cv2.imread(str(GOLD / "PCTCN2025075389-ftappb-I100297.jpg"))
    cols = ex._detect_columns(img)
    col = next(c for c in cols if c.name == "smiles")
    rows = ex._detect_row_boundaries(img)

    lines: list[str] = []
    lines.append("=== detector input geometry, several regimes ===")
    lines.append("")
    lines.append(f"page image            : {img.shape[1]} x {img.shape[0]}")
    lines.append(f"smiles column x       : {col.x1} .. {col.x2}")
    lines.append(f"detector config       : limit_side_len="
                 f"{engine.cfg.Det.get('limit_side_len')} "
                 f"limit_type={engine.cfg.Det.get('limit_type')}")
    lines.append("")

    band = None
    for y1, y2 in rows:
        crop = img[y1:y2, col.x1:col.x2]
        if crop.size == 0:
            continue
        t, _ = _recognise(engine, ex._upscale(ex._trim_to_content(crop)))
        if t.startswith("CC[C@H]"):
            band = (y1, y2, crop)
            break
    if band is None:
        lines.append("could not locate the 1036 band")
        io.open(OUT / "det_scale.txt", "w",
                encoding="utf-8").write("\n".join(lines))
        print("\n".join(lines))
        return
    y1, y2, crop = band
    trimmed = ex._trim_to_content(crop)
    up = ex._upscale(trimmed)

    lines.append(f"1036 cell crop        : {crop.shape[1]} x "
                 f"{crop.shape[0]}  (page rows {y1}..{y2})")
    lines.append(f"  after trim          : {trimmed.shape[1]} x "
                 f"{trimmed.shape[0]}")
    lines.append(f"  after 2x upscale    : {up.shape[1]} x {up.shape[0]}")
    lines.append("")

    ref = REFERENCES["1036"]["smiles"]

    regimes = []
    regimes.append(("production (trim->2x, limit 736)", up, 736))
    regimes.append(("same pixels, det limit_side_len=32", up, 32))

    canvas = np.full((img.shape[0], img.shape[1], 3), 255, np.uint8)
    canvas[y1:y2, col.x1:col.x2] = crop
    regimes.append(("cell on page canvas (native scale)", canvas, 736))
    regimes.append(("whole page (native scale)", img, 736))

    lines.append(f"{'regime':<42} {'det in':>12} {'scale':>6} "
                 f"{'edit':>5} {'rows':>5} {'len':>5}")
    for name, image, limit in regimes:
        h, w = image.shape[:2]
        dh, dw = detector_input_size(h, w, limit)
        txt, parts = _recognise(engine, image)
        edit = edit_ops(ref, txt)
        lines.append(f"{name:<42} {dw:>5}x{dh:<6} "
                     f"{effective_char_scale(h, w, limit):>6.3f} "
                     f"{edit:>5} {len(parts):>5} {len(txt):>5}")

    lines.append("")
    lines.append("edit distance is against the human-verified 1036 "
                 "reference (text check only).")

    io.open(OUT / "det_scale.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
