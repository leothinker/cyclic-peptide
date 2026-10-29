r"""Task C: visual evidence panels for the row-crop audit.

Builds, for a spread of flagged cells, a three-panel strip:

    left  the neighbourhood of the offending detector box, in place
    mid   the same neighbourhood with the box edges drawn
    right what the recogniser actually received, i.e. the box interior

so a reviewer can see the excluded glyph, see the box that excluded it and
see that the glyph is absent from the recogniser's input, without trusting
any number in the manifest.

Read-only. Writes only into data/audit/phase2/row_crop_audit/.

Run:
    .venv\Scripts\python.exe row_crop_audit_montage.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    build_engine,
)

from row_crop_audit import (  # noqa: E402
    INK,
    OUT,
    RULE_FRAC,
    RULE_PAD,
    box_rect,
    dilate,
    line_boxes,
    preprocess,
)
from row_crop_audit_batch import overhang_profile  # noqa: E402

PANELS = ("context", "box_marked", "as_recognised")
MONTAGE_MAX = 12


def build_panel(engine, rec, tag):
    """Three-panel strip for the first glyph_excluded line of one cell."""
    path = REPO / "data" / "audit" / "phase2" / rec["crop_path"]
    img = cv2.imread(str(path))
    if img is None:
        return None
    proc = preprocess(img, BASELINE_THRESHOLD)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rules = np.where((gray < INK).sum(axis=0) > h * RULE_FRAC)[0]
    excluded = dilate(rules, RULE_PAD, w)
    polys = line_boxes(engine(proc))
    target = None
    for li, poly in enumerate(polys):
        lp, _, _, lab = overhang_profile(gray, poly, excluded, "left")
        if lab == "glyph_excluded":
            target = (li, poly)
            break
    if target is None:
        return None
    li, poly = target
    x1, y1, x2, y2 = box_rect(poly)
    y0, y1b = max(0, y1 - 8), min(h, y2 + 8)
    pad = 120
    c0 = max(0, x1 - pad)
    c1 = min(w, x1 + 260)
    ctx = gray[y0:y1b, c0:c1]
    marked = cv2.cvtColor(ctx, cv2.COLOR_GRAY2BGR)
    cv2.line(marked, (x1 - c0, 0), (x1 - c0, marked.shape[0] - 1),
             (0, 0, 255), 2)
    asrec = gray[max(0, y1):min(h, y2 + 1), x1:min(w, x1 + 260)]
    return _assemble([ctx, marked, asrec], rec["cmpd_id"], li,
                      x1, int(rec["worst_left_overhang_px"]))


def _fit(img, target_h=150):
    if img.size == 0:
        return np.full((target_h, target_h, 3), 255, np.uint8)
    k = max(1, target_h // max(img.shape[0], 1))
    up = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    if up.shape[0] > target_h:
        top = (up.shape[0] - target_h) // 2
        up = up[top:top + target_h]
    if up.ndim == 2:
        up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    if up.shape[0] < target_h:
        up = cv2.copyMakeBorder(
            up, 0, target_h - up.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
    return up


def _label(img, text):
    bar = np.full((20, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _assemble(images, cid, li, x1, overhang):
    tiles = []
    for name, im in zip(PANELS, images):
        t = _label(_fit(im, 110), name)
        tiles.append(t)
    wmax = max(t.shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for t in tiles]
    body = np.vstack(padded)
    head = np.full((22, body.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"cmpd {cid}  L{li}  box x1={x1}  overhang "
                f"{overhang}px", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, body])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_header"] == "no"]
    flagged = [r for r in rows if int(r["n_lines_glyph_excluded"] or 0) > 0]

    seen: set[str] = set()
    picked = []
    for r in flagged:                     # spread across pages first
        if len(picked) >= MONTAGE_MAX:
            break
        if r["page"] not in seen:
            seen.add(r["page"])
            picked.append(r)
    for r in flagged:                     # then fill the remaining slots
        if len(picked) >= MONTAGE_MAX:
            break
        if r["cmpd_id"] not in {p["cmpd_id"] for p in picked}:
            picked.append(r)

    engine = build_engine()
    strips = []
    for r in picked:
        s = build_panel(engine, r, r["cmpd_id"])
        if s is not None:
            strips.append(s)
            cv2.imwrite(str(OUT / f"panel_{r['cmpd_id']}_L.png"), s)
    print(f"built {len(strips)} panels from {len(picked)} flagged cells")

    blocks = [s for s in strips if s is not None]
    width = max(b.shape[1] for b in blocks)
    blocks = [cv2.copyMakeBorder(b, 0, 0, 0, width - b.shape[1],
                                  cv2.BORDER_CONSTANT, value=(210, 210, 210))
              for b in blocks]
    height = max(b.shape[0] for b in blocks)
    blocks = [cv2.copyMakeBorder(b, 0, height - b.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(210, 210, 210))
              for b in blocks]
    cols = 4
    rows = [
        np.hstack(blocks[i:i + cols])
        for i in range(0, len(blocks), cols)
    ]
    montage = np.vstack(rows)
    cv2.imwrite(str(OUT / "flagged_montage.png"), montage)
    print(f"wrote {OUT / 'flagged_montage.png'}  {montage.shape}")


if __name__ == "__main__":
    main()
