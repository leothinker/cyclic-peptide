r"""Task 1 helper: group the per-event evidence strips into review grids.

Each cell shows the two panels that decide the question:

    top     the geometry panel, detector box in red and the excluded ink
            tinted, so it is visible whether the excluded ink is this row's
            own leading '=' or something else
    bottom  the re-crop panel, showing the same row with the left edge
            extended, which is what the A/B experiment will feed in

Grids are numbered so a reviewer's judgements can be cited back to an
image. Nothing here decides anything; the grids are only evidence.

Run:
    .venv\Scripts\python.exe pre_report_grids.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"

PER_GRID = 16
COLS = 4
CELL_W = 330


def load_strip(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    head = img[:22]
    body = img[22:]
    # body is [caption+panel] x 4 stacked
    per = body.shape[0] // 4
    geom = body[per:2 * per]
    recrop = body[3 * per:4 * per]
    return head, geom, recrop


def cell(head, geom, recrop, label):
    g = _resize_w(geom, CELL_W)
    r = _resize_w(recrop, CELL_W)
    h = max(g.shape[0], r.shape[0])
    g = _pad_h(g, h)
    r = _pad_h(r, h)
    body = np.vstack([g, r])
    top = _resize_w(head, CELL_W)
    top = _pad_h(top, body.shape[0] + 18)
    cap = np.full((18, CELL_W, 3), 255, np.uint8)
    cv2.putText(cap, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([cap, top[:18], body])


def _resize_w(img, w):
    k = w / img.shape[1]
    return cv2.resize(img, (w, max(1, int(img.shape[0] * k))),
                      interpolation=cv2.INTER_AREA)


def _pad_h(img, h):
    if img.shape[0] >= h:
        return img[:h]
    return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _pad_w(img, w):
    if img.shape[1] >= w:
        return img
    return cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1],
                              cv2.BORDER_CONSTANT, value=(190, 190, 190))


def main() -> None:
    with (OUTDIR / "review_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as fh:
        rows = list(csv.DictReader(fh))
    grid_dir = OUTDIR / "grids"
    grid_dir.mkdir(exist_ok=True)

    cells = []
    for r in rows:
        s = load_strip(OUTDIR / r["evidence"])
        if s is None:
            continue
        cells.append(cell(*s, r["event_id"]))

    n = 0
    for start in range(0, len(cells), PER_GRID):
        chunk = cells[start:start + PER_GRID]
        w = max(c.shape[1] for c in chunk)
        ch = max(c.shape[0] for c in chunk)
        padded = [
            cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, w - c.shape[1],
                               cv2.BORDER_CONSTANT, value=(190, 190, 190))
            for c in chunk
        ]
        rows_img = [
            np.hstack(padded[i:i + COLS])
            for i in range(0, len(padded), COLS)
        ]
        gh = max(r.shape[0] for r in rows_img)
        full_w = max(r.shape[1] for r in rows_img)
        rows_img = [_pad_w(_pad_h(r, gh), full_w) for r in rows_img]
        grid = np.vstack(rows_img)
        n += 1
        name = f"grid_{n:02d}_{chunk[0].shape[0] and ''}"[:9] + ".png"
        name = f"grid_{n:02d}.png"
        cv2.imwrite(str(grid_dir / name), grid)
        print(f"  {name}: {len(chunk)} events, {grid.shape}")
    print(f"{n} grids -> {grid_dir}")


if __name__ == "__main__":
    main()
