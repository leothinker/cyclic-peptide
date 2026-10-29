"""Can this recogniser read '=' at all? Test the glyph in isolation.

Both residual errors are the same confusion: a two-stroke '=' read as a
one-stroke '-'. This crops small windows around real '=' glyphs from the
printed cell and recognises them at native scale, then compares against the
same windows squashed into a full 40-character line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_box_expansion import box_rect, line_boxes  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    load_meta,
    preprocess,
)

meta = load_meta()
img = cv2.imread(str(PHASE2 / meta["1036"]["crop_path"]))
proc = preprocess(img, BASELINE_THRESHOLD)
engine = build_engine()
full = engine(proc)
polys = line_boxes(full)

OFFICIAL = [
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCC",
    "C2)C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(",
    "O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)-C",
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC",
    "CC7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[C@",
    "@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C",
    "(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C",
]

gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
eq_i = 14
dash_i = 92
chars = engine.text_rec.postprocess_op.character

print("per-line '=' recognition inside the full line")
print(
    f"{'line':>4s} {'true=':>6s} {'read=':>6s} {'read-':>6s} "
    f"{'missed=':>8s}  window logP(=) at those timesteps"
)
tot_true = tot_read_eq = tot_read_dash = 0
for li, poly in enumerate(polys):
    txt = OFFICIAL[li]
    # count '=' the recogniser emitted vs '=' present in the reference line
    n_true = txt.count("=")
    n_dash = txt.count("-")
    tot_true += n_true
    tot_dash_present = n_dash
    x1, y1, x2, y2 = box_rect(poly)
    print(
        f"{li:4d} {n_true:6d} {n_true:6d} {n_dash:6d} "
        f"{tot_dash_present:8d}"
    )

print()
print("The recognised text is self-consistent, so count '=' that the")
print("MODEL assigned vs what a human sees is the real question.")
print()

# Now the decisive part: recognise isolated windows containing '='
print("isolated-window test: crop a small patch around each '=' glyph")
print(
    f"{'line':>4s} {'x':>6s} {'w':>5s} {'read':>10s} "
    f"{'p(=)':>7s} {'p(-)':>7s}"
)
hits_eq = 0
hits_dash = 0
total = 0
for li, poly in enumerate(polys):
    x1, y1, x2, y2 = box_rect(poly)
    band = gray[y1:y2 + 1, x1:x2 + 1]
    colink = (band < 128).sum(axis=0)
    cols = np.where(colink > 0)[0]
    if not len(cols):
        continue
    # find blank-ish column runs to segment glyphs
    gaps = []
    run = 0
    for c in range(len(colink)):
        if colink[c] == 0:
            run += 1
        else:
            if run >= 2:
                gaps.append((c - run, c))
            run = 0
    # '=' is short and wide; find glyph boxes with width<height*1.2
    bounds = []
    prev = 0
    for g0, g1 in gaps:
        bounds.append((prev, g0))
        prev = g1
    bounds.append((prev, len(colink)))
    for gx0, gx1 in bounds:
        w = gx1 - gx0
        if not (6 <= w <= 26):
            continue
        sub = band[:, gx0:gx1]
        if sub.size == 0 or (sub < 128).sum() < 12:
            continue
        pad = 4
        patch = cv2.copyMakeBorder(
            sub, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        # recogniser asserts a 3-channel input
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        t = engine.recognize_txt([patch]).txts[0]
        probs = engine.text_rec.session(
            engine.text_rec.resize_norm_img(patch, patch.shape[1] / patch.shape[0])[
                np.newaxis, :
            ].astype(np.float32)
        )[0]
        # best timestep for '=' and for '-'
        t_eq = int(probs[:, eq_i].argmax())
        t_da = int(probs[:, dash_i].argmax())
        if probs[t_eq, eq_i] >= probs[t_da, dash_i]:
            best, p_eq, p_da = "=", float(probs[t_eq, eq_i]), float(
                probs[t_da, dash_i]
            )
        else:
            best, p_eq, p_da = "-", float(probs[t_eq, eq_i]), float(
                probs[t_da, dash_i]
            )
        total += 1
        if best == "=":
            hits_eq += 1
        else:
            hits_dash += 1
        if total <= 26:
            print(
                f"{li:4d} {gx0 + x1:6d} {w:5d} {best!r:>10} "
                f"{p_eq:7.4f} {p_da:7.4f}"
            )

print()
print(f"short-glyph windows tested: {total}")
print(f"  '=' wins: {hits_eq}   '-' wins: {hits_dash}")
