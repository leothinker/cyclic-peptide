"""Capture the exact recogniser tensors behind the baseline string.

Monkeypatches ``rec.session`` so we record (input, output) pairs from the
official ``recognize_txt`` call itself. No re-derivation of the padding
width, so captured posteriors provably correspond to the baseline text.
"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import line_boxes  # noqa: E402
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

lines = [get_rotate_crop_image(proc, np.array(p, dtype=np.float32)) for p in polys]
print(f"{len(lines)} rectified lines")
for i, ln in enumerate(lines):
    print(f"  line{i}: {ln.shape[1]}x{ln.shape[0]}  wh_ratio={ln.shape[1]/ln.shape[0]:.2f}")

rec = engine.text_rec
captured: list[tuple[np.ndarray, np.ndarray]] = []
real_session = rec.session


class Spy:
    have_key = staticmethod(real_session.have_key)
    get_character_list = staticmethod(real_session.get_character_list)
    get_input_names = staticmethod(real_session.get_input_names)
    get_output_names = staticmethod(real_session.get_output_names)

    def __call__(self, x):
        out = real_session(x)
        captured.append((x.copy(), out.copy()))
        return out

    def __getattr__(self, name):
        return getattr(real_session, name)


rec.session = Spy()
out = engine.recognize_txt(lines)
rec.session = real_session

txts = list(out.txts)
print()
print(f"captured {len(captured)} session call(s); rec_batch_num={rec.rec_batch_num}")
for bi, (x, y) in enumerate(captured):
    print(f"  batch{bi}: input {x.shape} -> output {y.shape}")
    print(f"           padded width = {x.shape[3]} "
          f"(= imgH {rec.rec_image_shape[1]} * ratio {x.shape[3]/rec.rec_image_shape[1]:.2f})")

chars = rec.postprocess_op.character
print(f"\ncharacter dict size = {len(chars)}; "
      f"'=' idx {chars.index('=')}, '-' idx {chars.index('-')}, blank idx 0")

# map batch -> lines, replicating the official width-sort
width_list = [ln.shape[1] / float(ln.shape[0]) for ln in lines]
indices = np.argsort(np.array(width_list))
bnum = rec.rec_batch_num
assigned: dict[int, list[int]] = {}
for beg in range(0, len(lines), bnum):
    end = min(len(lines), beg + bnum)
    for rno, ino in enumerate(indices[beg:end]):
        assigned.setdefault(beg // bnum, []).append((rno, int(ino)))

eq_i, dash_i, blank_i = chars.index("="), chars.index("-"), 0
print()
for bi, (x, y) in enumerate(captured):
    print(f"--- batch {bi}: lines {[l for _, l in assigned[bi]]} ---")
    for rno, lidx in assigned[bi]:
        preds = y[rno]
        idxs = preds.argmax(axis=1)
        sel = np.ones(len(idxs), dtype=bool)
        sel[1:] = idxs[1:] != idxs[:-1]
        sel &= idxs != blank_i
        kept = np.where(sel)[0]
        dec = "".join(chars[i] for i in idxs[sel])
        pe = preds[:, eq_i]
        pd_ = preds[:, dash_i]
        print(f"  line{lidx} steps={preds.shape[0]} kept={len(kept)} "
              f"match={dec == txts[lidx]} len={len(dec)}")
        print(f"    max p(=)={pe.max():.4f} at t={int(pe.argmax())}  "
              f"n steps p(=)>=.05={int((pe >= 0.05).sum())}  "
              f"n steps p(=)>=.10={int((pe >= 0.10).sum())}")
        print(f"    max p(-)={pd_.max():.4f} at t={int(pd_.argmax())}")
        # where '=' is at least 2nd choice
        rank = (preds > pe[:, None]).sum(axis=1)
        cand = np.where((pe >= 0.01) & (rank <= 2))[0]
        print(f"    timesteps with p(=)>=.01 and rank<=2: {cand.tolist()}")
        for t in cand:
            o = np.argsort(-preds[t])[:5]
            print(f"      t={t:3d} argmax={chars[int(preds[t].argmax())]!r} "
                  f"{[(chars[i], round(float(preds[t][i]), 4)) for i in o]}")
    print()

# ---- the deleted '=' sits on the line1/line2 junction ----
print("=== junction: line1 tail / line2 head, full detail ===")
per_line: dict[int, np.ndarray] = {}
for bi, (x, y) in enumerate(captured):
    for rno, lidx in assigned[bi]:
        per_line[lidx] = y[rno]

for lidx, ts, label in (
    (1, range(120, 144), "line1 tail"),
    (2, range(0, 24), "line2 head"),
):
    preds = per_line[lidx]
    print(f"--- {label} ---")
    for t in ts:
        p = preds[t]
        o = np.argsort(-p)[:4]
        print(f"  t={t:3d} argmax={chars[int(p.argmax())]!r:8s} "
              f"{[(chars[i], round(float(p[i]), 4)) for i in o]}")
