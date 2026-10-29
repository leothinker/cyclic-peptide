"""Inspect raw CTC timesteps for the dropped '=' at the end of line 1."""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from phase2_box_expansion import line_boxes  # noqa: E402
from phase2_threshold_scan import build_engine, preprocess  # noqa: E402

img = cv2.imread("data/audit/phase2/cells_text/I100297_1036_text.png")
proc = preprocess(img, 200)
engine = build_engine()
polys = line_boxes(engine(proc))
rec = engine.text_rec
chars = rec.postprocess_op.character
print("char dict len:", len(chars))
print("index 0 =", repr(chars[0]))
eq_i = chars.index("=")
dash_i = chars.index("-")
print(f"'=' idx {eq_i}, '-' idx {dash_i}")

imgH, imgW = rec.rec_image_shape[1], rec.rec_image_shape[2]
max_wh_ratio = imgW / imgH

for li in (1, 2):
    line = get_rotate_crop_image(proc, np.array(polys[li], dtype=np.float32))
    norm = rec.resize_norm_img(line, max_wh_ratio)[np.newaxis, :].astype(np.float32)
    preds = rec.session(norm)[0]
    print()
    print(f"=== line {li}: preds {preds.shape} ===")
    # greedy CTC path, but report every timestep including blanks
    for t in range(preds.shape[0]):
        p = preds[t]
        order = np.argsort(-p)[:4]
        top = [(chars[i], round(float(p[i]), 4)) for i in order if i < len(chars)]
        arg = int(p.argmax())
        label = chars[arg] if arg < len(chars) else "?"
        keep = "" if arg != 0 else "  <-- BLANK"
        print(f"  t={t:3d} argmax={label!r:6s} {top}{keep}")
    print(f"  -- timesteps where '=' ranks 2nd or 3rd (p>=0.02) --")
    for t in range(preds.shape[0]):
        p = preds[t]
        order = np.argsort(-p)
        rank = int(np.where(order == eq_i)[0][0])
        if rank <= 2 and p[eq_i] >= 0.02:
            arg = chars[int(p.argmax())]
            top = [(chars[i], round(float(p[i]), 4)) for i in order[:4]]
            print(
                f"     t={t:3d} argmax={arg!r} '=' rank{rank} "
                f"p={p[eq_i]:.4f} dash_p={p[dash_i]:.4f} {top}"
            )
