"""
test_rapidocr_hybrid.py — Chinese detection + English recognition.

Hypothesis: on dense patent text, the Chinese det model gives better boxes
than the English det model (which over-segments). But SMILES recognition
should use the English rec model (its dict is pure ASCII, not the 18k-char
Chinese dict). So: use the best of each.

USAGE:
      python test_rapidocr_hybrid.py                       # testocr.jpg
      python test_rapidocr_hybrid.py path/to/img.jpg
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType, LangCls, ModelType, OCRVersion, TaskType,
)

img_path = sys.argv[1] if len(sys.argv) > 1 else 'testocr.jpg'
img = cv2.imread(img_path)
print(f'input: {img_path}  shape: {img.shape}\n')

# ---------------------------------------------------------------------------
# Stage 1: Chinese detection (the ch_PP-OCRv6 model is robust on dense text)
# ---------------------------------------------------------------------------
print('stage 1: Chinese detection (ch_PP-OCRv6)')
det_engine = RapidOCR()  # default = ch_PP-OCRv6
t0 = time.perf_counter()
det_result = det_engine(img, use_cls=False, use_rec=False)
dt_det = time.perf_counter() - t0
print(f'  time: {dt_det:.2f}s,  regions found: '
      f'{len(det_result.boxes) if det_result.boxes is not None else 0}')

# also run with default to compare
print('stage 1b: English detection (en_PP-OCRv3) for comparison')
en_det_engine = RapidOCR(params={
    'Det.lang_type': 'en', 'Det.ocr_version': OCRVersion.PPOCRV4,
    'Det.model_type': ModelType.MOBILE,
    'Det.engine_type': EngineType.ONNXRUNTIME, 'Det.task_type': TaskType.DET,
})
en_det = en_det_engine(img, use_cls=False, use_rec=False)
print(f'  regions found: '
      f'{len(en_det.boxes) if en_det.boxes is not None else 0}')

# ---------------------------------------------------------------------------
# Stage 2: English recognition on each detected region
# ---------------------------------------------------------------------------
print('\nstage 2: English recognition on each detected region')
rec_engine = RapidOCR(params={
    'Det.lang_type': 'en', 'Det.ocr_version': OCRVersion.PPOCRV4,
    'Det.model_type': ModelType.MOBILE,
    'Det.engine_type': EngineType.ONNXRUNTIME, 'Det.task_type': TaskType.DET,
    'Rec.lang_type': 'en', 'Rec.ocr_version': OCRVersion.PPOCRV4,
    'Rec.model_type': ModelType.MOBILE,
    'Rec.engine_type': EngineType.ONNXRUNTIME, 'Rec.task_type': TaskType.REC,
    'Cls.lang_type': LangCls.CH.value,
    'Global.text_score': 0.0,
    'Global.return_word_box': False,
})


def recognise_with_boxes(boxes, name):
    """Crop each box, recognise it, return [(text, conf, box)]."""
    if boxes is None:
        return []
    out = []
    for i, box in enumerate(boxes):
        xs = box[:, 0]
        ys = box[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # add 4px margin
        x1 = max(0, x1 - 4); y1 = max(0, y1 - 2)
        x2 = min(img.shape[1], x2 + 4); y2 = min(img.shape[0], y2 + 2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Resize so the height is between 32 and 64 pixels, keeping aspect ratio
        h, w = crop.shape[:2]
        target_h = 48
        if h < target_h:
            scale = target_h / h
            crop = cv2.resize(crop, (int(w * scale), target_h),
                              interpolation=cv2.INTER_CUBIC)
        res = rec_engine(crop, use_det=False, use_cls=False)
        if res.txts and res.txts[0]:
            txt = res.txts[0]
            sc = float(res.scores[0]) if res.scores else 0.0
        else:
            txt, sc = '', 0.0
        out.append({'i': i, 'text': txt, 'score': round(sc, 4),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]})
    return out


# Try BOTH detectors, see which gives more usable text after EN recognition
t0 = time.perf_counter()
zh_then_en = recognise_with_boxes(det_result.boxes, 'zh_det')
en_then_en = recognise_with_boxes(en_det.boxes, 'en_det')
dt_rec = time.perf_counter() - t0
print(f'  recognition time (both detectors): {dt_rec:.2f}s')

# Filter: only keep recognitions with confidence > 0.3 and length > 0
def keep(r):
    return r['score'] > 0.3 and r['text']

zh_kept = [r for r in zh_then_en if keep(r)]
en_kept = [r for r in en_then_en if keep(r)]

print(f'\n  zh-det + en-rec  : {len(zh_kept)}/{len(zh_then_en)} kept')
print(f'  en-det + en-rec  : {len(en_kept)}/{len(en_then_en)} kept')

# Sort each by y coordinate so we read top-to-bottom
zh_kept.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
en_kept.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

print('\n  ZH-DET then EN-REC:')
for r in zh_kept:
    print(f"    y={r['bbox'][1]:>4}  conf={r['score']:.2f}  {r['text']!r}")

print('\n  EN-DET then EN-REC:')
for r in en_kept:
    print(f"    y={r['bbox'][1]:>4}  conf={r['score']:.2f}  {r['text']!r}")

# ---------------------------------------------------------------------------
# Stage 3: visualise - draw the boxes from each approach
# ---------------------------------------------------------------------------
out_vis = Path(img_path).with_name(Path(img_path).stem + '_hybrid.jpg')
vis = img.copy()
for r in zh_kept:
    x1, y1, x2, y2 = r['bbox']
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(vis, r['text'][:20], (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
cv2.imwrite(str(out_vis), vis)
print(f'\nvis image: {out_vis}')

# ---------------------------------------------------------------------------
# Stage 4: JSON dump
# ---------------------------------------------------------------------------
out_json = Path(img_path).with_name(Path(img_path).stem + '_hybrid.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({
        'image': str(img_path),
        'zh_det_then_en_rec': zh_kept,
        'en_det_then_en_rec': en_kept,
    }, f, ensure_ascii=False, indent=2)
print(f'JSON dump: {out_json}')
