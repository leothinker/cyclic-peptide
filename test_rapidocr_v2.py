"""
test_rapidocr_v2.py — full-image single-box test with newer/larger EN models.

Hypothesis: SMILES is pure ASCII, no Chinese needed. We treat the entire
test image as one bounding box (use_det=False) and compare three rec models:
   1. en_PP-OCRv4_rec_mobile   (default in test_rapidocr.py, ~10MB)
   2. en_PP-OCRv5_rec_mobile   (newer, same dict)
   3. en_PP-OCRv4_rec_server   (larger / more accurate, ~50MB)

For each model we dump: line text, per-char confidences, top-K ambiguity stats.

USAGE:
      python test_rapidocr_v2.py                       # default: testocr.jpg
      python test_rapidocr_v2.py path/to/img.jpg
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

# ---------------------------------------------------------------------------
# Hooks (same as test_rapidocr.py)
# ---------------------------------------------------------------------------
import rapidocr.ch_ppocr_rec.utils as rec_utils
import rapidocr.ch_ppocr_rec.main as rec_main

_RAW = {'preds': []}
_RECOUT = {'word_results': [], 'txts': [], 'scores': []}
_orig_ctc = rec_utils.CTCLabelDecode.__call__


def _capture_call(self, preds, return_word_box=False, **kwargs):
    _RAW['preds'].append(preds.copy())
    return _orig_ctc(self, preds, return_word_box, **kwargs)


rec_utils.CTCLabelDecode.__call__ = _capture_call

_orig_recog = rec_main.TextRecognizer.__call__


def _capture_recog(self, args):
    out = _orig_recog(self, args)
    if out.word_results:
        _RECOUT['word_results'].extend(out.word_results)
    if out.txts:
        _RECOUT['txts'].extend(out.txts)
    if out.scores:
        _RECOUT['scores'].extend(out.scores)
    return out


rec_main.TextRecognizer.__call__ = _capture_recog


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
# PP-OCRv5 server is the largest English rec model; download is ~50MB.
# The "rec_server" runs slower but typically has higher accuracy.
MODEL_VARIANTS = [
    {
        'name': 'en_PP-OCRv4_mobile',
        'params': {
            'Rec.ocr_version': OCRVersion.PPOCRV4,
            'Rec.model_type': ModelType.MOBILE,
        },
    },
    {
        'name': 'en_PP-OCRv5_mobile',
        'params': {
            'Rec.ocr_version': OCRVersion.PPOCRV5,
            'Rec.model_type': ModelType.MOBILE,
        },
    },
    {
        'name': 'en_PP-OCRv4_server',
        'params': {
            'Rec.ocr_version': OCRVersion.PPOCRV4,
            'Rec.model_type': ModelType.SERVER,
        },
    },
]


def make_engine(rec_params: dict) -> RapidOCR:
    """Build a recogniser-only engine. We disable detection by passing
    use_det=False at call-time, but the engine still needs the det config."""
    base = {
        'Det.lang_type': 'en',
        'Det.ocr_version': OCRVersion.PPOCRV4,
        'Det.model_type': ModelType.MOBILE,
        'Det.engine_type': EngineType.ONNXRUNTIME,
        'Det.task_type': TaskType.DET,
        'Rec.lang_type': 'en',
        'Rec.engine_type': EngineType.ONNXRUNTIME,
        'Rec.task_type': TaskType.REC,
        'Cls.lang_type': LangCls.CH.value,
        'Global.text_score': 0.0,
        'Global.return_word_box': True,
    }
    base.update(rec_params)
    return RapidOCR(params=base)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
img_path = sys.argv[1] if len(sys.argv) > 1 else 'testocr.jpg'
img = cv2.imread(img_path)
print(f'input image: {img_path}  shape={img.shape}\n')

# 1) optional: rough upscale before recognition (helps the rec model see
#    small character features; detection is disabled so it does not change
#    the "one box = whole image" semantics)
h0, w0 = img.shape[:2]
SCALE = 2
img_up = cv2.resize(img, (w0 * SCALE, h0 * SCALE), interpolation=cv2.INTER_CUBIC)
print(f'upscaled to: {img_up.shape}  (x{SCALE})\n')

all_results = {}

for variant in MODEL_VARIANTS:
    name = variant['name']
    print(f'=' * 78)
    print(f'  {name}')
    print(f'=' * 78)

    # clear captured state
    _RAW['preds'].clear()
    _RECOUT['word_results'].clear()
    _RECOUT['txts'].clear()
    _RECOUT['scores'].clear()

    try:
        engine = make_engine(variant['params'])
    except Exception as e:
        print(f'  build engine failed: {e}')
        continue

    t0 = time.perf_counter()
    result = engine(img_up, use_det=False, use_cls=False)
    dt = time.perf_counter() - t0

    if not result.txts:
        print(f'  no text recognised in {dt:.2f}s')
        continue

    txt = result.txts[0]
    sc = result.scores[0]

    # per-char confidence via captured WordInfo
    wr_list = _RECOUT['word_results']
    per_char = []
    if wr_list and wr_list[0] is not None and getattr(wr_list[0], 'confs', None):
        chars_in_line = ''.join(c for w in wr_list[0].words for c in w)
        for c, p in zip(chars_in_line, wr_list[0].confs):
            per_char.append((c, round(float(p), 4)))

    # ambiguity stats from raw softmax
    n_total = n_low = n_strong2 = n_similar2 = 0
    TOPK = 5
    if _RAW['preds']:
        preds = _RAW['preds'][0]                              # (1, T, 97)
        chars_dict = engine.text_rec.postprocess_op.character
        order = np.argsort(-preds[0], axis=-1)[:, :TOPK]      # (T, TOPK)
        probs = np.take_along_axis(preds[0], order, axis=-1)
        for t in range(preds.shape[1]):
            a_idx = int(order[t, 0])
            if chars_dict[a_idx] == 'blank':
                continue
            n_total += 1
            p1 = float(probs[t, 0])
            p2 = float(probs[t, 1])
            if chars_dict[int(order[t, 1])] != 'blank':
                if p2 > 0.05:
                    n_strong2 += 1
                if p2 > 0.3 * p1:
                    n_similar2 += 1
            if p1 < 0.8:
                n_low += 1

    print(f'  time           : {dt:.2f}s')
    print(f'  line score     : {sc:.4f}')
    print(f'  text length    : {len(txt)} chars')
    print(f'  text           : {txt!r}')
    print(f'  ambiguity      : {n_total} non-blank frames, '
          f'{n_low} low-p1, {n_strong2} strong-2nd, {n_similar2} similar-2nd')
    # show chars whose per-char conf < 0.7
    bad = [(i, c, p) for i, (c, p) in enumerate(per_char) if p < 0.7]
    print(f'  per-char confs < 0.7  : {len(bad)}/{len(per_char)}')
    if bad:
        sample = ', '.join(f'pos{i}={c}({p:.2f})' for i, c, p in bad[:10])
        print(f'     first 10        : {sample}')

    all_results[name] = {
        'time_sec': round(dt, 3),
        'line_score': round(float(sc), 6),
        'text': txt,
        'per_char': per_char,
        'ambiguity': {
            'non_blank_frames': n_total,
            'low_p1_lt_080': n_low,
            'strong_2nd_p2_gt_005': n_strong2,
            'similar_2nd_p2_gt_03p1': n_similar2,
        },
    }
    print()

# 2) Side-by-side text comparison
print('=' * 78)
print('  SIDE-BY-SIDE TEXT')
print('=' * 78)
ref = next((v['text'] for v in all_results.values() if v.get('text')), '')
for name, r in all_results.items():
    if not r.get('text'):
        continue
    print(f'\n[{name}]')
    print(f'  {r["text"]}')
    if ref and name != list(all_results.keys())[0]:
        # show char-level diff vs the first non-empty result
        a = list(all_results.values())[0]['text']
        b = r['text']
        diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if diffs:
            print(f'  diffs vs first: {len(diffs)}')
            for i, x, y in diffs[:5]:
                print(f'    pos {i}: {x!r} -> {y!r}')

# 3) Save JSON dump
out_json = Path(img_path).with_name(Path(img_path).stem + '_v2_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({'image': str(img_path), 'models': all_results},
              f, ensure_ascii=False, indent=2)
print(f'\nJSON dump: {out_json}')
