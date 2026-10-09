"""
test_rapidocr.py — single-image test for RapidOCR with EN model + per-character confidences + top-K.

USAGE:
      python test_rapidocr.py                       # default image:  data/processed/cells/PCTCN2025075389-ftappb-I100280_1001_struct.png
      python test_rapidocr.py path/to/img.jpg       # any image path

PREREQUISITES (first run will auto-download ~30 MB into .venv/Lib/site-packages/rapidocr/models/):
      det: en_PP-OCRv3_det_mobile.onnx
      rec: en_PP-OCRv4_rec_mobile.onnx
      dict: en_dict.txt (96 ASCII chars + specials)

OUTPUT (printed to stdout):
   - N recognised text regions: text + line-level score
   - per-character confidence from word_results
   - top-5 character alternatives per non-blank frame from the raw softmax
   - ambiguity statistics (how often is there a useful second-best candidate?)

HOW TO SWITCH MODELS:
   * For Chinese (default):            engine = RapidOCR()
   * For English:                      engine = RapidOCR(params={'Det.lang_type': 'en',
                                                                'Rec.lang_type': 'en'})
   * For EN + PP-OCRv5 (newer, less battle-tested):
                                       from rapidocr.utils.typings import OCRVersion
                                       params['Rec.ocr_version'] = OCRVersion.PPOCRV5
   * For digits-only (number recipes):                 engine = RapidOCR(params={'Rec.lang_type': 'en'})  # same EN model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    LangCls,
    ModelType,
    OCRVersion,
    TaskType,
)

# ===========================================================================
# 1. ENGINE INITIALISATION  (English model)
# ===========================================================================
#
# RapidOCR takes a dict of dotted-path params, e.g. 'Det.lang_type'.
# lang_type accepts a plain string ('en'); but engine_type / model_type /
# ocr_version / task_type REQUIRE the actual Enum (the parser validates).
#
# Available en models:
#   en_PP-OCRv4_rec_mobile  (default, well-tested, 96-char ASCII dict)
#   en_PP-OCRv5_rec_mobile  (newer, same dict)
#
engine = RapidOCR(params={
    # ---- Detection (English text detector) ----
    'Det.lang_type': 'en',
    'Det.ocr_version': OCRVersion.PPOCRV4,
    'Det.model_type': ModelType.MOBILE,
    'Det.engine_type': EngineType.ONNXRUNTIME,
    'Det.task_type': TaskType.DET,
    # ---- Recognition (English text recogniser) ----
    'Rec.lang_type': 'en',
    'Rec.ocr_version': OCRVersion.PPOCRV4,           # swap to PPOCRV5 for newer EN
    'Rec.model_type': ModelType.MOBILE,
    'Rec.engine_type': EngineType.ONNXRUNTIME,
    'Rec.task_type': TaskType.REC,
    # ---- Classifier (Chinese-only, kept for orientation) ----
    'Cls.lang_type': LangCls.CH.value,
    # ---- Engine-wide ----
    'Global.text_score': 0.0,                        # don't drop low-confidence text
    'Global.return_word_box': True,                  # enables per-character confidence
    'Global.use_cls': True,                          # optional: 0/180-degree rotation
})

# ===========================================================================
# 2. HOOKS: (a) raw softmax for top-K   (b) TextRecOutput with WordInfo.confs
# ===========================================================================
# Default flow inside rapidocr.ch_ppocr_rec.main.TextRecognizer.__call__:
#
#     preds = self.session(norm_img_batch)            # shape (B, T, num_classes)
#     line_results, word_results = self.postprocess_op(preds, ...)
#
# Default postprocess_op = CTCLabelDecode.__call__:
#     preds_idx = preds.argmax(axis=2)                # <-- keeps ONLY argmax char
#     preds_prob = preds.max(axis=2)                  # <-- keeps ONLY its prob
#     ... collapse repeats/blanks ...
#
# We replace __call__ with a wrapper that ALSO captures `preds` so we can pull
# top-K from the raw softmax BEFORE the default decoder discards it.
import rapidocr.ch_ppocr_rec.utils as rec_utils
import rapidocr.ch_ppocr_rec.main as rec_main

# The recognizer runs rec_batch_num (default 6) images at a time and
# overwrites preds each iteration. We accumulate every batch.
_RAW: dict = {'preds': [], 'batch_sizes': [], 'aspect_orders': []}
_RECOUT: dict = {'word_results': [], 'txts': [], 'scores': []}
_orig_ctc = rec_utils.CTCLabelDecode.__call__


def _capture_call(self, preds, return_word_box=False, **kwargs):
    _RAW['preds'].append(preds.copy())                 # (B, T, num_classes) per batch
    _RAW['batch_sizes'].append(preds.shape[0])
    return _orig_ctc(self, preds, return_word_box, **kwargs)


rec_utils.CTCLabelDecode.__call__ = _capture_call

_orig_recog = rec_main.TextRecognizer.__call__


def _capture_recog(self, args):
    out = _orig_recog(self, args)
    # out.word_results is tuple-of-WordInfo per batch; concat across batches
    if out.word_results:
        _RECOUT['word_results'].extend(out.word_results)
    if out.txts:
        _RECOUT['txts'].extend(out.txts)
    if out.scores:
        _RECOUT['scores'].extend(out.scores)
    return out


rec_main.TextRecognizer.__call__ = _capture_recog

# ===========================================================================
# 3. RUN OCR
# ===========================================================================
DEFAULT_IMG = 'data/processed/cells/PCTCN2025075389-ftappb-I100280_1001_struct.png'
img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
print(f'OCR input: {img_path}')
img = cv2.imread(img_path)
result = engine(img)

# ===========================================================================
# 4. WHAT YOU CAN READ FROM result (the easy stuff)
# ===========================================================================
# result.txt / result.txt            tuple[str, ...]  recognised text per region
# result.scores                      tuple[float, ...] line-level MEAN of per-char confidences
# result.boxes                       np.ndarray       4-point corner coordinates of each region
# result.word_results[i]             WordInfo         per-word info (incl. per-char confidence)
#   ├─ .words                         list[list[str]] grouped chars per word
#   ├─ .word_cols                     list[list[int]]  model-frame column index per char
#   ├─ .word_types                    list[WordType]  CN/EN_NUM per word
#   └─ .confs                         list[float]      per-character confidence (max prob)
# result.vis(path)                   np.ndarray       save annotated image
#
print(f'\n=== recognised {len(result.txts)} text regions (line-level score) ===')
for i, (txt, sc) in enumerate(zip(result.txts, result.scores)):
    flag = ''
    if sc < 0.7:
        flag = '  <-- low confidence'
    if '@' in txt or '=' in txt:
        flag += '   (SMILES-like)'
    print(f'  [{i:>2}]  conf={sc:.4f}   {txt[:80]!r}{"..." if len(txt)>80 else ""}{flag}')

# ===========================================================================
# 5. PER-CHARACTER CONFIDENCE  (from word_results)
# ===========================================================================
print('\n=== per-character confidence (WordInfo.confs, captured before rewrite) ===')
wr_list = _RECOUT.get('word_results', [])
for i, wr in enumerate(wr_list[:5]):
    if wr is None or not getattr(wr, 'confs', None):
        continue
    chars = ''.join(c for w in wr.words for c in w)
    confs = wr.confs
    pairs = ' '.join(f'{c}:{p:.2f}' for c, p in zip(chars, confs))
    print(f'  [{i}]  text={chars[:60]!r}{"..." if len(chars)>60 else ""}')
    print(f'        confs  = {pairs[:80]}{"..." if len(pairs)>80 else ""}')
print(f'  (showing first 5 of {len(wr_list)} regions)')

# also: word_results in the final result is (txt, score, bbox) tuples
print('\n=== final result.word_results format (overwritten by build_final_output) ===')
for i, wr in enumerate(result.word_results[:3]):
    if not wr:
        continue
    print(f'  [{i}]  {len(wr)} word(s)  '
          f'first word = {wr[0] if wr else None!r}')

# ===========================================================================
# 6. TOP-K PER FRAME  (from the raw softmax that the default decoder discards)
# ===========================================================================
preds_batches = _RAW['preds']                    # list[np.ndarray], one per batch
chars_dict = engine.text_rec.postprocess_op.character
rec_txts = _RECOUT['txts']
total_imgs = sum(b.shape[0] for b in preds_batches)
T = preds_batches[0].shape[1] if preds_batches else 0
C = preds_batches[0].shape[2] if preds_batches else 0
print(f'\n=== raw softmax across {len(preds_batches)} batch(es): '
      f'{total_imgs} imgs, T={T}, num_classes={C} ===')

TOPK = 5
print(f'\n=== top-5 alternatives per non-blank decoded frame ===')
print(f'  {"region":>6}  {"frame":>5}  {"decoded":>5}  {"top-5 candidates":<60}')

global_idx = 0
for bi, preds in enumerate(preds_batches):
    order = np.argsort(-preds, axis=-1)[:, :, :TOPK]            # (B, T, TOPK)
    probs_bi = np.take_along_axis(preds, order, axis=-1)        # (B, T, TOPK)
    for i in range(preds.shape[0]):
        line_text = rec_txts[global_idx] if global_idx < len(rec_txts) else ''
        prev_idx = None
        emit = 0
        for t in range(preds.shape[1]):
            a_idx = int(order[i, t, 0])
            a_ch = chars_dict[a_idx]
            if a_ch == 'blank' or a_idx == prev_idx:
                prev_idx = a_idx
                continue
            top = [(chars_dict[int(order[i, t, k])], float(probs_bi[i, t, k]))
                   for k in range(TOPK)]
            line = '  '.join(f'{c!r}:{p:.2f}' for c, p in top)
            decoded_ch = line_text[emit] if emit < len(line_text) else chr(0xB7)
            print(f'  {global_idx:>6}  {t:>5}  {decoded_ch!r:>5}  {line}')
            emit += 1
            prev_idx = a_idx
        global_idx += 1

# ===========================================================================
# 7. AMBIGUITY SUMMARY  (over ALL batches, non-blank frames)
# ===========================================================================
print('\n=== ambiguity statistics over all non-blank frames ===')
n = n_low = n_top2 = n_top2_smiles = 0
for preds in preds_batches:
    order = np.argsort(-preds, axis=-1)[:, :, :TOPK]
    probs_bi = np.take_along_axis(preds, order, axis=-1)
    for i in range(preds.shape[0]):
        for t in range(preds.shape[1]):
            a_idx = int(order[i, t, 0])
            if chars_dict[a_idx] == 'blank':
                continue
            n += 1
            p1 = float(probs_bi[i, t, 0])
            p2 = float(probs_bi[i, t, 1])
            second_char = chars_dict[int(order[i, t, 1])]
            if p1 < 0.8:
                n_low += 1
            if p2 > 0.05 and second_char != 'blank':
                n_top2 += 1
            if p2 > 0.3 * p1 and second_char != 'blank':
                n_top2_smiles += 1
print(f'  total non-blank frames: {n}')
print(f'  frames with low argmax (p1 < 0.80)                        : {n_low}')
print(f'  frames with strong non-blank 2nd-best (p2 > 0.05)          : {n_top2}')
print(f'  frames with SIMILAR non-blank 2nd-best (p2 > 0.3*p1)       : {n_top2_smiles}')
print('  (only the last category is genuinely worth K-best expansion)')

# ===========================================================================
# 7b. BUILD JSON DUMP  (captures everything that was printed to stdout)
# ===========================================================================
TOPK_FOR_JSON = TOPK
chars_for_json = chars_dict
preds_batches_for_json = preds_batches
rec_word_results_for_json = _RECOUT['word_results']
rec_txts_for_json = _RECOUT['txts']

dump = {
    'image': str(img_path),
    'engine': {
        'det_lang': engine.cfg.Det.lang_type.value
                    if hasattr(engine.cfg.Det.lang_type, 'value')
                    else str(engine.cfg.Det.lang_type),
        'rec_lang': engine.cfg.Rec.lang_type.value
                    if hasattr(engine.cfg.Rec.lang_type, 'value')
                    else str(engine.cfg.Rec.lang_type),
        'rec_model': f'{engine.cfg.Rec.ocr_version.value}_{engine.cfg.Rec.model_type.value}',
        'dict_size': len(chars_for_json),
    },
    'regions': [],
    'per_char': [],
    'topk_frames': [],
    'ambiguity_stats': {
        'total_non_blank_frames': n,
        'low_argmax_p1_lt_080': n_low,
        'strong_2nd_p2_gt_005': n_top2,
        'similar_2nd_p2_gt_03p1': n_top2_smiles,
    },
}

# region-level: text + line score + bbox
boxes = result.boxes  # shape (N, 4, 2): four corner points
for i, (txt, sc) in enumerate(zip(result.txts, result.scores)):
    bbox = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
    dump['regions'].append({
        'i': i,
        'text': txt,
        'line_score': round(float(sc), 6),
        'bbox': bbox,
    })

# per-char confidences (from WordInfo captured before rewrite)
for i, wr in enumerate(rec_word_results_for_json):
    if wr is None or not getattr(wr, 'confs', None):
        continue
    chars_in_line = ''.join(c for w in wr.words for c in w)
    for j, (c, p) in enumerate(zip(chars_in_line, wr.confs)):
        dump['per_char'].append({
            'region': i,
            'pos': j,
            'char': c,
            'conf': round(float(p), 6),
        })

# top-K per non-blank frame (from raw softmax)
global_idx = 0
for bi, preds in enumerate(preds_batches_for_json):
    order = np.argsort(-preds, axis=-1)[:, :, :TOPK_FOR_JSON]
    probs_bi = np.take_along_axis(preds, order, axis=-1)
    for i in range(preds.shape[0]):
        line_text = rec_txts_for_json[global_idx] if global_idx < len(rec_txts_for_json) else ''
        prev_idx = None
        emit = 0
        for t in range(preds.shape[1]):
            a_idx = int(order[i, t, 0])
            a_ch = chars_for_json[a_idx]
            if a_ch == 'blank' or a_idx == prev_idx:
                prev_idx = a_idx
                continue
            cands = [
                {'char': chars_for_json[int(order[i, t, k])],
                 'prob': round(float(probs_bi[i, t, k]), 6)}
                for k in range(TOPK_FOR_JSON)
            ]
            decoded_ch = line_text[emit] if emit < len(line_text) else ''
            dump['topk_frames'].append({
                'region': global_idx,
                'frame': int(t),
                'decoded_char': decoded_ch,
                'topk': cands,
            })
            emit += 1
            prev_idx = a_idx
        global_idx += 1

# ===========================================================================
# 8. SAVE JSON + VISUALISATION
# ===========================================================================
out_json = Path(img_path).with_name(Path(img_path).stem + '_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(dump, f, ensure_ascii=False, indent=2)

out_vis = Path(img_path).with_name(Path(img_path).stem + '_vis.jpg')
result.vis(str(out_vis))

print(f'\nJSON dump : {out_json}')
print(f'vis image : {out_vis}')
