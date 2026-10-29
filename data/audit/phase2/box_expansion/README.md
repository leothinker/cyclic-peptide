# Phase 2 - right-edge detector-box expansion

PRELIMINARY. Scored against a single reference (cmpd 1036, supplied by
the researcher). Nothing here supports a claim about dataset accuracy.

## VERDICT: NO IMPROVEMENT. Baseline retained.

The hypothesis (detector boxes clip the trailing `=`, so widening them
recovers the character) is **falsified**. See "Why" below.

## Command

```powershell
.venv\Scripts\python.exe phase2_box_expansion.py
```

Writes only inside this directory.

## Pipeline (reproduced first, asserted)

2x `INTER_CUBIC` upscale -> threshold 200 -> PP-OCRv4 mobile `en`.

- `baseline_detrec`: CER 0.00738, edit distance 2, RDKit parses. Matches
  the earlier `binarize_up2_en` baseline exactly.
- `baseline_reconly`: byte-identical string to `baseline_detrec`. This is
  the control that makes every expansion row comparable.

Both are asserted in code; the script aborts if either fails.

## Results

| config | pad (processed px) | mode | len | edits | CER | RDKit |
|---|---|---|---|---|---|---|
| baseline_detrec | 0 | det+rec | 270 | 2 | 0.00738 | yes |
| baseline_reconly | 0 | rec-only | 270 | 2 | 0.00738 | yes |
| pad4_reconly | 4 | rec-only | 270 | 2 | 0.00738 | yes |
| pad8_reconly | 8 | rec-only | 274 | 13 | 0.04797 | no |
| pad12_reconly | 12 | rec-only | 271 | 10 | 0.0369 | no |
| recover_white8_reconly | ink+4 then +8 white | rec-only | 275 | 14 | 0.051661 | no |

Pads are in processed pixels; the processed image is 2x the source crop,
so pad 8 here is 4 source pixels.

Determinism: the best-scoring expansion config was re-run 3x and gave
edit distance 2 every time.

## Why it fails

`box_vs_ink_extent.txt` measures each detector box against the real
text-ink extent of its own line:

| line | box_x2 | ink_x2 | deficit |
|---|---|---|---|
| 0 | 1645 | 1640 | -5 |
| 1 | 1642 | 1636 | -6 |
| 2 | 1667 | 1666 | -1 |
| 3 | 1647 | 1644 | -3 |
| 4 | 1675 | 1674 | -1 |
| 5 | 1671 | 1666 | -5 |
| 6 | 1234 | 1232 | -2 |

Every box already extends 1-6 px **past** its last inked column. No glyph
is clipped. Total ink in the rectified line is constant across pads
(28016 -> 28014 px for pad 0 -> 24), confirming expansion adds only white.

`tail_zoom/line1_tail_pad0.png` and `..._pad12.png` show the same tail
content at both pads: the trailing `C(` is intact and nothing new appears
to its right.

So the two residual errors are **recogniser-level** confusions, not
geometric truncation:

1. `@77` `C(=O)` -> `C(O)` : the `=` is fully present in pixels and is
   dropped by the recogniser.
2. `@115` `C(F)=C5` -> `C(F)-C5` : the `=` is fully present and read as
   `-`.

Both are `=` vs `-` confusions. Widening only degrades PP-OCR's
aspect-ratio prior, which is why edits rise from 2 to 10-14 and spurious
`)` / `]` appear at other line ends.

## Method notes

Two measurement traps were hit and fixed; both would have produced wrong
conclusions if left:

- Crops are rectified with the pipeline's own
  `get_rotate_crop_image`. An axis-aligned rectangle drops the
  rectification on this skewed table text and changes the string on its
  own.
- Recognition goes through `engine.recognize_txt([...])`, the exact entry
  point the det+rec path uses. `engine(img)` with `use_det=False` looks
  equivalent but re-runs the image preprocessor and returns a different
  string (e.g. `[C@@H]` -> `N|C@@H]`), which made every expansion config
  look worse for an unrelated reason.

Every crop is re-extracted from the full processed cell image; no config
pads whitespace onto an already-clipped box. Expansion is clamped to the
cell's own right boundary (processed width 1702, page rule at 1679-1684),
so no neighbouring column can leak in.

## Files

- `box_expansion.csv` - one row per config
- `box_expansion_report.txt` - tables + per-config character diffs
- `box_vs_ink_extent.txt` - the geometric falsification evidence
- `raw_ocr/<cmpd>__<config>.txt` - raw recognised string per config
- `raw_ocr/<cmpd>__<config>_diff.json` - character-level opcodes
- `visuals/<cmpd>__<config>_boxes.png` - red = detector box,
  green = expanded box
- `visuals/<cmpd>__<config>_lines.png` - the actual crops fed to the
  recogniser
- `tail_zoom/line1_tail_pad*.png` - 4x tail crops at pad 0/8/12/24

## What is worth testing next

Not more geometry. The errors are `=` vs `-` confusions inside a fully
pixel-complete glyph, so the next candidates are recogniser-side:
top-k candidate re-ranking on the `=` position, or a recogniser trained on
this font. Both need a second verified reference before they can be
scored - see the caveat below.

## Caveat

Only cmpd 1036 has a reference that passed transcription QC. 1037 is
withheld pending a human re-read; 1038/1040/2173 failed transcription QC
(mass-gate deltas -14.2 / +41.9 / +13.9 Da, i.e. whole residues misread).
