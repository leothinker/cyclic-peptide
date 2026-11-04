# WO2022251372 — can the current pipeline read the printed SMILES?

Status: **audit only**.  No production code, no formal CSV and no Git
history were touched.  Every number below is recomputed from the files
listed at the end of this document.

## Short answer

**No.**  The shipped pipeline does not read this patent at all, and it
also cannot be pointed at it with a small tweak:

1. end to end, the pipeline extracts **0 compounds and 0 assays** from
   the 20 scanned pages, for four independent reasons;
2. even after cropping the SMILES cells correctly, the current OCR
   route yields **0 of 28** RDKit-parseable SMILES;
3. the root cause is *not* the recogniser model.  It is that the ink
   band is cropped flush against the last ink pixel, so the final
   glyph of every printed line is read wrong or dropped;
4. a one-line geometry change (a 6 px white margin around the line)
   plus the English recogniser takes the same 28 cells to **21/28
   bracket balanced and 17/28 RDKit-parseable**, with a character error
   rate of 0.0 % on 3 of the 6 hand-transcribed compounds.

This is an input/layout adaptation problem with a known, measurable
fix, not a "the OCR model is bad" problem.

## 1. The input

| property | value |
| - | - |
| files | 20 TIFF, `000107.tif` … `000126.tif` |
| geometry | 2478 x 3506 px, 300 dpi |
| Table 1 | pages 000107-000115, 8 columns: `No. | SMILES String | MDM2 | MDM4 | MOLM-13 | MV4-11 | WaGa | MKL-1` |
| Table 1a | pages 000116-000126, 3 columns: `No. | SMILES String | Chemical Structure` |
| compounds | 1 - 28 (not four-digit ids) |
| activity data | letter grades (A/B/C), not numeric values |

The same 28 SMILES are printed **twice**: once in Table 1 and once in
Table 1a.  Table 1a was used for the measurement below because its
rows are not split across page breaks.

The source PDF (`WO2022251372-PAMPH-20221201-7554.pdf`) contains 152
image streams and **zero fonts**, i.e. it is a rasterised document.
There is no text layer, so a human reading of the page images is the
only possible ground truth.

## 2. Why the pipeline returns nothing

Four separate blockers, each verified in the current source:

| # | blocker | evidence |
| - | - | - |
| 1 | the input scanner only looks at `*.jpg` | `main.py:94` — `img_dir.glob("*.jpg")`; the 20 TIFFs are never listed |
| 2 | compound ids must be four digits | `table_extractor.py:775` — `re.fullmatch(r"\d{4}", ...)`; the ids here are `1`-`28` |
| 3 | only 3, 6 and 7 column layouts exist | `table_extractor.py:226-230`; Table 1 has 8 columns |
| 4 | the 3-column layout has the columns the wrong way round | the old layout is `Structure | SMILES`, Table 1a is `SMILES | Structure` |

Feeding the representative pages `000108` and `000116` straight into
`RapidOcrTableExtractor` gives **0 compounds / 0 assays**.

## 3. What the OCR does once the crop is correct

Table 1a, all 28 compounds, correctly cropped SMILES column
(`x = 371..915`):

| configuration | brackets balanced | RDKit parses | edit distance to the hand transcription (cmpd 1, 2, 3, 4, 10, 28) |
| - | - | - | - |
| current `validated` route (PP-OCRv6, 2x, no margin) | 3/28 | **0/28** | 4, 4, 4, 4, 7, 8 |
| English PP-OCRv4, no margin | 1/28 | 1/28 | 5, 6, 6, 5, 4, 5 |
| **English PP-OCRv4 + 6 px margin** | **21/28** | **17/28** | **0, 1, 0, 0, 2, 2** |

At line level (60 printed lines of the 6 transcribed compounds):

| configuration | exact lines | total line edits | mean line CER |
| - | - | - | - |
| PP-OCRv6, no margin | 36/60 | 31 | 2.70 % |
| English PP-OCRv4, no margin | 33/60 | 31 | 2.42 % |
| **English PP-OCRv4 + 6 px margin** | **56/60** | **5** | **0.37 %** |

The upstream default recogniser is also *unstable* without a margin:
`PP-OCRv6` at 1x scores a 12.4 % CER on compound 10 because one line
collapses completely, while the same model at 2x is back at 3.5 %.
That volatility is a further reason to prefer the margin over a model
swap.

## 4. Root cause

The band diagnosis (`band_diag/`) shows that the **segmentation is
perfect**: all 60 printed lines are found, with correct boundaries.
The damage is made afterwards, by the recogniser, and it is
concentrated on the last glyph of each line.

`text_span()` returns the first and the last column that contains *any*
ink, so the cropped band ends exactly on the last ink pixel of the
final glyph.  That glyph is therefore flush against the image border
with no margin at all, and the recogniser mis-reads it:

```
0 px margin ->  'H](Cc2ccc(Cl)cc2)NC(=O)'     the '[' is lost
4 px margin ->  'H](Cc2ccc(Cl)cc2)NC(=O)['    exact
```

Reproduced by `experiments/phase4_wo2022251372_border_effect.py`, which
prints the whole margin series for that one line: 0 px loses the `[`,
2 px turns it into `]`, 4-16 px are exact, and 24 px turns it into
`I`.  The pixel data never changes -- only the margin does.

This is the same *class* of defect as the row-start truncation measured
on WO2025162428, but the mechanism is different and worth stating
precisely:

| | WO2025162428 (row start) | WO2022251372 (line end) |
| - | - | - |
| symptom | the `=` is missing from the OCR input | the last glyph is mis-read or dropped |
| mechanism | the detector box excluded the pixels | the pixels are present, the margin is missing |
| fix | has to change detection | a few pixels of white border |

The margin curve for the English recogniser over the 60 reference
lines runs 31 line edits at 0 px, 26 at 2 px, 13 at 4 px, **5 at
6 px**, then 8-11 for 8-20 px and 21 at 24 px.  Between 6 and 20 px
the margin is not critical; below 4 px it clearly is.  The optimum is
sharp, so 6 px is a candidate, not a settled constant.

## 5. A defect in the shipped repair chain

With the geometry fixed, the OCR text is good enough to expose a bug
in `cpd.ocr_rewrite.rewrite_oxo`.

That rule rewrites every `X=O)` into `X(=O)`.  In this patent the
lactam carbonyl is printed as `C1=O)`: the ring-closure digit sits
between the carbon and the `=`, so the rule inserts a spurious `(`
into a perfectly good string.

| stage | parses (of 28) | edit distance to the hand transcription |
| - | - | - |
| raw OCR | 17 | 0, 1, 0, 0, 2, 2 |
| + shipped `rewrite_oxo` | **9** | 1, 2, 1, 1, 2, 2 — **worse** |
| + guarded `rewrite_oxo` (skip a ring-closure digit) | 17 | 0, 1, 0, 0, 2, 2 — unchanged |
| + guarded `rewrite_oxo` + `=0` -> `=O` | **19** | 0, 1, 0, 0, 2, 2 — unchanged |

The `=0` -> `=O` rewrite is safe by construction: zero is not an
element, so `=0` is never valid SMILES.

Note that the guarded rewrite is still a proposal.  It is measured
here on one patent and has not been re-validated against the
WO2025162428 rows the original rule was tuned on.

## 6. What is still open

* **9 of 28 compounds still do not parse.**  Two damage patterns are
  left: `C(=O)` read as `C=O)` (the bracket is lost) and a lost
  trailing ring-closure digit (`c1ccccc1` -> `c1ccccc`).
* **"Parses" is not "is the right structure".**  6 of 28 compounds
  have a hand transcription.  On those, the candidate configuration is
  character-exact on 3 (cmpd 1, 3, 4) and off by one character on
  cmpd 2, and by two on cmpd 10 and 28.
* **Table 1 has not been adapted at all** (8 columns, letter-grade
  activity values, rows that span page breaks).
* The assay extractor expects numeric values; this patent prints
  letter grades, so the activity columns need a different parser.
* No mass gate is possible: this patent prints no m/z.

## 7. Suggested next steps

1. Implement the 6 px margin in `cpd.ocr.printed_smiles` behind the
   existing route switch, and re-validate the WO2025162428 rows
   before switching the default.  The margin must not be applied to
   the `production` route, which uses the detector, not ink bands.
2. Fix the `rewrite_oxo` false positive, then re-run the existing
   tests plus the WO2025162428 repair audit.
3. Add a patent-specific input adapter (TIFF, 1-2 digit ids, 8-column
   Table 1, letter-grade assays) rather than special-casing inside
   the generic extractor.
4. Extend the hand-transcribed benchmark towards 10-15 compounds so
   the margin choice and the repair rules are measured on more than
   one page layout.

## 8. Files and commands

All artefacts are under `data/audit/phase4/wo2022251372/`; the
experiment scripts are under `experiments/`.

| artefact | contents |
| - | - |
| `table1a_ocr_results.csv` | per-compound OCR of the 28 Table 1a cells, with the hand transcription and the CER / edit distance |
| `table1a_cells/` | the 28 cropped SMILES cells |
| `figure_table1a_smiles_cells.png` | the cropped cells |
| `visual_checks/` | the renderings the hand transcriptions were read from, with a README |
| `band_diag/` | line-level segmentation and recognition, band by band |
| `recognizer_matrix/` | recogniser x scale, no margin |
| `padding_matrix/` | recogniser x scale x margin sweep, 60 reference lines |
| `candidate_config/` | the three configurations over all 28 compounds |
| `repair_ladder/` | the repair chain stage by stage, with edit distances |
| `summary.json` | end-to-end compatibility probe and headline counts |

`data/audit/phase4/tools/` holds three small diagnostic scripts used
while writing this report (failure-mode summary, false-positive
locator, margin-curve printer).

```
.venv\Scripts\python.exe experiments\phase4_wo2022251372_audit.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_band_diag.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_border_effect.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_recognizer_matrix.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_padding_matrix.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_candidate_config.py
.venv\Scripts\python.exe experiments\phase4_wo2022251372_repair_ladder.py
```

The hand transcriptions live in `experiments/wo2022251372_references.py`
and are marked as *text verified by hand*.  That is not a statement
about chemical structure validation.
