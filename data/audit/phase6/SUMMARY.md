# Phase 6 — patent adapter layer

This phase answers one question: **can a second patent be processed by
changing configuration instead of code?**  Two patents are now served by
one extractor, one OCR layer, one repair chain and one merge code.

## The split

### Patent-specific (data, in `src/cpd/parsers/profiles.py`)

| What | Field | WO2025162428 | WO2022251372 |
| --- | --- | --- | --- |
| Input kind | `source_mode` | `table_image` | `full_page` |
| Column layouts | `layouts` | activity / dense / smiles | activity8 / smiles3 |
| Column count → layout | `layout_by_column_count` | 3 / 6 / 7 | 3 / 8 |
| Header rows to skip | `LayoutSpec.header_bands` | 0 | 1 |
| Row may span pages | `LayoutSpec.continuation` | False | True |
| Compound-number syntax | `cmpd_id_pattern` | `\d{4}` | `\d{1,2}` |
| Number cell geometry | `cmpd_id_top_px`, `cmpd_id_inset_px` | None, 0 | 90, 4 |
| Activity fields and their type | `assay_fields` | 5 numeric/text | 6 letter grades |
| Closed value set | `AssayFieldSpec.allowed` | — | ("A","B","C","D") |
| Printed-SMILES route | `smiles_route` | `production` | `padded` |
| Cross-table merge rule | `merge_compounds` | `first_valid` | `prefer_structure` |

### Shared (code, unchanged by the patent)

* page/table locating and column-boundary measurement
  (`src/cpd/parsers/page_tables.py`);
* the row reader, the OCR cell readers and the compound/assay records
  (`src/cpd/parsers/table_extractor.py`);
* the printed-SMILES recognisers — all three routes use the same
  `PrintedSmilesOCR` class and the same RapidOCR engine
  (`src/cpd/ocr/printed_smiles.py`);
* the bracket-repair chain and RDKit validation (`src/cpd/chem.py`);
* the record merge (`src/cpd/merge.py`);
* the DECIMER structure pass and the combined-CSV writer (`main.py`).

## Evidence

### 1. The shipped patent is unchanged

`experiments/phase6_profile_snapshot.py` runs the real extractor over
every golden table of a patent and writes a canonical JSON snapshot.
The snapshot was taken before and after the profile refactor:

```
before_production.json   260 701 bytes
after_production.json    260 701 bytes
SHA256 (both)            2284F5FEB4FB6A0C3EBE9B1328FCED45C09F1CAEFDAA125104C16889CEA0003D
```

The two files are byte-for-byte identical: 103 images, 335 compounds,
222 assays, unchanged.  The adapter layer is a refactor, not a
regression.

### 2. The second patent runs end to end

`experiments/phase6_wo2022251372_extract.py`:

```
n_images           20
n_compounds        28
n_rdkit_valid      23
n_with_structure   28
n_assays           28
```

All six activity columns are read, and every value is a letter grade
from the closed set A–D.  A grade that cannot be read is left empty
rather than written out as a wrong letter (`src/cpd/ocr/constrained.py`).

### 3. Text accuracy against hand transcriptions

Six compounds were transcribed by hand from the pages
(`experiments/wo2022251372_references.py`).  Those are a *text*
reference only: they say nothing about whether the drawn molecule is
the intended one.

```
                         raw OCR        after repair
exact matches            4 / 6          4 / 6
total edit distance      6              6
aggregate CER            0.48 %         0.48 %
RDKit-parseable          5 / 6          5 / 6
```

`experiments/phase6_diff_detail.py` shows where the remaining errors
are.  Every one of them is the first or last character of a printed
line.  The two tables of the patent print the same SMILES, and where
they disagree the more complete reading is the one kept.

### 4. The remaining errors are *not* a cropping problem

`experiments/phase6_boundary_measure.py` measures the distance from
the printed text to the vertical column rule that bounds the cell,
with the rule columns removed first:

```
printed lines measured                       141
lines whose text touches a column rule         0
smallest gap to a rule                     22 px
```

This matters because the *first* patent has a confirmed cropping
problem (the row-leading `=` excluded by the detection box, 215
events).  The second patent does **not** have it: its text sits well
inside the cell.  The remaining character errors are therefore
recogniser errors on a very condensed serif face, not geometry.  The
same fix will not apply to both patents, which is exactly why the
route is configuration rather than a hard-coded constant.

## What this does not establish

* RDKit-parseable is not structure-correct.  23 of 28 strings parse;
  none of them has been compared with the drawn structure.
* The text reference is verified by hand.  It is not a chemical
  ground truth.
* Six hand-transcribed compounds are a small sample.  The 0.48 % CER
  describes those six, not the whole patent.
* The `production` route is still the route of the shipped CSV.  The
  second patent's `padded` route was measured on this patent and is
  known to be *worse* on the first patent, so the two must not be
  merged into one setting.

## Reproducing

```powershell
# regression: the shipped patent must not change
.venv\Scripts\python.exe experiments\phase6_profile_snapshot.py `
    --patent WO2025162428 --out data\audit\phase6\after_production.json

# the second patent, end to end
.venv\Scripts\python.exe experiments\phase6_wo2022251372_extract.py

# text accuracy against the hand transcriptions
.venv\Scripts\python.exe experiments\phase6_verify_accuracy.py

# where the remaining characters differ
.venv\Scripts\python.exe experiments\phase6_diff_detail.py

# is the remaining error geometry or recognition?
.venv\Scripts\python.exe experiments\phase6_boundary_measure.py
```

## Files

| Path | What |
| --- | --- |
| `src/cpd/parsers/profiles.py` | the profile registry (the patent-specific data) |
| `src/cpd/parsers/page_tables.py` | full-page table locating and row bands |
| `src/cpd/parsers/table_extractor.py` | the shared reader; header skip and cross-page continuation |
| `src/cpd/ocr/constrained.py` | closed-set cell reader for grade fields |
| `src/cpd/ocr/printed_smiles.py` | the three SMILES routes on one recogniser |
| `tests/test_profiles.py` | locks the old patent's behaviour and checks the new one |
| `data/audit/phase6/` | all measurements above |
