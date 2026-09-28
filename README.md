# Cyclic Peptide Dataset Pipeline

Extract cyclic-peptide SMILES and RAS bioactivity data from patent
characterization tables (e.g. WO2025162428).

## Pipeline Workflow

1. **Header aliasing (`cpd.parsers.headers`)**:
   Patent authors write the same column under different names
   (`Cmpd #` vs `Compound #`, `G12V GDP KD nM` vs `KRAS G12V GDP KD (nM)`).
   A small dictionary collapses every variant to one canonical field.
2. **Unified table extraction (`cpd.parsers.table_extractor`)**:
   OCR the top header band to learn the column layout, then for every
   row OCR each text cell and crop the `structure` column to a PNG.
   One code path handles all three layouts found in golden tables:
   3-col SMILES, 7-col activity (with structure), and 6-col dense
   activity (no structure).
3. **Chemical validation (`cpd.chem`)**:
   Auto-repair common OCR artefacts (amide carbonyls, unclosed chiral
   brackets) and validate the canonicalised SMILES with RDKit.
4. **Dual-table output (`main.py`)**:
   Write `compounds.csv` and `assays.csv` joined on `cmpd_id`, plus a
   `cells/` folder of cropped structure PNGs.

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2. Run the pipeline (reads data/raw/WO2025162428/golden_tables by default)
uv run python main.py

# 3. Run the test suite
uv run pytest
```

## Key Outputs

- `data/processed/compounds.csv` &mdash; one row per cyclic peptide
  (cmpd_id, canonical SMILES, RDKit descriptors, cropped structure PNG
  path).
- `data/processed/assays.csv` &mdash; one row per KD/LCMS measurement
  (cmpd_id, kd_nm, rt_min, ms_mz, lcms_method, ms_polarity).
- `data/processed/cells/{cmpd_id}_struct.png` &mdash; cropped structure
  image per compound.

The two CSVs share `cmpd_id`; either or both may be populated for a
given compound depending on which patent pages cover it.

## Project Structure

```text
cyclic-peptide/
├── pyproject.toml              # Project dependencies and configurations
├── main.py                     # Pipeline entry point
├── src/cpd/
│   ├── chem.py                 # RDKit validation and SMILES syntax repair
│   ├── merge.py                # cmpd_id normalisation + row dedupe
│   └── parsers/
│       ├── headers.py          # Header alias map (Cmpd # / Compound # / ...)
│       └── table_extractor.py  # RapidOCR + OpenCV cell-aligned extractor
├── tests/                      # pytest suite
└── data/
    ├── raw/                    # Raw patent images (FullText, golden_tables, ...)
    └── processed/              # Generated compounds.csv / assays.csv / cells/
```

### Package Installation

```shell
uv sync --extra dev
```
