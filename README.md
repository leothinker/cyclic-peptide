# Cyclic Peptide Dataset Pipeline

Extract cyclic-peptide SMILES and RAS bioactivity data from patent
characterization tables (e.g. WO2025162428).

## Pipeline Workflow

1. **Header aliasing (`cpd.parsers.headers`)**
   Patent authors write the same column under different names
   (`Cmpd #` vs `Compound #`, `G12V GDP KD nM` vs `KRAS G12V GDP KD (nM)`).
   A small dictionary collapses every variant to one canonical field.
2. **Unified table extraction (`cpd.parsers.table_extractor`)**
   OCR the top header band to learn the column layout, then for every
   row OCR each text cell and crop the `structure` column to a PNG.
   One code path handles all three layouts found in golden tables:
   3-col SMILES, 7-col activity (with structure), and 6-col dense
   activity (no structure).
3. **Chemical validation (`cpd.chem`)**
   Auto-repair common OCR artefacts (amide carbonyls, unclosed chiral
   brackets) and validate the canonicalised SMILES with RDKit.
4. **Optional DECIMER OCSR (`cpd.decimer`)**
   When the DECIMER package is installed, run it on every cropped
   structure PNG to get a SECOND SMILES string. The module returns
   `None` and the column stays empty when DECIMER is not available.
5. **Combined-table output (`main.py`)**
   `main.py` runs OCR + DECIMER in a single pass and writes one
   `compounds_combined.csv` with both SMILES streams side-by-side,
   RDKit canonical forms, an `smiles_agree` flag, and the activity
   columns.

## Quick Start

```powershell
# 1. Install dependencies (one-time)
uv sync

# 2. Optional: install DECIMER for OCSR SMILES (pulls PyTorch ~2 GB).
#    Skip it if you only need the OCR SMILES column.
uv pip install --python .venv/Scripts/python.exe DECIMER

# 3. Run the pipeline (reads data/raw/WO2025162428/golden_tables by default)
$env:PYTHONIOENCODING = "utf-8"
uv run python main.py

# 4. Run the test suite
uv run pytest
```

When DECIMER is not installed the `smiles_decimer` column stays empty;
install the package and re-run to fill it.

## Key Outputs

- `data/processed/compounds_combined.csv` -- **one row per cmpd**,
  joining the OCR SMILES (table text) with the DECIMER SMILES (OCSR
  over the cropped structure image), plus RDKit canonical versions
  and an `smiles_agree` flag. This is the main artifact -- a quick
  `pandas.read_csv("data/processed/compounds_combined.csv")` gives
  you a joined table ready for QSAR / clustering. Columns:

  ```text
  cmpd_id, smiles, canonical_smiles,
  smiles_decimer, canonical_smiles_decimer, smiles_agree,
  molecular_weight, heavy_atom_count, is_valid,
  structure_image, source_image,
  kd_nm, rt_min, ms_mz, lcms_method, ms_polarity, assay_source_image
  ```

- `data/processed/cells/{src_stem}_{cmpd_id}_struct.png` -- cropped
  structure image per compound, with a 12 px white margin so DECIMER (or
  any other OCSR model) does not get confused by black borders.

## Project Structure

```text
cyclic-peptide/
├── pyproject.toml              # Project dependencies and configurations
├── main.py                     # Pipeline entry point (OCR + DECIMER -> one CSV)
├── src/cpd/
│   ├── chem.py                 # RDKit validation and SMILES syntax repair
│   ├── decimer.py              # Optional DECIMER (OCSR) wrapper; no-op when absent
│   ├── merge.py                # cmpd_id normalisation + row dedupe
│   └── parsers/
│       ├── headers.py          # Header alias map (Cmpd # / Compound # / ...)
│       └── table_extractor.py  # RapidOCR + OpenCV cell-aligned extractor
├── tests/                      # pytest suite (chem / merge / table / combined)
├── references/                 # Read-only upstream repos for the OCSR phase
│   ├── RapidOCR/               # the OCR engine we wrap
│   ├── DECIMER-Image_Transformer/
│   └── Hiro-OCSR/              # patsnap's patent-domain OCSR (alternative)
└── data/
    ├── raw/                    # Raw patent images (FullText, golden_tables, ...)
    └── processed/              # Generated compounds_combined.csv + cells/
```

`references/` is **read-only** -- we clone the upstream repos there so
we can read their source while modifying the OCR and OCSR wrappers
under `src/cpd/`. Do not commit local changes to them; if we need to
patch an upstream, copy the relevant file into `src/cpd/` instead.

### Package Installation

```shell
uv sync --extra dev
```
