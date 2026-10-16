# Cyclic Peptide Dataset Pipeline

Extract cyclic-peptide SMILES and KRAS G12V GDP KD bioactivity data from
patent characterization tables (e.g. WO2025162428). One combined CSV is the
main deliverable: 406 cmpds x 27 columns (SMILES, DECIMER 2D re-render,
LCMS RT / MS m/z / method / polarity, KRAS G12V GDP KD, OCR trust scores).

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2. Optional: install DECIMER for OCSR SMILES (~2 GB PyTorch)
uv pip install --python .venv/Scripts/python.exe DECIMER

# 3. Run the pipeline (reads data/raw/WO2025162428/golden_tables by default)
$env:PYTHONIOENCODING = "utf-8"
uv run python main.py

# 4. Run the test suite
uv run pytest
```

When DECIMER is not installed the `smiles_decimer` column stays empty;
install the package and re-run to fill it.

## Main Output

- `data/processed/compounds_combined.csv` -- one row per cmpd, 27 columns:
  cmpd_id, smiles, canonical_smiles, smiles_decimer,
  canonical_smiles_decimer, smiles_best, smiles_best_source, smiles_agree,
  molecular_weight, heavy_atom_count, is_valid, structure_image,
  source_image, kd_nm, rt_min, ms_mz, lcms_method, ms_polarity,
  assay_source_image, decimer_min_conf, decimer_low_conf_count,
  decimer_render_path, ocr_avg_entropy, ocr_min_top1, ocr_low_conf_count,
  data_quality_flags, data_quality.
- `data/processed/cells/` -- cropped structure image per compound.
- `data/processed/rendered/` -- DECIMER-canonicalised 2D re-render PNG per
  valid structure (327 of 406 cmpds).

## Project Structure

```text
cyclic-peptide/
 +- main.py                       # Pipeline entry point
 +- src/cpd/
 |   +- chem.py                   # RDKit validation + SMILES repair
 |   +- decimer.py                # Optional DECIMER OCSR wrapper
 |   +- merge.py                  # cmpd_id normalisation + dedupe
 |   +- render.py                 # RDKit Mol -> PNG renderer
 |   +- parsers/
 |       +- headers.py            # Header alias map
 |       +- table_extractor.py    # RapidOCR + OpenCV cell extractor
 +- tests/                        # pytest suite
 +- data/
 |   +- raw/WO2025162428/
 |       +- WO2025162428-PAMPH-20250807-0047.pdf
 |       +- golden_tables/        # 103 source pages used by the pipeline
 |   +- processed/
 |       +- compounds_combined.csv   # main artifact (406 x 27)
 |       +- cells/                   # 406 cropped structure PNGs
 |       +- rendered/                # 327 DECIMER re-render PNGs
 +- references/                   # Read-only upstream OCSR source clones
 +- pyproject.toml
 +- LICENSE
 +- .gitignore
 +- README.md
```

`data/processed/` is gitignored -- regenerate any artifact with `main.py`.
`references/RapidOCR`, `references/DECIMER-Image_Transformer`,
`references/Hiro-OCSR` are local read-only clones used while developing the
OCSR wrappers in `src/cpd/`.

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
 over the cropped structure image), plus RDKit canonical versions,
 an `smiles_agree` flag, an OCR entropy trust score, and a
 DECIMER per-token confidence score. This is the main artifact --
 a quick `pandas.read_csv("data/processed/compounds_combined.csv")`
 gives you a joined table ready for QSAR / clustering. Columns:

  ```text
  cmpd_id, smiles, canonical_smiles,
 smiles_decimer, canonical_smiles_decimer,
 smiles_best, smiles_best_source, smiles_agree,
  molecular_weight, heavy_atom_count, is_valid,
  structure_image, source_image,
 kd_nm, rt_min, ms_mz, lcms_method, ms_polarity, assay_source_image,
 decimer_min_conf, decimer_low_conf_count, decimer_render_path,
 ocr_avg_entropy, ocr_min_top1, ocr_low_conf_count,
 data_quality_flags, data_quality
  ```

- `data/processed/cells/{src_stem}_{cmpd_id}_struct.png` -- cropped
  structure image per compound, with a 12 px white margin so DECIMER (or
  any other OCSR model) does not get confused by black borders.
- `data/processed/rendered/{cmpd_id}.png` -- DECIMER-canonicalised
  re-render of every valid structure (CoordGen + linearised macrocycle),
  ~327 PNGs for the WO2025162428 set.
- `demo_*/` -- one directory per report diagram (e.g.
  `demo_pipeline_render/grid_12.png`,
  `demo_roundtrip/cmpd_1036_side_by_side.png`).

## Project Structure

```text
cyclic-peptide/
 ├── pyproject.toml              # Project dependencies and configurations
 ├── main.py                     # Pipeline entry point (OCR + DECIMER -> one CSV)
 ├── src/cpd/
 │   ├── chem.py                 # RDKit validation and SMILES syntax repair
 │   ├── decimer.py              # Optional DECIMER (OCSR) wrapper; no-op when absent
 │   ├── merge.py                # cmpd_id normalisation + row dedupe
 │   ├── render.py               # RDKit Mol -> PNG renderer (CoordGen + macrocycle)
 │   └── parsers/
 │       ├── headers.py          # Header alias map (Cmpd # / Compound # / ...)
 │       └── table_extractor.py  # RapidOCR + OpenCV cell-aligned extractor
 ├── tests/                      # pytest suite (chem / merge / table / combined)
 ├── demo_*/                     # one-shot reporting scripts + their PNG outputs
 ├── MORNING_BRIEF.md            # at-a-glance status, refreshed each morning
 ├── PROGRESS_REPORT.md          # detailed progress log
 ├── REPORT_MATERIAL.md          # distilled report content
 ├── references/                 # Read-only upstream clones; we read but do not modify
 │   ├── RapidOCR/               # the OCR engine we wrap
 │   ├── DECIMER-Image_Transformer/
 │   └── Hiro-OCSR/              # patent-domain OCSR (awaiting Hugging Face access)
 └── data/
     ├── raw/WO2025162428/       # patent images + golden_tables/*.jpg (103 source pages)
     └── processed/              # pipeline outputs
         ├── compounds_combined.csv  # main artifact (406 rows, 27 cols)
         ├── decimer_cache.json      # 405/405 DECIMER predictions (cached)
         ├── cells/                  # 406 cropped structure PNGs
         └── rendered/               # 327 DECIMER-re-rendered PNGs
```

`references/` is **read-only** -- we clone the upstream repos there so
we can read their source while modifying the OCR and OCSR wrappers
under `src/cpd/`. Do not commit local changes to them; if we need to
patch an upstream, copy the relevant file into `src/cpd/` instead.

### Package Installation

```shell
uv sync --extra dev
```
