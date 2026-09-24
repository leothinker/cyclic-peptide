# Cyclic Peptide Dataset Pipeline

Automated extraction of cyclic-peptide chemical structures (SMILES) and RAS bioactivity data ($K_D$) from patent characterization tables (e.g. WO2025162428).

## Pipeline Workflow

1. **Table Identification (`cpd.parsers.table_finder`)**:
   Scans FullText patent images and classifies them into SMILES tables and bioactivity ($K_D$) tables based on image dimensions and aspect ratios.
2. **Cell-Aligned OCR Extraction (`cpd.parsers.table_extractor`)**:
   Detects table grid lines with OpenCV morphological operations, performs cell-level text recognition via RapidOCR, and extracts compound rows.
3. **Chemical Validation & Auto-Repair (`cpd.chem`)**:
   Auto-repairs common OCR wrapping artifacts (e.g. amide carbonyl bonds and unclosed chiral brackets) and validates cyclic peptide descriptors using RDKit.
4. **Data Assembly (`main.py`)**:
   Joins SMILES structures with quantitative $K_D$ values on `Cmpd #` and exports the final benchmark dataset.

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2. Run the full extraction pipeline
uv run python main.py

# 3. Run the test suite
uv run pytest
```

## Key Outputs

- `data/processed/cyclic_peptides_benchmark.csv`: Merged dataset containing compound IDs, canonical SMILES, molecular weights, heavy atom counts, and G12V GDP $K_D$ (nM) affinities.

## Project Structure

```text
cyclic-peptide/
├── pyproject.toml              # Project dependencies and configurations
├── main.py                     # Single-entry pipeline execution script
├── src/cpd/
│   ├── chem.py                 # RDKit validation and SMILES syntax repair
│   ├── merge.py                # Dataset join and export utilities
│   ├── parsers/
│   │   ├── table_finder.py     # Table image classifier
│   │   ├── table_extractor.py  # RapidOCR-backed table row extractor
│   │   └── wipo_xml.py         # FullText body XML parser
│   └── ocr/
│       ├── base.py             # StructureReader interface
│       └── decimer.py          # DECIMER backup engine (for structure-only patents)
├── tests/                      # Core pytest test suite
└── data/
    ├── raw/                    # Raw patent images and documents
    └── processed/              # Generated benchmark datasets
```

### Package Installation

**Recommended: Using uv (faster and more reliable)**

```shell
# Install uv if you haven't already
pip install uv

# Install the package with development dependencies
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate
```

**Alternative: Using pip**

```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
# Ensure you have a recent version of pip (required for editable installs with pyproject.toml)
$ python3 -m pip install --upgrade pip
# Install the package in editable mode
$ pip install -e .
```
