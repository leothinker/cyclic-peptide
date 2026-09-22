# cyclic-peptide

Build a structured dataset of cyclic peptide compounds (e.g. KRAS inhibitors from
patents like [WO2025162428](https://patentscope.wipo.int/search/en/WO2025162428))
by extracting SMILES and bioactivity values (e.g. G12V-GDP `Kd` nM) from patent
PDFs and tables.

## Quick start

```bash
# install uv (once): https://docs.astral.sh/uv/getting-started/installation/
uv sync                       # base install: data models + CLI
uv sync --extra chem          # add RDKit
uv sync --extra ocr           # add DECIMER
uv sync --extra pdf           # add PDF + table extraction
uv sync --extra scrape        # add Playwright + httpx
uv sync --all-extras          # everything (heavy)
```

Run the CLI:

```bash
uv run cpd --help
uv run cpd fetch WO2025162428
uv run cpd extract data/raw/WO2025162428.pdf
uv run cpd ocr data/images/example.png
```

## Layout

```
cyclic-peptide/
├── data/
│   ├── raw/         # downloaded patent PDFs / HTML
│   ├── images/      # cropped 2D structure figures
│   └── processed/   # final CSV / JSON dataset
├── notebooks/       # exploratory OCSR / LLM prompts
├── src/cpd/
│   ├── cli.py       # `cpd` entrypoint
│   ├── models/      # Pydantic schemas: Compound, Bioactivity
│   ├── scrapers/    # WIPO / SureChEMBL fetchers
│   ├── extractors/  # PDF table + image crop
│   └── ocr/         # DECIMER + LLM-based structure readers
└── tests/
```

## Pipeline (planned)

1. **fetch** — pull patent PDF from WIPO / Google Patents.
2. **extract** — split into per-page images and tables (PyMuPDF + Camelot).
3. **ocr** — run DECIMER on cropped 2D structure images → SMILES.
4. **parse** — read Markush / R-group tables via LLM into structured rows.
5. **merge** — align SMILES with `Kd` / `IC50` rows into a CSV dataset.

See `docs/` (TBD) for the full design notes.
