# Read-only reference repos for modifying the OCSR pipeline

These three upstream repos are vendored here purely as **read-only
source references** so we can read the implementation while modifying
`src/cpd/parsers/table_extractor.py` and `src/cpd/decimer.py`.

* **`RapidOCR/`** -- the ONNX OCR engine that
  `cpd.parsers.table_extractor.RapidOcrTableExtractor` wraps. We read
  it when we need a confidence score, top-K, or a different rec model.
* **`DECIMER-Image_Transformer/`** -- the upstream DECIMER package
  source we import via `cpd.decimer._ensure_engine`. We read it to
  understand `predict_SMILES`, the checkpoint loads, and the confidence
  flag.
* **`Hiro-OCSR/`** -- patsnap's patent-domain OCSR model, kept around
  in case we want to swap it in for DECIMER. Not imported by the
  pipeline; we would build a `cpd.hiro` wrapper mirroring
  `cpd.decimer` if we go this route.

Do not commit local changes to these repos. Edit the vendored files
back upstream, or copy the bits we need into `src/cpd/`.
