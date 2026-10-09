"""Combined-table pipeline: OCR + optional DECIMER + assays in ONE CSV.

Per image:
  1. ``RapidOcrTableExtractor.extract_records`` returns CompoundRecords
     (SMILES + structure PNG) and AssayRecords (KD + LCMS).
  2. Compound-id spelling variants collapse through ``normalise_cmpd_id``.
  3. Optional DECIMER (OCSR) generates a SECOND SMILES for every cropped
     structure PNG. When DECIMER is not present the column stays empty --
     install ``DECIMER`` and re-run to fill it.
  4. The two SMILES streams + activity stream merge on ``cmpd_id`` into a
     single wide ``compounds.csv``. Legacy ``compounds.csv`` (smiles only)
     and ``assays.csv`` (kd/rt/ms) are also written so anything downstream
     still works.

Combined ``compounds.csv`` columns::

    cmpd_id,
    smiles, canonical_smiles, smiles_decimer, canonical_smiles_decimer,
    smiles_agree,
    molecular_weight, heavy_atom_count, is_valid,
    structure_image, source_image,
    kd_nm, rt_min, ms_mz, lcms_method, ms_polarity, assay_source_image

Empty string when a value is missing (CSV-friendly).

Run:  python main.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

# Windows consoles default to cp1252 which cannot encode the status emojis;
# force UTF-8 so the banner prints cleanly even when piped to a file.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


from cpd.chem import validate_and_enrich
from cpd.decimer import is_available, predict_smiles
from cpd.merge import dedupe_assays, dedupe_compounds, normalise_cmpd_id
from cpd.parsers.table_extractor import (
    AssayRecord,
    Column,
    CompoundRecord,
    RapidOcrTableExtractor,
)


log = logging.getLogger("cpd")

# Combined output: one row per cmpd_id, both SMILES streams side by side.
COMBINED_COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "smiles",
    "canonical_smiles",
    "smiles_decimer",
    "canonical_smiles_decimer",
    "smiles_agree",
    "molecular_weight",
    "heavy_atom_count",
    "is_valid",
    "structure_image",
    "source_image",
    "kd_nm",
    "rt_min",
    "ms_mz",
    "lcms_method",
    "ms_polarity",
    "assay_source_image",
)


def _iter_jpgs(img_dir: Path) -> Iterable[Path]:
    return sorted(img_dir.glob("*.jpg"))


def _safe_extract(
    extractor: RapidOcrTableExtractor,
    image_path: Path,
    cells_dir: Path,
    prev_columns: list[Column],
) -> tuple[list[CompoundRecord], list[AssayRecord], list[Column]]:
    try:
        return extractor.extract_records(
            image_path, cells_dir=cells_dir, prev_columns=prev_columns
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("skipping %s: extractor crashed: %s", image_path.name, exc)
        return [], [], prev_columns


def _extract_all(
    img_dir: Path,
    cells_dir: Path,
) -> tuple[list[CompoundRecord], dict[str, AssayRecord], dict[str, str]]:
    """Run the extractor over every image, deduping across pages.

    Returns ``(compounds, assays, failures)``:
      * ``compounds`` is a list of unique CompoundRecord (one per cmpd_id).
      * ``assays``    is a dict keyed by cmpd_id for fast lookup in
        ``_build_combined_rows``.
    """
    extractor = RapidOcrTableExtractor()
    compounds: dict[str, CompoundRecord] = {}
    assays: dict[str, AssayRecord] = {}
    failures: dict[str, str] = {}
    prev_columns: list[Column] = []

    jpgs = list(_iter_jpgs(img_dir))
    for idx, image_path in enumerate(jpgs, start=1):
        page_compounds, page_assays, columns = _safe_extract(
            extractor, image_path, cells_dir, prev_columns
        )
        if columns:
            prev_columns = columns

        if not (page_compounds or page_assays):
            failures[image_path.name] = "no rows"

        page_compounds = [c for c in page_compounds if normalise_cmpd_id(c.cmpd_id)]
        page_assays = [a for a in page_assays if normalise_cmpd_id(a.cmpd_id)]

        for cid, rec in dedupe_compounds(page_compounds).items():
            compounds.setdefault(cid, rec)
        for cid, rec in dedupe_assays(page_assays).items():
            assays.setdefault(cid, rec)

        print(
            f"   [{idx}/{len(jpgs)}] {image_path.name} -> "
            f"+{len(page_compounds)} cmpd, +{len(page_assays)} assay "
            f"| total: {len(compounds)} cmpd, {len(assays)} assay",
            end="\r",
        )
    print()
    return list(compounds.values()), assays, failures


def _run_decimer(
    compounds: list[CompoundRecord],
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Run DECIMER over every cropped structure PNG.

    Returns ``(decimer_raw, decimer_canon)`` keyed by ``cmpd_id``:

    * ``decimer_raw``    = the SMILES string DECIMER returned, or ``""``.
    * ``decimer_canon``  = the RDKit-canonical form (empty when invalid).

    When DECIMER is not installed the function logs once and returns two
    empty dicts so the rest of the pipeline still runs.
    """
    raw: dict[str, str] = {}
    canon: dict[str, str] = {}
    if not is_available():
        print("   [DECIMER] not installed -> smiles_decimer column will be empty")
        print("              install with:  uv pip install --python .venv/Scripts/python.exe DECIMER")
        return raw, canon

    targets = [c for c in compounds if c.structure_image]
    print(f"   [DECIMER] running on {len(targets)} cropped structures ...")
    cells_root = out_dir  # structure_image paths are relative to out_dir.
    for i, c in enumerate(targets, start=1):
        img_path = cells_root / c.structure_image  # type: ignore[operator]
        smi = predict_smiles(img_path)
        if not smi:
            continue
        raw[c.cmpd_id] = smi
        chem = validate_and_enrich(smi)
        if chem is not None:
            canon[c.cmpd_id] = chem.canonical_smiles
        if i % 25 == 0 or i == len(targets):
            print(f"      [{i}/{len(targets)}] parsed", end="\r")
    print()
    print(f"   [DECIMER] got SMILES for {len(raw)} / {len(targets)} structures")
    return raw, canon


def _build_combined_rows(
    compounds: list[CompoundRecord],
    assays: dict[str, AssayRecord],
    decimer_raw: dict[str, str],
    decimer_canon: dict[str, str],
) -> list[dict]:
    """Flatten compounds + assays + DECIMER into COMBINED_COLUMNS-shaped dicts."""
    rows: list[dict] = []
    seen_assay_ids: set[str] = set()
    for c in compounds:
        a = assays.get(c.cmpd_id)
        if a is not None:
            seen_assay_ids.add(c.cmpd_id)
        d_raw = decimer_raw.get(c.cmpd_id, "")
        d_can = decimer_canon.get(c.cmpd_id, "")
        agree = (
            bool(c.canonical_smiles and d_can and c.canonical_smiles == d_can)
            if c.canonical_smiles and d_can
            else False
        )
        rows.append({
            "cmpd_id": c.cmpd_id,
            "smiles": c.smiles or "",
            "canonical_smiles": c.canonical_smiles or "",
            "smiles_decimer": d_raw,
            "canonical_smiles_decimer": d_can,
            "smiles_agree": "True" if agree else "False",
            "molecular_weight": c.molecular_weight if c.molecular_weight is not None else "",
            "heavy_atom_count": c.heavy_atom_count if c.heavy_atom_count is not None else "",
            "is_valid": "True" if c.is_valid else "False",
            "structure_image": c.structure_image or "",
            "source_image": c.source_image or "",
            "kd_nm": a.kd_nm if a is not None and a.kd_nm is not None else "",
            "rt_min": a.rt_min if a is not None and a.rt_min is not None else "",
            "ms_mz": a.ms_mz if a is not None and a.ms_mz is not None else "",
            "lcms_method": (a.lcms_method or "") if a is not None else "",
            "ms_polarity": (a.ms_polarity or "") if a is not None else "",
            "assay_source_image": (a.source_image or "") if a is not None else "",
        })
    # Cmpds that appear only in the assay stream (no compound record).
    for cid, a in assays.items():
        if cid in seen_assay_ids:
            continue
        rows.append({
            "cmpd_id": cid,
            "smiles": "",
            "canonical_smiles": "",
            "smiles_decimer": "",
            "canonical_smiles_decimer": "",
            "smiles_agree": "False",
            "molecular_weight": "",
            "heavy_atom_count": "",
            "is_valid": "False",
            "structure_image": "",
            "source_image": "",
            "kd_nm": a.kd_nm if a.kd_nm is not None else "",
            "rt_min": a.rt_min if a.rt_min is not None else "",
            "ms_mz": a.ms_mz if a.ms_mz is not None else "",
            "lcms_method": a.lcms_method or "",
            "ms_polarity": a.ms_polarity or "",
            "assay_source_image": a.source_image or "",
        })
    return rows


def _write_csv(
    rows: Iterable[dict],
    path: Path,
    columns: tuple[str, ...],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def run_pipeline(
    img_dir: str = "data/raw/WO2025162428/golden_tables",
    out_dir: str = "data/processed",
) -> None:
    """Run extraction -> DECIMER -> write all three CSVs in order.

    Prints a short banner between every step so progress is visible
    even when stdout is piped to a log file.
    """
    img_path = Path(img_dir)
    out_path = Path(out_dir)
    cells_dir = out_path / "cells"

    print(f"[1/4] scanning {img_path} ...")
    jpgs = list(_iter_jpgs(img_path))
    print(f"   -> {len(jpgs)} JPGs")

    print(f"\n[2/4] extracting SMILES + structure + KD/LCMS ...")
    compounds, assays, failures = _extract_all(img_path, cells_dir)

    print(f"\n[3/4] running DECIMER (OCSR) on cropped structures ...")
    decimer_raw, decimer_canon = _run_decimer(compounds, out_path)

    print(f"\n[4/4] writing combined dataset to {out_path} ...")
    combined_rows = _build_combined_rows(compounds, assays, decimer_raw, decimer_canon)
    n_combined = _write_csv(
        combined_rows,
        out_path / "compounds_combined.csv",
        COMBINED_COLUMNS,
    )

    n_decimer = len(decimer_raw)
    n_agree = sum(1 for r in combined_rows if r["smiles_agree"] == "True")

    print("\n" + "=" * 60)
    print(f"compounds_combined.csv : {n_combined} rows")
    print(f"DECIMER SMILES         : {n_decimer} rows; agreement: {n_agree}")
    if failures:
        print(f"[!] images without rows ({len(failures)}): {', '.join(list(failures)[:5])}"
              + ("..." if len(failures) > 5 else ""))
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
