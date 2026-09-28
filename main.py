"""Extract dual-table dataset (compounds.csv + assays.csv) from patent JPGs.

Pipeline overview
-----------------
For every JPG in ``img_dir`` (default: WO2025162428's golden_tables):

  1. :class:`RapidOcrTableExtractor.extract_records` returns CompoundRecords
     (SMILES + structure PNG) and AssayRecords (KD + LCMS).
  2. Compound-id spelling variants collapse through
     :func:`cpd.merge.normalise_cmpd_id`.
  3. The two streams are written to ``data/processed/compounds.csv`` and
     ``data/processed/assays.csv``. Structure PNGs land in
     ``data/processed/cells/``.

Output schema
-------------
``compounds.csv``::

    cmpd_id, smiles, canonical_smiles, molecular_weight,
    heavy_atom_count, is_valid, structure_image, source_image

``assays.csv``::

    cmpd_id, kd_nm, rt_min, ms_mz, lcms_method, ms_polarity, source_image

The two tables share ``cmpd_id``; a compound may appear in either one
or both depending on which patent pages cover it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from cpd.merge import dedupe_assays, dedupe_compounds, normalise_cmpd_id
from cpd.parsers.table_extractor import (
    AssayRecord,
    Column,
    CompoundRecord,
    RapidOcrTableExtractor,
)


COMPOUND_COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "smiles",
    "canonical_smiles",
    "molecular_weight",
    "heavy_atom_count",
    "is_valid",
    "structure_image",
    "source_image",
)

ASSAY_COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "kd_nm",
    "rt_min",
    "ms_mz",
    "lcms_method",
    "ms_polarity",
    "source_image",
)


def _iter_jpgs(img_dir: Path) -> Iterable[Path]:
    """Yield every ``*.jpg`` in ``img_dir``, sorted by filename."""
    return sorted(img_dir.glob("*.jpg"))


def _extract_all(
    img_dir: Path,
    cells_dir: Path,
) -> tuple[list[CompoundRecord], list[AssayRecord]]:
    """Run the extractor over every image, deduping across pages."""
    extractor = RapidOcrTableExtractor()
    compounds: dict[str, CompoundRecord] = {}
    assays: dict[str, AssayRecord] = {}
    prev_columns: list[Column] = []

    for idx, image_path in enumerate(_iter_jpgs(img_dir), start=1):
        page_compounds, page_assays, columns = extractor.extract_records(
            image_path, cells_dir=cells_dir, prev_columns=prev_columns
        )
        prev_columns = columns or prev_columns

        page_compounds = [c for c in page_compounds if normalise_cmpd_id(c.cmpd_id)]
        page_assays = [a for a in page_assays if normalise_cmpd_id(a.cmpd_id)]

        for cid, rec in dedupe_compounds(page_compounds).items():
            compounds.setdefault(cid, rec)  # first image wins
        for cid, rec in dedupe_assays(page_assays).items():
            assays.setdefault(cid, rec)

        print(
            f"   [{idx}] {image_path.name} -> "
            f"+{len(page_compounds)} cmpd, +{len(page_assays)} assay "
            f"| total: {len(compounds)} cmpd, {len(assays)} assay",
            end="\r",
        )
    print()
    return list(compounds.values()), list(assays.values())


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
    img_path = Path(img_dir)
    out_path = Path(out_dir)
    cells_dir = out_path / "cells"

    print(f"🚀 [1/3] 扫描目录中的表格图片: {img_path} ...")
    jpgs = list(_iter_jpgs(img_path))
    print(f"   -> 共发现 {len(jpgs)} 张 JPG")

    print(f"\n📦 [2/3] 正在提取 SMILES + 结构图，并解析 KD/LCMS ...")
    compounds, assays = _extract_all(img_path, cells_dir)

    print(f"\n🧪 [3/3] 写入双表数据集到 {out_path} ...")
    n_compounds = _write_csv(
        (asdict(c) for c in compounds),
        out_path / "compounds.csv",
        COMPOUND_COLUMNS,
    )
    n_assays = _write_csv(
        (asdict(a) for a in assays),
        out_path / "assays.csv",
        ASSAY_COLUMNS,
    )

    n_valid = sum(1 for c in compounds if c.is_valid)
    n_struct = sum(1 for c in compounds if c.structure_image)

    print("\n" + "=" * 55)
    print("🎉 双表数据集构建完成！")
    print(f"📊 化合物表 compounds.csv : {n_compounds} 条 (RDKit 有效: {n_valid}, 含结构图: {n_struct})")
    print(f"🎯 活性表 assays.csv       : {n_assays} 条")
    print(f"📁 结构图裁剪目录          : {cells_dir}")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
