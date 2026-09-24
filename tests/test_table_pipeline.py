"""Tests for table classification and real OCR row extraction."""

from pathlib import Path
from cpd.parsers.table_extractor import ActivityRow, RapidOcrTableExtractor, SmilesRow
from cpd.parsers.table_finder import TableType, _classify_size


def test_classify_table_images_by_size() -> None:
    # 验证无需 OCR 即可区分 SMILES 表和活性表
    assert _classify_size(1945, 1500, 600_000) == TableType.SMILES
    assert _classify_size(1910, 1500, 300_000) == TableType.ACTIVITY
    assert _classify_size(1000, 1000, 50_000) == TableType.STRUCTURE


def test_rapidocr_table_extractor_golden_image() -> None:
    image_path = Path(
        "data/raw/WO2025162428/golden_tables/PCTCN2025075389-ftappb-I100302.jpg"
    )
    if not image_path.exists():
        return  # 避免缺少本地图片时 CI 报错

    extractor = RapidOcrTableExtractor()
    rows = extractor.extract_smiles(image_path)
    assert len(rows) == 4
    # 验证 1055, 1057, 1058 中至少有 3 个分子成功验证通过
    valid_count = sum(1 for r in rows if r.is_valid)
    assert valid_count >= 3
