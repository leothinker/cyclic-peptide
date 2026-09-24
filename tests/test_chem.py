"""Tests for chemical validation and peptide SMILES auto-repair."""

from cpd.chem import auto_repair_smiles, validate_and_enrich


def test_auto_repair_smiles() -> None:
    # 1. 测试修复丢失的等号 C(O)N -> C(=O)N
    assert auto_repair_smiles("CC(O)NCC") == "CC(=O)NCC"
    # 2. 测试修复丢失的中括号 [C@H( -> [C@H](
    assert auto_repair_smiles("CC[C@H(CC") == "CC[C@H](CC"
    # 3. 测试清理制表符杂质
    assert auto_repair_smiles("CC(=O)N╬▓") == "CC(=O)N"


def test_validate_and_enrich_valid_cyclic_peptide() -> None:
    # 闭合五元内酰胺环（合法分子）
    info = validate_and_enrich("C1CC(=O)NC1")
    assert info is not None
    assert info.canonical_smiles == "O=C1CCCN1"
    assert info.molecular_weight > 80.0
    assert info.num_rings == 1


def test_validate_and_enrich_invalid_unclosed_ring() -> None:
    # 未闭合环必须判定为无效返回 None
    info = validate_and_enrich("C1CC(=O)NC")
    assert info is None
