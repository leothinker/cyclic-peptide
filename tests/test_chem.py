"""Tests for chemical validation and peptide SMILES auto-repair."""

from cpd.chem import auto_repair_smiles, validate_and_enrich


def test_auto_repair_smiles() -> None:
    # 1. 测试修复丢失的等号 C(O)N -> C(=O)N
    assert auto_repair_smiles("CC(O)NCC") == "CC(=O)NCC"
    # 2. 测试修复丢失的中括号 [C@H( -> [C@H](
    assert auto_repair_smiles("CC[C@H(CC") == "CC[C@H](CC"
    # 3. 测试清理制表符杂质
    assert auto_repair_smiles("CC(=O)N╬▓") == "CC(=O)N"


def test_auto_repair_reads_lactam_zero_as_oxygen() -> None:
    # A lactam carbonyl is printed as ``C1=O)``: the ring-closure
    # digit sits between the carbon and the =O, and the recogniser
    # reads that O as a zero.  With a single zero in the string the
    # zero has no ring-closure partner, so it is a misread oxygen.
    assert auto_repair_smiles("C1=0)") == "C1=O)"
    assert auto_repair_smiles("CC(=O)NC1=0)C") == "CC(=O)NC1=O)C"


def test_auto_repair_keeps_paired_ring_closure_zero() -> None:
    # Two zeros means both are ring-closure digits, each with a
    # partner.  They must not be turned into oxygens.
    raw = "C0CCCCC0"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_restores_a_dropped_chiral_bracket() -> None:
    # The recogniser drops the "]" of "[C@@H]", so the printed
    # "[C@@H]4" is read as "[C@@H4".  A digit cannot sit inside a
    # bracket atom, so the bracket is restored and the digit stays
    # where it was.
    # The digit must have a partner: it is a ring closure.
    assert (
        auto_repair_smiles("[C@@H4C(=O)NC1CCC4")
        == "[C@@H]4C(=O)NC1CCC4"
    )


def test_auto_repair_leaves_a_chiral_bracket_alone() -> None:
    # A bracket atom that is already closed must not be touched.
    raw = "[C@@H]4C(=O)NC1CCC4"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_keeps_an_unpaired_digit() -> None:
    # The digit is only restored when it is a ring closure, and a
    # ring closure always has a partner.  Without a partner the
    # repair would be a guess, so the string is left alone.
    raw = "[C@@H4C(=O)NC"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_restores_a_mirrored_bracket() -> None:
    # The recogniser read the opening bracket of "[C@@H](C)" as a
    # closing one.  One bracket read as its mirror image moves the
    # bracket difference by two, so the string holds two more "]" than
    # "[", the parentheses still balance, and exactly one position can
    # be flipped back into a string that parses.
    raw = (
        "CCC[C@@H]1N(C)C(=O)]C@@H](C)N(C)C(=O)[C@H](Cc2cncs2)"
        "NC(=O)[C@H](C)N(C)C(=O)[C@H](CCOC)NC(=O)[C@H]2CCCN2C"
        "(=O)[C@H](Cc2ccccc2)N(C)C(=O)[C@H](Cc2ccc(Cl)cc2)NC(=O)"
        "[C@H](C(C)C)N(C)C(=O)[C@H](Cc2ccc(F)cc2)NC1=O"
    )
    fixed = auto_repair_smiles(raw)
    assert fixed != raw
    assert fixed.count("[") == fixed.count("]")
    assert "C(=O)[C@@H](C)" in fixed
    assert validate_and_enrich(fixed) is not None


def test_auto_repair_leaves_a_single_bracket_alone() -> None:
    # A mirror-image misread changes the bracket difference by *two*.
    # A difference of one therefore cannot come from that mistake, and
    # flipping a bracket would only make the string worse.
    # The string below holds one "]" more than "[", and it does not
    # parse, so it is left exactly as it is.
    raw = "CC[C@H](C)C(=O)N(C)C(=O)]N"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_leaves_a_balanced_bracket_pair_alone() -> None:
    # A string whose brackets balance is not missing a bracket.
    raw = "CC[C@H](C)C(=O)N(C)C(=O)[C@@H](C)N"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_mirrored_bracket_keeps_a_ring_closure_a_ring() -> None:
    # Turning the bracket into an opening one would pull the "1"
    # inside it, and a digit inside a bracket atom is an isotope, not
    # a ring closure.  That is a different molecule, so the flip is
    # refused and the string is left alone.
    raw = "CC[C@H](C)C(=O)N(C)C(=O)]1CC]"
    assert auto_repair_smiles(raw) == raw


def test_auto_repair_mirrored_bracket_needs_balanced_parentheses() -> None:
    # The parentheses must balance, or the confusion is wider than a
    # single bracket and the repair would be a guess.
    raw = "CC[C@H](C)C(=O)N(C)C(=O)]C@@H](C N"
    assert auto_repair_smiles(raw) == raw


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
