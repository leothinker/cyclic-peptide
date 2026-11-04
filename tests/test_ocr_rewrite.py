import sys
sys.path.insert(0, '"'"'src'"'"')
from rdkit import Chem
from cpd.ocr_rewrite import rewrite_oxo, repair_by_rewrite, _OXO


def test_rewrite_preserves_intact_carbonyls():
    # ``C(=O)`` already has its opening bracket -- the regex matches the
    # ``=O)`` substring, but the substitution must skip so the existing
    # bracket is not doubled into ``C((=O)``.
    text, n = rewrite_oxo("C(=O)")
    assert n == 0
    assert text == "C(=O)"


def test_rewrite_repairs_dash_oxo_close():
    text, n = rewrite_oxo("C-O)")
    assert n == 1
    assert text == "C(=O)"


def test_rewrite_repairs_equal_oxo_close():
    text, n = rewrite_oxo("C=O)")
    assert n == 1
    assert text == "C(=O)"


def test_rewrite_does_not_touch_intact_carbonyls():
    text, n = rewrite_oxo("CC(=O)N")
    assert n == 0
    assert text == "CC(=O)N"


def test_rewrite_leaves_lactam_with_ring_closure_alone():
    # ``C1=O)`` is a lactam printed correctly: the ``1`` closes the
    # ring and ``=O`` is the carbonyl.  Inserting a bracket there
    # would break a string that was never damaged.
    text, n = rewrite_oxo("C1=O)")
    assert n == 0
    assert text == "C1=O)"


def test_rewrite_leaves_lactam_with_ring_closure_alone_in_context():
    text, n = rewrite_oxo("C(=O)NC1=O)")
    assert n == 0
    assert text == "C(=O)NC1=O)"


def test_rewrite_handles_real_1036_corruption():
    ref = (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
        "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
        "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)"
        "[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)"
        "C9=C(F)C(F)=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
    )
    corrupted = ref.replace("C(=O)", "C-O)", 4).replace("C(=O)", "C=O)", 2)
    fixed, n = rewrite_oxo(corrupted)
    # six carbonyls were damaged; six remained intact.
    assert n == 6
    assert fixed == ref


def test_repair_by_rewrite_parses_after_fix():
    ref = "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N(C)CC(=O)N1"
    corrupted = ref.replace("C(=O)", "C-O)", 1).replace("C(=O)", "C=O)", 1)
    result = repair_by_rewrite(corrupted, normalise=False)
    assert result["parses"] is True
    assert result["n_fixes"] == 2
    assert result["formula"].startswith("C")


def test_repair_by_rewrite_returns_mass_diagnostics():
    out = repair_by_rewrite("CCO", normalise=False)
    assert out["parses"] is True
    assert out["formula"] == "C2H6O"
    assert out["num_rings"] == 0
    assert out["largest_ring"] == 0
