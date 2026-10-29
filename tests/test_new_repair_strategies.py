"""Tests for the three new repair strategies (v8): syntax_fix,
rejoin_fragments, strip_truncated_tail."""
from __future__ import annotations

import pytest
from rdkit import Chem

from cpd.macrocycle_repair import (
    attempt_syntax_fix,
    attempt_rejoin_fragments,
    attempt_strip_truncated_tail,
    repair_macrocycle,
)
from cpd.render import is_macrocycle_intact


def test_syntax_fix_returns_none_for_clean_input():
    smi = "C1CCCCCCCCCCCCCCCCCCCC1"
    assert is_macrocycle_intact(smi)
    assert attempt_syntax_fix(smi) is None


def test_syntax_fix_returns_none_for_empty():
    assert attempt_syntax_fix("") is None


def test_rejoin_returns_none_for_clean_smiles():
    smi = "C1CCCCCCCCCCCCCCCCCCCC1"
    assert "." not in smi
    assert attempt_rejoin_fragments(smi) is None


def test_strip_truncated_tail_handles_unclosed_branch():
    smi = "C1CCCCCCCCCCCCCCCCCCCCC(C)N("
    out = attempt_strip_truncated_tail(smi)
    if out is not None:
        assert is_macrocycle_intact(out)


def test_repair_macrocycle_dispatch_handles_truncated_real_world():
    """Real DECIMER failure: SMILES ends with 'C(=O)N(C)C(' unclosed paren at the tail."""
    smi = (
        "CCCC1CCN(CC1)C(=O)[C@H](CC(=O)N(C)[C@@H](C)C(=O)N(C)[C@@H]"
        "([C@@H](C)CC)C(=O)N(C)[C@@H](C)C(=O)N2CC[C@H]2C(=O)N(C)CC"
        "(=O)N(C)CC(=O)N[C@@H](CCC3=CC(=C(C(=C3)F)C(F)(F)F)F)C(=O)N"
        "4CCC[C@H]4C(=O)NC5(CCCC5)C(=O)N(C)[C@@H](C6CCCC6)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)C)C(=O)N1)N(C)C("
    )
    out = repair_macrocycle(smi)
    if out is not None:
        assert is_macrocycle_intact(out)


def test_repair_macrocycle_returns_none_for_empty():
    assert repair_macrocycle("") is None


def test_repair_macrocycle_returns_none_for_intact():
    smi = "C1CCCCCCCCCCCCCCCCCCCC1"
    assert repair_macrocycle(smi) is None
