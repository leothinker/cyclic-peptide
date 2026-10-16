"""Unit tests for the macrocycle-repair heuristic."""

from __future__ import annotations

import pytest

from cpd.macrocycle_repair import (
    attempt_remove_extra_digits,
    attempt_ring_closure_digit_swap,
    attempt_terminal_bridge,
    repair_macrocycle,
)
from cpd.render import is_macrocycle_intact


def test_repair_returns_none_for_intact() -> None:
    """Already-intact SMILES are not 'repaired' -- returns None."""
    smi = "C1CCCCCCCCCCC1"  # 13-ring, intact
    assert repair_macrocycle(smi) is None


def test_repair_returns_none_for_empty() -> None:
    assert repair_macrocycle("") is None


def test_repair_returns_none_for_garbage() -> None:
    assert repair_macrocycle("not_a_smiles") is None


def test_swap_attempt_returns_none_for_intact() -> None:
    smi = "C1CCCCCCCCCCC1"
    assert attempt_ring_closure_digit_swap(smi) is None


def test_swap_does_not_repair_a_clean_single_ring() -> None:
    """A correctly-formed cyclohexane has no over-used digits to swap."""
    smi = "C1CCCCC1"
    assert attempt_ring_closure_digit_swap(smi) is None


def test_bridge_attempt_returns_none_for_intact() -> None:
    smi = "C1CCCCCCCCCCC1"
    assert attempt_terminal_bridge(smi) is None


def test_bridge_returns_none_when_no_terminals() -> None:
    """A molecule where every atom is in a ring has no degree-1 atoms."""
    # Cubane: every atom is in multiple rings.
    smi = "C12C3C4C1C5C2C3C45"
    assert attempt_terminal_bridge(smi) is None


def test_repair_handles_broken_real_world_pattern() -> None:
    """The DECIMER failure pattern (digit re-use) should be repairable.

    This is a synthetic example that mimics the DECIMER failure:
    two small rings plus a chain.  Whether the heuristic repairs it
    is implementation-dependent, but the call should never raise.
    """
    smi = "CCC1CC1CCC1CC1"  # 4 x '1' (over-used)
    result = repair_macrocycle(smi)
    # We don't strictly require a result, just no exception.
    # If a result IS returned, it should be intact.
    if result is not None:
        assert is_macrocycle_intact(result)
    if result is not None:
        assert is_macrocycle_intact(result)


def test_remove_extra_digits_returns_none_for_intact() -> None:
    smi = "C1CCCCCCCCCCC1"
    assert attempt_remove_extra_digits(smi) is None


def test_remove_extra_digits_keeps_valid_smiles_intact() -> None:
    """A clean SMILES with all digits appearing twice stays unchanged."""
    smi = "C1CCC1C2CCC2"  # two 3-rings sharing an edge
    assert attempt_remove_extra_digits(smi) is None
