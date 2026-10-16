"""Unit tests for the 2D structure renderer and macrocycle diagnostic."""

from __future__ import annotations

import pytest

from cpd.render import (
    MACROCYCLE_MIN_SIZE,
    is_macrocycle_intact,
    render_mol,
    render_mol_pil,
    ring_closure_audit,
)


# ---------- ring_closure_audit ----------

def test_audit_counts_simple_ring_once() -> None:
    """A single cyclohexane uses digit '1' exactly twice."""
    audit = ring_closure_audit("C1CCCCC1")
    assert audit["counts"] == {"1": 2}
    assert audit["unclosed"] == {}
    assert audit["len"] == len("C1CCCCC1")


def test_audit_handles_adjacent_ring_digits() -> None:
    """Decalin: 'C12CCCCC1CCCCC2' uses '1' and '2' each twice."""
    audit = ring_closure_audit("C12CCCCC1CCCCC2")
    assert audit["counts"] == {"1": 2, "2": 2}
    assert audit["unclosed"] == {}


def test_audit_handles_multi_digit_percent_rings() -> None:
    """%10 ... %10 should be counted as the two-digit token '10'."""
    audit = ring_closure_audit("C%10CCCCC%10")
    assert audit["counts"] == {"10": 2}
    assert audit["unclosed"] == {}


def test_audit_flags_reused_digit() -> None:
    """A digit used 4 times (the DECIMER failure mode) is flagged."""
    # CCC1CC1CCC1CC1 has two 3-rings using '1' each twice -> '1' = 4.
    # That's a valid SMILES but the digit re-use (4 != 2) is the
    # DECIMER failure mode we want to surface.
    smi = "CCC1CC1CCC1CC1"
    audit = ring_closure_audit(smi)
    assert audit["counts"]["1"] == 4
    assert audit["unclosed"] == {"1": 4}


def test_audit_ignores_digits_inside_brackets() -> None:
    """[13C] and [Fe+2] are isotope / charge, not ring closures."""
    audit = ring_closure_audit("[13C]CC.CC.[Fe+2]")
    # No ring-closure digits, so counts and unclosed are empty.
    assert audit["counts"] == {}
    assert audit["unclosed"] == {}


def test_audit_empty_string() -> None:
    audit = ring_closure_audit("")
    assert audit["counts"] == {}
    assert audit["unclosed"] == {}
    assert audit["len"] == 0


# ---------- is_macrocycle_intact ----------

def test_intact_cyclohexane_at_min_threshold() -> None:
    assert is_macrocycle_intact("C1CCCCC1", min_ring_size=6) is True


def test_broken_cyclohexane_below_threshold() -> None:
    assert is_macrocycle_intact("C1CCCCC1", min_ring_size=12) is False


def test_intact_cyclic_peptide_default_threshold() -> None:
    # 13-membered ring -> intact at the default 12-atom threshold.
    smi = "C1CCCCCCCCCCCC1"
    assert is_macrocycle_intact(smi) is True


@pytest.mark.parametrize("bad", ["", "not_a_smiles", "C1CC1", None])
def test_invalid_inputs_return_false(bad) -> None:
    assert is_macrocycle_intact(bad) is False  # type: ignore[arg-type]


# ---------- render_mol / render_mol_pil ----------

def test_render_simple_molecule_returns_pil_image() -> None:
    img = render_mol("CCO", size=(200, 150))
    assert img is not None
    assert img.size == (200, 150)


def test_render_macrocycle_does_not_raise() -> None:
    """A real macrocycle renders even when no amide bond is present."""
    # All-carbon 13-ring; the linearize step should leave the molecule
    # untouched (no amide C-N bond) and still produce an image.
    img = render_mol("C1CCCCCCCCCCCC1", size=(300, 200))
    assert img is not None
    assert img.size == (300, 200)


def test_render_empty_returns_none() -> None:
    assert render_mol("") is None
    assert render_mol_pil("") is None


def test_render_garbage_returns_none() -> None:
    assert render_mol("not_a_smiles") is None
    assert render_mol_pil("not_a_smiles") is None


def test_render_pil_wrapper_is_alias() -> None:
    img_a = render_mol("CCO", size=(220, 160))
    img_b = render_mol_pil("CCO", size=(220, 160))
    assert img_a is not None and img_b is not None
    assert img_a.size == img_b.size


def test_macrocycle_min_size_constant_is_sensible() -> None:
    """The threshold is high enough to skip pyrrolidines and small rings."""
    # 5-membered pyrrolidine ring (small, not a macrocycle).
    assert is_macrocycle_intact("C1CCNC1") is False
    # 12-membered ring (just at threshold).
    assert is_macrocycle_intact("C1CCCCCCCCCCC1") is True
    # And the constant value is what we expect.
    assert MACROCYCLE_MIN_SIZE == 12
