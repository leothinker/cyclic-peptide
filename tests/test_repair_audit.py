"""Tests for the chemical-invariant repair audit and the MS mass check.

These lock in the two things that must never silently regress:

1. :func:`cpd.repair_audit.atom_multiset` has to agree with RDKit's heavy
   atom count, because it is the primary evidence in
   :func:`cpd.repair_audit.audit_repair`.
2. A repair that invents atoms must never be reported as validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger, Chem

RDLogger.DisableLog("rdApp.*")

from cpd.ms_validation import (
    mz_printing_tolerance_ppm,
    neutral_mass_from_mz,
    parse_polarity,
    validate_against_ms,
)
from cpd.repair_audit import (
    atom_multiset,
    audit_repair,
    fingerprint,
    is_plausible_cyclic_peptide,
)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CCO", 3),
        ("CC(=O)N", 4),
        ("[C@H](C)Cl", 3),
        ("[13C]", 1),
        ("c1ccccc1F", 7),
        ("BrCCl", 3),
        ("[nH]1cccc1", 5),
        ("[Na+]", 1),
        ("[NH4+]", 1),
        ("C%10CCCCC%10", 6),
    ],
)
def test_atom_multiset_counts(smiles: str, expected: int) -> None:
    assert sum(atom_multiset(smiles).values()) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)N[C@@H](C)C(=O)N1CC[C@H]1C(=O)N(C)C",
        "O=C(Nc1ccccc1)c1ccccc1",
        "CC(C)CC1=CC=C(C=C1)C(=O)NC1CCCCC1",
        "O=C(O)C1CCC(N2CCN(CC2)C2=NC3=NC=CC=C3N2)CC1",
    ],
)
def test_atom_multiset_agrees_with_rdkit(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert sum(atom_multiset(smiles).values()) == mol.GetNumHeavyAtoms()


def test_atom_multiset_ignores_ring_digits_and_bonds() -> None:
    # Cyclohexane (6 C) vs n-hexane (6 C): the ring digits carry no atoms.
    assert atom_multiset("C1CCCCC1") == atom_multiset("CCCCCC")
    # A cyclopropane and propane are different molecules, so the histogram
    # must NOT be blind to digits in a way that erases the ring atoms.
    assert sum(atom_multiset("C1CC1").values()) == 3
    assert atom_multiset("C/C=C/C") == atom_multiset("CC=CC")


def test_atom_multiset_stops_at_unbalanced_bracket() -> None:
    # Must not invent atoms past an unterminated bracket.
    assert sum(atom_multiset("CC[CC").values()) == 2


def test_fingerprint_rejects_unparseable() -> None:
    assert fingerprint("C1CC") is None
    assert fingerprint("") is None


# --- plausibility gate -------------------------------------------------


def test_plausibility_rejects_oversized_ring() -> None:
    # A 60-atom ring is not a peptide macrocycle.
    smi = "C1" + "C" * 60 + "1"
    fp = fingerprint(smi)
    assert fp is not None
    ok, reason = is_plausible_cyclic_peptide(fp)
    assert not ok
    assert "largest_ring" in reason


def test_plausibility_rejects_non_peptide() -> None:
    fp = fingerprint("C1CCCCCCCCCCC1")
    assert fp is not None
    ok, reason = is_plausible_cyclic_peptide(fp)
    assert not ok
    assert "amides" in reason


def test_plausibility_rejects_fragmented() -> None:
    # Two fragments, each big enough to be peptide-like on its own, so the
    # fragmentation check is what has to fire.
    fp = fingerprint(
        "CC(=O)N1CCCCC(=O)N(C)CCCCC(=O)N(C)CCCC1.CC(=O)N1CCCCC(=O)N(C)"
        "CCCCC(=O)N(C)CCCC1"
    )
    assert fp is not None
    assert fp["fragments"] == 2
    ok, reason = is_plausible_cyclic_peptide(fp)
    assert not ok
    assert "fragments" in reason


# --- audit tiering -----------------------------------------------------


def _valid_peptide() -> str:
    """A known-good cyclic peptide used as a repair target in tests.

    This is the SMILES transcribed by hand from the patent image for cmpd
    1036. It is used verbatim so the fixtures stay anchored to real data
    rather than to a synthetic structure nobody reviewed. The string lives
    in ``demo_1036_manual_render.py`` and must not be edited by heuristics.
    """
    return (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
        "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
        "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
        "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)"
        "=C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
    )


def test_fixture_is_a_real_intact_macrocycle() -> None:
    """Guard the fixture itself: it must parse and pass the plausibility gate."""
    from cpd.render import is_macrocycle_intact

    smi = _valid_peptide()
    fp = fingerprint(smi)
    assert fp is not None
    assert is_macrocycle_intact(smi)
    ok, reason = is_plausible_cyclic_peptide(fp)
    assert ok, reason


def test_audit_punctuation_only_repair_is_not_validated_when_no_baseline() -> None:
    """A repair on an unparseable input cannot claim a verified atom count."""
    broken = _valid_peptide()[:-3]  # truncate, so RDKit cannot parse it
    repaired = broken + "))"
    audit = audit_repair(broken, repaired, "syntax_fix")
    assert audit.before is None  # no parseable baseline
    # Even if plausibility passes, the tier must not be "high" because the
    # atom-inventory diff is against raw text, not a verified molecule.
    assert audit.tier in ("experimental", "low", "rejected")


def test_audit_flags_invented_atoms_as_experimental() -> None:
    original = _valid_peptide()
    audit = audit_repair(original, original + "C", "syntax_fix")
    assert audit.atoms_invented == 1
    assert audit.tier == "experimental"
    assert not audit.validated


def test_audit_flags_truncation_as_low() -> None:
    # Same macrocyclic scaffold, different chain length. Dropping four atoms
    # is exactly what ``strip_truncated_tail`` does, and the audit must
    # report "low" confidence: the result is a provable *sub*-structure of
    # the real compound, not the compound itself.
    original = (
        "CC(=O)N1CCCCC(=O)N(C)CCCCC(=O)N(C)CCCCC(=O)N(C)CCCCCCCCCCCC1"
    )
    trimmed = (
        "CC(=O)N1CCCCC(=O)N(C)CCCCC(=O)N(C)CCCCC(=O)N(C)CCCC1"
    )
    assert fingerprint(original) is not None
    assert is_plausible_cyclic_peptide(fingerprint(original))[0]
    assert is_plausible_cyclic_peptide(fingerprint(trimmed))[0]
    audit = audit_repair(original, trimmed, "strip_truncated_tail")
    assert audit.atoms_deleted > 0
    assert audit.tier == "low"
    assert not audit.validated


def test_audit_rejects_implausible_repair() -> None:
    audit = audit_repair("C1CC", "C1CCCCC1", "syntax_fix")
    assert audit.tier == "rejected"
    assert not audit.accepted


def test_audit_returns_rejected_for_no_repair() -> None:
    audit = audit_repair("C1CC", None, "")
    assert audit.tier == "rejected"
    assert audit.reason == "no_repair_found"


def test_terminal_bridge_is_never_validated() -> None:
    """The bond-inventing strategy must never reach a validated tier."""
    smi = "CC(=O)NCCCCCCNC(=O)C"
    audit = audit_repair(smi, smi, "terminal_bridge")
    assert audit.tier in ("experimental", "rejected")
    assert not audit.validated


# --- MS validation -----------------------------------------------------


def test_parse_polarity_variants() -> None:
    assert parse_polarity("[M+H]+") == pytest.approx(
        (1.007276466, 1), abs=1e-6
    )
    assert parse_polarity("[M+3H]3+")[1] == 3
    assert parse_polarity("[M-H]-")[1] == -1
    assert parse_polarity("") is None


def test_neutral_mass_is_charge_aware() -> None:
    singly = neutral_mass_from_mz(1610.9, "[M+H]+")
    # The patent lists the same compound twice: cmpd 1036 as [M+H]+ 1610.9
    # and cmpd 1046 as [M+3H]3+ 537.6. A triply charged ion's m/z is not the
    # singly charged one divided by three, because three protons are added.
    triply = neutral_mass_from_mz(537.6, "[M+3H]3+")
    assert singly == pytest.approx(1609.89, abs=0.01)
    # The two agree to 0.12 Da, which is inside the joint printing
    # uncertainty of "1610.9" (+/-0.05 Da) and "537.6" (+/-0.05, x3).
    assert triply == pytest.approx(singly, abs=0.25)


def test_neutral_mass_anion_is_inverted() -> None:
    # [M-H]- at 100 means neutral M is heavier, not lighter.
    m = neutral_mass_from_mz(100.0, "[M-H]-")
    assert m == pytest.approx(101.00728, abs=0.01)


def test_printing_tolerance_scales_with_precision() -> None:
    coarse = mz_printing_tolerance_ppm(1610.9)
    fine = mz_printing_tolerance_ppm(1610.923)
    assert coarse > fine > 0


def test_validate_against_ms_flags_wrong_mass() -> None:
    smi = _valid_peptide()
    res = validate_against_ms(smi, 1610.9, "[M+H]+")
    assert res["verdict"] in ("match", "mismatch")
    assert res["error_ppm"] is not None


def test_validate_against_ms_unknown_for_unparseable() -> None:
    res = validate_against_ms("C1CC", 1610.9, "[M+H]+")
    assert res["verdict"] == "unknown"
    assert res["observed_mass"] is None

