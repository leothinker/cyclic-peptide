"""Tests for the incremental SMILES prefix validator.

The validator underpins every repair search, so the properties that matter
are asymmetric and must both hold:

* **No false negatives.** If RDKit accepts a string, we must not reject it.
  A false negative prunes the correct answer out of the beam, which is
  indistinguishable from "the search failed".
* **No free passes on obviously broken input.** A validator that accepts
  everything would silently turn the repair beam into a random search.

Because the validator is also *cloneable* and used incrementally, there is
an extra property no whole-string test can catch: advancing must not mutate
the receiver, or every beam level would contaminate its siblings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpd.parsers.prefix_validator import (
    PrefixValidator,
    extend_ok,
    first_violation,
)

RDLogger.DisableLog("rdApp.*")


def _walk(smiles: str) -> Optional[PrefixValidator]:
    v = PrefixValidator()
    for ch in smiles:
        v = v.advance(ch)
        if v is None:
            return None
    return v


VALID = [
    "C",
    "CCO",
    "CC(C)C",
    "c1ccccc1",
    "C1CCCCC1",
    "C1CCCCC1C",
    "CC(=O)N",
    "C(=O)N",
    "[C@@H](C)C(=O)N",
    "[C@H]1(C)CC[C@@H](C)CC1",
    "Cl",
    "Br",
    "CCl",
    "CBr",
    "[13C]",
    "[2H]",
    "[NH4+]",
    "[O-]",
    "[O-]C=O",
    "[nH]1cccc1",
    "[s]1cccc1",
    "[Fe+2]",
    "C%10CCCCC%10",
    "[C@H](F)(Cl)Br",
    "C1CC2CCC1CC2",
    "CC(=O)N1CCCCC1",
    "C/C=C\\C",
    "[C@@H]1(C(=O)N)CCCC1",
    "[C@TH1](C)(N)O",
    "[C@TB15](O)N",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "*",
    "[*]",
    "[CH4]",
    "CCO.CCO",
]


MALFORMED = [
    "",
    "CC(",              # unclosed branch
    "CC)",              # branch close with nothing open
    "CC[",              # unterminated bracket
    "CC]",              # stray bracket close
    "CC1CCC",           # dangling ring open
    "C(=)C",            # bond with no atom after it
    "CC..CC",           # doubled dot
    ".CC",              # leading dot
    "CC(=O",            # branch left open mid-carbonyl
    "C1CC1CC2",         # ring 2 opened and never closed
]


@pytest.mark.parametrize("smiles", VALID)
def test_accepts_valid(smiles):
    v = _walk(smiles)
    assert v is not None, f"rejected {smiles}"
    assert v.is_complete()


@pytest.mark.parametrize("smiles", MALFORMED)
def test_rejects_malformed(smiles):
    v = _walk(smiles)
    assert v is None or not v.is_complete(), f"accepted {smiles!r}"


def test_agreement_with_rdkit_on_valid_set():
    """Every string in VALID must also satisfy RDKit -- guards the fixture."""
    for s in VALID:
        assert Chem.MolFromSmiles(s) is not None, f"fixture is not real: {s}"


def test_rdkit_agrees_on_malformed_set():
    """Conversely the MALFORMED set must be rejected by RDKit too."""
    for s in MALFORMED:
        if not s:
            continue
        assert Chem.MolFromSmiles(s) is None, f"fixture is not malformed: {s}"


def test_advance_does_not_mutate_receiver():
    """The beam holds one validator per candidate; aliasing would corrupt it."""
    # Ends mid-branch so there is room for the sibling edits below.
    base = _walk("CC(=O")
    assert base is not None
    before = (base.prev, base.depth, dict(base.rings), base.trail_cl, base.sym)

    branch = base.advance("C")
    assert branch is not None
    child_before = (branch.prev, branch.depth, dict(branch.rings))
    assert branch.advance(")") is not None
    assert branch.advance("(") is not None
    assert (branch.prev, branch.depth, dict(branch.rings)) == child_before

    assert (base.prev, base.depth, dict(base.rings), base.trail_cl, base.sym) == before


def test_lowercase_l_cannot_start_an_atom():
    """'Cl' is one atom, but after '(' a bare 'l' starts nothing."""
    assert _walk("Cl") is not None
    v = PrefixValidator().advance("C")
    v = v.advance("(")
    assert v is not None
    assert v.advance("l") is None


def test_extend_ok_matches_walk():
    assert extend_ok("CC", "O")
    assert not extend_ok("CC", ")")
    assert extend_ok("C(", "C")
    assert extend_ok("C(=O", ")")     # closing after an atom is legal
    assert extend_ok("C(=O", "C")     # a new atom continues the chain
    assert extend_ok("C=", "N")       # C=N is fine
    assert not extend_ok("C(=)", ")")  # a bond before a branch close is not
    assert not extend_ok("C=(", ")")   # nor may a bond be left dangling


def test_first_violation_locates_break():
    assert first_violation("CC)") == 2
    # Dangling ring open is incomplete rather than wrong: reported at len.
    assert first_violation("CC1CCC") == 6
    assert first_violation("CCO") is None


def test_walk_is_incremental_over_whole_string():
    # A real fragment with every ring actually closed.
    smi = "CC[C@H](C)[C@@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)C(=O)N(C)C1=O"
    v = _walk(smi)
    assert v is not None and v.is_complete()


def test_bracket_symbol_boundaries():
    """A bracket symbol is greedy over letters but stops at any non-letter."""
    assert _walk("[NH4+]") is not None      # N, then H4 as hydrogen count
    assert _walk("[Na+]") is not None       # Na, not N then a
    assert _walk("[Sc]") is not None        # Sc, not S then c
    assert _walk("[Se]") is not None
    assert _walk("[Q]") is None             # not an element
    assert _walk("[]") is None
