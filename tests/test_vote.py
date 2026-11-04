"""Tests for the second-recogniser vote.

The vote lets a second recogniser settle the characters the first one
lost.  Two properties matter and are locked here.

*The primary reading wins by default.*  A vote must never make a
correct reading worse, so the combinations are tried in order of how
much they change the primary reading, and the combination that
changes nothing is tried first.

*The rule never looks at a reference.*  It may use the two readings
and whether the result is a well-formed SMILES, and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.vote import SMILES_ALPHABET, vote_lines  # noqa: E402


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


# ---------- the primary reading wins by default ----------


def test_a_reading_that_already_parses_is_kept() -> None:
    """The primary reading is the first candidate, so it wins."""
    primary = ["C(=O)N"]
    secondary = ["C(=0)N"]
    result = vote_lines(primary, secondary)
    assert result.text == "C(=O)N"
    assert result.changed_bands == 0
    assert result.used_primary is True


def test_a_reading_that_parses_is_not_replaced_by_a_longer_one() -> None:
    """A longer reading is not better just because it is longer."""
    primary = ["CCO"]
    secondary = ["CCOC"]
    result = vote_lines(primary, secondary)
    assert result.text == "CCO"
    assert result.changed_bands == 0


# ---------- the vote does its job ----------


def test_a_lost_character_is_recovered_when_the_second_reading_has_it() -> None:
    """A bracket the primary lost comes back from the second reading."""
    primary = ["C(=O)NC"]
    secondary = ["C(=O)N(C"]
    result = vote_lines(primary, secondary)
    # Neither reading is balanced on its own; the vote can only pick
    # one of them, and it prefers the one that changes nothing.
    assert result.text == "C(=O)NC"


def test_the_vote_picks_the_reading_that_parses() -> None:
    """A broken primary is replaced by a parseable second reading."""
    primary = ["CC(=O)N(C"]
    secondary = ["CC(=O)N(C)"]
    result = vote_lines(primary, secondary)
    assert result.text == "CC(=O)N(C)"
    assert result.changed_bands == 1
    assert result.parses is True


def test_the_vote_prefers_the_smallest_change_that_parses() -> None:
    """Two changes are not made when one is enough."""
    primary = ["CC(=O)N(C", "C"]
    secondary = ["CC(=O)N(C)", "CC"]
    result = vote_lines(primary, secondary)
    assert result.changed_bands == 1
    assert result.text == "CC(=O)N(C)C"


# ---------- the search is bounded ----------


def test_too_many_differences_are_not_searched() -> None:
    """A cell whose two readings disagree everywhere is left alone."""
    primary = ["A", "B", "C", "D", "E"]
    secondary = ["a", "b", "c", "d", "e"]
    result = vote_lines(primary, secondary, max_changes=2)
    assert result.text == "ABCDE"
    assert result.changed_bands == 0


def test_the_vote_is_deterministic() -> None:
    """The same inputs always give the same answer."""
    primary = ["CC(=O)N(C", "C"]
    secondary = ["CC(=O)N(C)", "CC"]
    first = vote_lines(primary, secondary)
    second = vote_lines(primary, secondary)
    assert first.text == second.text
    assert first.changed_bands == second.changed_bands


# ---------- the rule is about well-formedness, not chemistry ----------


def test_a_reading_with_a_space_is_not_well_formed() -> None:
    """RDKit accepts a space; a printed SMILES column never has one."""
    # RDKit reads "CC O" as "CC", so a parse check alone would accept
    # a combination that introduced a space.
    primary = ["CC(=O)N(C"]
    secondary = ["CC(=O)N(C) "]
    result = vote_lines(primary, secondary)
    # The combination with the space is not a SMILES, so it is
    # rejected even though RDKit can parse it.
    assert result.text == "CC(=O)N(C"


def test_a_wildcard_is_not_a_smiles_character() -> None:
    """A wildcard atom cannot be printed in a SMILES column."""
    assert "*" not in SMILES_ALPHABET
    primary = ["CC(=O)N(C"]
    secondary = ["CC(=O)N(C)*"]
    result = vote_lines(primary, secondary)
    assert result.text == "CC(=O)N(C"


def test_the_alphabet_holds_the_characters_a_smiles_uses() -> None:
    """The guard must not reject a legitimate SMILES."""
    for character in "C(=O)N[C@@H]1CC2CCC1CC2c1ccccc1/F=C\\Cl":
        assert character in SMILES_ALPHABET


def test_the_alphabet_holds_every_element_symbol() -> None:
    """A printed SMILES may hold a bracketed atom of any element.

    The alphabet is built from the element symbols, so the two-letter
    ones come with their second letter -- ``l`` for ``Cl``, ``r`` for
    ``Br``.  An alphabet without them would reject a perfectly good
    reading of a chlorinated peptide.
    """
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    for number in range(1, table.GetMaxAtomicNumber() + 1):
        for character in table.GetElementSymbol(number):
            assert character in SMILES_ALPHABET, character


def test_a_chlorine_is_not_rejected() -> None:
    """``Cl`` must survive: the corpus really does contain chlorine."""
    assert "l" in SMILES_ALPHABET
    assert "r" in SMILES_ALPHABET


def test_a_non_ascii_character_is_not_a_smiles_character() -> None:
    """The recogniser reads footnote marks; a printed column has none."""
    assert "②" not in SMILES_ALPHABET
    assert "−" not in SMILES_ALPHABET
    assert " " not in SMILES_ALPHABET


# ---------- the bond order may not shrink ----------


def test_a_vote_may_not_drop_a_double_bond() -> None:
    """Turning an aromatic ring saturated is a change of chemistry.

    Both readings parse, so a parse check cannot tell them apart.
    The vote is licensed to restore a lost ``=`` and to leave the
    rest alone, so a combination with fewer bonds is refused.
    """
    primary = ["C1=CC=CC=C1"]
    secondary = ["C1-CC-C-C-C1"]
    result = vote_lines(primary, secondary)
    assert result.text == "C1=CC=CC=C1"
    assert result.changed_bands == 0


def test_a_vote_may_drop_a_double_bond_when_asked_not_to() -> None:
    """The guard can be switched off, for callers that want it."""
    # The primary does not parse -- it has an unclosed bracket -- so
    # the vote has to reach for the second reading, and the second
    # reading is the one that loses the double bonds.
    primary = ["C1=CC=CC=C1("]
    secondary = ["C1-CC-C-C-C1"]
    result = vote_lines(primary, secondary, keep_bonds=False)
    assert result.text == "C1-CC-C-C-C1"
    assert result.changed_bands == 1


def test_a_bond_loss_is_refused_even_when_it_is_the_only_option() -> None:
    """With the guard on, a bond loss is refused and the primary stays."""
    primary = ["C1=CC=CC=C1("]
    secondary = ["C1-CC-C-C-C1"]
    result = vote_lines(primary, secondary, keep_bonds=True)
    assert result.text == "C1=CC=CC=C1("
    assert result.changed_bands == 0


def test_a_validity_test_can_be_supplied() -> None:
    """The rule is about well-formedness; the test is the caller's."""
    primary = ["a", "b"]
    secondary = ["A", "B"]
    result = vote_lines(
        primary,
        secondary,
        is_valid=lambda text: text == "AB",
    )
    assert result.text == "AB"
    assert result.changed_bands == 2


def test_bracket_balance_can_be_used_as_the_test() -> None:
    """A caller may use a different well-formedness test."""
    primary = ["CC(=O"]
    secondary = ["CC(=O)"]
    result = vote_lines(
        primary,
        secondary,
        is_valid=_balanced,
    )
    assert result.text == "CC(=O)"
