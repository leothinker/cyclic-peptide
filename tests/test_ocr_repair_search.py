"""Tests for the OCR repair search, on cases where the answer is known.

The search is only worth trusting if it can recover a known answer. These
tests build damaged strings *from* a reference by deleting the characters the
edit model says OCR deletes, then check the search puts them back.

The important property is not "it returns something" but "it returns the
reference". A mass-verified macrocycle that differs from the reference would
be a silent failure, because the mass gate cannot tell the two apart.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpd.mass_constrained_search import bracket_deficit, repair_from_ocr
from cpd.ocr_mass_gate import MassGate, raw_element_counts

RDLogger.DisableLog("rdApp.*")


def damage(text: str, drop: str, count: int, seed: int = 0) -> str:
    """Delete ``count`` random occurrences of characters in ``drop``."""
    rng = random.Random(seed)
    positions = [i for i, c in enumerate(text) if c in drop]
    rng.shuffle(positions)
    victims = set(positions[:count])
    return "".join(c for i, c in enumerate(text) if i not in victims)


def mass_of(smiles: str) -> float:
    """Exact monoisotopic mass, used to stand in for a reported m/z."""
    return rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles))


def mz_of(smiles: str) -> float:
    """The [M+H]+ m/z a mass spectrometer would print for ``smiles``.

    The repair API takes an observed m/z, not a neutral mass, so tests have to
    add the proton back. Getting this wrong shifts the window by 1.007 Da and
    makes every assertion fail for the wrong reason.
    """
    return mass_of(smiles) + 1.007276466


def neutral_of(smiles: str) -> float:
    """Neutral monoisotopic mass, which is what MassGate expects."""
    return mass_of(smiles)


def canonical(smiles: str) -> str:
    """Canonical form, so atom ordering cannot mask a difference."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# A 19-membered cyclic hexapeptide: the macrocycle is larger than the
# 12-atom acceptance threshold, and the backbone carries the amide and
# carboxyl brackets that OCR destroys. The second fixture adds a fused
# ring, so a repair has to get two ring systems right at once.
SIMPLE = "CC1CC(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)N1"
BICYCLIC = ("CC(C)C1CC(=O)NC(Cc2ccccc2)C(=O)NC(C)C(=O)NC(C)C(=O)"
            "NC(C)C(=O)N1")


def test_bracket_deficit_counts_unmatched_closers():
    assert bracket_deficit("C(=O)N")["total"] == 0
    # One "(" against two ")" means one closer has no opener.
    d = bracket_deficit("CC(C=O)N)O")
    assert d["open_paren"] == 1
    assert d["total"] == 1
    assert bracket_deficit("CC(C)")["total"] == 0


def test_raw_counts_survive_unbalanced_bracket():
    """The strict tokenizer drops atoms after an unclosed "[", so we do not."""
    broken = "CC[C@H]C)CC[Ca@H]N"
    counts = raw_element_counts(broken)
    assert counts["C"] == broken.count("C")


def test_gate_confirms_composition_of_reference():
    gate = MassGate(target=neutral_of(SIMPLE), polarity="[M+H]+")
    score = gate.score_text(SIMPLE)
    assert score["verdict"] == "on_integer_h"
    assert score["carbon"] == raw_element_counts(SIMPLE)["C"]


def test_undamaged_reference_passes_through():
    res = repair_from_ocr(SIMPLE, mz_of(SIMPLE), "[M+H]+", max_seconds=20.0)
    assert res["smiles"] is not None
    assert res["trace"] == "already_valid_and_in_window"


@pytest.mark.parametrize("drop_count", [1, 2, 3])
def test_recovers_deleted_carbonyl_open_brackets(drop_count):
    """Deleting "(" is the dominant real damage; the search must put it back."""
    damaged = damage(SIMPLE, "(", drop_count, seed=drop_count)
    assert damaged != SIMPLE
    res = repair_from_ocr(damaged, mz_of(SIMPLE), "[M+H]+",
                          max_seconds=25.0)
    assert res["smiles"] is not None, f"failed to recover: {res['trace']}"
    assert canonical(res["smiles"]) == canonical(SIMPLE)


@pytest.mark.parametrize("drop_count", [1, 2])
def test_recovers_bicyclic_after_damage(drop_count):
    """The macrocycle must come back even where the exact branch may not.

    This fixture is the honest limit of the approach. Deleting one "(" can
    leave two different branches that are both valid, both macrocyclic, and
    both the same formula -- so the mass gate cannot choose between them, and
    the search returns one of them. What is guaranteed is the atom inventory,
    the macrocycle, and the mass; what is *not* guaranteed is which of several
    mass-equivalent connectivities comes back.
    """
    damaged = damage(BICYCLIC, "(", drop_count, seed=10 + drop_count)
    res = repair_from_ocr(damaged, mz_of(BICYCLIC), "[M+H]+",
                          max_seconds=25.0)
    assert res["smiles"] is not None, f"failed to recover: {res['trace']}"
    assert raw_element_counts(res["smiles"]) == raw_element_counts(BICYCLIC)
    assert abs(mass_of(res["smiles"]) - mass_of(BICYCLIC)) < 0.6
    mol = Chem.MolFromSmiles(res["smiles"])
    biggest = max(len(r) for r in mol.GetRingInfo().AtomRings())
    assert biggest >= 12, "macrocycle was not restored"


def test_insertion_only_repair_preserves_atom_inventory():
    """Brackets cannot change the formula, so a repair must not either."""
    damaged = damage(SIMPLE, "(", 2, seed=7)
    res = repair_from_ocr(damaged, mz_of(SIMPLE), "[M+H]+",
                          max_seconds=25.0)
    assert res["smiles"] is not None
    assert raw_element_counts(damaged) == raw_element_counts(res["smiles"])


def test_mass_gate_rejects_wrong_molecule():
    """A well-formed but wrong molecule must not pass the gate."""
    wrong = "CC1CC(=O)NC(C)CCCCCCCCCCC(=O)NC(C)C(=O)NC(C)C(=O)N1"
    assert Chem.MolFromSmiles(wrong) is not None
    res = repair_from_ocr(wrong, mz_of(SIMPLE), "[M+H]+", max_seconds=10.0)
    if res["smiles"] is not None:
        assert abs(mass_of(res["smiles"]) - mass_of(SIMPLE)) < 1.0
