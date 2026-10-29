"""Deterministic carbonyl-rewrite repair for OCR'd printed SMILES.

Why this exists
---------------
``analysis/align_ocr_to_reference.py`` measured the real OCR damage on
cmpd 1036 and found it is *not* arbitrary character loss. The dominant
mechanism is a single, recognisable substitution:

* ``C(=O)``  is printed as  ``C-O)``   -- the OCR read ``(=`` as ``-``

plus a second, simpler one:

* ``C(=O)``  is printed as  ``C=O)``   -- only the ``(`` was lost

Together those account for 5 of the 6 substitutions and 1 of the 23
deletions measured. Repairing them is a *rewrite*, not a search: a
cyclic peptide SMILES is dominated by amide carbonyls, so ``-O)`` and
``=O)`` are high-precision anchors for exactly the characters that
went missing.

Why this beats the A* search
----------------------------
Inserting 15 brackets into a 244-character string is C(244, 15), about
10**23 candidate placements. No state budget makes that tractable, and
the search measured on real rows confirms it: even at 15 s and 1.5 M
states it returns nothing. A rewrite that keys on a two-character
anchor is O(n) and has no combinatorial blow-up at all.

What it cannot do
-----------------
Brackets that were lost where no ``O)`` anchor survives. Those stay
unrepaired, and the remaining mass gap is reported honestly so the
caller can tell how much is left rather than assuming success.

Public API
----------
* :func:`rewrite_oxo`     -- the anchor rewrite, returns (text, n_fixes)
* :func:`repair_by_rewrite` -- rewrite plus mass/formula diagnostics
"""

from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


_OXO = re.compile(r"(.)[-=](O\))")


def rewrite_oxo(text: str) -> tuple:
    """Repair ``-O)`` / ``=O)`` back into ``(=O)``.

    Returns ``(new_text, n_fixes)``. ``n_fixes`` counts the rewrites so a
    caller can report how much was recovered instead of assuming the row
    came back whole.
    """
    count = 0

    def sub(m: "re.Match") -> str:
        # If the captured char is ``(`` the carbonyl still has its opener:
        # the ``=O)`` substring is the *inside* of ``C(=O)``, not a
        # damaged carbonyl. Leaving it alone keeps intact amides intact.
        if m.group(1) == "(":
            return m.group(0)
        nonlocal count
        count += 1
        return f"{m.group(1)}(=O)"

    return _OXO.sub(sub, text), count


def repair_by_rewrite(
    text: str,
    *,
    normalise: bool = True,
) -> dict:
    """Rewrite-damaged carbonyls, then report what RDKit makes of it.

    ``normalise`` first applies the lexical fixes from
    :mod:`cpd.ocr_edit_model` (the ``Ca`` -> ``[C@@H]`` misread, which is a
    separate measured defect and is safe to fix before parsing).

    Returns a dict with ``smiles`` (canonical or ``None``), ``formula``,
    ``exact_mass``, ``largest_ring``, ``num_rings``, ``n_fixes`` and the
    rewritten ``text``.
    """
    working = text
    notes: dict = {}
    if normalise:
        from cpd.ocr_edit_model import normalise_lexical
        working, notes = normalise_lexical(text)

    rewritten, n_fixes = rewrite_oxo(working)

    mol = Chem.MolFromSmiles(rewritten)
    if mol is None:
        return {
            "smiles": None,
            "formula": None,
            "exact_mass": None,
            "largest_ring": 0,
            "num_rings": 0,
            "n_fixes": n_fixes,
            "text": rewritten,
            "parses": False,
            "lexical_notes": notes,
        }

    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        "smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        "largest_ring": max(rings) if rings else 0,
        "num_rings": len(rings),
        "n_fixes": n_fixes,
        "text": rewritten,
        "parses": True,
        "lexical_notes": notes,
    }


__all__ = ["repair_by_rewrite", "rewrite_oxo"]
