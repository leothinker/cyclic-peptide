"""Graph-repair experiments for DECIMER's broken cyclic-peptide SMILES.

Background
----------
DECIMER-AI was trained mostly on small molecules from PubChem / ChEMBL.
On a fused bicyclic peptide (two rings sharing an edge) it occasionally
emits the same ring-closure digit three or more times in close
proximity, creating small side rings instead of carrying the macrocycle
digit through the full backbone.  After RDKit parses the SMILES, the
largest ring is typically 4-6 atoms (a 5/6 ring) instead of the 25-40
atoms a real cyclic peptide should have.

Diagnostic
----------
:func:`cpd.render.is_macrocycle_intact` returns False for any SMILES
whose largest ring is below :data:`cpd.render.MACROCYCLE_MIN_SIZE` (12).
141 of the 328 DECIMER-valid cmpds in WO2025162428 trip that gate.

Repair attempts (this module)
-----------------------------
The repair is a research-grade heuristic, not a production fix.  The
two strategies below both fail on real-world cases more often than
they succeed, but they are useful as a starting point for the team
that takes the next step.

Strategy 1 -- :func:`attempt_ring_closure_digit_swap` -- for every
over-used ring-closure digit, try the alternative interpretation where
the third+ occurrence is re-coded to a fresh digit.  The hypothesis:
the extra appearances of the same digit are creating the small side
rings; re-coding them to a fresh digit frees the original digit to
serve as the macrocycle closer.

Strategy 2 -- :func:`attempt_terminal_bridge` -- the parsed molecule
is a tree-like assembly of small rings.  Find the two degree-1
atoms (terminal atoms at the ends of branches) that are chemically
farthest apart in the molecular graph, add a single bond between
them, and re-emit the SMILES.  The hypothesis: those two terminals
should be bonded to close the macrocycle.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Optional

from rdkit import Chem

from cpd.render import MACROCYCLE_MIN_SIZE, is_macrocycle_intact


def _pick_fresh_digit(smiles: str) -> str:
    """Return a ring-closure digit that does not appear in ``smiles``."""
    used = set(re.findall(r"\d", smiles))
    for d in "3456789":
        if d not in used:
            return d
    return "9"  # best-effort fallback


def attempt_ring_closure_digit_swap(smi: str) -> Optional[str]:
    """Heuristic: re-code extra appearances of over-used digits.

    For every digit that appears != 2 times in the SMILES, replace
    the third+ occurrence with a fresh digit and check if the new
    SMILES parses to a molecule with an intact macrocycle.  If so,
    return the canonical SMILES; otherwise ``None``.

    Limitations
    -----------
    * Replacing an extra digit with a fresh one creates an extra ring
      rather than removing the side ring the digit was forming.  This
      rarely repairs the macrocycle in practice.
    * Some multi-digit ring closures (%10, %11, ...) use 2 characters
      in the raw string; we only handle single-digit re-coding here.
    """
    if not smi or is_macrocycle_intact(smi):
        return None
    counts = Counter(re.findall(r"\d", smi))
    for digit, n in counts.items():
        if n <= 2:
            continue
        fresh = _pick_fresh_digit(smi)
        new_smi = []
        seen = 0
        for ch in smi:
            if ch == digit:
                seen += 1
                new_smi.append(fresh if seen >= 3 else ch)
            else:
                new_smi.append(ch)
        candidate = "".join(new_smi)
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None and is_macrocycle_intact(Chem.MolToSmiles(mol)):
            return Chem.MolToSmiles(mol)
    return None


def _terminal_atoms(mol: Chem.Mol) -> list[int]:
    """Return atom indices of degree-1 atoms in ``mol``."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1]


def _graph_distance(mol: Chem.Mol, a: int, b: int) -> int:
    """Unweighted shortest-path distance between two atoms. -1 if disconnected."""
    if a == b:
        return 0
    visited = {a}
    q = deque([(a, 0)])
    while q:
        node, d = q.popleft()
        for nb in mol.GetAtomWithIdx(node).GetNeighbors():
            j = nb.GetIdx()
            if j == b:
                return d + 1
            if j not in visited:
                visited.add(j)
                q.append((j, d + 1))
    return -1


def attempt_terminal_bridge(smi: str) -> Optional[str]:
    """Heuristic: connect the two most-distant degree-1 atoms.

    The hypothesis is that the broken DECIMER SMILES describes a
    tree-like assembly of small rings whose branch tips are the two
    ends of the macrocycle.  Bonding them closes the macrocycle.

    Limitations
    -----------
    * Cyclic peptides are NOT always tree-like; sometimes the broken
      DECIMER output is a small fused ring system with no degree-1
      atoms at all.  In that case this strategy is a no-op.
    * Adding a bond arbitrarily to a peptide backbone can create
      unreasonable chemistry.  We only attempt this when the bond
      length is "large enough" to plausibly span a macrocycle.
    """
    if not smi or is_macrocycle_intact(smi):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    terminals = _terminal_atoms(mol)
    if len(terminals) < 2:
        return None
    best: Optional[tuple] = None
    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            d = _graph_distance(mol, terminals[i], terminals[j])
            if d < 0:
                continue
            if best is None or d > best[2]:
                best = (terminals[i], terminals[j], d)
    if best is None or best[2] < MACROCYCLE_MIN_SIZE:
        return None
    a, b, _ = best
    rw = Chem.RWMol(mol)
    rw.AddBond(a, b, Chem.BondType.SINGLE)
    try:
        fixed = rw.GetMol()
        Chem.SanitizeMol(fixed)
    except Exception:
        return None
    canon = Chem.MolToSmiles(fixed)
    if is_macrocycle_intact(canon):
        return canon
    return None


def repair_macrocycle(smi: str) -> Optional[str]:
    """Try every repair strategy in order; return the first that succeeds.

    Returns ``None`` when no strategy repairs the macrocycle.  This
    is the public entry point callers should use.
    """
    if not smi or is_macrocycle_intact(smi):
        return None
    repaired = attempt_ring_closure_digit_swap(smi)
    if repaired is not None:
        return repaired
    return attempt_terminal_bridge(smi)


__all__ = [
    "attempt_ring_closure_digit_swap",
    "attempt_terminal_bridge",
    "repair_macrocycle",
]
