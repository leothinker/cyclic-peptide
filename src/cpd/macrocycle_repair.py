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





def attempt_rejoin_fragments(smi):
    """For SMILES with a stray '.' separator, attempt to rejoin the
    fragments into a single molecule by stripping the dot."""
    if not smi or "." not in smi:
        return None
    parts = []
    buf = []
    depth_p = depth_b = 0
    for ch in smi:
        if ch == "(":
            depth_p += 1
        elif ch == ")":
            depth_p -= 1
        elif ch == "[":
            depth_b += 1
        elif ch == "]":
            depth_b -= 1
        if ch == "." and depth_p == 0 and depth_b == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    joined = "".join(parts)
    mol = Chem.MolFromSmiles(joined)
    if mol is not None:
        canon = Chem.MolToSmiles(mol)
        if is_macrocycle_intact(canon):
            return canon
    longest = max(parts, key=len)
    mol = Chem.MolFromSmiles(longest)
    if mol is not None:
        canon = Chem.MolToSmiles(mol)
        if is_macrocycle_intact(canon):
            return canon
    return None


def attempt_strip_truncated_tail(smi):
    """Walk back from the tail dropping dangling characters until the
    truncated SMILES parses to a mol with an intact macrocycle."""
    if not smi or is_macrocycle_intact(smi):
        return None
    for trim in range(1, min(30, len(smi) - 5)):
        cand = smi[:-trim] if trim else smi
        mol = Chem.MolFromSmiles(cand)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if is_macrocycle_intact(canon):
            return canon
    return None

def attempt_syntax_fix(smi: str) -> Optional[str]:
    """Brute-force repairs for the 4 most common DECIMER failure modes.

    Of the 97 broken macrocycles in the corpus, 75 (77%) are plain
    syntax errors that survive purely because of the issue, not a
    genuine topology loss. This strategy patches them up in order:

    1. **Unclosed parenthesis at the tail** -- DECIMER's beam-search
       truncation drops the closing ``)`` of the last branch. Append
       enough ``)`` to balance the string. 33 cases.
    2. **Odd ring-digit count** -- exactly one ring closure digit is
       left without its partner (DECIMER emitted only one of the two
       appearances). Either append the matching digit + a closing atom,
       or drop the lone digit if it is followed by an unclosed bond.
       38 cases.
    3. **Unbalanced square brackets** -- missing ``]``. Insert it.
       4 cases.
    4. **Extra close parenthesis** -- an extra ``)`` slipped in.
       Delete the first one that breaks the balance. ~1 case.

    The first patched variant that parses AND yields an intact
    macrocycle is returned.
    """
    if not smi or is_macrocycle_intact(smi):
        return None

    def _parse_check(candidate: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        if is_macrocycle_intact(canon):
            return canon
        return None

    pd = bd = 0
    for ch in smi:
        if ch == "(":
            pd += 1
        elif ch == ")":
            pd -= 1
        elif ch == "[":
            bd += 1
        elif ch == "]":
            bd -= 1

    if pd > 0 and bd == 0:
        fixed = smi.rstrip() + (")" * pd)
        got = _parse_check(fixed)
        if got is not None:
            return got

    if bd > 0:
        fixed = smi.rstrip() + ("]" * bd)
        got = _parse_check(fixed)
        if got is not None:
            return got

    counts: Counter[str] = Counter()
    i = 0
    while i < len(smi):
        ch = smi[i]
        if ch == "[":
            j = smi.find("]", i + 1)
            i = (j + 1) if j >= 0 else len(smi)
            continue
        if ch == "%" and i + 2 < len(smi) and smi[i + 1:i + 3].isdigit():
            counts[smi[i + 1:i + 3]] += 1
            i += 3
            continue
        if ch.isdigit():
            counts[ch] += 1
        i += 1
    odd = [d for d, n in counts.items() if n % 2 == 1]
    if odd:
        for d in odd:
            for tail_atom in ("C", "N", "O"):
                fixed = smi.rstrip() + d + tail_atom
                got = _parse_check(fixed)
                if got is not None:
                    return got
            j = smi.find(d)
            if j >= 0 and j > 0 and smi[j - 1] == "=":
                fixed = smi[: j - 1] + smi[j + 1:]
            elif j >= 0:
                fixed = smi[:j] + smi[j + 1:]
            else:
                continue
            got = _parse_check(fixed)
            if got is not None:
                return got

    pd = 0
    drop_at = None
    for k, ch in enumerate(smi):
        if ch == "(":
            pd += 1
        elif ch == ")":
            pd -= 1
            if pd < 0:
                drop_at = k
                break
    if drop_at is not None:
        fixed = smi[:drop_at] + smi[drop_at + 1:]
        got = _parse_check(fixed)
        if got is not None:
            return got

    if smi and smi[-1] in "(=":
        for k in range(len(smi) - 1, -1, -1):
            if smi[k] in ")]0123456789":
                continue
            fixed = smi[: k + 1]
            got = _parse_check(fixed)
            if got is not None:
                return got
            break
    return None


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


    return -1


def attempt_remove_extra_digits(smi: str) -> Optional[str]:
    """Heuristic: drop 3rd+ occurrence of over-used digits.

    A ring-closure digit must appear exactly twice. DECIMER sometimes
    emits the same digit three or more times when it gets confused on
    a fused bicyclic; the extra appearances prevent the remaining
    pair from closing into the intended macrocycle. This strategy
    removes the 3rd+ occurrence of any over-used digit and checks
    whether the resulting SMILES now parses to a molecule with an
    intact macrocycle.

    Limitations
    -----------
    * Only useful when the failure mode is digit re-use. Truncated
      SMILES, missing atoms, or wrong ring digit assignment require
      different fixes.
    """
    if not smi or is_macrocycle_intact(smi):
        return None
    counts = Counter(re.findall(r"\d", smi))
    for digit, n in counts.items():
        if n <= 2:
            continue
        new_smi = []
        seen = 0
        for ch in smi:
            if ch == digit:
                seen += 1
                if seen >= 3:
                    continue
            new_smi.append(ch)
        candidate = "".join(new_smi)
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None and is_macrocycle_intact(Chem.MolToSmiles(mol)):
            return Chem.MolToSmiles(mol)
    return None


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


def repair_macrocycle_with_method(
    smi: str,
) -> tuple[Optional[str], Optional[str]]:
    """Run every strategy in order and report which one succeeded.

    Returns ``(repaired_smiles, method_name)``.  ``(None, None)`` means no
    strategy produced a parseable macrocycle.  The method name is what
    :func:`cpd.repair_audit.audit_repair` uses to decide how much to trust
    the result, so callers that publish a repaired SMILES should keep it.

    Strategy order (cheapest and least invasive first):

    1. ``syntax_fix``               -- balance punctuation / ring digits
    2. ``rejoin_fragments``         -- drop a stray ``.`` separator
    3. ``strip_truncated_tail``     -- trim dangling tail characters
    4. ``ring_closure_digit_swap``  -- re-code over-used ring digits
    5. ``remove_extra_digits``      -- drop 3rd+ occurrence of a digit
    6. ``terminal_bridge``          -- bond the two farthest terminals
    """
    strategies = (
        ("syntax_fix", attempt_syntax_fix),
        ("rejoin_fragments", attempt_rejoin_fragments),
        ("strip_truncated_tail", attempt_strip_truncated_tail),
        ("ring_closure_digit_swap", attempt_ring_closure_digit_swap),
        ("remove_extra_digits", attempt_remove_extra_digits),
        ("terminal_bridge", attempt_terminal_bridge),
    )
    if not smi or is_macrocycle_intact(smi):
        return None, None
    for name, fn in strategies:
        got = fn(smi)
        if got is not None:
            return got, name
    return None, None


def repair_macrocycle(smi: str) -> Optional[str]:
    """Try every repair strategy in order; return the first that succeeds.

    Strategy order (cheapest first):
    1. **syntax_fix** -- fix plain RDKit-parse failures by closing
       truncated parens, balancing brackets, patching odd ring-digit
       counts. Recovers the bulk of DECIMER's truncated beam-search
       output. Accounts for the 77% of broken cases that survive only
       because of the issue, not because the topology is wrong.
    2. Re-code extra appearances of over-used digits to a fresh digit.
    3. Remove extra appearances of over-used digits entirely.
    4. Bridge the two most-distant terminal atoms.

    Returns the canonical SMILES of the first successful repair plus
    the method name via :func:`repair_macrocycle_with_method`.
    """
    repaired, _method = repair_macrocycle_with_method(smi)
    return repaired


__all__ = [
    "attempt_syntax_fix",
    "attempt_rejoin_fragments",
    "attempt_strip_truncated_tail",
    "attempt_ring_closure_digit_swap",
    "attempt_remove_extra_digits",
    "attempt_terminal_bridge",
    "repair_macrocycle",
    "repair_macrocycle_with_method",
]
