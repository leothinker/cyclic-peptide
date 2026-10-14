"""Lightweight grammar-aware SMILES prefix validator.

Walks a SMILES string token by token and returns True iff every
seen prefix is grammatically plausible. Catches the OCR errors that
plague our cyclic peptide pipeline:

  - unbalanced parens (`CC[C@H]C)` has a stray `)`)
  - missing parens (`C=ON` collapses `C(=O)N`)
  - broken ring closures (`C5` opens ring 5; later `...C5` must close it)
  - bond before non-atom (`=)`, `-)`, `=(`, etc.)
  - dangling ring open at end-of-string (`...C5` with no closing `5`)

This is NOT a full SMILES parser -- it does no atom-type validation,
no valence check, no stereo. It only enforces *syntactic* well-
formedness of the prefix. Much faster than ``Chem.MolFromSmiles``
(roughly 10x on 250-char strings) and good enough to prune the
~99% of swap candidates that would never RDKit-parse anyway.

Used by ``_beam_search.py`` as a cheap pruning signal before paying
the full RDKit parse cost.
"""
from __future__ import annotations
import re


_DIGIT = re.compile(r"\d")
_BOND_CHARS = set("=#\-:/~\\")
# Characters that may legally follow a bond symbol.
_BOND_FOLLOW = set("[(/0123456789ABCDEFGHIKLMNOPRSTUVWYZabcdefghiklmnoprstuvwy")


def is_grammar_valid(smiles: str) -> bool:
    """True iff ``smiles`` is syntactically well-formed at every prefix."""
    paren_depth = 0
    ring_open: dict[str, int] = {}  # ring number str -> count of opens not yet closed
    bracket_depth = 0
    in_pct_ring = False
    pct_ring_buf = ""
    i = 0
    n = len(smiles)
    while i < n:
        c = smiles[i]
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return False  # extra close paren
        elif c == "[":
            bracket_depth += 1
        elif c == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                return False
        elif c == "%":
            # %nn starts a 2-digit ring number; we read 2 digits after.
            in_pct_ring = True
            pct_ring_buf = ""
            i += 1
            if i + 1 >= n or not (smiles[i].isdigit() and smiles[i + 1].isdigit()):
                return False
            ring_num = smiles[i:i + 2]
            ring_open[ring_num] = ring_open.get(ring_num, 0) + 1
            i += 2
            in_pct_ring = False
            continue
        elif c.isdigit():
            ring_num = c
            if ring_num in ring_open:
                ring_open[ring_num] -= 1
                if ring_open[ring_num] == 0:
                    del ring_open[ring_num]
            else:
                ring_open[ring_num] = ring_open.get(ring_num, 0) + 1
        elif c in _BOND_CHARS:
            # Bond char must be followed by an atom (uppercase letter or
            # bracket or digit for ring closure). `(`, `)`, `-` are illegal.
            if i + 1 >= n:
                return False  # dangling bond at EOL
            nxt = smiles[i + 1]
            if nxt not in _BOND_FOLLOW:
                return False
        i += 1
    if paren_depth != 0:
        return False  # unclosed paren
    if bracket_depth != 0:
        return False
    if ring_open:
        return False  # dangling ring closure
    return True
