"""Relative costs for the OCR repair search.

The beam in :mod:`cpd.mass_constrained_search` ranks candidate repairs by
what it had to invent. These costs are *relative*, not probabilities: they
only have to order candidates consistently, so they are expressed as small
penalties that sum along a path.

They are derived from the measured edit model
(``analysis/align_ocr_to_reference.py``), which found that ``(`` accounts for
69% of all deleted characters and that a lost ``(`` sitting next to a ``=``
is usually rendered as a single ``-``. So:

* inserting ``(`` right after a ``=`` is nearly free -- it completes a
  carbonyl, by far the most common bracket in a peptide;
* reading a ``-`` as ``(=`` is cheap, because that is the merge we measured;
* an arbitrary bracket in an arbitrary position is expensive, because we
  have no evidence for it.

Keeping this in its own module lets the numbers be unit-tested directly
instead of being buried inside a search loop.
"""

from __future__ import annotations

# Substitution costs: turning what OCR read into what was actually printed.
HYPHEN_COST = {"-": 0.45, "=": 0.30, "(=": 0.15}
A_COST = {"a": 0.45, "@": 0.30}
GENERIC_SUB = 0.40

# Insertion costs.
COST_CARBONYL_OPEN = 0.02    # "(" directly after "=" -> completes C(=O)
COST_PLAIN_OPEN = 0.35
COST_PLAIN_CLOSE = 0.30
COST_CHIRAL_BRACKET = 0.08   # "[" right before a chiral atom, i.e. [C@@H]
COST_PLAIN_BRACKET = 0.20
COST_BOND = 0.25

# Dropping a character is almost never right (the measured model had zero
# deletions by OCR), so it must be cheaper to insert than to delete.
COST_DELETE = 0.90


def expansion_cost(src: str, chosen: str) -> float:
    """Cost of emitting ``chosen`` in place of the read ``src``."""
    if chosen == src:
        return 0.0
    if src == "-":
        return HYPHEN_COST.get(chosen, GENERIC_SUB)
    if src == "a":
        return A_COST.get(chosen, GENERIC_SUB)
    return GENERIC_SUB


def expansions_for(ch: str) -> tuple:
    """Ways to emit ``ch``: as read, or as one of the documented misreads."""
    if ch == "-":
        # "(=" renders as "-" because the recogniser merges the two glyphs,
        # so a hyphen may be a single bond, a lost "=", or a lost "(=".
        return ("-", "=", "(=")
    if ch == "a":
        # The chiral marker "@" is frequently read as a lowercase "a".
        return ("a", "@")
    return (ch,)


def insert_cost(ch: str, emitted: str) -> float:
    """Cost of inserting ``ch`` at the end of the text emitted so far.

    Context matters, so the already-emitted text is consulted: the same
    character is nearly free in one position and expensive in another.
    """
    if ch == "(":
        if len(emitted) >= 2 and emitted[-2] == "=":
            return COST_CARBONYL_OPEN
        return COST_PLAIN_OPEN
    if ch == ")":
        return COST_PLAIN_CLOSE
    if ch == "[":
        # "[C@@H]" -- a bracket immediately before a chiral atom.
        if emitted.endswith("H") or emitted.endswith("@"):
            return COST_CHIRAL_BRACKET
        return COST_PLAIN_BRACKET
    if ch == "]":
        return COST_PLAIN_BRACKET
    return COST_BOND


__all__ = [
    "COST_BOND",
    "COST_CARBONYL_OPEN",
    "COST_CHIRAL_BRACKET",
    "COST_DELETE",
    "COST_PLAIN_BRACKET",
    "COST_PLAIN_CLOSE",
    "COST_PLAIN_OPEN",
    "expansion_cost",
    "expansions_for",
    "insert_cost",
]
