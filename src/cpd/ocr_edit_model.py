"""What the OCR actually does to a printed SMILES.

This module encodes the edit model measured in
``analysis/align_ocr_to_reference.py``, which diffs cmpd 1036's OCR text
against its hand transcription. That alignment is unusually informative
because 1036 is the one row whose answer is not in doubt.

What it found
-------------
On a 271-character reference, OCR produced 244 characters through
**23 deletions, 0 insertions and 6 substitutions**:

=========================  ====  =========================================
deletion                   count  note
=========================  ====  =========================================
``(``                        16  69% of all deletions
``)``                         5
``=``                         1
``[``                         1
=========================  ====  =========================================

=========================  ====  =========================================
substitution               count  reference -> OCR
=========================  ====  =========================================
``(=``  -> ``-``              4  the bracket merges with the bond symbol
``=``   -> ``-``              1  a lone bond symbol also reads as a hyphen
``@``   -> ``a``              1  the chiral marker reads as a small a
=========================  ====  =========================================

Three consequences drive the whole pipeline.

1. **Zero insertions.** OCR never invents a character here, so a repair that
   only *inserts* is not a heuristic -- it is the complete edit set for this
   row. That is why this module exposes an insertion alphabet rather than a
   general edit menu.

2. **``(`` next to ``=`` becomes ``-``.** This single mechanism explains the
   dominant ``C(=O)`` -> ``C-O)`` damage. It also means a bare ``-`` in the
   OCR text is ambiguous: it may be a real single bond, a lost ``=``, or a
   lost ``(=``. The repair has to keep all three options alive.

3. **Brackets do not change the atom inventory.** ``C(=O)N`` and ``C-O)N``
   carry identical heavy atoms, so the heavy-atom mass is invariant under
   the damage. That makes the patent's own m/z usable as a gate *before*
   the connectivity is known -- see :mod:`cpd.ocr_mass_gate`.

Public API
----------
* :data:`INSERT_ALPHABET` -- characters a repair may put back
* :data:`HYBRID_PAIRS`    -- ``(``+``=`` collapsed into ``-``, and friends
* :func:`normalise_lexical` -- deterministic pre-pass (Ca, a@H, hybrid bonds)
* :func:`count_insertions` -- expected insertion count from the text alone
"""

from __future__ import annotations

import re
from collections import Counter

# Characters an insertion-only repair is allowed to put back. Derived from
# the 1036 alignment; every one of these was observed as a deletion.
INSERT_ALPHABET = "()[]="

# Prior probability of each insertion, used to rank beam candidates.
# Brackets dominate the observed damage, so they are cheap; a lost "=" or
# "[" costs more, which biases the beam toward the common explanations.
INSERT_COST = {
    "(": 0.06,
    ")": 0.06,
    "[": 0.25,
    "]": 0.25,
    "=": 0.20,
}

# Two-character groups the recogniser collapses into a single glyph.
# "(=" renders as "-", so a hyphen may stand for either character, or for
# both of them together.
HYBRID_PAIRS = {
    "-": ("=", "(=", ""),   # real single bond / lost = / lost (= / nothing
    "a": ("@", "", "a"),
}

# OCR reads the chiral marker "@" as a lowercase "a" often enough that the
# pattern "a@H" and "a@@H" is unambiguous in context.
CHIRAL_FIXES = (
    (re.compile(r"\[Ca(@@?)H\]"), r"[C\1H]"),
    (re.compile(r"\bCa(@@?)H\]"), r"[C\1H]"),
    (re.compile(r"\[C(\s*)a@H\]"), r"[C@@H]"),
    (re.compile(r"\[C(\s*)a@@H\]"), r"[C@@H]"),
)

# "Ca" appears only as a misread of "[C" plus a chiral marker. A real
# calcium in this corpus would not be followed by "@H]".
CA_CHIRAL = re.compile(r"Ca(@@?)H\]")


def count_insertions(text: str) -> dict:
    """How many of each character a repair must insert to rebalance ``text``.

    Uses bracket balance only, so it is a *lower bound* on the true edit
    count: it cannot see the ``(`` lost together with a ``=`` into a single
    ``-``, nor a ``[`` whose loss left no trace. It is still a useful budget
    signal, because an under-estimate is far less harmful than a wild
    over-estimate in a beam search.
    """
    paren = 0
    bracket = 0
    for ch in text:
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
    return {
        "(": max(0, -paren),      # unmatched ")" need a "(" before them
        ")": max(0, paren),
        "[": max(0, -bracket),
        "]": max(0, bracket),
        "=": 0,                   # invisible when merged with "(" into "-"
        "min_total": max(0, -paren) + max(0, paren) + max(0, -bracket)
                      + max(0, bracket),
    }


def hybrid_sites(text: str) -> dict:
    """Count the collapsed-glyph sites that a plain bracket count misses.

    Every ``-`` in OCR text is a candidate for "this was ``=``" or "this was
    ``(=``", and every ``a`` immediately before ``@H`` is a candidate for a
    misread chiral marker.
    """
    return {
        "hyphen": text.count("-"),
        "a_at_h": len(re.findall(r"a@H", text)),
        "ca_chiral": len(CA_CHIRAL.findall(text)),
    }


def normalise_lexical(text: str) -> tuple:
    """Deterministic pre-pass. Returns ``(new_text, notes)``.

    These are the edits that carry no ambiguity at all: a ``Ca`` immediately
    followed by ``@H]`` can only be a misread ``[C@@H]``, because a real
    calcium would not be followed by a hydrogen count and a bracket close.
    Everything uncertain is left for the search.
    """
    notes: Counter = Counter()
    out = text
    new, n = CA_CHIRAL.subn(r"[C\1H]", out)
    if n:
        notes["Ca_to_C_chiral"] += n
        out = new
    # A stray "[Ca@H]" where the "[" survived the merge.
    new, n = re.subn(r"\[Ca(@@?)H\]", r"[C\1H]", out)
    if n:
        notes["bracket_Ca_to_chiral"] += n
        out = new
    return out, dict(notes)


def edit_model_report(text: str) -> dict:
    """Everything the search needs to know about one OCR string, cheaply."""
    ins = count_insertions(text)
    hyb = hybrid_sites(text)
    return {
        "length": len(text),
        "insertions": ins,
        "hybrid": hyb,
        # A hyphen that sits right before ")" is almost always a lost "(=".
        "hyphen_before_close": len(re.findall(r"-\)", text)),
        "bare_equals_close": len(re.findall(r"=[ONS]\)", text)),
    }


__all__ = [
    "CA_CHIRAL",
    "INSERT_ALPHABET",
    "INSERT_COST",
    "HYBRID_PAIRS",
    "count_insertions",
    "edit_model_report",
    "hybrid_sites",
    "normalise_lexical",
]
