"""Incremental, cloneable SMILES prefix validator.

Why this exists
---------------
``grammar.py`` answers "is this whole string well formed?" in one shot.
That is useless for search: a beam that repairs a 250-character OCR string
has to reject a bad prefix *the moment* it becomes bad, not after building
the rest of the string. Re-validating whole prefixes is O(n) per edit, which
turns an incremental search into an O(n^2) one and makes the repair
intractable in Python.

``PrefixValidator`` carries just enough state to advance one character at a
time in O(1) and to be copied cheaply, so a beam can expand millions of
prefixes in seconds.

What it enforces
----------------
Token-level well-formedness only -- the same scope as ``grammar.py``, but
incremental:

* an atom may not directly follow another atom;
* a bond symbol may not follow another bond, ``(`` or a dot;
* ``(`` may only follow an atom, a branch close or a ring-closure digit;
* every ring-closure number must be opened before it is closed;
* bracket atoms must be terminated and must contain a legal symbol.

It does **not** check valence, aromaticity or stereochemistry -- that is
RDKit's job at the end of the beam, where a full parse is affordable.

Public API
----------
* :class:`PrefixValidator` -- cloneable walker with :meth:`advance`
* :func:`extend_ok`         -- can ``prefix`` accept ``char``?
* :func:`first_violation`   -- index of the first offending character
"""

from __future__ import annotations

from typing import Optional

# Token kinds tracked as ``prev``.
PREV_NONE = 0
PREV_ATOM = 1
PREV_BOND = 2
PREV_BRANCH_OPEN = 3
PREV_BRANCH_CLOSE = 4
PREV_RING = 5
PREV_DOT = 6
PREV_BRACKET = 7

# Bare organic-subset symbols; lowercase forms are aromatic.
ORGANIC = frozenset("BCNOPSFI*bcnops")

# Leading characters of the two-letter halogens Cl and Br.
TWO_LETTER_HEAD = frozenset("C B".replace(" ", ""))

# Inside a bracket atom: what are we waiting for?
_B_SYMBOL = 0        # reading the element symbol (may be two letters)
_B_AFTER_SYMBOL = 1  # symbol complete; @ / H / charge / ] may follow
_B_CHIRAL_TAIL = 2   # one "@" consumed, a second or the class may follow
_B_CHIRAL_CLASS = 3  # first class letter of @TH / @AL / @SP / @TB / @OH
_B_CHIRAL_NUM = 4    # second class letter; a number may follow
_B_AFTER_H = 5       # H or Hn consumed
_B_CHARGE = 6
_B_CLASS = 7

# Element symbols legal inside a bracket atom. A symbol is complete as soon
# as it is a prefix of no longer element name, so "F" alone is finished but
# "Fe" needs the 'e' before we can accept the ']'.
ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg
Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D T
""".split())

# Aromatic symbols are legal inside brackets ("[nH]", "[se]", "[c-]").
# The only multi-letter aromatic symbol is selenium.
AROMATIC_BRACKET = frozenset("cnosbpcasete")


class PrefixValidator:
    """A cloneable SMILES prefix walker.

    :meth:`advance` returns a *new* validator and leaves the receiver
    untouched, so a beam can hold one validator per candidate without any
    copy bookkeeping. It returns ``None`` the moment a character would make
    the prefix ungrammatical -- that is the pruning signal the search needs.
    """

    __slots__ = ("prev", "depth", "brk", "rings", "pct", "trail_cl", "sym")

    def __init__(
        self,
        prev: int = PREV_NONE,
        depth: int = 0,
        brk: int = _B_SYMBOL,
        rings: Optional[dict] = None,
        pct: int = 0,
        trail_cl: str = "",
        sym: str = "",
    ) -> None:
        self.prev = prev
        self.depth = depth
        self.brk = brk
        self.rings = dict(rings) if rings else {}
        self.pct = pct
        self.trail_cl = trail_cl
        self.sym = sym

    def copy(self) -> "PrefixValidator":
        return PrefixValidator(self.prev, self.depth, self.brk, self.rings,
                               self.pct, self.trail_cl, self.sym)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PrefixValidator(prev={self.prev}, depth={self.depth}, "
                f"open_rings={sorted(self.rings)})")

    # -- context predicates ---------------------------------------------
    def can_atom(self) -> bool:
        return self.prev in (PREV_NONE, PREV_ATOM, PREV_BOND,
                             PREV_BRANCH_OPEN, PREV_BRANCH_CLOSE,
                             PREV_DOT, PREV_RING)

    def can_bond(self) -> bool:
        # "C(=O)" is legal: OpenSMILES allows an optional bond as the first
        # token inside a branch. This is extremely common in peptides, so
        # forbidding it would reject almost every real structure here.
        return self.prev in (PREV_ATOM, PREV_BRANCH_OPEN, PREV_BRANCH_CLOSE,
                             PREV_RING)

    def can_ring(self) -> bool:
        # Ring numbers may chain: "C12CC1CC2" opens 1 and then 2 back to back.
        return self.prev in (PREV_ATOM, PREV_BRANCH_CLOSE, PREV_RING)

    def can_branch(self) -> bool:
        return self.prev in (PREV_ATOM, PREV_BRANCH_CLOSE, PREV_RING)

    # -- the transition function ----------------------------------------
    def advance(self, ch: str) -> Optional["PrefixValidator"]:
        """Return the validator after reading ``ch``, or ``None`` if illegal."""
        if self.prev == PREV_BRACKET:
            new = self.copy()
            return new if _advance_bracket(new, ch) else None

        if ch == "[":
            if not self.can_atom():
                return None
            new = self.copy()
            new.prev = PREV_BRACKET
            new.brk = _B_SYMBOL
            return new

        if ch == "]":
            return None

        if ch == "(":
            if not self.can_branch():
                return None
            new = self.copy()
            new.depth += 1
            new.prev = PREV_BRANCH_OPEN
            new.trail_cl = ""
            return new

        if ch == ")":
            # A branch close must follow an atom or a nested branch, never a
            # bond symbol -- otherwise "C(=)C" would be accepted.
            if self.depth <= 0 or self.prev not in (PREV_ATOM, PREV_BRANCH_CLOSE,
                                                    PREV_RING):
                return None
            new = self.copy()
            new.depth -= 1
            new.prev = PREV_BRANCH_CLOSE
            new.trail_cl = ""
            return new

        if ch in "-=#$:/\\~":
            if not self.can_bond():
                return None
            new = self.copy()
            new.prev = PREV_BOND
            new.trail_cl = ""
            return new

        if ch == ".":
            if self.prev not in (PREV_ATOM, PREV_BRANCH_CLOSE, PREV_RING):
                return None
            new = self.copy()
            new.prev = PREV_DOT
            new.trail_cl = ""
            return new

        if ch == "%":
            if not self.can_ring() or self.pct:
                return None
            new = self.copy()
            new.pct = 1
            new.prev = PREV_RING
            return new

        if ch.isdigit():
            new = self.copy()
            if self.pct == 1:
                new.pct = 2        # first digit seen, one more to go
                new.prev = PREV_RING
                return new
            if self.pct == 2:
                # %10 .. %99 all share one bucket. We only care that a
                # two-digit closure is opened as often as it is closed.
                new.pct = 0
                if "%" in new.rings:
                    left = new.rings["%"] - 1
                    if left:
                        new.rings["%"] = left
                    else:
                        del new.rings["%"]
                else:
                    new.rings["%"] = 1
                new.prev = PREV_RING
                return new
            if self.pct:
                return None
            if not self.can_ring():
                return None
            d = ch
            if d in new.rings:
                left = new.rings[d] - 1
                if left:
                    new.rings[d] = left
                else:
                    del new.rings[d]
            else:
                new.rings[d] = new.rings.get(d, 0) + 1
            new.prev = PREV_RING
            return new

        # An atom: a single symbol, or the second half of Cl / Br.
        two_letter = self.trail_cl in ("C", "B") and ch in ("l", "r")
        if ch in ORGANIC or two_letter:
            if not self.can_atom():
                return None
            new = self.copy()
            new.prev = PREV_ATOM
            new.trail_cl = ch if ch in ("C", "B") else ""
            return new
        return None

    # -- completion ------------------------------------------------------
    def is_complete(self) -> bool:
        """True iff the string consumed so far is a finished SMILES."""
        if self.prev in (PREV_NONE, PREV_BOND, PREV_BRANCH_OPEN, PREV_BRACKET):
            return False
        return self.depth == 0 and not self.rings

    def open_rings(self) -> tuple:
        return tuple(sorted(self.rings))


def _advance_bracket(v: PrefixValidator, ch: str) -> bool:
    """Advance the mini-grammar inside ``[...]``. Mutates ``v`` in place.

    The element symbol is accumulated in ``v.sym`` rather than guessed from a
    single character, because most symbols are two letters and a symbol may
    only be closed once it is no longer a prefix of a longer element name:
    ``F`` is complete, ``Fe`` needs its ``e`` before ``]`` is legal.

    ``]`` always terminates the bracket in the same step, so ``v.prev`` can
    never be left stuck on :data:`PREV_BRACKET`.
    """
    state = v.brk

    if ch == "]":
        # "]" is not a letter, so it always terminates the element symbol.
        # "[13C]" is therefore fine; "[]" and "[Q]" are not.
        if state == _B_SYMBOL:
            if v.sym not in ELEMENTS and v.sym not in AROMATIC_BRACKET:
                return False
        elif state not in (_B_AFTER_SYMBOL, _B_CHIRAL_TAIL, _B_CHIRAL_CLASS,
                           _B_CHIRAL_NUM, _B_AFTER_H, _B_CHARGE, _B_CLASS):
            return False
        return _close_bracket(v)

    if state == _B_SYMBOL:
        if ch.isdigit():
            return True                       # isotope number
        if ch == "*":
            v.sym = "*"
            v.brk = _B_AFTER_SYMBOL
            return True
        if ch.isalpha():
            grown = v.sym + ch
            if grown in ELEMENTS or grown in AROMATIC_BRACKET:
                v.sym = grown
                # Greedy: hold the symbol open, since a following letter
                # could still extend it ("C" -> "Ca").
                v.brk = _B_SYMBOL
                return True
            if v.sym not in ELEMENTS and v.sym not in AROMATIC_BRACKET:
                return False
            # The symbol was already complete and this letter cannot extend
            # it: "[NH4+]" is nitrogen bearing four hydrogens, not "Nh".
            v.brk = _B_AFTER_SYMBOL
            return _advance_bracket(v, ch)
        # Not a letter: the symbol is over, and this character has to be
        # re-read as the next thing in the bracket ("[C@H]" -> "@").
        if v.sym not in ELEMENTS and v.sym not in AROMATIC_BRACKET:
            return False
        v.brk = _B_AFTER_SYMBOL
        return _advance_bracket(v, ch)

    if state == _B_AFTER_SYMBOL:
        # Chiral class: @, @@, @TH1, @AL2, @SP3, @TB15, @OH12 ...
        if ch == "@":
            v.brk = _B_CHIRAL_TAIL
            return True
        if ch.isdigit():
            return True                       # leftover chiral class number
        if ch in ("H", "h"):
            v.brk = _B_AFTER_H
            return True
        if ch in "+-":
            v.brk = _B_CHARGE
            return True
        if ch == ":":
            v.brk = _B_CLASS
            return True
        return False

    if state == _B_CHIRAL_TAIL:
        # A second "@" makes it "@@". Then @TH1 / @AL2 / @SP3 / @TB15 / @OH12:
        # two letters then an optional number.
        if ch == "@":
            v.brk = _B_CHIRAL_CLASS
            return True
        if ch.isalpha():
            v.brk = _B_CHIRAL_CLASS
            return True
        return False

    if state == _B_CHIRAL_CLASS:
        # Second letter of the class name (@TH, @AL, @SP, @TB, @OH), then
        # the class number, then the bracket may close.
        if ch.isalpha():
            v.brk = _B_CHIRAL_NUM
            return True
        if ch.isdigit():
            v.brk = _B_AFTER_SYMBOL
            return True
        return False

    if state == _B_CHIRAL_NUM:
        if ch.isdigit():
            v.brk = _B_AFTER_SYMBOL
            return True
        return False

    if state == _B_AFTER_H:
        if ch.isdigit():
            return True                       # H2, H3 ...
        if ch in "+-":
            v.brk = _B_CHARGE
            return True
        if ch == ":":
            v.brk = _B_CLASS
            return True
        return False

    if state == _B_CHARGE:
        if ch.isdigit() or ch in "+-":
            return True
        if ch == ":":
            v.brk = _B_CLASS
            return True
        return False

    if state == _B_CLASS:
        if ch.isdigit():
            return True
        return False

    return False


def _close_bracket(v: PrefixValidator) -> bool:
    """Finish a bracket atom, leaving the walker in an ``atom`` context."""
    v.prev = PREV_ATOM
    v.brk = _B_SYMBOL
    v.trail_cl = ""
    v.sym = ""
    return True


def _has_longer_element(symbol: str) -> bool:
    """True iff ``symbol`` could still grow into a longer element name."""
    n = len(symbol)
    return any(len(e) > n and e.startswith(symbol) for e in ELEMENTS)


def extend_ok(prefix: str, ch: str) -> bool:
    """True iff ``prefix + ch`` is still a grammatical SMILES prefix."""
    v = PrefixValidator()
    for c in prefix:
        v = v.advance(c)
        if v is None:
            return False
    return v.advance(ch) is not None


def first_violation(smiles: str) -> Optional[int]:
    """Index of the first character that breaks the grammar, or ``None``.

    Returns ``len(smiles)`` when the string is merely unfinished (unclosed
    branch or ring) rather than wrong, which lets a caller tell "damaged
    here" apart from "just incomplete".
    """
    v = PrefixValidator()
    for i, ch in enumerate(smiles):
        nxt = v.advance(ch)
        if nxt is None:
            return i
        v = nxt
    return None if v.is_complete() else len(smiles)


__all__ = ["PrefixValidator", "extend_ok", "first_violation"]
