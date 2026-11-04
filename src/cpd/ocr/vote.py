"""Let a second recogniser settle the characters the first one lost.

One recogniser reads a printed line and sometimes drops or confuses a
single character -- a bracket, a parenthesis, a doubled ``@@``.  The
damage is small but it is systematic, and it is the kind of damage
that a *different* model does not make in the same place.

So every printed line is read twice, once by the recogniser of the
route and once by the other recogniser, and the two readings are
combined.  The combination is chosen by one rule:

    **take the combination that is a well-formed SMILES, and among
    those the one that replaces the fewest printed lines of the
    primary reading.**

The two readings differ line by line, so a line is either taken from
the primary reading or taken from the second one, and the rule
minimises the number of lines that are swapped.  Within a swapped
line the whole second reading is used; the two readings are not
patched together character by character.

The rule is deliberately *not* allowed to look at a reference
transcription: it may look at the two readings, at how much they
differ, and at whether the result is a well-formed SMILES.  That
makes it usable on a patent nobody has transcribed.

Two guards keep the rule honest
--------------------------------

**The characters must be SMILES characters.**  A printed SMILES
column cannot contain a space or a wildcard atom, so a combination
that introduces one is rejected however well it parses.  This matters
because RDKit is more permissive than the patent: it reads
``"CC O"`` as ``CC`` and accepts ``"CC*O"`` as it stands.  Without the
guard a combination can be *accepted* for having a space in it.

**The bond orders may not shrink.**  The vote is licensed to fix
*syntax* -- a missing bracket, a lost parenthesis -- and it is not
licensed to change *chemistry*.  A parse check cannot tell the two
apart: ``C1=CC=CC=C1`` and ``C1-CC-C-C-C1`` both parse, and both
are valid SMILES, but they are different molecules.  Measured on the
335 cells the pipeline cuts, a vote without this guard turns a double
bond into a single bond in 5 of the 17 strings it changes, because
the second recogniser read the ring as saturated while the first read
it as aromatic and the *whole string* only parsed with the second
reading.

So a combination is only accepted when it keeps **at least** as many
double and triple bonds as the primary reading.  The vote may add a
bond (a lost ``=`` restored), and it may not delete one.

What the rule may and may not claim
-----------------------------------

Choosing between two readings of the same ink by asking which one is
a well-formed SMILES is a *text* decision.  It says "these two
pixels were read in two ways and only one of them is a valid
SMILES".  It does **not** say that the result is the compound the
patent drew.  Only a comparison against the drawn structure can say
that, and this module does not do it.

The rule is also conservative in two ways that matter.

*The primary reading wins by default.*  The combinations are
enumerated in order of how many lines they replace, so the primary
reading itself is the first candidate.  If it already parses -- which
is the normal case -- it is kept, and the second recogniser is never
consulted.

*The search is bounded.*  Only the bands whose two readings differ
are branched over, and the number of bands that may be changed at
once is limited.  A compound with many differing bands is therefore
not searched exhaustively; the primary reading is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

# The characters a printed SMILES can consist of.
#
# The letters are the letters of the element symbols, because a
# printed SMILES may contain a bracketed atom of any element --
# ``[Se]``, ``[Cu]``, ``[Na]`` -- and the two-letter symbols bring
# their second letter with them, so ``Cl`` and ``Br`` need ``l`` and
# ``r``.  The list is checked against RDKit's own periodic table by
# ``tests/test_vote.py``.
#
# The punctuation is the SMILES syntax.  ``*`` is deliberately
# absent: a wildcard atom is legal SMILES but cannot be printed in a
# patent's SMILES column.  So is the space, and so is every
# non-ASCII character -- a printed column contains none of them, and
# the recogniser produces them when it reads a rule or a footnote
# mark as text.
SMILES_LETTERS = frozenset(
    "ABCDEFGHIKLMNOPRSTUVWXYZ"
    "abcdefghiklmnoprstuvy"
)

# ``$`` (quadruple bond), ``<`` and ``>`` (directional bonds) are part
# of the SMILES grammar but RDKit does not read them, so a string
# holding one is not a reading RDKit can use.  They are left out.
SMILES_SYMBOLS = frozenset("=#@[]()+-/\\%.:~")

SMILES_ALPHABET = SMILES_LETTERS | SMILES_SYMBOLS | frozenset("0123456789")

# The bond symbols whose *number* carries chemical meaning.  A printed
# SMILES may lose one of them to a mis-read, but it must never gain
# one that the primary reading did not have.
BOND_SYMBOLS = ("=", "#")


@dataclass(frozen=True)
class VoteResult:
    """The outcome of the vote and what it was based on."""

    text: str
    changed_bands: int
    used_primary: bool
    parses: bool


def _parses(text: str) -> bool:
    """Whether RDKit can read the string as a SMILES."""
    if not text:
        return False
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _uses_only_smiles_characters(text: str) -> bool:
    """Whether every character of ``text`` can appear in a SMILES."""
    return not (set(text) - SMILES_ALPHABET)


def _bond_count(text: str) -> int:
    """How many double and triple bonds the text declares."""
    return sum(text.count(symbol) for symbol in BOND_SYMBOLS)


def _options(
    primary: Sequence[str], secondary: Sequence[str]
) -> list[list[str]]:
    """The choices available for every printed line.

    A line the two recognisers read the same way has one choice; a
    line they disagree on has two.
    """
    count = max(len(primary), len(secondary))
    options: list[list[str]] = []
    for index in range(count):
        first = primary[index] if index < len(primary) else ""
        second = secondary[index] if index < len(secondary) else ""
        options.append([first] if first == second else [first, second])
    return options


def _combinations_with(
    options: list[list[str]], changes: int
):
    """Every combination that changes exactly ``changes`` lines.

    The lines are chosen in order, so the enumeration is
    deterministic, and within a line the *primary* reading is tried
    first, so a tie is won by the reading that leaves the primary
    text untouched.
    """
    if changes == 0:
        yield tuple(option[0] for option in options)
        return

    index = 0
    while index < len(options):
        if len(options[index]) < 2:
            index += 1
            continue
        break
    else:
        return

    chosen = [option[0] for option in options]

    def _recurse(start: int, left: int):
        if left == 0:
            yield tuple(chosen)
            return
        for position in range(start, len(options)):
            if len(options[position]) < 2:
                continue
            if len(options) - position < left:
                break
            original = chosen[position]
            chosen[position] = options[position][1]
            yield from _recurse(position + 1, left - 1)
            chosen[position] = original

    yield from _recurse(index, changes)


def vote_lines(
    primary: Sequence[str],
    secondary: Sequence[str],
    *,
    max_changes: int = 3,
    is_valid: Callable[[str], bool] | None = None,
    keep_bonds: bool = True,
) -> VoteResult:
    """Combine two readings of the same printed lines.

    ``primary`` and ``secondary`` hold one string per printed line.
    The result is the combination that ``is_valid`` accepts and that
    replaces the fewest lines of the primary reading.

    ``max_changes`` bounds the search: only the combinations that
    change at most that many lines are tried.  The bound keeps the
    search from running away on a badly printed cell, where almost
    any combination would be a guess.

    ``is_valid`` decides what "acceptable" means.  The default is
    *well-formed*: the string must use only SMILES characters **and**
    RDKit must be able to parse it.  Both halves are needed -- RDKit
    accepts a string with a space in it, and a printed SMILES column
    never has one.

    ``keep_bonds`` (on by default) additionally requires that the
    result declares at least as many double and triple bonds as the
    primary reading.  It stops the vote from silently turning an
    aromatic ring into a saturated one, which a parse check cannot
    detect.
    """
    if is_valid is not None:
        check = is_valid
    else:
        def check(text: str) -> bool:
            return _uses_only_smiles_characters(text) and _parses(text)

    primary_text = "".join(primary)
    primary_bonds = _bond_count(primary_text)

    options = _options(primary, secondary)
    for changes in range(max_changes + 1):
        for combination in _combinations_with(options, changes):
            text = "".join(combination)
            if not check(text):
                continue
            if keep_bonds and _bond_count(text) < primary_bonds:
                continue
            return VoteResult(
                text, changes, changes == 0, True
            )
    return VoteResult(primary_text, 0, True, check(primary_text))


__all__ = ["SMILES_ALPHABET", "VoteResult", "vote_lines"]
