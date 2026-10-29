"""Score an unparseable OCR SMILES against the patent's measured mass.

The problem this solves
-----------------------
Every quality signal we had required a finished SMILES. That is backwards:
OCR text never parses, so the gate could only ever say "no", and the repair
search had nothing to steer by.

The key observation is that **brackets do not change which atoms are
present**. ``C(=O)N``, ``C=O)N`` and ``CON`` all carry exactly the same
heavy atoms. Since the dominant OCR damage is missing brackets, the
heavy-atom composition survives the damage intact -- and that composition
can be tested against the measured mass without knowing the connectivity.

For a peptide that leaves a clean signature:

* heavy-atom mass is fully determined by the element counts;
* the leftover mass budget is the hydrogen count, which is an integer;
* so a candidate is plausible iff ``target - heavy_mass`` lands on a whole
  number of hydrogens in a chemically sane range.

That is a real gate, not a formality: a wrong formula is off by tens of
daltons and cannot land on a hydrogen count by accident.

Public API
----------
* :class:`MassGate`   -- build once from an m/z and polarity
* :func:`heavy_atom_mass`
* :func:`hydrogen_budget`
* :func:`score_text`  -- score a raw (possibly unparseable) string
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from cpd.ms_validation import neutral_mass_from_mz
from cpd.repair_audit import atom_multiset

MONO = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "F": 18.99840316273,
    "P": 30.97376151,
    "S": 31.9720711744,
    "Cl": 34.96885271,
    "Br": 78.9183376,
    "I": 126.904473,
    "Si": 27.9769265325,
    "Se": 79.9165218,
    "B": 11.0093054,
}

ORGANIC = frozenset("CNOSFPB")


def raw_element_counts(text: str) -> Counter:
    """Element histogram of a SMILES string that may not be well formed.

    Deliberately more forgiving than :func:`cpd.repair_audit.atom_multiset`,
    which stops its bracket walk at an unbalanced ``[`` and therefore drops
    every atom after it. That strictness is right when diffing two *parseable*
    structures, and wrong here: OCR routinely leaves a bracket unclosed, and a
    silently truncated histogram makes a good molecule look like it is missing
    a carbon.

    So this counts letters directly. ``Cl``/``Br`` are still read as single
    atoms, a bracket's contents are read as the element symbol, and characters
    that cannot be an element (``@``, ``=``, digits, brackets) are skipped.
    Nothing here depends on the string being grammatical.
    """
    counts: Counter = Counter()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        two = text[i:i + 2]
        if two in ("Cl", "Br"):
            counts[two] += 1
            i += 2
            continue
        if ch == "[":
            # Read to the next ']' if there is one; otherwise to the end.
            j = text.find("]", i + 1)
            span = text[i + 1:j] if j > 0 else text[i + 1:]
            for c in span:
                if c in ORGANIC:
                    counts[c] += 1
                    break
            i = (j + 1) if j > 0 else n
            continue
        if ch in ORGANIC:
            counts[ch] += 1
        i += 1
    return counts

# A peptide's hydrogens are bounded by its carbons: no carbon carries more
# than 3, and a saturated chain averages about 2. Wide on purpose -- the gate
# is meant to catch "wildly wrong", not to adjudicate borderline cases.
H_PER_C_MIN = 0.5
H_PER_C_MAX = 3.0

# Hydrogen mass is ~1.008, so a whole hydrogen count is a very tight
# signature. This is how far off the budget may be and still count.
H_TOLERANCE = 0.25


def heavy_atom_mass(counts: Counter) -> float:
    """Monoisotopic mass of the heavy atoms only. Unknown elements ignored."""
    return sum(MONO[el] * n for el, n in counts.items() if el in MONO)


def hydrogen_budget(counts: Counter, target_mass: float) -> Optional[float]:
    """Leftover mass after the heavy atoms, expressed in hydrogen units."""
    remainder = target_mass - heavy_atom_mass(counts)
    if remainder < 0:
        return None
    return remainder / MONO["H"]


@dataclass(frozen=True)
class MassGate:
    """A reusable gate for one row's reported mass.

    ``slack_da`` absorbs the uncertainty implied by how many decimals the
    patent printed: a cell reading ``1610.9`` is only good to +/-0.05 Da,
    which is about half a hydrogen and must not be treated as an error.
    """

    target: float
    polarity: str
    slack_da: float = 0.06

    @classmethod
    def from_row(cls, mz: str, polarity: str) -> Optional["MassGate"]:
        try:
            value = float(str(mz).replace(",", ""))
        except (TypeError, ValueError):
            return None
        target = neutral_mass_from_mz(value, polarity)
        if target is None:
            return None
        text = str(mz).replace(",", "")
        decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
        slack = max(0.06, 0.5 * (10.0 ** -decimals))
        return cls(target=target, polarity=polarity, slack_da=slack)

    @property
    def hydrogen_tolerance(self) -> float:
        return self.slack_da / MONO["H"] + H_TOLERANCE

    def score_counts(self, counts: Counter) -> dict:
        """Judge an element histogram against the measured mass."""
        unknown = sorted(e for e in counts if e not in MONO)
        heavy = heavy_atom_mass(counts)
        carbon = counts.get("C", 0)
        result = {
            "target": round(self.target, 4),
            "heavy_mass": round(heavy, 4),
            "carbon": carbon,
            "unknown_elements": unknown,
            "verdict": "unknown",
            "implied_h": None,
            "h_error": None,
        }
        if unknown:
            # "Ca" from a misread [C@@H] is handled upstream; anything else
            # means we do not know what we are measuring.
            result["verdict"] = "unknown_elements" if unknown != ["Ca"] else "needs_ca_fix"
            return result

        implied = hydrogen_budget(counts, self.target)
        if implied is None:
            result["verdict"] = "too_heavy"
            return result

        nearest = round(implied)
        err = abs(implied - nearest)
        result["implied_h"] = implied
        result["h_error"] = err
        if not (H_PER_C_MIN * carbon <= implied <= H_PER_C_MAX * carbon):
            result["verdict"] = "implausible_h"
            return result
        if err <= self.hydrogen_tolerance:
            result["verdict"] = "on_integer_h"
        elif err <= 0.75:
            result["verdict"] = "near_integer_h"
        else:
            result["verdict"] = "off_integer_h"
        return result

    def score_text(self, text: str) -> dict:
        """Convenience: histogram a raw string, then score it.

        Uses :func:`raw_element_counts` rather than
        :func:`cpd.repair_audit.atom_multiset` because OCR text routinely has
        unbalanced brackets, and the strict tokenizer drops every atom after
        the first unclosed ``[``.
        """
        return self.score_counts(raw_element_counts(text))


__all__ = [
    "H_PER_C_MAX",
    "H_PER_C_MIN",
    "H_TOLERANCE",
    "MONO",
    "MassGate",
    "heavy_atom_mass",
    "hydrogen_budget",
]
