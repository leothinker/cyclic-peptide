"""Mass-constrained repair of an OCR'd printed SMILES.

What the edit model says
------------------------
``analysis/align_ocr_to_reference.py`` diffs cmpd 1036's OCR text against its
hand transcription and measures the damage exactly: **23 deletions, 0
insertions, 6 substitutions.** The deleted characters are ``(`` (16), ``)``
(5), ``=`` (1) and ``[`` (1). The substitutions are ``(=`` -> ``-`` four
times, ``=`` -> ``-`` once, and ``@`` -> ``a`` once.

Two consequences shape this search.

*Brackets are missing, not wrong.* So the repair is overwhelmingly an
**insertion** problem, and restricting the search to insertions keeps it
small. The previous version of this module enumerated subsets of edit sites
with ``itertools.combinations`` over every low-confidence position; with ~40
edits across ~244 positions that space is astronomically large, which is why
it had never produced a result on real data.

*The mass cannot choose between bracket placements.* Inserting a parenthesis
does not change the molecular formula, so no mass test can prefer one correct
placement over another. What discriminates is the ring structure: a cyclic
peptide must close a large macrocycle and keep its ring-closure digits
balanced. The mass gate's real job -- established separately in
``analysis/heavy_atom_mass_gate.py`` -- is to confirm the **atom inventory**
is right; grammar plus the macrocycle test select the **connectivity**.

The search
-----------
A best-first (A*) walk over the string. At each position the search may emit
the next source character, drop it, or insert a character from
:data:`INSERT_ALPHABET` before continuing. Every partial output is advanced
through :class:`cpd.parsers.prefix_validator.PrefixValidator`, which rejects a
prefix the instant it becomes ungrammatical -- that is the pruning signal
which keeps the search tractable.

Why A* and not a level-synchronous beam
---------------------------------------
A beam ranked by cumulative cost throws away precisely the states that are
investing in a repair. Repairing 15 brackets costs around 5, while a state
that has inserted nothing costs 0, so a 240-wide beam fills up with states
that insert nothing and then die at the first unmatched ``)``. Measured on
1036, that search explored 58 415 states and found nothing.

Best-first ordering by *estimated total* cost fixes it: a state that has
already paid for several brackets is queued ahead of one that has paid for
none but is about to become invalid. The heuristic is the number of unmatched
``)`` still ahead, times the cheapest possible ``(`` -- admissible, because
every one of those closers needs an opener before it.

Costs come from :mod:`cpd.ocr_repair_costs`, so the search loop stays
readable and the numbers stay testable.

Acceptance
----------
A finished string is accepted only when RDKit parses it, its largest ring is a
plausible macrocycle, and its monoisotopic mass lands inside the window the
patent's reported m/z implies. That last test is what stops the search from
"repairing" a mis-read structure into a different but well-formed molecule.

Public API
----------
* :func:`repair_from_ocr`         -- lexical pre-pass plus search (main entry)
* :func:`search_mass_constrained` -- the beam, on already-prepared text
* :func:`bracket_deficit`         -- how many brackets a text is missing
* :data:`INSERT_ALPHABET`
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cpd.ms_validation import neutral_mass_from_mz
from cpd.ocr_edit_model import normalise_lexical
from cpd.ocr_repair_costs import (COST_CARBONYL_OPEN, COST_DELETE,
                                       expansion_cost, expansions_for,
                                       insert_cost)
from cpd.parsers.prefix_validator import PrefixValidator

# Best-first has no beam to size, but it still needs a ceiling on how many
# states it may expand, so a pathological row cannot spin indefinitely.
DEFAULT_MAX_STATES = 3_000_000
DEFAULT_SECONDS = 30.0

# How far a candidate's exact mass may sit from the window centre.
#
# This is deliberately loose. The patent prints m/z to one decimal, which
# only implies +/-0.05 Da, but the hand-transcribed reference for cmpd 1036
# sits 0.20 Da *below* the centre of that window, and the same compound
# reported as [M+3H]3+ on a neighbouring row sits 0.11 Da below it too. Two
# independent rows agreeing on a ~0.15 Da shortfall says the offset is
# systematic, not a transcription error, so a tight window would reject the
# one answer we know to be correct.
#
# Loosening the mass does not make the gate useless. It still rejects the
# wrong-molecule failures that motivated it -- DECIMER's output is ~66 Da
# off, and no plausible bracket placement moves a formula by more than a few
# tenths. What the mass cannot do is *rank* bracket placements against each
# other, because brackets do not change the formula. That job belongs to the
# macrocycle test and the cost model.
DEFAULT_MASS_SLACK = 0.60

# Brackets and bond symbols the edit model observed being lost.
INSERT_ALPHABET = "()[]="


def bracket_deficit(text: str) -> dict:
    """How many brackets this text is missing.

    A negative paren balance means unmatched ``)``, each needing an inserted
    ``(`` somewhere before it. This is a *lower bound* on the true insertion
    count: it cannot see a ``(`` that was swallowed together with a ``=`` into
    a single ``-``, nor a ``[`` that left no trace at all. It is still the
    right thing to budget against, because under-estimating prunes far less
    than over-estimating does.
    """
    paren = bracket = 0
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
        "open_paren": max(0, -paren),
        "close_paren": max(0, paren),
        "open_bracket": max(0, -bracket),
        "close_bracket": max(0, bracket),
        "total": (max(0, -paren) + max(0, paren)
                   + max(0, -bracket) + max(0, bracket)),
    }


def mass_window(mz: float, polarity: str) -> tuple:
    """Neutral-mass window implied by a printed m/z.

    A value printed to ``d`` decimals carries about +/- 0.5 * 10**-d of
    uncertainty, so the window follows the table's own formatting.
    """
    expected = neutral_mass_from_mz(mz, polarity)
    if expected is None:
        raise ValueError(f"cannot interpret polarity {polarity!r}")
    text = repr(float(mz))
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    half = 0.5 * (10.0 ** -decimals)
    return expected - half, expected + half


def _evaluate(smiles: str, lo: float, hi: float, slack: float,
               min_ring: int) -> Optional[dict]:
    """Accept only a parseable macrocycle whose mass fits the window."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    largest = max(rings) if rings else 0
    if largest < min_ring:
        return None
    mass = rdMolDescriptors.CalcExactMolWt(mol)
    if not (lo - slack) <= mass <= (hi + slack):
        return None
    return {
        "mass": mass,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "largest_ring": largest,
        "num_rings": len(rings),
        "atoms": mol.GetNumAtoms(),
    }


def search_mass_constrained(
    text: str,
    mz: float,
    polarity: str,
    *,
    max_inserts: Optional[int] = None,
    max_states: int = DEFAULT_MAX_STATES,
    mass_slack: float = DEFAULT_MASS_SLACK,
    min_ring: int = 12,
    max_seconds: float = DEFAULT_SECONDS,
    allow_subs: bool = True,
    allow_dels: bool = True,
) -> dict:
    """Search a valid macrocyclic SMILES out of damaged OCR text.

    Returns ``smiles`` (canonical, or ``None``), its ``formula`` and ``mass``,
    the edit counts, and a ``trace`` explaining the outcome.

    Best-first over ``cost_so_far + heuristic``, where the heuristic counts
    the unmatched ``)`` still ahead times the cheapest ``(``. That is
    admissible: each of those closers needs an opener before it, and no opener
    is cheaper than :data:`COST_CARBONYL_OPEN`.
    """
    started = time.time()
    deficit = bracket_deficit(text)
    if max_inserts is None:
        # The observed deficit, plus headroom for the invisible losses: a "("
        # swallowed by a "-", or a "[" that left no trace.
        max_inserts = deficit["total"] + 14

    result = {
        "smiles": None,
        "mass": None,
        "formula": None,
        "largest_ring": 0,
        "n_inserts": 0,
        "n_subs": 0,
        "n_dels": 0,
        "states": 0,
        "trace": "",
    }
    if not text:
        result["trace"] = "empty_input"
        return result
    try:
        lo, hi = mass_window(mz, polarity)
    except ValueError as exc:
        result["trace"] = f"bad_polarity:{exc}"
        return result

    info = _evaluate(text, lo, hi, mass_slack, min_ring)
    if info:
        result.update(
            smiles=Chem.MolToSmiles(Chem.MolFromSmiles(text)),
            mass=round(info["mass"], 4),
            formula=info["formula"],
            largest_ring=info["largest_ring"],
            trace="already_valid_and_in_window",
        )
        return result

    n = len(text)
    # How many ")" still lie ahead of each source index, used by the heuristic.
    closes_ahead = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        closes_ahead[i] = closes_ahead[i + 1] + (1 if text[i] == ")" else 0)

    def heuristic(depth: int, i: int) -> float:
        """Cost still owed: one "(" for every closer nothing will cover.

        Admissible, because each of those closers needs an opener before it,
        and no opener costs less than COST_CARBONYL_OPEN.
        """
        uncovered = closes_ahead[min(i, n)] - depth
        return COST_CARBONYL_OPEN * max(0, uncovered)

    # A state is (emitted, validator, source_index, inserts, subs, dels, cost).
    counter = itertools.count()
    start = ("", PrefixValidator(), 0, 0, 0, 0, 0.0)
    queue = [(heuristic(0, 0), 0.0, next(counter), start)]
    best = None
    seen = 0

    # Cap on the open priority queue. Without this, the heap can grow to
    # millions of entries on heavy rows (e.g. 1036 with 15 missing "(")
    # and dominate runtime on heap operations instead of useful search.
    # The cap is intentionally larger than the state budget so we never
    # drop a state we might still want to expand.
    MAX_QUEUE = 4 * max_states

    # Evicting the *worst* state needs the largest key, and heapq only offers
    # a min-heap, so a mirrored max-heap (negated keys) tracks the eviction
    # order. Eviction marks an entry dead rather than deleting it from the
    # frontier, because removing an arbitrary element from a heap costs a
    # full sift. Popping the frontier skips the dead entries instead, which
    # keeps every operation O(log n) and -- unlike calling ``heappop`` on the
    # frontier -- never discards a cheap state we still want to expand.
    evicted: set = set()
    live = 1  # frontier entries that are neither expanded nor evicted
    eviction_queue: list = [(-queue[0][0], -queue[0][1], -0, 0)]

    while queue:
        _est, _spent, _tie, state = heapq.heappop(queue)
        if _tie in evicted:
            continue
        live -= 1
        if seen >= max_states:
            result["trace"] = "state_budget_hit"
            break
        if time.time() - started > max_seconds:
            result["trace"] = "time_budget_hit"
            break

        out, v, i, ins, subs, dels, cost = state
        seen += 1

        # Goal: the whole source consumed and the result a finished,
        # mass-verified macrocycle. Best-first ordering means the first such
        # state popped is the cheapest one reachable.
        if i >= n and v.is_complete():
            got = _evaluate(out, lo, hi, mass_slack, min_ring)
            if got:
                best = (state, got)
                break

        children = []
        if i < n:
            src = text[i]
            if allow_subs:
                for chosen in expansions_for(src):
                    vv = v
                    for ch in chosen:
                        vv = vv.advance(ch) if vv is not None else None
                        if vv is None:
                            break
                    if vv is None:
                        continue
                    children.append((out + chosen, vv, i + 1, ins,
                                      subs + (1 if chosen != src else 0), dels,
                                      cost + expansion_cost(src, chosen)))
            else:
                # Pure emission. Faster than checking expansions: emits src
                # iff the validator accepts it.
                vv = v.advance(src)
                if vv is not None:
                    children.append((out + src, vv, i + 1, ins, subs, dels,
                                      cost + 1.0))
            # Dropping is almost never right -- the measured edit model had
            # zero OCR deletions -- so it is heavily penalised and capped.
            if allow_dels and dels < 2:
                children.append((out, v, i + 1, ins, subs, dels + 1,
                                  cost + COST_DELETE))

        if ins < max_inserts:
            for ch in INSERT_ALPHABET:
                vv = v.advance(ch)
                if vv is None:
                    continue
                children.append((out + ch, vv, i, ins + 1, subs, dels,
                                  cost + insert_cost(ch, out)))

        for child in children:
            tie = next(counter)
            est = child[6] + heuristic(child[1].depth, child[2])
            heapq.heappush(queue, (est, child[6], tie, child))
            heapq.heappush(eviction_queue, (-est, -child[6], -tie, tie))
            live += 1

        # Trim from the expensive end. This is the only safe direction: the
        # cheapest states on the frontier are the ones most likely to reach a
        # complete, mass-verified molecule.
        while live > MAX_QUEUE and eviction_queue:
            _nf, _ng, _nt, tie = heapq.heappop(eviction_queue)
            if tie == 0 or tie in evicted:
                # The start state has already been expanded, and a repeated
                # tie means this slot was accounted for by an earlier trim.
                continue
            evicted.add(tie)

    result["states"] = seen
    result["evicted"] = len(evicted)
    if best is None:
        if not result["trace"]:
            result["trace"] = "no_mass_matching_candidate"
        return result

    (out, _v, _i, ins, subs, dels, _cost), info = best
    result.update(
        smiles=Chem.MolToSmiles(Chem.MolFromSmiles(out)),
        mass=round(info["mass"], 4),
        formula=info["formula"],
        largest_ring=info["largest_ring"],
        n_inserts=ins,
        n_subs=subs,
        n_dels=dels,
        trace="accepted",
    )
    return result


def repair_from_ocr(
    text: str,
    mz: float,
    polarity: str,
    *,
    apply_lexical: bool = True,
    **kwargs,
) -> dict:
    """Run the deterministic lexical pre-pass, then the beam search."""
    notes: dict = {}
    prepared = text
    if apply_lexical:
        prepared, notes = normalise_lexical(text)
    out = search_mass_constrained(prepared, mz, polarity, **kwargs)
    out["lexical_notes"] = notes
    return out


__all__ = [
    "DEFAULT_MAX_STATES",
    "INSERT_ALPHABET",
    "bracket_deficit",
    "mass_window",
    "repair_from_ocr",
    "search_mass_constrained",
]
