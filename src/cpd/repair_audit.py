"""Chemical-invariant auditing for macrocycle repair candidates.

Why this module exists
----------------------
:mod:`cpd.macrocycle_repair` contains six *heuristic* strategies.  Some of
them are genuinely conservative -- appending a missing ``)`` changes not a
single atom of the molecular graph.  Others are not: appending a ring digit
plus a ``C`` invents an atom that was never in the DECIMER output, and
trimming a truncated tail deletes real atoms to manufacture a macrocycle.

Reporting a single "recovery rate" over all six strategies therefore
overstates the result.  This module measures *what each repair actually did
to the molecular graph* and assigns a confidence tier, so the pipeline can
separate validated repairs from speculative ones.

Invariant fingerprint
---------------------
:func:`fingerprint` computes the observables that must not move unless a
repair genuinely rebuilt the molecule:

* ``formula``          -- Hill-order molecular formula
* ``heavy_atoms``      -- non-hydrogen atom count
* ``mw``               -- exact molecular weight
* ``rings``            -- RDKit ring count (SSSR)
* ``largest_ring``     -- size of the biggest SSSR ring
* ``amides``           -- C(=O)-N bonds (the peptide-backbone marker)
* ``stereo_centres``   -- assigned tetrahedral stereocentres
* ``charge``           -- net formal charge
* ``fragments``        -- number of disconnected components

Atom-multiset check
--------------------
Most broken DECIMER outputs do not parse, so there is no parseable "before"
molecule to diff against -- :func:`fingerprint` returns ``None`` and an
invariant comparison becomes impossible.  Falling back to "assume zero atoms
changed" would be unsound, because
:func:`cpd.macrocycle_repair.attempt_syntax_fix` really does append
``C``/``N``/``O`` atoms in one of its branches.

:func:`atom_multiset` therefore tallies element symbols straight out of the
raw SMILES text (bracket atoms by their element, organic-subset atoms by
themselves, ring digits and bond symbols ignored).  It works on strings that
RDKit cannot parse at all, which is exactly the population we care about.
Diffing the two multisets answers the only question that matters: did this
repair invent atoms, delete atoms, or leave the atom inventory untouched?

Confidence tiers
----------------
``high``
    The repair is purely syntactic: brackets/parens were balanced or a
    ring-closure digit was re-balanced.  The raw atom multiset is unchanged,
    so no atom was invented or deleted.
``medium``
    The repair removed a ``.`` fragment separator, or re-coded an
    over-used ring-closure digit.  Atom count is unchanged; the graph
    connectivity may differ, so the result needs visual confirmation.
``low``
    The repair truncated the tail.  Atoms were deleted, so the molecule is
    provably a sub-structure of the real compound.
``experimental``
    The repair invented atoms or a bond.  The macrocycle may be an
    artefact of the heuristic rather than a property of the compound.
``rejected``
    The repair failed a chemistry plausibility gate (implausible ring
    size, too few amides, disconnected graph) and must not be promoted to
    ``smiles_best``.

Public API
----------
* :func:`fingerprint`                  -- invariants of one molecule/SMILES
* :func:`atom_multiset`                -- element histogram of raw SMILES text
* :func:`audit_repair`                 -- before/after comparison plus a tier
* :func:`is_plausible_cyclic_peptide`  -- the plausibility gate
* :func:`tier_rank`                    -- ordering helper for sorting tiers
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# A cyclic peptide in this patent series is a macrocycle whose ring runs
# from roughly 12 up to ~45 atoms.  The fused bicyclic members carry an
# extra small ring, but the macrocycle itself never exceeds 45.
PLAUSIBLE_MACROCYCLE_RANGE: tuple[int, int] = (12, 45)

# A real cyclic peptide has at least a handful of amide bonds; anything
# below this is not peptide-like even if a large ring happened to form.
MIN_AMIDE_COUNT: int = 4

# Confidence tiers, most to least trustworthy.
TIERS: tuple[str, ...] = ("high", "medium", "low", "experimental", "rejected")


# Anchored on the opening bracket so it can be matched at the bracket
# position itself; an optional isotope prefix ([13C]) is skipped and the
# element symbol captured.  Two-letter elements are tried first via the
# greedy optional second character.
_BRACKET_ATOM_RE = re.compile(r"\[\s*(?:\d+)?\s*([A-Z][a-z]?|[a-z])")

# Two-character organic-subset symbols must be tried before the
# single-character ones, otherwise "Cl" would be read as "C" plus a stray
# "l" that is not an element.
_TWO_CHAR_SYMBOLS = ("Cl", "Br")
_ONE_CHAR_SYMBOLS = set("BCNOPSFIbcnops")


def atom_multiset(smiles: str) -> Counter[str]:
    """Return a ``Counter`` of element symbols in a raw SMILES string.

    Works on strings RDKit cannot parse, which is the whole point: the
    broken DECIMER outputs that most need auditing are exactly the ones
    :func:`fingerprint` refuses.

    * bracket atoms ``[C@H]``, ``[nH+]``, ``[13C]``, ``[Na+]`` contribute
      their element symbol (case preserved, so aromatic ``n`` stays
      distinct from aliphatic ``N``);
    * outside brackets ``Cl``/``Br`` are read as single atoms and the
      organic subset ``B C N O P S F I b c n o p s`` likewise;
    * ring-closure digits, bond symbols, branches, dots and whitespace are
      ignored, since they carry no atoms.

    An unbalanced ``[`` stops the walk rather than guessing, which keeps a
    malformed input from producing a silently wrong histogram.
    """
    counts: Counter[str] = Counter()
    if not smiles:
        return counts
    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "[":
            close = smiles.find("]", i + 1)
            if close < 0:
                break
            m = _BRACKET_ATOM_RE.match(smiles, i)
            if m:
                counts[m.group(1)] += 1
            i = close + 1
            continue
        two = smiles[i : i + 2]
        if two in _TWO_CHAR_SYMBOLS:
            counts[two] += 1
            i += 2
            continue
        if ch in _ONE_CHAR_SYMBOLS:
            counts[ch] += 1
        i += 1
    return counts


def tier_rank(tier: str) -> int:
    """Return a sort rank for ``tier``; lower is more trustworthy."""
    try:
        return TIERS.index(tier)
    except ValueError:
        return len(TIERS)


def _amide_count(mol: Chem.Mol) -> int:
    """Count C(=O)-N bonds in ``mol`` (the peptide-backbone marker)."""
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7:
            carbon = a
        elif a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6:
            carbon = b
        else:
            continue
        for nb in carbon.GetBonds():
            if nb.GetIdx() == bond.GetIdx():
                continue
            if (
                nb.GetBondType() == Chem.BondType.DOUBLE
                and nb.GetOtherAtom(carbon).GetAtomicNum() == 8
            ):
                count += 1
                break
    return count


def _stereo_centre_count(mol: Chem.Mol) -> int:
    """Return the number of assigned tetrahedral stereocentres."""
    return sum(1 for atom in mol.GetAtoms() if atom.HasProp("_CIPCode"))


def fingerprint(mol_or_smiles: Chem.Mol | str) -> Optional[dict]:
    """Return the invariant fingerprint of a molecule or SMILES string.

    Returns ``None`` when the input cannot be parsed or sanitised, which
    callers should treat as "no baseline available".
    """
    if isinstance(mol_or_smiles, str):
        if not mol_or_smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(mol_or_smiles)
        except Exception:
            return None
    else:
        mol = mol_or_smiles
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        "rings": mol.GetRingInfo().NumRings(),
        "largest_ring": max(ring_sizes) if ring_sizes else 0,
        "amides": _amide_count(mol),
        "stereo_centres": _stereo_centre_count(mol),
        "charge": Chem.GetFormalCharge(mol),
        "fragments": len(Chem.GetMolFrags(mol)),
    }


def is_plausible_cyclic_peptide(fp: Optional[dict]) -> tuple[bool, str]:
    """Gate a fingerprint on peptide-like macrocycle expectations.

    Returns ``(ok, reason)``.  ``reason`` is empty when ``ok`` is True and
    otherwise names the first violated expectation.
    """
    if fp is None:
        return False, "unparseable"
    lo, hi = PLAUSIBLE_MACROCYCLE_RANGE
    if not (lo <= fp["largest_ring"] <= hi):
        return False, f"largest_ring={fp['largest_ring']}_outside_{lo}_{hi}"
    if fp["amides"] < MIN_AMIDE_COUNT:
        return False, f"amides={fp['amides']}_below_{MIN_AMIDE_COUNT}"
    if fp["fragments"] != 1:
        return False, f"fragments={fp['fragments']}"
    return True, ""


# Strategies whose only edit is textual punctuation cannot change the graph.
_PUNCTUATION_ONLY_METHODS = {"syntax_fix"}

# Strategies that delete real atoms from the tail.
_TRUNCATING_METHODS = {"strip_truncated_tail"}

# Strategies that create a bond outright.
_INVENTING_METHODS = {"terminal_bridge"}


@dataclass
class RepairAudit:
    """Audit result for one repair candidate."""

    cmpd_id: Optional[int] = None
    method: str = ""
    before: Optional[dict] = None
    after: Optional[dict] = None
    tier: str = "rejected"
    reason: str = ""
    atoms_delta: int = 0
    atoms_invented: int = 0
    atoms_deleted: int = 0
    amides_delta: int = 0
    stereo_delta: int = 0
    formula_changed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """True when the repair may be promoted to a production column."""
        return self.tier != "rejected"

    @property
    def validated(self) -> bool:
        """True only for tiers we are willing to call a real success."""
        return self.tier in ("high", "medium")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["accepted"] = self.accepted
        data["validated"] = self.validated
        return data


def audit_repair(
    original: str,
    repaired: Optional[str],
    method: str = "",
    cmpd_id: Optional[int] = None,
) -> RepairAudit:
    """Audit one repair candidate against its DECIMER input.

    ``original`` is the raw DECIMER SMILES (which usually does *not*
    parse), ``repaired`` the candidate produced by
    :func:`cpd.macrocycle_repair.repair_macrocycle`.

    The tier is decided by what the repair did to the atom inventory, read
    straight out of the two raw strings via :func:`atom_multiset`.  That
    works even when neither side parses, which is the common case:

    * identical atom multisets -> nothing invented, nothing deleted, so
      ``high`` when the strategy is punctuation-only and ``medium``
      otherwise;
    * fewer heavy atoms -> ``low`` (real atoms were deleted);
    * more heavy atoms -> ``experimental`` (atoms were invented);
    * any chemistry-plausibility failure -> ``rejected``, overriding the
      above.
    """
    audit = RepairAudit(cmpd_id=cmpd_id, method=method)

    if not repaired:
        audit.tier = "rejected"
        audit.reason = "no_repair_found"
        return audit

    before_fp = fingerprint(original)
    after_fp = fingerprint(repaired)
    audit.before = before_fp
    audit.after = after_fp

    if after_fp is None:
        audit.tier = "rejected"
        audit.reason = "repair_unparseable"
        return audit

    # Primary evidence: compare the raw atom inventory.  This is the only
    # comparison that works when the DECIMER input does not parse.
    raw_before = atom_multiset(original)
    raw_after = atom_multiset(repaired)
    invented = raw_after - raw_before
    deleted = raw_before - raw_after
    audit.atoms_invented = sum(invented.values())
    audit.atoms_deleted = sum(deleted.values())
    audit.atoms_delta = audit.atoms_invented - audit.atoms_deleted

    if before_fp is not None:
        audit.amides_delta = after_fp["amides"] - before_fp["amides"]
        audit.stereo_delta = after_fp["stereo_centres"] - before_fp["stereo_centres"]
        audit.formula_changed = before_fp["formula"] != after_fp["formula"]
    else:
        # The raw atom diff above still stands on its own, so a tier is
        # reachable; we only lose the formula/stereo cross-checks.
        audit.notes.append("original_unparseable_no_formula_check")

    ok, reason = is_plausible_cyclic_peptide(after_fp)
    if not ok:
        audit.tier = "rejected"
        audit.reason = reason
        return audit

    # Plausibility passed. Now classify what the repair actually changed.
    if audit.atoms_invented > 0 or method in _INVENTING_METHODS:
        audit.tier = "experimental"
        audit.reason = f"invented_{audit.atoms_invented}_atoms"
        return audit
    if audit.atoms_deleted > 0 or method in _TRUNCATING_METHODS:
        audit.tier = "low"
        audit.reason = f"truncated_{audit.atoms_deleted}_atoms"
        return audit
    if audit.stereo_delta < 0:
        audit.tier = "low"
        audit.reason = f"lost_{abs(audit.stereo_delta)}_stereocentres"
        return audit
    if method in _PUNCTUATION_ONLY_METHODS and not audit.formula_changed:
        audit.tier = "high"
        return audit
    if method == "rejoin_fragments":
        audit.tier = "medium"
        audit.notes.append("check_fragment_bond_order_manually")
        return audit
    if audit.formula_changed:
        audit.tier = "medium"
        audit.notes.append("formula_changed_despite_matched_atom_counts")
        return audit
    audit.tier = "medium"
    return audit


__all__ = [
    "MIN_AMIDE_COUNT",
    "PLAUSIBLE_MACROCYCLE_RANGE",
    "TIERS",
    "RepairAudit",
    "atom_multiset",
    "audit_repair",
    "fingerprint",
    "is_plausible_cyclic_peptide",
    "tier_rank",
]
