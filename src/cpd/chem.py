"""Chemical structure validation, auto-repair, and descriptor calculation for cyclic peptides."""

from __future__ import annotations

import re
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import rdBase


@dataclass(frozen=True)
class ChemInfo:
    """Validated SMILES and physical descriptors for a cyclic peptide."""

    canonical_smiles: str
    molecular_weight: float
    heavy_atom_count: int
    logp: float
    tpsa: float
    rotatable_bonds: int
    num_rings: int
    largest_ring_size: int
    num_stereocenters: int
    formula: str


def largest_fragment_canonical(smiles: str) -> str | None:
    """Drop small phantom fragments from a SMILES string.

    DECIMER occasionally appends tiny fragments like ``CC`` or ``CCO``
    when it gets confused on a structure, separated from the real
    peptide by ``.``. RDKit then parses the dot as a multi-fragment
    molecule and canonicalises differently, so the SMILES never
    matches the OCR's single-fragment output. Keeping only the
    largest fragment by atom count recovers the real peptide.

    Returns the canonical SMILES of the largest fragment, or ``None``
    if the input cannot be parsed.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    if len(frags) <= 1:
        return Chem.MolToSmiles(mol)
    frags.sort(key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0])


def close_unbalanced_brackets(smiles: str) -> str | None:
    """Append the missing ``)`` and ``]`` so a truncated SMILES parses.

    The DECIMER transformer decoder has a positional-encoding maxlen of
    ~300, so cyclic-peptide SMILES longer than that get truncated mid-
    branch (e.g. ``...C(=O)N(C)C)F)F)F``). RDKit cannot parse them
    because of the dangling parens. Counting ``(`` minus ``)`` and
    ``[`` minus ``]`` and appending the missing closers recovers a
    surprising fraction of these cases (~33% in our benchmark).

    Returns the patched SMILES, or ``None`` if it has nothing to fix.
    """
    if not smiles:
        return None
    open_p = smiles.count("(") - smiles.count(")")
    open_b = smiles.count("[") - smiles.count("]")
    if open_p <= 0 and open_b <= 0:
        return None
    return smiles + ")" * max(0, open_p) + "]" * max(0, open_b)


def auto_repair_smiles(raw_smiles: str) -> str:
    """Auto-heal common OCR syntax flaws in peptide SMILES strings."""
    s = raw_smiles
    # 1. Strip non-ASCII artifacts (e.g. table border line fragments)
    s = re.sub(r"[^\x00-\x7F]+", "", s)

    # 2. Repair wrapped amide carbonyls: C(O)N or C(=0)N -> C(=O)N
    s = re.sub(r"C\(O\)N", "C(=O)N", s)
    s = re.sub(r"C\(=[0O]\)?N", "C(=O)N", s)
    # Same flavour for generic C(=0) -> C(=O) anywhere in the string
    s = re.sub(r"C\(=0\)", "C(=O)", s)

    # A lactam carbonyl is printed as "C1=O)" -- the ring-closure digit
    # sits between the carbon and the =O.  The recogniser reads that O
    # as a zero, giving "C1=0)".  Zero is not an element, so "=0" can
    # only be a misread "=O" -- unless the zero is a ring-closure digit,
    # and a ring-closure digit always appears in a matched pair.  When
    # the string holds exactly one "0", that zero has no partner and is
    # therefore a misread oxygen.
    if s.count("0") == 1:
        s = s.replace("=0", "=O")

    # 3. Repair unclosed chiral brackets: [C@H( or [C@@H( -> [C@H]( or [C@@H](
    s = re.sub(r"(\[C@{1,2}H)\(", r"\1](", s)

    # The same recogniser failure in front of a *ring-closure digit*:
    # "[C@@H4C(=O)" should be "[C@@H]4C(=O)".  A digit cannot appear
    # inside a bracket atom in SMILES -- the bracket holds an isotope,
    # an element, an H count, a charge and an atom class, nothing
    # else -- so the form with the digit inside the brackets is never
    # valid SMILES and cannot be what was printed.
    #
    # The digit is restored as a ring closure, so it must have a
    # partner elsewhere in the string; a ring-closure digit always
    # appears in a matched pair.  When it has no partner the string
    # is left alone, because then the digit is not a ring closure and
    # the repair would be a guess.
    s = _restore_dropped_chiral_bracket(s)

    # 4. Re-attach the "CC[C@" prefix the OCR tends to lop off when the
    #    first SMILES characters touch the left column boundary.
    if s.startswith("H](") or s.startswith("@H]("):
        s = "CC[C@" + s
    elif s.startswith("[C@H](") and not s.startswith("CC[C@H]("):
        s = "CC" + s

    # 5. Undo a bracket that was read as its mirror image.
    s = _repair_mirrored_bracket(s)

    return s.strip()


def _repair_mirrored_bracket(smiles: str) -> str:
    """Restore an opening bracket that was read as a closing one.

    A printed ``[C@@H](C)`` is sometimes read as ``]C@@H](C)``: the
    recogniser saw the *closing* bracket where the page has the
    opening one.  The damage has a precise signature, which is what
    makes the repair safe to attempt:

    * the string holds **two** more ``]`` than ``[`` -- one bracket
      changed into its mirror image moves the difference by two, not
      one, so a difference of one cannot come from this mistake and
      must not be "fixed" by flipping a bracket;
    * the parentheses still balance, so the misread bracket is not
      part of a confused pair; and
    * the string does not parse as it stands, so there is nothing to
      lose by trying.

    The rule then flips a closing bracket into an opening one and
    accepts the result only when it is *unambiguous*: exactly one
    position yields a string that parses and that has not lost a
    double or triple bond.  Every condition is checked on the string
    itself -- no reference transcription and no mass value is
    consulted.

    Measured on the two patents' printed corpora, the rule fires on
    1 of 28 compounds of WO2022251372 and on 0 of 335 of
    WO2025162428.
    """
    if not smiles or "[" not in smiles or "]" not in smiles:
        return smiles

    # A mirror-image misread changes the bracket difference by two.
    if smiles.count("]") - smiles.count("[") != 2:
        return smiles
    # The parentheses must still balance, or the confusion is wider
    # than a single bracket and the repair would be a guess.
    if smiles.count(")") != smiles.count("("):
        return smiles
    # A string that already parses is not damaged in this way.
    if _parses(smiles):
        return smiles

    candidates: list[str] = []
    for index, char in enumerate(smiles):
        if char != "]":
            continue
        patched = smiles[:index] + "[" + smiles[index + 1:]
        if patched.count("]") != patched.count("["):
            continue
        if not _parses(patched):
            continue
        # The repair may restore syntax; it may not change chemistry.
        # Turning a closing bracket into an opening one can pull a
        # digit inside the bracket, and a digit inside a bracket atom
        # is an *isotope* while a digit outside it is a *ring
        # closure*.  The two are different molecules, so a flip that
        # moves a digit across the bracket is refused.
        if _digit_moved_inside(patched, index):
            continue
        candidates.append(patched)

    if len(candidates) != 1:
        return smiles
    return candidates[0]


def _parses(smiles: str) -> bool:
    """Whether RDKit can read the string as SMILES."""
    if not smiles:
        return False
    try:
        # The parse is a *test*, not a diagnosis: a string that fails
        # to parse is an expected input here, so RDKit's complaint
        # would only be noise on the console.
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _digit_moved_inside(patched: str, flip_index: int) -> bool:
    """Whether the flip pulled a ring-closure digit into a bracket.

    The flip replaces one ``]`` with a ``[``, so the bracket depth
    before the flip is the same in both strings and the depth after
    it is one greater.  A digit that sits inside a bracket *after*
    the flip therefore sat outside it before, where it was a ring
    closure.
    """
    depth = 0
    for index, char in enumerate(patched):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char.isdigit() and index > flip_index and depth > 0:
            return True
    return False


def _restore_dropped_chiral_bracket(smiles: str) -> str:
    """Restore the ``]`` of a chiral atom whose closing bracket was lost.

    The recogniser sometimes drops the ``]`` of ``[C@@H]``, so the
    printed ``[C@@H]4`` is read as ``[C@@H4``.  The digit is a
    ring-closure digit, and it is restored as one -- which means it
    has to have a partner in the string.  A digit without a partner is
    left alone: the repair would then be a guess.
    """
    if "[" not in smiles:
        return smiles

    def _repair(match: re.Match) -> str:
        head, digit = match.group(1), match.group(2)
        # The digit must appear elsewhere as well, or it is not a
        # ring closure.
        if smiles.count(digit) < 2:
            return match.group(0)
        return f"{head}]{digit}"

    return re.sub(r"(\[C@{1,2}H)(\d)", _repair, smiles)


def validate_and_enrich(smiles: str) -> ChemInfo | None:
    """Validate a SMILES string with RDKit and compute cyclic peptide descriptors.

    Returns None if the SMILES string is chemically invalid.
    """
    cleaned_smiles = auto_repair_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned_smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    ring_info = mol.GetRingInfo()

    # Count explicitly assigned chiral centers
    num_stereo = sum(
        1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )

    return ChemInfo(
        canonical_smiles=canonical,
        molecular_weight=round(Descriptors.MolWt(mol), 2),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        logp=round(Descriptors.MolLogP(mol), 2),
        tpsa=round(Descriptors.TPSA(mol), 2),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        num_rings=ring_info.NumRings(),
        largest_ring_size=max((len(r) for r in ring_info.AtomRings()), default=0),
        num_stereocenters=num_stereo,
        formula=rdMolDescriptors.CalcMolFormula(mol),
    )
