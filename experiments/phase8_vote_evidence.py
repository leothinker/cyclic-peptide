"""The evidence behind the vote's two guards.

Two questions are answered here, and both are answered from the
corpus rather than from a reference transcription:

1. *Which characters does a printed SMILES actually contain?*  The
   alphabet guard rejects a combination holding a character a printed
   column cannot have.  The alphabet is built from the element
   symbols -- so it holds ``l`` for ``Cl`` and ``r`` for ``Br`` --
   and this script shows why that matters: the corpus contains real
   chlorines, and the other characters it shows are OCR garbage that
   co-occurs with a non-ASCII footnote mark.

2. *What does the vote change?*  The rule is allowed to fix syntax
   and not chemistry.  This script lists every cell the vote changed,
   as a difference, and counts the cells in which a double or triple
   bond was lost.  The answer must be zero.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_vote_evidence.py
"""

from __future__ import annotations

import csv
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.vote import SMILES_ALPHABET  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT = REPO / "data" / "audit" / "phase8"
CORPUS = OUT / "pipeline_cells_route_compare.csv"

# The characters a letter may be confused with, for the census.
TOKEN = re.compile(r"Cl|Br|[BCNOPSFIbcnops]|[A-Z][a-z]?|\d")


def _parses(text: str) -> bool:
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _bonds(text: str) -> int:
    return text.count("=") + text.count("#")


def _atoms(text: str) -> Counter:
    return Counter(TOKEN.findall(text))


def _load() -> dict[str, dict[str, str]]:
    by_route: dict[str, dict[str, str]] = {}
    with CORPUS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_route.setdefault(row["route"], {})[row["cmpd_id"]] = row["text"]
    return by_route


def _alphabet_census(by_route: dict[str, dict[str, str]]) -> None:
    print("=" * 68)
    print("1. Which characters occur outside the SMILES alphabet?")
    print("=" * 68)
    counter: Counter[str] = Counter()
    holders: dict[str, list[str]] = {}
    for cid, text in by_route["validated"].items():
        for character in set(text) - SMILES_ALPHABET:
            counter[character] += 1
            holders.setdefault(character, []).append(cid)

    if not counter:
        print("  (none)")
    for character, count in counter.most_common():
        cid = holders[character][0]
        text = by_route["validated"][cid]
        index = text.index(character)
        window = text[max(0, index - 12): index + 12]
        print(
            f"  {character!r:>10}  cells={count:<4} "
            f"parses={_parses(text)!s:<5} e.g. {cid}: ...{window}..."
        )

    print()
    print("  The two-letter element symbols need their second letter,")
    print("  and the corpus contains real chlorines:")
    chlorine = [
        cid for cid, text in by_route["validated"].items() if "Cl" in text
    ]
    print(f"    cells holding 'Cl': {len(chlorine)}")
    print(f"    'l' in the alphabet: {'l' in SMILES_ALPHABET}")
    print(f"    'r' in the alphabet: {'r' in SMILES_ALPHABET}")


def _vote_changes(by_route: dict[str, dict[str, str]]) -> None:
    print()
    print("=" * 68)
    print("2. What does the vote change?")
    print("=" * 68)
    primary = by_route["validated"]
    vote = by_route["validated_vote"]
    changed = [cid for cid in primary if primary[cid] != vote[cid]]
    broke = [
        cid for cid in primary
        if _parses(primary[cid]) and not _parses(vote[cid])
    ]
    lost_bond = [
        cid for cid in changed
        if _bonds(vote[cid]) < _bonds(primary[cid])
    ]

    for cid in sorted(changed, key=int):
        before, after = primary[cid], vote[cid]
        matcher = difflib.SequenceMatcher()
        matcher.set_seqs(before, after)
        print(f"--- {cid}  parses {_parses(before)} -> {_parses(after)}")
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            print(f"     {tag:<8} {before[i1:i2]!r} -> {after[j1:j2]!r}")

    print()
    print(f"  cells changed                       : {len(changed)}")
    print(f"  of them, parsed before              : "
          f"{sum(1 for cid in changed if _parses(primary[cid]))}")
    print(f"  readings that stopped parsing       : {len(broke)}")
    print(f"  cells that lost a double/triple bond: {len(lost_bond)}")
    if lost_bond:
        print("    THE BOND GUARD FAILED:", lost_bond)

    # The atom census: what the vote changed at the level of atoms.
    print()
    print("  the atom symbols of the changed cells:")
    for cid in sorted(changed, key=int):
        before = _atoms(primary[cid])
        after = _atoms(vote[cid])
        difference = {
            key: (before.get(key, 0), after.get(key, 0))
            for key in set(before) | set(after)
            if before.get(key, 0) != after.get(key, 0)
        }
        print(f"    {cid:>6}  {difference}")


def main() -> int:
    by_route = _load()
    _alphabet_census(by_route)
    _vote_changes(by_route)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
