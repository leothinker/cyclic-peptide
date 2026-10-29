"""Check the incremental prefix validator against RDKit and the old grammar.

Three things have to agree before the repair beam can be trusted:

1. every SMILES RDKit accepts must be grammatically complete for us;
2. every string the old ``grammar.is_grammar_valid`` accepts must be
   complete for the new incremental walker (no regression);
3. the walker's verdict on the broken OCR strings must agree with the old
   one, because that is the population the beam actually operates on.

Run from the repo root::

    .venv/Scripts/python.exe analysis/validate_prefix_validator.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger

from cpd.parsers.grammar import is_grammar_valid
from cpd.parsers.prefix_validator import PrefixValidator, first_violation

RDLogger.DisableLog("rdApp.*")

CSV_PATH = ROOT / "data" / "processed" / "compounds_combined.csv"

EXTRA = [
    "CCO",
    "c1ccccc1",
    "C1CCCCC1",
    "[C@@H](C)C(=O)N",
    "Cl",
    "Br",
    "[13C]",
    "[nH]1cccc1",
    "[NH4+]",
    "[O-]C=O",
    "C%10CCCCC%10",
    "CC(C)(C)c1ccc(O)cc1",
    "[C@H]1(C)CC[C@@H](C)CC1",
    "C(F)(F)(F)",
    "C1=CC2=CC=CC=C2C=C1",
    "[2H]OC",
    "C/C=C\\C",
    "[Fe+2]",
    "CC(=O)N1CCCCC1",
]


def incremental_complete(smiles: str) -> bool:
    v = PrefixValidator()
    for ch in smiles:
        v = v.advance(ch)
        if v is None:
            return False
    return v.is_complete()


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    cols = ("smiles_decimer", "canonical_smiles_decimer", "smiles",
            "smiles_best", "smiles_candidate_repair")
    corpus: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for c in cols:
            v = (r.get(c) or "").strip()
            if v and v not in seen:
                seen.add(v)
                corpus.append(v)

    rdkit_ok = [s for s in corpus if Chem.MolFromSmiles(s) is not None]
    print(f"corpus strings          : {len(corpus)}")
    print(f"of which RDKit parses   : {len(rdkit_ok)}")
    print()

    # 1. no false negatives against RDKit
    bad = []
    for s in rdkit_ok:
        if not incremental_complete(s):
            bad.append(s)
    print(f"[1] RDKit-accepted but validator rejects : {len(bad)}")
    for s in bad[:10]:
        print(f"      at {first_violation(s)}: {s[:90]}")

    # 2. agreement with the old whole-string grammar
    g_ok = [s for s in corpus if is_grammar_valid(s)]
    both = sum(1 for s in g_ok if incremental_complete(s))
    only_new = sum(1 for s in corpus if incremental_complete(s) and
                   not is_grammar_valid(s))
    only_old = len(g_ok) - both
    print(f"[2] old grammar accepts                  : {len(g_ok)}")
    print(f"    of those, validator also accepts      : {both}")
    print(f"    old-only (validator stricter)        : {only_old}")
    print(f"    new-only (validator more lenient)    : {only_new}")
    for s in [x for x in corpus if incremental_complete(x)
              and not is_grammar_valid(x)][:6]:
        print(f"      new-only: {s[:90]}")

    # 3. edge cases the corpus does not cover
    edge_bad = [s for s in EXTRA if Chem.MolFromSmiles(s) is not None
                and not incremental_complete(s)]
    edge_neg = [s for s in EXTRA if Chem.MolFromSmiles(s) is None
                and incremental_complete(s)]
    print(f"[3] hand-written edge cases failing      : {len(edge_bad)}")
    for s in edge_bad:
        print(f"      rejects but RDKit accepts: {s}")
    print(f"    accepts but RDKit rejects (fine, syntax-only): {len(edge_neg)}")
    for s in edge_neg:
        print(f"      {s}")

    # 4. damaged population: where does the OCR break first?
    ocr = [(r["cmpd_id"], r["smiles"].strip()) for r in rows
           if (r.get("smiles") or "").strip()]
    agree = sum(1 for _, s in ocr
                if (first_violation(s) is None) == is_grammar_valid(s))
    print()
    print(f"[4] broken OCR rows: {len(ocr)}, "
          f"verdict agrees with old grammar on {agree}")
    first_bad = {}
    for cid, s in ocr:
        i = first_violation(s)
        first_bad.setdefault(i, []).append(cid)
    print("    position of first violation (top 10):")
    for i, ids in sorted(first_bad.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"      pos {i:>4}: {len(ids):>3} rows  e.g. {ids[:4]}")

    ok = not bad and not edge_bad and only_old == 0
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
