r"""Demo: 1036 printed-page OCR top-k + grammar check + beam search.

Reads the recovered ``demo_rapidocr_topk/topk_printed_smiles.json``,
concatenates the 13 SMILES fragments into 1036's full printed SMILES,
prints the per-char top-3 with softmax probabilities, flags chars
below 0.9 confidence, runs the grammar-aware validator, and finally
runs the beam search to attempt a repair.

Run::
    .venv\\Scripts\\python.exe -u demo_1036_ocr_topk.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpd.parsers.grammar import is_grammar_valid
from cpd.parsers.beam_search import beam_search_smiles

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

TOPK_JSON = (Path(__file__).resolve().parents[1] / "data" / "audit"
             / "demos" / "topk" / "topk_printed_smiles.json")
ENGINE = "ch_PP-OCRv4"
UNCERTAIN_THRESHOLD = 0.9


def _safe_char(ch):
    try:
        ch.encode("cp1252")
        return repr(ch)
    except UnicodeEncodeError:
        return f"U+{ord(ch):04X}"


def main():
    d = json.load(open(TOPK_JSON, encoding="utf-8"))
    lines = d.get(ENGINE) or []
    full = "".join(line.get("text", "") for line in lines)
    chars = []
    flat_pos = 0
    for line in lines:
        for c in line.get("chars", []):
            chars.append({
                "flat_pos": flat_pos,
                "page_pos": c.get("pos"),
                "char": c.get("char"),
                "top3": c.get("top3") or [],
            })
            flat_pos += 1

    print("=== 1036 printed-page OCR reconstruction ===")
    print(f"engine       : {ENGINE}")
    print(f"lines joined : {len(lines)}")
    print(f"len(full)    : {len(full)}")
    print(f"full text    : {full}")
    print()
    uncertain = [c for c in chars
                 if c["top3"] and c["top3"][0]["prob"] < UNCERTAIN_THRESHOLD]
    print(f"chars with top-1 < {UNCERTAIN_THRESHOLD}: {len(uncertain)} / {len(chars)}")
    for c in uncertain[:8]:
        top3 = c["top3"]
        top = ", ".join(f"{_safe_char(t['char'])} {t['prob']:.3f}" for t in top3)
        print(f"  flat={c['flat_pos']:>3} page={c['page_pos']:>3} "
              f"chosen={_safe_char(c['char'])} p={top3[0]['prob']:.3f}  "
              f"alts: [{top}]")

    print()
    print("=== grammar check ===")
    ok = is_grammar_valid(full)
    print(f"is_grammar_valid({full[:40]}...) = {ok}")

    print()
    print("=== RDKit parse check ===")
    mol = Chem.MolFromSmiles(full)
    if mol:
        print(f"Chem.MolFromSmiles(...) = OK, {mol.GetNumAtoms()} atoms")
    else:
        print("Chem.MolFromSmiles(...) = None (unparseable)")

    print()
    print("=== beam search repair ===")
    parts = []
    top_k_lines = []
    for line in lines:
        parts.append(line.get("text", ""))
        per_char_top = []
        for c in line.get("chars", []):
            alts = c.get("top3") or []
            if not alts:
                per_char_top.append([(c.get("char", ""), 0.0)])
                continue
            per_char_top.append([(t["char"], t["prob"]) for t in alts])
        top_k_lines.append(per_char_top)
    t0 = time.time()
    repaired, n_swaps = beam_search_smiles(parts, top_k_lines, max_2_swap=2000)
    dt = time.time() - t0
    print(f"wall time    : {dt:.2f}s")
    print(f"n_swaps tried: {n_swaps}")
    if repaired:
        ok_g = is_grammar_valid(repaired)
        mol2 = Chem.MolFromSmiles(repaired)
        print(f"repaired     : {repaired}")
        print(f"is_grammar   : {ok_g}")
        if mol2:
            print(f"is_rdkit     : OK, {mol2.GetNumAtoms()} atoms, "
                  f"largest_ring={max((len(r) for r in mol2.GetRingInfo().AtomRings()), default=0)}")
        else:
            print("is_rdkit     : None (still unparseable)")
    else:
        print("repaired     : (no valid SMILES found by beam search)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
