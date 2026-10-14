"""Implement a beam-search-style SMILES repair:
  1. Try the argmax (greedy).
  2. Try all 1-swaps, sorted by confidence drop (smaller drop = preferred).
     Return the highest-confidence RDKit-valid swap.
  3. If no 1-swap works, try 2-swaps (positions×alts²) within a budget.
  4. If nothing works, return the original argmax text (preserves 'smiles'
     column even when 'is_valid' is False).

This is a pragmatic beam search: at each step we keep a 'beam' of the
top-B hypotheses by confidence, but since RDKit parse is binary we just
enumerate candidates in confidence order and stop at the first valid.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from cpd.chem import validate_and_enrich
from cpd.parsers._grammar import is_grammar_valid


@dataclass
class _Alt:
    pos: int
    alt_ch: str
    orig_ch: str
    p_alt: float
    p_orig: float
    drop: float  # = -log(p_alt) - (-log(p_orig)) = log(p_orig/p_alt)

    def __lt__(self, other: "_Alt") -> bool:
        return self.drop < other.drop


def _build_alts(parts: list[str], top_k_lines: list) -> list[_Alt]:
    """Flatten (parts × top_k_lines) into a single list of single-char alts."""
    alts: list[_Alt] = []
    flat_pos = 0
    for row_text, row_alts in zip(parts, top_k_lines or []):
        row_alts = row_alts or []
        for c_i, ch in enumerate(row_text):
            if c_i < len(row_alts) and len(row_alts[c_i]) > 1:
                top1, p1 = row_alts[c_i][0]
                for alt_ch, p_alt in row_alts[c_i][1:]:
                    if alt_ch == top1:
                        continue
                    alts.append(_Alt(
                        pos=flat_pos,
                        alt_ch=alt_ch,
                        orig_ch=top1,
                        p_alt=p_alt,
                        p_orig=p1,
                        drop=math.log(max(p1, 1e-9) / max(p_alt, 1e-9)),
                    ))
            flat_pos += 1
    alts.sort()
    return alts


def beam_search_smiles(
    parts: list[str],
    top_k_lines: list,
    max_2_swap: int = 2000,
) -> tuple[Optional[str], int]:
    """Try to repair the OCR'd SMILES using beam search over top-k alts.

    Returns ``(repaired_canonical_smiles_or_None, n_swaps_used)``.
    - ``n_swaps_used`` is 0 when the argmax itself is valid, 1 for a
      successful 1-swap, 2 for a 2-swap, etc.
    - Returns ``(None, 0)`` when no repair was attempted (empty input).
    """
    full_text = "".join(parts)
    if not full_text:
        return None, 0

    # Pass 0: argmax as-is
    chem = validate_and_enrich(full_text) if is_grammar_valid(full_text) else None
    if chem is not None:
        return chem.canonical_smiles, 0

    alts = _build_alts(parts, top_k_lines)
    if not alts:
        return None, 0

    # Pass 1: all 1-swaps in confidence-drop order
    chars = list(full_text)
    for a in alts:
        if a.alt_ch == a.orig_ch:
            continue
        chars[a.pos] = a.alt_ch
        cand = "".join(chars)
        if is_grammar_valid(cand):
            chem = validate_and_enrich(cand)
        else:
            chem = None
        chars[a.pos] = a.orig_ch
        if chem is not None:
            return chem.canonical_smiles, 1

    # Pass 2: 2-swaps, positions ordered by single-pos drop
    # Cap the search to keep runtime sane.
    seen: set[tuple[int, int, str, str]] = set()
    tried = 0
    for i, a1 in enumerate(alts):
        chars1 = list(full_text)
        chars1[a1.pos] = a1.alt_ch
        for a2 in alts[i + 1:]:
            if a2.pos == a1.pos:
                continue
            key = (a1.pos, a2.pos, a1.alt_ch, a2.alt_ch)
            if key in seen:
                continue
            seen.add(key)
            chars2 = list(chars1)
            chars2[a2.pos] = a2.alt_ch
            cand = "".join(chars2)
            tried += 1
            chem = validate_and_enrich(cand) if is_grammar_valid(cand) else None
            if chem is not None:
                return chem.canonical_smiles, 2
            if tried >= max_2_swap:
                return None, 0

    return None, 0
