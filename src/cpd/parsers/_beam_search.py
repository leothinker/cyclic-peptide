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
    alt_ch: str          # the candidate character (or '' for delete)
    orig_ch: str         # the original character (or '' for insert)
    p_alt: float
    p_orig: float
    drop: float          # log(p_orig / p_alt) when both present;
                         # for insert/delete we treat drop = log(1/0.5)
                         # = 0.69 nats (a constant prior cost).
    op: str = "sub"      # "sub" | "ins" | "del"

    def __lt__(self, other: "_Alt") -> bool:
        return self.drop < other.drop


def _build_alts(parts: list[str], top_k_lines: list) -> list[_Alt]:
    """Flatten (parts × top_k_lines) into a single list of single-char alts."""
    full_text = "".join(parts)
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

    # Insert/delete alts at every character position. These are
    # chemistry-meaningful single-character edits the OCR is
    # likely to have missed/added: missing open paren, stray
    # close paren, broken bracket, dangling single bond.
    PRIOR_P = 0.5  # a flat prior for the new char; drop = log(0.99/0.5) ~ 0.68
    INSERT_CHARS = "(["
    DELETE_CHARS = ")-"
    for i, ch in enumerate(full_text):
        if ch in DELETE_CHARS:
            alts.append(_Alt(pos=i, alt_ch="", orig_ch=ch,
                             p_alt=PRIOR_P, p_orig=0.99,
                             drop=math.log(0.99 / PRIOR_P), op="del"))
    # for inserts we need a position between chars; flat_pos is
    # the index in the joined string, so insert BEFORE flat_pos
    # is the i-th position in ``full_text``.
    for i in range(len(full_text) + 1):
        for ch in INSERT_CHARS:
            alts.append(_Alt(pos=i, alt_ch=ch, orig_ch="",
                             p_alt=PRIOR_P, p_orig=0.99,
                             drop=math.log(0.99 / PRIOR_P), op="ins"))
    # re-sort after adding ins/del entries
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

    # Pass 1: all 1-edits (sub / ins / del) in confidence-drop order
    def _apply(a, chars):
        """Apply edit a in-place to chars list; return (cand, restore)."""
        if a.op == "sub":
            chars[a.pos] = a.alt_ch
            return "".join(chars), lambda: chars.__setitem__(a.pos, a.orig_ch)
        if a.op == "del":
            saved = chars.pop(a.pos)
            return "".join(chars), lambda: chars.insert(a.pos, saved)
        if a.op == "ins":
            chars.insert(a.pos, a.alt_ch)
            return "".join(chars), lambda: chars.pop(a.pos)
        raise ValueError(a.op)

    chars = list(full_text)
    for a in alts:
        if a.op == "sub" and a.alt_ch == a.orig_ch:
            continue
        cand, restore = _apply(a, chars)
        if is_grammar_valid(cand):
            chem = validate_and_enrich(cand)
        else:
            chem = None
        restore()
        if chem is not None:
            return chem.canonical_smiles, 1

    # Pass 2: 2-swaps, positions ordered by single-pos drop
    # Cap the search to keep runtime sane.
    seen: set[tuple[int, int, str, str]] = set()
    tried = 0
    sub_alts = [a for a in alts if a.op == "sub"]
    for i, a1 in enumerate(sub_alts):
        chars1 = list(full_text)
        chars1[a1.pos] = a1.alt_ch
        for a2 in sub_alts[i + 1:]:
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
