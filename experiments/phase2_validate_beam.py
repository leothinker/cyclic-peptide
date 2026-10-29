"""Validate the CTC prefix beam search against brute-force CTC forward.

The beam search keeps preferring strings with EXTRA characters (e.g.
'[C@@@H]'). Either the beam search is wrong, or those strings genuinely
carry more CTC probability than the greedy string. Both possibilities have
very different consequences, so decide it with an independent algorithm:
the exact CTC forward recursion, summed over all alignments.
"""

from __future__ import annotations

from pathlib import Path

import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_recog_candidates import ctc_prefix_beam_search  # noqa: E402

NEG = float("-inf")


def ctc_logprob_exact(probs: np.ndarray, target: str, char_list) -> float:
    """Exact log P(target | posteriors) via the CTC forward algorithm.

    alpha[t][s] = log P(emitted target[:s] | x[0..t]) with s indexing
    OUTPUT length, not path length. From state s we either emit blank and
    stay at s, or emit target[s] and move to s+1, so repeated characters
    are handled without any special case.
    """
    idx = {}
    for i, c in enumerate(char_list):
        if c in ("blank", " "):
            continue
        idx[c] = i
    if any(c not in idx for c in target):
        return NEG

    lab = [idx[c] for c in target]
    S = len(lab)
    T = probs.shape[0]
    logp = np.log(np.clip(probs, 1e-30, 1.0))

    alpha = np.full((T, S + 1), NEG)
    alpha[0, 0] = logp[0, 0]
    if S >= 1:
        alpha[0, 1] = logp[0, lab[0]]
    for t in range(1, T):
        for s in range(S + 1):
            # Graves' recursion. All three terms matter: omitting the third
            # makes consecutive equal labels unreachable, which sends any
            # string containing '))' or 'CC' to absurdly low probability.
            v = alpha[t - 1, s] + logp[t, 0]
            if s >= 1:
                v = np.logaddexp(v, alpha[t - 1, s - 1] + logp[t, lab[s - 1]])
                if s >= 2 and lab[s - 1] != lab[s - 2]:
                    v = np.logaddexp(v, alpha[t - 1, s] + logp[t, lab[s - 1]])
            alpha[t, s] = v
    if S == 0:
        return float(alpha[T - 1, 0])
    return float(np.logaddexp(alpha[T - 1, S], alpha[T - 1, S - 1]))


GREEDY = {
    0: "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCC",
    1: ")C(=O)N3[C@@H](CC3)C(=O)N(C)[C@@H]4C(",
    2: "O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)-C",
    3: "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC",
    4: "7)C(=O)N(C)[C@@H](C8CCCC8)C(=O)N(C)[C@@H](CC",
    5: "(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)=C(SC4)C(",
    6: "=C9F)C(=O)N1)C(=O)N(C)C",
}


def greedy_decode(probs, char_list):
    idx = probs.argmax(axis=1)
    sel = np.ones(len(idx), dtype=bool)
    sel[1:] = idx[1:] != idx[:-1]
    sel &= idx != 0
    return "".join(char_list[i] for i in idx[sel])

data = np.load(
    "data/audit/phase2/recognizer_candidates/posteriors/"
    "1036_baseline_posteriors.npz",
    allow_pickle=True,
)
chars = list(data["characters"])

print(
    f"{'line':>4s} {'beam_top':>12s} {'exact_beam':>12s} "
    f"{'exact_greedy':>13s} {'match':>6s} {'greedy>beam':>12s}"
)
for li in range(7):
    probs = data[f"line{li}"]
    beam = ctc_prefix_beam_search(probs, chars, beam_width=32)
    bt, bs = beam[0]
    g = greedy_decode(probs, chars)
    eb = ctc_logprob_exact(probs, bt, chars)
    eg = ctc_logprob_exact(probs, g, chars)
    ok = abs(eb - bs) < 1e-3
    print(
        f"{li:4d} {bs:12.4f} {eb:12.4f} {eg:13.4f} "
        f"{str(ok):>6s} {str(eg > eb):>12s}  greedy={g[:26]!r}"
    )

print()
print("exact_greedy > beam_top means the greedy string really is the more")
print("probable one and the beam search is wrong; the reverse means CTC")
print("likelihood itself prefers the longer strings.")
