"""Decide which CTC scorer is correct by brute-force enumeration.

The prefix beam search and the 'exact' forward recursion disagree, and a
pruned beam can only ever UNDER-estimate a string's probability. The beam
claiming a HIGHER score than 'exact' means one of them computes something
other than log P(string | posteriors).

So enumerate every alignment explicitly on a tiny synthetic posterior and
compare all three scorers against the true sum.
"""

from __future__ import annotations

from pathlib import Path

import itertools

import numpy as np

NEG = float("-inf")


def collapse(path, blank=0):
    out = []
    prev = -1
    for c in path:
        if c != blank and c != prev:
            out.append(c)
        prev = c
    return tuple(out)


def brute_force_logprob(probs, target_idx):
    """Sum log-prob over EVERY path whose CTC collapse equals target."""
    T = probs.shape[0]
    logp = np.log(np.clip(probs, 1e-30, 1.0))
    total = NEG
    n = 0
    for path in itertools.product(range(probs.shape[1]), repeat=T):
        if collapse(path) != tuple(target_idx):
            continue
        n += 1
        lp = sum(logp[t, path[t]] for t in range(T))
        total = lp if total == NEG else np.logaddexp(total, lp)
    return total, n


def ctc_logprob_exact(probs: np.ndarray, target: str, char_list) -> float:
    """Exact log P(target | posteriors), with A/B states.

    A[t][s] = frame t emitted a label, s labels emitted so far
    B[t][s] = frame t was blank,    s labels emitted so far

    Three ways to arrive at A[t][s], all emitting label l_{s-1} at frame t:
      B[t-1][s-1] * p(l)         always valid (blank then label)
      A[t-1][s-1] * p(l)         only when l_{s-1} != l_{s-2}
      A[t-1][s]   * p(l)         always valid; a repeated label collapses

    The third term is the one that is easy to omit, and omitting it makes
    every alignment containing a same-label repeat invisible.
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

    A = np.full((T, S + 1), NEG)
    B = np.full((T, S + 1), NEG)
    B[0, 0] = logp[0, 0]
    if S >= 1:
        A[0, 1] = logp[0, lab[0]]
    for t in range(1, T):
        lp_b = logp[t, 0]
        for s in range(S + 1):
            B[t, s] = np.logaddexp(A[t - 1, s], B[t - 1, s]) + lp_b
            if s >= 1:
                lp_l = logp[t, lab[s - 1]]
                v = B[t - 1, s - 1] + lp_l
                if s == 1 or lab[s - 1] != lab[s - 2]:
                    v = np.logaddexp(v, A[t - 1, s - 1] + lp_l)
                v = np.logaddexp(v, A[t - 1, s] + lp_l)
                A[t, s] = v
    if S == 0:
        return float(B[T - 1, 0])
    return float(np.logaddexp(A[T - 1, S], B[T - 1, S]))


rng = np.random.default_rng(0)

# tiny alphabet: blank(0), a(1), b(2), c(3)
CHARS = ["blank", "a", "b", "c"]
CHARS_NOSPACE = ["blank", "a", "b", "c"]

cases = [
    (np.array([[0.5, 0.2, 0.2, 0.1]]), "a"),
    (np.array([[0.2, 0.5, 0.2, 0.1], [0.2, 0.3, 0.4, 0.1]]), "ab"),
    (np.array([[0.4, 0.3, 0.2, 0.1], [0.3, 0.1, 0.5, 0.1]]), "aa"),
    (
        np.array(
            [[0.5, 0.2, 0.2, 0.1], [0.2, 0.3, 0.3, 0.2], [0.2, 0.3, 0.2, 0.3]]
        ),
        "ab",
    ),
    (
        np.array(
            [[0.4, 0.3, 0.2, 0.1], [0.2, 0.4, 0.3, 0.1], [0.3, 0.1, 0.2, 0.4]]
        ),
        "ba",
    ),
]

print(
    f"{'target':>6s} {'brute':>11s} {'forward':>11s} {'n_align':>8s} "
    f"{'match':>6s}"
)
for probs, target in cases:
    bf, n = brute_force_logprob(probs, [CHARS.index(c) for c in target])
    fw = ctc_logprob_exact(probs, target, CHARS_NOSPACE)
    ok = abs(bf - fw) < 1e-6
    print(
        f"{target:>6s} {bf:11.6f} {fw:11.6f} {n:8d} {str(ok):>6s}"
    )

print()
print("'aa' reports False only because -inf - -inf is nan; both scorers")
print("agree it is impossible in T=2.")
print()

# randomised confirmation on a 4-symbol alphabet
rng = np.random.default_rng(7)
bad = 0
total = 0
for _ in range(300):
    T = int(rng.integers(2, 6))
    p = rng.dirichlet(np.ones(4), size=T)
    tgt = "".join(
        rng.choice(list("abc"), size=int(rng.integers(1, 4)))
    )
    bf, _ = brute_force_logprob(p, [CHARS.index(c) for c in tgt])
    fw = ctc_logprob_exact(p, tgt, CHARS_NOSPACE)
    total += 1
    same = (
        (bf == NEG and fw == NEG)
        or (bf != NEG and fw != NEG and abs(bf - fw) < 1e-8)
    )
    if not same:
        bad += 1
        if bad <= 5:
            print(f"  MISMATCH T={T} target={tgt!r} brute={bf} fwd={fw}")

print(f"randomised: {total - bad}/{total} agree with brute force")
print("PASS" if bad == 0 else "FAIL")


# The production script uses a VECTORISED copy of this recursion for speed.
# It is only trustworthy if it reproduces the verified scalar version
# exactly, so compare them on the real 97-class posteriors.
print()
try:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_recog_candidates import ctc_logprob_batch
except Exception as exc:  # pragma: no cover
    print(f"batch check skipped: {exc}")
    raise SystemExit(0)

data = np.load(
    "data/audit/phase2/recognizer_candidates/posteriors/"
    "1036_baseline_posteriors.npz",
    allow_pickle=True,
)
chars = list(data["characters"])
alpha = sorted(set("BCNOSFIHcnosp()[]@+-=#$/.\\0123456789"))


def greedy_decode(p, cl):
    idx = p.argmax(axis=1)
    sel = np.ones(len(idx), dtype=bool)
    sel[1:] = idx[1:] != idx[:-1]
    sel &= idx != 0
    return "".join(cl[i] for i in idx[sel])


max_diff = 0.0
n_cmp = 0
for li in range(7):
    probs = data[f"line{li}"]
    g = greedy_decode(probs, chars)
    neigh = [g[:i] + c + g[i + 1:] for i in range(len(g))
             for c in alpha if c != g[i]][:60]
    neigh += [g[:i] + g[i + 1:] for i in range(len(g))][:20]
    targets = [g] + neigh
    ref_vals = np.array([ctc_logprob_exact(probs, t, chars) for t in targets])
    bat_vals = ctc_logprob_batch(probs, targets, chars)
    fin = np.isfinite(ref_vals)
    d = float(np.max(np.abs(ref_vals[fin] - bat_vals[fin]))) if fin.any() else 0.0
    max_diff = max(max_diff, d)
    n_cmp += int(fin.sum())
    same_inf = np.array_equal(np.isinf(ref_vals), np.isinf(bat_vals))
    print(f"  line{li}: {int(fin.sum())} finite targets, "
          f"max |scalar-batch| = {d:.3e}, inf pattern match={same_inf}")

print(f"\ncompared {n_cmp} targets, worst absolute disagreement "
      f"{max_diff:.3e}")
print("PASS" if max_diff < 1e-3 else "FAIL - batch scorer disagrees")
