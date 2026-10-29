"""Price each of the two residual errors under the verified exact scorer.

Both errors live on printed line 2. Score the line's greedy string, the
full reference span, and each single-fix variant, so the report can say
exactly what the recogniser's own likelihood says about each fix.
"""

from __future__ import annotations

from pathlib import Path

import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from phase2_recog_candidates import ctc_logprob_exact  # noqa: E402

meta = json.loads(
    open(
        "data/audit/phase2/recognizer_candidates/raw/"
        "1036_baseline_lines.json",
        encoding="utf-8",
    ).read()
)
official = [meta["lines"][str(i)] for i in range(len(meta["lines"]))]

data = np.load(
    "data/audit/phase2/recognizer_candidates/posteriors/"
    "1036_baseline_posteriors.npz",
    allow_pickle=True,
)
chars = list(data["characters"])
probs = data["line2"]
g = official[2]

REF = "=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C"

fix_insert = "=" + g
fix_replace = g[:37] + "=" + g[38:]
fix_both = "=" + g[:37] + "=" + g[38:]

variants = [
    ("greedy (baseline)", g),
    ("+ insert '=' at head", fix_insert),
    ("+ replace '-' with '='", fix_replace),
    ("+ both fixes (= reference)", fix_both),
]

base = ctc_logprob_exact(probs, g, chars)
print(f"line2 greedy = {g!r}")
print(f"line2 reference span = {REF!r}")
print(f"reconstructed both-fixes string == reference span: "
      f"{fix_both == REF}\n")

print(
    f"{'variant':28s} {'len':>4s} {'logP':>10s} {'d(logP)':>9s} "
    f"{'ratio':>9s}"
)
for name, s in variants:
    lp = ctc_logprob_exact(probs, s, chars)
    d = lp - base
    print(
        f"{name:28s} {len(s):4d} {lp:10.4f} {d:9.4f} "
        f"{np.exp(d):9.4f}"
    )

print()
print("A positive d(logP) means the model's own likelihood PREFERS the")
print("fix; a large negative value means the posterior actively rejects it.")
print()

# Where does the '=' sit in the posterior for each fix?
eq_i = chars.index("=")
dash_i = chars.index("-")
for t in (34, 96, 117, 118, 141):
    p = probs[t]
    o = np.argsort(-p)[:4]
    print(
        f"  t={t:3d} argmax={chars[int(p.argmax())]!r:7s} "
        f"p(=)={p[eq_i]:.4f} p(-)={p[dash_i]:.4f} "
        f"top={[(chars[i], round(float(p[i]), 3)) for i in o]}"
    )
