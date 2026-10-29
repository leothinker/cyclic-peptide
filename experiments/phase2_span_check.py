"""Print the per-line reference spans verbatim.

The two residual errors sit at ref positions 77 and 115. Confirm which
detector line owns each one and that the span boundaries used for scoring
are correct, rather than trusting the alignment helper.
"""

from __future__ import annotations

from pathlib import Path

import difflib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from phase2_recog_candidates import (  # noqa: E402
    ctc_logprob_exact,
    ref_spans_by_line,
)

meta = json.loads(
    open(
        "data/audit/phase2/recognizer_candidates/raw/"
        "1036_baseline_lines.json",
        encoding="utf-8",
    ).read()
)
official = [meta["lines"][str(i)] for i in range(len(meta["lines"]))]
joined = meta["joined"]
ref = meta["reference"]
bounds = meta["line_bounds"]

assert "".join(official) == joined
assert bounds[-1] == len(joined)

spans = ref_spans_by_line(joined, ref, bounds)

for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
    a=joined, b=ref
).get_opcodes():
    if tag == "equal":
        continue
    line = next(
        (
            li
            for li in range(len(bounds) - 1)
            if i1 >= bounds[li] or (i1 == bounds[li] and li == 0)
        ),
        None,
    )
    print(
        f"{tag:8s} joined[{i1}:{i2}]={joined[i1:i2]!r} "
        f"ref[{j1}:{j2}]={ref[j1:j2]!r} context={ref[max(0,j1-10):j2+10]!r}"
    )

print()
for li in range(len(official)):
    print(
        f"line{li}: greedy[{bounds[li]}:{bounds[li+1]}] "
        f"{joined[bounds[li]:bounds[li+1]]!r}"
    )
    print(f"        refspan  {spans[li]!r}  ({len(spans[li])} ch)")

print()
print("reconstructed from spans == reference:",
      "".join(spans.values()) == ref)

data = np.load(
    "data/audit/phase2/recognizer_candidates/posteriors/"
    "1036_baseline_posteriors.npz",
    allow_pickle=True,
)
chars = list(data["characters"])
for li in range(len(official)):
    probs = data[f"line{li}"]
    g = official[li]
    r = spans[li]
    same = g == r
    print(
        f"line{li}: identical={same} "
        f"logP_greedy={ctc_logprob_exact(probs, g, chars):.4f} "
        f"logP_refspan={ctc_logprob_exact(probs, r, chars):.4f}"
    )
    if not same:
        sm = difflib.SequenceMatcher(a=g, b=r)
        for tag, x1, x2, y1, y2 in sm.get_opcodes():
            if tag != "equal":
                print(
                    f"    {tag} greedy[{x1}:{x2}]={g[x1:x2]!r} "
                    f"-> ref[{y1}:{y2}]={r[y1:y2]!r}"
                )
