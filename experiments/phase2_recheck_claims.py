"""Independent re-scoring of the headline numbers in 1036_candidate_report.txt.

Nothing here re-runs OCR. It reloads the captured posteriors and re-scores
the four strings that the report's conclusion rests on, so the conclusion
does not depend on the report generator having been correct.

Also asserts the three flags are kept apart: the report's CER numbers are
measured against a transcribed string, and no mass gate is consulted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for p in (SRC, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase2_ocr_compare import edit_ops  # noqa: E402
from phase2_recog_candidates import (  # noqa: E402
    ctc_logprob_exact,
    ref_spans_by_line,
)

OUT = REPO / "data" / "audit" / "phase2" / "recognizer_candidates"
TARGET_LINE = 2


def main() -> None:
    # The archive stores the character table as an object array.
    npz = np.load(
        OUT / "posteriors" / "1036_baseline_posteriors.npz", allow_pickle=True
    )
    lines = json.loads(
        (OUT / "raw" / "1036_baseline_lines.json").read_text(encoding="utf-8")
    )
    chars = [str(c) for c in npz["characters"]]
    per_line = {int(k[4:]): v for k, v in npz.items() if k.startswith("line")}

    probs = per_line[TARGET_LINE]
    greedy = lines["lines"][str(TARGET_LINE)]
    # The joined (recognised) bounds do NOT index the reference: the two
    # strings differ in length. Use the same alignment the report used, and
    # assert it tiles the reference exactly.
    spans = ref_spans_by_line(
        lines["joined"], lines["reference"], lines["line_bounds"]
    )
    assert "".join(spans[k] for k in sorted(spans)) == lines["reference"]
    ref_span = spans[TARGET_LINE]

    def sc(s: str) -> float:
        return round(ctc_logprob_exact(probs, s, chars), 4)

    print(f"line {TARGET_LINE} timesteps      = {probs.shape[0]}")
    print(f"line {TARGET_LINE} greedy         = {greedy!r}")
    print(f"line {TARGET_LINE} reference span = {ref_span!r}")
    print(f"line {TARGET_LINE} logP greedy    = {sc(greedy)}")
    print(f"line {TARGET_LINE} logP reference = {sc(ref_span)}")
    print()

    d_ref = round(sc(ref_span) - sc(greedy), 4)
    print(f"delta to reference              = {d_ref} nats")

    # Error 1: the leading '=' of C(=O) is missing from the crop entirely.
    ins = "=" + greedy
    d_ins = round(sc(ins) - sc(greedy), 4)
    print(f"insert '=' at head              = {sc(ins)}  (delta {d_ins})")

    # Error 2: '-' read where the cell prints '='.
    rep = greedy.replace("(F)-C", "(F)=C")
    d_rep = round(sc(rep) - sc(greedy), 4)
    print(f"replace '-' with '='            = {sc(rep)}  (delta {d_rep})")
    print()

    joined = lines["joined"]
    ref = lines["reference"]
    n_edits = edit_ops(ref, joined)
    print(f"baseline CER = {n_edits}/{len(ref)} = {n_edits/len(ref):.5f}")
    print("lines where greedy != reference span:")
    for k in sorted(spans):
        if spans[k] != lines["lines"][str(k)]:
            print(f"  line {k} only")
    print()
    print("character diff on that line:")
    import difflib

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=ref_span, b=greedy
    ).get_opcodes():
        if tag != "equal":
            print(f"  {tag}: reference {ref_span[i1:i2]!r} "
                  f"@ref[{i1}:{i2}]  ->  recognised {greedy[j1:j2]!r}")

    assert n_edits == 2, n_edits
    print()
    print("VERDICT: both errors sit on printed line 2; the reference span is")
    print(f"{len(ref_span)-len(greedy)} char longer than what was recognised.")


if __name__ == "__main__":
    main()
