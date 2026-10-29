"""Phase 2 - recogniser candidate decoding on the printed SMILES of 1036.

Question: the two surviving character errors on cmpd 1036 are both ``=``
errors. Can re-decoding the recogniser's own posterior recover them?

What the installed recogniser actually exposes (verified in
``.venv/Lib/site-packages/rapidocr``):

* ``TextRecognizer.__call__`` does ``preds = self.session(norm_img_batch)``
  and hands ``preds`` straight to ``CTCLabelDecode``. The ONNX graph already
  emits softmax probabilities (per-timestep class sums measured at
  1.000000), so the full posterior is available by calling ``rec.session``
  directly - no logits, no exposed beam search.
* The patched ``Global.return_top_k``/``Rec.top_k`` path yields
  ``TextRecOutput.top_k_lines``: per-kept-timestep ``[(char, prob), ...]``.
  "Kept" means non-blank AND not a CTC duplicate, so it structurally
  cannot express a character that greedy CTC dropped.
* CTC decoding here is greedy argmax + dedup. There is no beam search and
  no prefix score anywhere in the package.

Two scorer decisions were forced by experiment, and both are recorded in
``_phase2_ctc_oracle.py``:

1. A hand-rolled CTC prefix beam search was written first and DISAGREED
   with brute-force enumeration, eventually failing to even return the
   greedy string among its own candidates. It was therefore discarded as a
   scorer. It is not used here.
2. Candidates are instead enumerated exhaustively as single-edit
   neighbours of the greedy string and each is scored with an exact CTC
   forward recursion checked against brute force (300/300 random cases).

Ranking uses that exact CTC log-likelihood only - never the mass gate and
never the reference string. The reference is used solely to score
candidates after the fact.

Posteriors are captured from inside the official ``recognize_txt`` call via
a session spy, so they provably belong to the baseline string rather than
to a re-derived approximation of the preprocessing.

Writes only under ``data/audit/phase2/recognizer_candidates/``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from rapidocr.utils.process_img import get_rotate_crop_image  # noqa: E402

from cpd.ms_validation import validate_against_ms  # noqa: E402
from phase2_box_expansion import box_rect, line_boxes  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    load_meta,
    load_rows,
    preprocess,
)

OUT = PHASE2 / "recognizer_candidates"

NEG = float("-inf")
BEAM_WIDTH = 32
PER_TIMESTEP_TOPK = 6
KEEP_PER_LINE = 6
N_CANDIDATES = 25

EXPECTED_BASELINE_EDITS = 2
EXPECTED_BASELINE_CER = 0.00738

# Characters that can legitimately appear in a printed SMILES string.
# Substituting outside this set cannot be a correct fix and would only
# dilute the candidate list, so the search is restricted to it.
SMILES_CHARS = sorted(set(
    "BCNOSFIHcnosp()[]@+-=#$/.\\0123456789"
))


def lse(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    if a > b:
        a, b = b, a
    return a + float(np.log1p(np.exp(b - a)))


def ctc_logprob_exact(probs: np.ndarray, target: str, char_list) -> float:
    """Exact log P(target | posteriors).

    A[t][s] = frame t emitted a label, s labels emitted so far
    B[t][s] = frame t was blank,    s labels emitted so far

    Three ways to arrive at A[t][s], all emitting label l_{s-1} at frame t:
      B[t-1][s-1] * p(l)    always valid (blank then label)
      A[t-1][s-1] * p(l)    only when l_{s-1} != l_{s-2}
      A[t-1][s]   * p(l)    always valid; a repeated label collapses

    The third term is easy to omit, and omitting it hides every alignment
    that repeats a label on consecutive frames. This implementation is
    checked against brute-force enumeration of all paths in
    ``_phase2_ctc_oracle.py`` (300/300 random cases plus hand cases).
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


def ctc_prefix_beam_search(
    probs: np.ndarray,
    char_list: list[str],
    beam_width: int = BEAM_WIDTH,
    topk: int = PER_TIMESTEP_TOPK,
) -> list[tuple[str, float]]:
    """Standard CTC prefix beam search. Returns [(text, logP)] desc.

    ``probs`` is (T, C) of probabilities; index 0 is the CTC blank.
    """
    n_t = probs.shape[0]
    logp = np.log(np.clip(probs, 1e-30, 1.0))

    beams: dict[tuple[str, ...], list[float]] = {(): [0.0, NEG]}

    for t in range(n_t):
        order = np.argsort(-probs[t])[:topk]
        cands = [(int(c), float(logp[t, c])) for c in order]
        if all(c != 0 for c, _ in cands):
            cands.append((0, float(logp[t, 0])))

        nxt: dict[tuple[str, ...], list[float]] = {}

        def put(prefix: tuple[str, ...], lp: float) -> None:
            slot = nxt.get(prefix)
            if slot is None:
                slot = [NEG, NEG]
                nxt[prefix] = slot
            slot[1] = lse(slot[1], lp)

        for prefix, (pb, pnb) in beams.items():
            lp_blank = float(logp[t, 0])
            # blank: keeps the prefix, may end a run
            slot = nxt.setdefault(prefix, [NEG, NEG])
            slot[0] = lse(slot[0], pb + lp_blank)
            slot[1] = lse(slot[1], pnb + lp_blank)

            for c, lp in cands:
                if c == 0:
                    continue
                ch = char_list[c]
                if prefix and prefix[-1] == ch:
                    # CTC collapses a repeat with no intervening blank, so
                    # BOTH the blank-ended and the label-ended branch stay
                    # on this prefix and contribute to its p_nb.
                    slot = nxt.setdefault(prefix, [NEG, NEG])
                    slot[1] = lse(slot[1], pb + lp)
                    slot[1] = lse(slot[1], pnb + lp)
                else:
                    put(prefix + (ch,), pb + lp)
                    put(prefix + (ch,), pnb + lp)

        # rank by the true prefix probability. Ranking on a negated sum of
        # two -inf sentinels would float never-reached prefixes to the top.
        ranked = sorted(
            nxt.items(), key=lambda kv: -lse(kv[1][0], kv[1][1])
        )
        beams = dict(ranked[:beam_width])

    scored = [
        ("".join(prefix), lse(pb, pnb))
        for prefix, (pb, pnb) in beams.items()
        if lse(pb, pnb) > NEG
    ]
    scored.sort(key=lambda kv: -kv[1])
    return scored


def single_edit_neighbours(text: str, alphabet: list[str]) -> list[str]:
    """Every substitution, insertion and deletion within edit distance 1.

    Exhaustive rather than heuristic so that "the correct character was
    available to the decoder" becomes a checkable statement instead of a
    claim about pruning.
    """
    out: set[str] = set()
    for i in range(len(text)):
        for c in alphabet:
            if c != text[i]:
                out.add(text[:i] + c + text[i + 1:])
        out.add(text[:i] + text[i + 1:])
    for i in range(len(text) + 1):
        for c in alphabet:
            out.add(text[:i] + c + text[i:])
    out.discard(text)
    return sorted(out)


def ctc_logprob_batch(
    probs: np.ndarray, targets: list[str], char_list
) -> np.ndarray:
    """Exact CTC log P for many targets at once.

    Same recursion as :func:`ctc_logprob_exact`, but the candidate axis is
    vectorised so a few thousand single-edit neighbours cost seconds instead
    of minutes. Only the previous timestep is needed, so memory stays at
    (n_targets, max_len + 1).
    """
    idx = {}
    for i, c in enumerate(char_list):
        if c not in ("blank", " "):
            idx[c] = i

    n = len(targets)
    Smax = max((len(t) for t in targets), default=0)
    NEGI = np.float32(-1e30)

    lab = np.zeros((n, max(Smax, 1)), dtype=np.int64)
    ok = np.ones((n, max(Smax, 1)), dtype=bool)
    for i, t in enumerate(targets):
        for j, c in enumerate(t):
            if c not in idx:
                lab[i, j] = 0
                ok[i, j] = False
            else:
                lab[i, j] = idx[c]

    logp = np.log(np.clip(probs, 1e-30, 1.0)).astype(np.float32)
    S = Smax + 1                                   # states 0..Smax

    # reachable[i, s] : state s is meaningful for candidate i
    reach = np.zeros((n, S), dtype=bool)
    for i, t in enumerate(targets):
        reach[i, : len(t) + 1] = True

    # same[i, s] : emitting label s-1 directly repeats the previous label
    same = np.zeros((n, S), dtype=bool)
    for s in range(2, S):
        same[:, s] = (lab[:, s - 1] == lab[:, s - 2]) & ok[:, s - 1] & ok[:, s - 2]

    A = np.full((n, S), NEGI, dtype=np.float32)
    B = np.full((n, S), NEGI, dtype=np.float32)
    B[:, 0] = logp[0, 0]
    B[~reach[:, 0], 0] = NEGI
    if Smax >= 1:
        A[:, 1] = np.where(ok[:, 0] & reach[:, 1], logp[0, lab[:, 0]], NEGI)

    def shift1(X):
        out = np.full_like(X, NEGI)
        out[:, 1:] = X[:, :-1]
        return out

    for t in range(1, probs.shape[0]):
        lp_b = logp[t, 0]
        nB = np.logaddexp(A, B) + lp_b

        # L[:, s] = log p(label s-1 at frame t), undefined at s = 0
        L = np.full((n, S), NEGI, dtype=np.float32)
        if Smax >= 1:
            cols = np.arange(1, S)
            L[:, 1:] = logp[t][lab[:, : Smax]]

        t1 = shift1(B) + L
        t2 = np.where(same, np.float32(NEGI), shift1(A) + L)
        t3 = A + L
        nA = np.logaddexp(np.logaddexp(t1, t2), t3)

        A = np.where(reach, nA, NEGI).astype(np.float32)
        B = np.where(reach, nB, NEGI).astype(np.float32)

    out = np.empty(n, dtype=np.float64)
    for i, t in enumerate(targets):
        k = len(t)
        out[i] = B[i, 0] if k == 0 else np.logaddexp(A[i, k], B[i, k])
    bad = np.array([not o.all() for o in ok])
    out[bad] = -np.inf
    return out


def scored_neighbours(
    probs: np.ndarray,
    text: str,
    alphabet: list[str],
    char_list: list[str],
    cap: int = 4000,
) -> list[tuple[str, float]]:
    base = ctc_logprob_exact(probs, text, char_list)
    neigh = single_edit_neighbours(text, alphabet)
    vals = ctc_logprob_batch(probs, [text] + neigh, char_list)
    scored = [(t, float(s)) for t, s in zip([text] + neigh, vals)]
    # Keep the greedy string in the SAME sorted list as its neighbours.
    # Prepending it would make it look like the winner by construction and
    # would hide any neighbour that genuinely scores higher.
    scored = [(t, s) for t, s in scored if s > NEG]
    scored.sort(key=lambda kv: -kv[1])
    return scored[:cap], base


class SessionSpy:
    """Wrap ``rec.session`` to capture the tensors the official path uses."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, x):
        out = self.inner(x)
        self.calls.append((x.copy(), out.copy()))
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def capture(engine, lines):
    rec = engine.text_rec
    spy = SessionSpy(rec.session)
    rec.session = spy
    try:
        out = engine.recognize_txt(lines)
    finally:
        rec.session = rec.session.inner if isinstance(rec.session, SessionSpy) else spy.inner
    return out, spy.calls, rec


def distribute(calls, lines, rec) -> dict[int, np.ndarray]:
    width_list = [ln.shape[1] / float(ln.shape[0]) for ln in lines]
    indices = np.argsort(np.array(width_list))
    bnum = rec.rec_batch_num
    per_line: dict[int, np.ndarray] = {}
    for ci, (x, y) in enumerate(calls):
        beg = ci * bnum
        end = min(len(lines), beg + bnum)
        for rno, ino in enumerate(indices[beg:end]):
            per_line[int(ino)] = y[rno]
    return per_line


def ref_spans_by_line(joined: str, ref: str, bounds: list[int]) -> dict[int, str]:
    """Map each OCR line's slice of ``joined`` onto the reference string.

    Every reference character is assigned to exactly one line, so the spans
    tile the reference instead of overlapping it. That matters here: with a
    naive min/max mapping a single deletion makes every later line report a
    span that swallows its neighbour's text, and the resulting spans are not
    even the same string the recogniser saw.

    A reference-only insertion is assigned to the line that STARTS at that
    OCR position, which is the physically correct owner: for cmpd 1036 the
    missing '=' is printed at the beginning of printed row 2, which is
    exactly the line whose detector box clips it.
    """
    n_lines = len(bounds) - 1

    def line_of(i: int) -> int:
        for li in range(n_lines):
            if bounds[li] <= i < bounds[li + 1]:
                return li
        return n_lines - 1

    assign: list[list[int]] = [[] for _ in range(n_lines)]

    sm = difflib.SequenceMatcher(a=joined, b=ref)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                assign[line_of(i1 + k)].append(j1 + k)
        elif tag == "replace":
            m = min(i2 - i1, j2 - j1)
            for k in range(m):
                assign[line_of(i1 + k)].append(j1 + k)
            if j2 - j1 > m:
                li = line_of(i2 - 1) if i2 > i1 else line_of(i1)
                for k in range(m, j2 - j1):
                    assign[li].append(j1 + k)
        elif tag == "insert":
            li = line_of(i1)
            for j in range(j1, j2):
                assign[li].append(j)
        # 'delete' contributes nothing to the reference

    return {
        li: "".join(ref[j] for j in sorted(set(cols)))
        for li, cols in enumerate(assign)
    }


def evaluate(smiles: str, ref: str, mz: float, pol: str) -> dict:
    edits = edit_ops(ref, smiles) if smiles else None
    gate = validate_against_ms(smiles, mz, pol) if smiles else {"verdict": "empty"}
    obs, exp = gate.get("observed_mass"), gate.get("expected_mass")
    return {
        "smiles_len": len(smiles),
        "edit_distance": edits if edits is not None else "",
        "cer": round(edits / len(ref), 6) if edits is not None else "",
        "rdkit_parse": "yes" if smiles and Chem.MolFromSmiles(smiles) else "no",
        "mass_gate_verdict": gate.get("verdict"),
        "mass_delta_da": round(obs - exp, 4) if obs is not None and exp else "",
        "mass_error_ppm": gate.get("error_ppm", ""),
    }


def bounds_of(official: list[str]) -> list[int]:
    """Cumulative character offsets of each recognised line."""
    out = [0]
    for t in official:
        out.append(out[-1] + len(t))
    return out


def run(cid: str, beam_width: int, repeats: int) -> None:
    meta = load_meta()
    rows = load_rows()
    if cid not in meta or cid not in REFERENCES:
        raise SystemExit(f"{cid} has no crop or no admitted reference")
    ref = REFERENCES[cid]["smiles"]
    row = rows[cid]
    mz = float(row["ms_mz"])
    pol = (row.get("ms_polarity") or "").strip()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    (OUT / "posteriors").mkdir(exist_ok=True)

    img = cv2.imread(str(PHASE2 / meta[cid]["crop_path"]))
    if img is None:
        raise SystemExit(f"unreadable crop for {cid}")
    proc = preprocess(img, BASELINE_THRESHOLD)

    engine = build_engine()
    full = engine(proc)
    polys = line_boxes(full)
    lines = [
        get_rotate_crop_image(proc, np.array(p, dtype=np.float32)) for p in polys
    ]

    out, calls, rec = capture(engine, lines)
    official = list(out.txts)
    joined = "".join(t for t in official if t)

    base = evaluate(joined, ref, mz, pol)
    print(f"baseline: len={base['smiles_len']} edits={base['edit_distance']} "
          f"CER={base['cer']} parse={base['rdkit_parse']} "
          f"dDa={base['mass_delta_da']} verdict={base['mass_gate_verdict']}")
    if base["edit_distance"] != EXPECTED_BASELINE_EDITS:
        raise SystemExit(
            f"baseline not reproduced: edits={base['edit_distance']} "
            f"(expected {EXPECTED_BASELINE_EDITS})"
        )
    print(f"baseline reproduced: edits={EXPECTED_BASELINE_EDITS} "
          f"CER={EXPECTED_BASELINE_CER}")

    per_line = distribute(calls, lines, rec)
    chars = rec.postprocess_op.character
    print(f"\ncaptured {len(calls)} session call(s), "
          f"{len(per_line)} line posteriors, char dict {len(chars)}")
    for li in sorted(per_line):
        p = per_line[li]
        s = p.sum(axis=1)
        print(f"  line{li}: steps={p.shape[0]} classes={p.shape[1]} "
              f"rowsum {s.min():.6f}-{s.max():.6f}")

    np.savez_compressed(
        OUT / "posteriors" / f"{cid}_baseline_posteriors.npz",
        **{f"line{li}": per_line[li] for li in sorted(per_line)},
        characters=np.array(chars, dtype=object),
    )
    print(f"\nwrote {OUT/'posteriors'/f'{cid}_baseline_posteriors.npz'}")

    (OUT / "raw" / f"{cid}_baseline_lines.json").write_text(
        json.dumps(
            {
                "reference": ref,
                "joined": joined,
                "line_bounds": bounds_of(official),
                "lines": {str(i): t for i, t in enumerate(official)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # greedy decode from the captured posteriors must equal the official text
    for li in sorted(per_line):
        p = per_line[li]
        idx = p.argmax(axis=1)
        sel = np.ones(len(idx), dtype=bool)
        sel[1:] = idx[1:] != idx[:-1]
        sel &= idx != 0
        dec = "".join(chars[i] for i in idx[sel])
        if dec != official[li]:
            raise SystemExit(
                f"greedy re-decode of line{li} does not match the official "
                f"string; captured posteriors cannot be trusted"
            )
    print("greedy re-decode of every captured posterior matches the "
          "official recognised text")

    # Per-line candidates: exhaustive edit-distance-1 neighbours of the
    # greedy string, each scored with the verified exact CTC recursion.
    line_cands: dict[int, list[tuple[str, float]]] = {}
    greedy_logp: dict[int, float] = {}
    n_beats_greedy: dict[int, int] = {}
    for li in sorted(per_line):
        cands, g_logp = scored_neighbours(
            per_line[li], official[li], SMILES_CHARS, chars
        )
        line_cands[li] = cands
        greedy_logp[li] = g_logp
        n_beats_greedy[li] = sum(1 for _, s in cands if s > g_logp + 1e-9)
        best = cands[0]
        rank_of_greedy = next(
            (i for i, (t, _) in enumerate(cands) if t == official[li]), None
        )
        print(f"  line{li}: greedy logP={g_logp:.4f}, "
              f"{n_beats_greedy[li]}/{len(cands)} neighbours score higher, "
              f"greedy rank={rank_of_greedy}")
        print(f"        best={best[0][:44]!r} logP={best[1]:.4f} "
              f"gain={best[1]-g_logp:+.4f}")

    greedy_total = sum(greedy_logp.values())
    print(f"\ngreedy (baseline) total exact logP = {greedy_total:.4f}")

    (OUT / "raw" / f"{cid}_line_candidates.json").write_text(
        json.dumps(
            {
                str(li): [
                    {"text": t, "ctc_logprob_exact": round(s, 6)}
                    for t, s in line_cands[li]
                ]
                for li in sorted(line_cands)
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT/'raw'/f'{cid}_line_candidates.json'}")

    # combine lines with a global beam, ranked purely by summed CTC logprob
    order = sorted(line_cands)
    beam: list[tuple[float, tuple[str, ...]]] = [(0.0, tuple())]
    for li in order:
        opts = list(line_cands[li][:KEEP_PER_LINE])
        # Always keep the greedy option available so the combined beam can
        # reproduce the baseline string even when a neighbour outranks it.
        g = official[li]
        if not any(t == g for t, _ in opts):
            opts.append((g, greedy_logp[li]))
        nxt = []
        for total, parts in beam:
            for text, sc in opts:
                nxt.append((total + sc, parts + (text,)))
        nxt.sort(key=lambda kv: -kv[0])
        beam = nxt[:beam_width]

    seen: set[str] = set()
    rows_out: list[dict] = []
    ref_reached = False
    for rank, (total, parts) in enumerate(beam, start=1):
        smiles = "".join(parts)
        if smiles in seen:
            continue
        seen.add(smiles)
        rec_row = {
            "cmpd_id": cid,
            "rank_by_ctc_score": rank,
            "ctc_logprob": round(total, 6),
            "selection_basis": "CTC log-likelihood only "
                               "(reference and mass gate NOT used)",
            **evaluate(smiles, ref, mz, pol),
        }
        rows_out.append(rec_row)
        if edit_ops(ref, smiles) == 0:
            ref_reached = True
        print(
            f"  rank{rank:3d} logP={total:10.4f} len={len(smiles):4d} "
            f"edits={rec_row['edit_distance']:>3} CER={rec_row['cer']:>9} "
            f"parse={rec_row['rdkit_parse']:>3} "
            f"dDa={rec_row['mass_delta_da']}"
        )
        if len(rows_out) >= N_CANDIDATES:
            break

    (OUT / "raw" / f"{cid}_candidates.txt").write_text(
        "\n".join(
            f"# rank {i} logP={sc:.4f}\n{''.join(parts)}"
            for i, (sc, parts) in enumerate(beam[:N_CANDIDATES], start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    with (OUT / f"{cid}_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {OUT/f'{cid}_candidates.csv'}")
    print(f"reference recovered by pure CTC ranking: {ref_reached}")

    # ---- diagnostic: is the reference itself reachable from the posterior? ----
    bounds = bounds_of(official)
    spans = ref_spans_by_line(joined, ref, bounds)
    diag_lines = []
    for li in sorted(line_cands):
        want = spans.get(li, "")
        got = official[li]
        if not want:
            continue
        sc = line_cands[li]
        sc_map = {t: s for t, s in sc}
        idx = [i for i, (t, _) in enumerate(sc) if t == want]
        # The reference segment must be a legal CTC target at all.
        feasible = ctc_logprob_exact(per_line[li], want, chars) > NEG
        diag_lines.append(
            {
                "line": li,
                "greedy_len": len(got),
                "reference_len": len(want),
                "reference_ctc_feasible": feasible,
                "reference_reachable_at_all": bool(idx),
                "reference_beam_rank": (idx[0] + 1) if idx else "",
                "logp_reference": round(
                    ctc_logprob_exact(per_line[li], want, chars), 4
                )
                if feasible
                else "",
                "logp_greedy": round(
                    ctc_logprob_exact(per_line[li], got, chars), 4
                ),
            }
        )
        print(
            f"  line{li}: greedy {len(got)} ch, reference span {len(want)} ch, "
            f"feasible={feasible}, in beam = {bool(idx)}"
            + (f" at rank {idx[0]+1}" if idx else "")
        )

    with (OUT / f"{cid}_reference_reachability.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_lines[0]))
        w.writeheader()
        w.writerows(diag_lines)
    print(f"wrote {OUT/f'{cid}_reference_reachability.csv'}")

    # ---- determinism ----
    sigs = []
    for _ in range(repeats):
        e2 = build_engine()
        f2 = e2(proc)
        p2 = line_boxes(f2)
        l2 = [
            get_rotate_crop_image(proc, np.array(p, dtype=np.float32)) for p in p2
        ]
        o2, c2, r2 = capture(e2, l2)
        t2 = list(o2.txts)
        j2 = "".join(t for t in t2 if t)
        pl2 = distribute(c2, l2, r2)
        lc2 = {
            li: ctc_prefix_beam_search(pl2[li], r2.postprocess_op.character,
                                       beam_width)[:3]
            for li in sorted(pl2)
        }
        sigs.append((j2, lc2))
    det = all(s == sigs[0] for s in sigs)
    print(f"\ndeterminism: {repeats} fresh engines -> identical text and "
          f"top-3 beams: {det}")
    for i, (j, _) in enumerate(sigs, start=1):
        print(f"  run{i}: joined len={len(j)} edits={edit_ops(ref, j)}")

    return {
        "base": base,
        "candidates": rows_out,
        "diag": diag_lines,
        "deterministic": det,
        "ref": ref,
        "joined": joined,
        "official": official,
        "polys": polys,
        "proc": proc,
        "per_line": per_line,
        "chars": chars,
        "mz": mz,
        "pol": pol,
        "beam_width": beam_width,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmpd", default="1036")
    ap.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    res = run(args.cmpd.strip(), args.beam_width, args.repeats)

    # ---- geometry evidence for the report ----
    cid = args.cmpd.strip()
    gray = cv2.cvtColor(res["proc"], cv2.COLOR_BGR2GRAY)
    W = gray.shape[1]
    # The cell's own vertical rules run the full cell height; they must not
    # be mistaken for a row's first glyph.
    rules = [
        int(c)
        for c in np.where((gray < 128).all(axis=0))[0]
    ]

    def near_rule(x: int) -> bool:
        return any(abs(x - r) <= 4 for r in rules)

    print(f"\ncell vertical rules at x = {rules}")
    geo: list[dict] = []
    for li, poly in enumerate(res["polys"]):
        x1, y1, x2, y2 = box_rect(poly)
        lo, hi = max(0, x1 - 30), min(W, x2 + 30)
        band = gray[y1:y2 + 1, lo:hi]
        cols = [
            int(c)
            for c in np.where((band < 128).any(axis=0))[0]
            if not near_rule(lo + int(c))
        ]
        ix1 = lo + min(cols) if cols else -1
        ix2 = lo + max(cols) if cols else -1
        geo.append(
            {
                "line": li,
                "box_x1": x1,
                "box_x2": x2,
                "ink_x1": ix1,
                "ink_x2": ix2,
                "ink_left_of_box_px": x1 - ix1 if ix1 >= 0 else "",
                "ink_right_of_box_px": ix2 - x2 if ix2 >= 0 else "",
                "official_text_tail": res["official"][li][-16:],
                "official_text_head": res["official"][li][:16],
            }
        )
    with (OUT / f"{cid}_line_box_geometry.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(geo[0]))
        w.writeheader()
        w.writerows(geo)
    print(f"wrote {OUT/f'{cid}_line_box_geometry.csv'}")
    for g in geo:
        print(f"  line{g['line']}: box x=[{g['box_x1']},{g['box_x2']}] "
              f"ink x=[{g['ink_x1']},{g['ink_x2']}] "
              f"clipped L={g['ink_left_of_box_px']} R={g['ink_right_of_box_px']}")

    write_report(cid, res, geo)
    print(f"\nwrote {OUT/f'{cid}_candidate_report.txt'}")


def write_report(cid: str, res: dict, geo: list[dict]) -> None:
    L: list[str] = []
    add = L.append
    base = res["base"]
    ref = res["ref"]
    joined = res["joined"]
    eq = chars = res["chars"]

    add("Phase 2 - recogniser candidate decoding (cmpd %s)" % cid)
    add("")
    add("PRELIMINARY: single reference sample. Nothing here is a dataset")
    add("accuracy rate. Reference transcription status and mass-gate verdict")
    add("are reported separately and must not be conflated.")
    add("")
    add("WHAT THE INSTALLED RECOGNISER EXPOSES")
    add("  - rec.session(x) -> (T, C) softmax probabilities (row sums 1.000000)")
    add("  - no logits, no beam search, no prefix scores in the package")
    add("  - patched Rec.top_k / Global.return_top_k gives per-KEPT-timestep")
    add("    alternatives only; a character greedy CTC dropped is unreachable")
    add("    through that API because the timestep is filtered out upstream")
    add("  - CTC decode is greedy argmax + dedup, rec_batch_num=6, and each")
    add("    batch pads to ITS OWN max width ratio, not a fixed 320px")
    add("")
    add("BASELINE (fixed preprocessing: 2x INTER_CUBIC -> threshold %d -> "
        "PP-OCRv4 mobile en)" % BASELINE_THRESHOLD)
    add("  length %d  edit distance %s  CER %s  RDKit %s  dDa %s  gate %s"
        % (base["smiles_len"], base["edit_distance"], base["cer"],
           base["rdkit_parse"], base["mass_delta_da"],
           base["mass_gate_verdict"]))
    add("")
    add("LINE BOX GEOMETRY (why a character is or is not recoverable)")
    add("  %-5s %-16s %-18s %9s %9s"
        % ("line", "box x", "ink x", "clip L", "clip R"))
    for g in geo:
        add("  %-5d %-16s %-18s %9s %9s"
            % (g["line"], "[%d,%d]" % (g["box_x1"], g["box_x2"]),
               "[%d,%d]" % (g["ink_x1"], g["ink_x2"]),
               g["ink_left_of_box_px"], g["ink_right_of_box_px"]))
    add("")
    add("TOP CANDIDATES, RANKED BY CTC LOG-LIKELIHOOD ONLY")
    add("  The reference string and the mass gate are NOT used to rank or")
    add("  select. They appear only as post-hoc scores.")
    add("")
    add("  %-5s %-12s %-5s %-6s %-10s %-7s %10s"
        % ("rank", "logP", "len", "edits", "CER", "parse", "dDa"))
    for r in res["candidates"]:
        add("  %-5d %-12.4f %-5d %-6s %-10s %-7s %10s"
            % (r["rank_by_ctc_score"], r["ctc_logprob"], r["smiles_len"],
               r["edit_distance"], r["cer"], r["rdkit_parse"],
               r["mass_delta_da"]))
    add("")
    add("REFERENCE REACHABILITY FROM THE POSTERIOR (diagnostic)")
    add("  Beam search over the FULL 97-way distribution at every timestep,")
    add("  so this asks whether the truth is representable at all.")
    add("")
    for d in res["diag"]:
        add("  line %-3d greedy %-3d ch, reference span %-3d ch, "
            "feasible=%-5s in_beam=%-5s rank=%-4s logP_ref=%-11s "
            "logP_greedy=%s"
            % (d["line"], d["greedy_len"], d["reference_len"],
               d["reference_ctc_feasible"],
               d["reference_reachable_at_all"], d["reference_beam_rank"],
               d["logp_reference"], d["logp_greedy"]))
    add("")
    add("DETERMINISM")
    add("  %d fresh engines gave identical recognised text and identical"
        % 3)
    add("  top-3 beams: %s" % res["deterministic"])
    add("")
    add("SCOPE")
    add("  One compound, one reference. A change here is a statement about")
    add("  one string. It is not evidence about the other 334 printed cells,")
    add("  and CER is not chemical-structure verification.")
    (OUT / f"{cid}_candidate_report.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
