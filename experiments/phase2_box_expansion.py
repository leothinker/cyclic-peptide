"""Phase 2: right-edge detector-box expansion on printed-SMILES cells.

Hypothesis under test: the two surviving character errors on cmpd 1036
both sit at the *right edge of a detector line box*, where the final
``=`` is clipped. If the box is widened before recognition, those
characters should come back.

Two rules keep the experiment honest:

1. Every crop handed to the recogniser is re-extracted from the *full*
   processed cell image, never padded on top of an already-clipped box.
2. Expansion is clamped to the SMILES cell's own right boundary, so no
   neighbouring column can leak in.

The baseline is reproduced first and asserted (CER 0.00738, edit
distance 2) before any expansion is measured, and the winner is re-run
three times to confirm determinism.

Writes only under ``data/audit/phase2/box_expansion/``.
"""

from __future__ import annotations

import argparse
import csv
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
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    PHASE2,
    build_engine,
    diff_ops,
    load_meta,
    load_rows,
    preprocess,
)

OUT = PHASE2 / "box_expansion"

# Pads are in *processed* image pixels (the space the detector reports).
# The processed image is 2x the source crop, so a pad of P here equals
# P/2 source pixels.
PADS = (4, 8, 12)
WHITE_MARGIN = 8
INK_PAD = 4
UPSCALE = 2


def load_meta() -> dict[str, dict]:
    from phase2_threshold_scan import META

    with META.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {
        r["cmpd_id"]: r
        for r in rows
        if r["extraction_status"] == "ok" and r["crop_path"]
    }


def line_boxes(result) -> list[np.ndarray]:
    """Detector boxes sorted top-to-bottom, left-to-right."""
    boxes = np.asarray(result.boxes, dtype=float)
    order = sorted(
        range(len(boxes)),
        key=lambda i: (float(boxes[i][:, 1].min()), float(boxes[i][:, 0].min())),
    )
    return [boxes[i] for i in order]


def box_rect(box: np.ndarray) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def ink_extent_x(proc: np.ndarray, y1: int, y2: int) -> int | None:
    """Right-most inked column inside a horizontal band."""
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    band = gray[max(0, y1):y2, :]
    if band.size == 0:
        return None
    cols = np.where((band < 128).sum(axis=0) > 0)[0]
    return int(cols.max()) if len(cols) else None


def extract_line(
    proc: np.ndarray,
    poly: np.ndarray,
    pad_right: int,
    cell_right: int,
    white_right: int = 0,
) -> np.ndarray:
    """Re-extract one text line from the full processed cell image.

    Uses the *same* perspective rectification as the official pipeline
    (``get_rotate_crop_image``). An axis-aligned rectangle would drop the
    rectification on this slightly skewed table text and change the
    recognised string on its own, which would confound the experiment:
    only the right edge may vary here.

    The right edge is the detector edge plus ``pad_right``, clamped to
    ``cell_right`` so no neighbouring column can leak in. With
    ``white_right`` an extra white margin is appended afterwards.
    """
    pts = np.array(poly, dtype=np.float32).copy()
    for i in (1, 2):
        pts[i][0] = min(float(cell_right), float(pts[i][0]) + pad_right)
    # Keep the polygon valid: both right-hand corners share the new x.
    if pts[1][0] < pts[0][0]:
        pts[1][0] = pts[0][0]
    if pts[2][0] < pts[3][0]:
        pts[2][0] = pts[3][0]
    line = get_rotate_crop_image(proc, pts)
    if white_right:
        line = cv2.copyMakeBorder(
            line, 0, 0, 0, white_right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return line


def rec_only(engine, lines: list[np.ndarray]) -> str:
    """Recognise rectified line crops the same way the pipeline does.

    ``engine.recognize_txt([...])`` is the exact entry point the official
    det+rec path uses, so it is the only one that reproduces the baseline
    string. Calling ``engine(img)`` with ``use_det=False`` looks
    equivalent but re-runs the image preprocessor and silently returns a
    different string (e.g. ``[C@@H]`` -> ``N|C@@H]``), which would make
    every expansion config look worse for a reason unrelated to the
    experiment.
    """
    if not lines:
        return ""
    out = engine.recognize_txt(lines)
    txts = out.txts
    if isinstance(txts, str):
        return txts
    return "".join(t for t in txts if t)


def run_config(
    engine,
    proc: np.ndarray,
    polys: list[np.ndarray],
    cell_right: int,
    pad_right: int,
    white_right: int = 0,
    ink_pad: int = 0,
) -> tuple[str, list[np.ndarray]]:
    lines: list[np.ndarray] = []
    for poly in polys:
        p = poly
        if ink_pad:
            x1, y1, x2, y2 = box_rect(poly)
            hit = ink_extent_x(proc, y1 - 4, y2 + 4)
            if hit is not None:
                new_x2 = min(cell_right, max(x2, hit + ink_pad))
                p = np.array(poly, dtype=np.float32).copy()
                p[1][0] = p[2][0] = float(new_x2)
        line = extract_line(proc, p, pad_right, cell_right, white_right)
        lines.append(line)
    return rec_only(engine, lines), lines


def save_overlay(
    proc: np.ndarray,
    polys: list[np.ndarray],
    pad_right: int,
    cell_right: int,
    out_png: Path,
) -> None:
    vis = proc.copy()
    for poly in polys:
        x1, y1, x2, y2 = box_rect(poly)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        xr = min(cell_right, x2 + pad_right)
        cv2.rectangle(vis, (x1, y1), (xr, y2), (0, 200, 0), 3)
    cv2.imwrite(str(out_png), vis)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmpd", default="1036")
    args = ap.parse_args()
    cid = args.cmpd.strip()

    meta = load_meta()
    rows = load_rows()
    if cid not in meta or cid not in REFERENCES:
        raise SystemExit(f"{cid} has no crop or no admitted reference")
    ref = REFERENCES[cid]["smiles"]
    row = rows[cid]
    mz = float(row["ms_mz"])
    pol = (row.get("ms_polarity") or "").strip()

    OUT.mkdir(parents=True, exist_ok=True)
    vis_dir = OUT / "visuals"
    raw_dir = OUT / "raw_ocr"
    vis_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    crop_path = PHASE2 / meta[cid]["crop_path"]
    img = cv2.imread(str(crop_path))
    if img is None:
        raise SystemExit(f"unreadable crop {crop_path}")
    proc = preprocess(img, BASELINE_THRESHOLD)
    proc_h, proc_w = proc.shape[:2]
    cell_right = proc_w
    print(f"{cid}: crop={img.shape[1]}x{img.shape[0]} "
          f"processed={proc_w}x{proc_h} cell_right={cell_right}")

    engine = build_engine()
    full = engine(proc)
    det_txts = "".join(t for t in full.txts if t)
    polys = line_boxes(full)

    configs: list[tuple[str, int, int, int, str]] = [
        ("baseline_detrec", 0, 0, 0, "detector boxes, det+rec (unchanged)"),
        ("baseline_reconly", 0, 0, 0, "detector boxes, rec-only"),
    ]
    for p in PADS:
        configs.append(
            (f"pad{p}_reconly", p, 0, 0, f"box right +{p}px, rec-only")
        )
    configs.append(
        (
            "recover_white8_reconly",
            0,
            WHITE_MARGIN,
            INK_PAD,
            "extend box to full ink (+4px), then 8px white margin, rec-only",
        )
    )

    results: list[dict] = []
    texts: dict[str, str] = {}

    for name, pad, white, inkp, desc in configs:
        if name == "baseline_detrec":
            raw = det_txts
            lines = []
        else:
            raw, lines = run_config(
                engine, proc, polys, cell_right,
                pad, white, inkp,
            )
        texts[name] = raw
        edits = edit_ops(ref, raw) if raw else None
        cer = round(edits / len(ref), 6) if edits is not None else None
        gate = validate_against_ms(raw, mz, pol) if raw else {"verdict": "empty"}
        obs, exp = gate.get("observed_mass"), gate.get("expected_mass")
        delta = round(obs - exp, 4) if obs is not None and exp else None
        ops = diff_ops(ref, raw) if raw else []

        (raw_dir / f"{cid}__{name}.txt").write_text(raw + "\n", encoding="utf-8")
        (raw_dir / f"{cid}__{name}_diff.json").write_text(
            json.dumps(ops, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if name != "baseline_detrec":
            save_overlay(
                proc, polys, pad, cell_right,
                vis_dir / f"{cid}__{name}_boxes.png",
            )
            if lines:
                width = max(line.shape[1] for line in lines)
                padded = [
                    cv2.copyMakeBorder(
                        line, 0, 0, 0, width - line.shape[1],
                        cv2.BORDER_CONSTANT, value=(255, 255, 255),
                    )
                    for line in lines
                ]
                strip = np.concatenate(padded, axis=0)
                cv2.imwrite(str(vis_dir / f"{cid}__{name}_lines.png"), strip)

        results.append(
            {
                "cmpd_id": cid,
                "config": name,
                "description": desc,
                "pad_right_px": pad,
                "pad_source_px": pad / UPSCALE,
                "white_margin_px": white,
                "ink_pad_px": inkp,
                "mode": "det+rec" if name == "baseline_detrec" else "rec-only",
                "reference_len": len(ref),
                "raw_len": len(raw),
                "edit_distance": edits if edits is not None else "",
                "cer": cer if cer is not None else "",
                "n_diff_ops": len(ops),
                "n_ops_in_carbonyl": sum(1 for o in ops if o["in_carbonyl"]),
                "rdkit_parse": "yes" if raw and Chem.MolFromSmiles(raw) else "no",
                "ms_mz": mz,
                "ms_polarity": pol,
                "mass_gate_verdict": gate.get("verdict"),
                "mass_delta_da": delta if delta is not None else "",
                "raw_ocr_file": f"raw_ocr/{cid}__{name}.txt",
                "diff_file": f"raw_ocr/{cid}__{name}_diff.json",
            }
        )
        print(
            f"{name:24s} len={len(raw):4d} edits={str(edits):>4s} "
            f"CER={str(cer):>9s} parse={results[-1]['rdkit_parse']:>3s} "
            f"dDa={delta}"
        )

    base = next(r for r in results if r["config"] == "baseline_detrec")
    assert base["edit_distance"] == 2, (
        f"baseline not reproduced: edit_distance={base['edit_distance']} "
        f"(expected 2), CER={base['cer']} (expected 0.00738)"
    )
    print("baseline reproduced: CER=0.00738 edit_distance=2")

    reconly = next(
        r for r in results if r["config"] == "baseline_reconly"
    )
    if texts["baseline_reconly"] != texts["baseline_detrec"]:
        raise SystemExit(
            "rec-only path does not reproduce the det+rec baseline string; "
            "expansion results would not be comparable"
        )
    print("rec-only path reproduces the det+rec baseline string exactly")

    with (OUT / "box_expansion.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    # determinism: re-run the best-scoring expansion config three times
    cands = [r for r in results if r["config"] != "baseline_detrec"]
    best = min(cands, key=lambda r: (r["edit_distance"] is None,
                                     r["edit_distance"] or 10**9))
    cfg = next(c for c in configs if c[0] == best["config"])
    repeats = []
    for _ in range(3):
        raw, _ = run_config(
            engine, proc, polys, cell_right, cfg[1], cfg[2], cfg[3]
        )
        repeats.append(edit_ops(ref, raw) if raw else None)
    det = len(set(repeats)) == 1
    print(
        f"determinism: best config {best['config']} repeated 3x -> "
        f"edit distances {repeats}, deterministic={det}"
    )

    write_report(cid, ref, results, best, repeats, det, polys, proc)
    print(f"\nwrote {OUT/'box_expansion.csv'}")
    print(f"wrote {OUT/'box_expansion_report.txt'}")


def write_report(
    cid: str,
    ref: str,
    results: list[dict],
    best: dict,
    repeats: list,
    deterministic: bool,
    polys: list,
    proc: np.ndarray,
) -> None:
    L: list[str] = []
    add = L.append
    add("Phase 2 - right-edge detector-box expansion")
    add("")
    add(f"PRELIMINARY: single reference sample (cmpd {cid}). Any improvement")
    add("here is a statement about one string, not a dataset accuracy rate.")
    add("")
    add(f"crop / preprocess: 2x INTER_CUBIC -> threshold "
        f"{BASELINE_THRESHOLD} -> PP-OCRv4 mobile en")
    add(f"detected lines: {len(polys)}   processed width: {proc.shape[1]}")
    add("")
    add("Each crop is re-extracted from the FULL processed cell image; no")
    add("config pads whitespace onto an already-clipped box. Expansion is")
    add("clamped to the cell's own right boundary.")
    add("")
    add(
        f"{'config':24s} {'pad':>4s} {'mode':>9s} {'len':>5s} {'edits':>6s} "
        f"{'CER':>9s} {'parse':>6s} {'dDa':>9s}"
    )
    for r in results:
        add(
            f"{r['config']:24s} {r['pad_right_px']:4d} {r['mode']:>9s} "
            f"{r['raw_len']:5d} {str(r['edit_distance']):>6s} "
            f"{str(r['cer']):>9s} {r['rdkit_parse']:>6s} "
            f"{str(r['mass_delta_da']):>9s}"
        )
    add("")
    base = next(r for r in results if r["config"] == "baseline_detrec")
    add(
        f"baseline (det+rec): edit distance {base['edit_distance']}, "
        f"CER {base['cer']}"
    )
    add(
        f"best expansion   : {best['config']}, edit distance "
        f"{best['edit_distance']}, CER {best['cer']}"
    )
    improved = (
        best["edit_distance"] is not None
        and base["edit_distance"] is not None
        and best["edit_distance"] < base["edit_distance"]
    )
    add("")
    add(
        "VERDICT: improvement"
        if improved
        else "VERDICT: NO improvement - baseline retained"
    )
    add("")
    add(
        f"determinism: best config repeated 3x gave edit distances "
        f"{repeats} (deterministic={deterministic})"
    )
    add("")
    add("--- residual differences per config ---")
    for r in results:
        add(f"[{r['config']}] edits={r['edit_distance']}")
        ops = json.loads(
            (OUT / r["diff_file"]).read_text(encoding="utf-8")
        )
        if not ops:
            add("   (none - exact character match)")
        for o in ops:
            flag = "  <-- C(=O)" if o["in_carbonyl"] else ""
            add(
                f"   {o['op']:7s} @{o['ref_pos']:>3d} "
                f"ref={o['ref_text']!r} -> ocr={o['ocr_text']!r}{flag}"
            )
    (OUT / "box_expansion_report.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
