"""Phase 2: morphological dilation on top of the best binarisation.

Run only after the threshold sweep, and kept as a separate experiment so
the two effects are never conflated. Threshold stays pinned at the sweep
winner (200) and the only variable is a small grayscale dilation kernel
applied after binarisation, which is aimed at the residual failure mode:
``=`` collapsing to ``-`` and a thin ``=`` being dropped outright.

Preprocessing here deliberately reuses
``phase2_threshold_scan.preprocess`` so the pinned threshold is provably
the same operation the sweep measured.

Writes only under ``data/audit/phase2/threshold_scan/dilate/``.
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

from cpd.ms_validation import validate_against_ms  # noqa: E402
from phase2_ocr_compare import REFERENCES, edit_ops  # noqa: E402
from phase2_threshold_scan import (  # noqa: E402
    BASELINE_THRESHOLD,
    CSV_PATH,
    OUT,
    PHASE2,
    build_engine,
    diff_ops,
    load_meta,
    load_rows,
    preprocess,
    recognise,
)

DILATE_OUT = OUT / "dilate"
KERNELS = ((1, 1), (2, 2), (3, 1), (3, 2))


def apply_dilate(img: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    if ksize == (1, 1):
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.dilate(img, kernel, iterations=1)


def run(cmpds: list[str]) -> None:
    meta = load_meta()
    rows = load_rows()
    DILATE_OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = DILATE_OUT / "raw_ocr"
    raw_dir.mkdir(exist_ok=True)

    targets = [
        c for c in cmpds if c in meta and c in REFERENCES and c in rows
    ]
    if not targets:
        raise SystemExit("no compound has both a crop and an admitted reference")

    engine = build_engine()
    out_rows: list[dict] = []

    for cid in targets:
        ref = REFERENCES[cid]["smiles"]
        img = cv2.imread(str(PHASE2 / meta[cid]["crop_path"]))
        if img is None:
            print(f"WARNING: unreadable crop for {cid}")
            continue
        base_proc = preprocess(img, BASELINE_THRESHOLD)
        row = rows[cid]
        mz = float(row["ms_mz"])
        pol = (row.get("ms_polarity") or "").strip()

        for ksize in KERNELS:
            proc = apply_dilate(base_proc, ksize)
            raw = recognise(engine, proc)
            edits = edit_ops(ref, raw) if raw else None
            cer = round(edits / len(ref), 6) if edits is not None else None
            gate = (
                validate_against_ms(raw, mz, pol)
                if raw
                else {"verdict": "empty"}
            )
            obs = gate.get("observed_mass")
            exp = gate.get("expected_mass")
            delta = round(obs - exp, 4) if obs is not None and exp else None
            ops = diff_ops(ref, raw) if raw else []

            tag = f"{ksize[0]}x{ksize[1]}"
            (raw_dir / f"{cid}_k{tag}.txt").write_text(raw + "\n", encoding="utf-8")
            (raw_dir / f"{cid}_k{tag}_diff.json").write_text(
                json.dumps(ops, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            cv2.imwrite(str(raw_dir / f"{cid}_k{tag}_proc.png"), proc)

            out_rows.append(
                {
                    "cmpd_id": cid,
                    "threshold": BASELINE_THRESHOLD,
                    "dilate_kernel": tag,
                    "reference_len": len(ref),
                    "raw_len": len(raw),
                    "edit_distance": edits if edits is not None else "",
                    "cer": cer if cer is not None else "",
                    "n_diff_ops": len(ops),
                    "n_ops_in_carbonyl": sum(1 for o in ops if o["in_carbonyl"]),
                    "rdkit_parse": "yes"
                    if raw and Chem.MolFromSmiles(raw)
                    else "no",
                    "ms_mz": mz,
                    "ms_polarity": pol,
                    "mass_gate_verdict": gate.get("verdict"),
                    "mass_delta_da": delta if delta is not None else "",
                    "raw_ocr_file": f"raw_ocr/{cid}_k{tag}.txt",
                    "diff_file": f"raw_ocr/{cid}_k{tag}_diff.json",
                }
            )
            print(
                f"{cid} dilate={tag} len={len(raw):4d} edits={edits} "
                f"CER={cer} parse={out_rows[-1]['rdkit_parse']}"
            )

    with (DILATE_OUT / "dilate_scan.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    L: list[str] = []
    add = L.append
    add("Phase 2 - morphological dilation on top of the best threshold")
    add("")
    add("PRELIMINARY: single reference (cmpd 1036). Not a dataset accuracy.")
    add(f"pinned    : threshold={BASELINE_THRESHOLD} upscale=2x strip=0 "
        f"rec_lang=en")
    add("variable  : dilation kernel only (1x1 == no dilation == sweep best)")
    add("")
    for cid in targets:
        add(f"=== cmpd {cid} ===")
        add(
            f"{'kernel':>7s} {'len':>5s} {'edits':>6s} {'CER':>9s} "
            f"{'parse':>6s} {'dDa':>9s}  verdict"
        )
        for r in out_rows:
            if r["cmpd_id"] != cid:
                continue
            add(
                f"{r['dilate_kernel']:>7s} {r['raw_len']:5d} "
                f"{str(r['edit_distance']):>6s} {str(r['cer']):>9s} "
                f"{r['rdkit_parse']:>6s} {str(r['mass_delta_da']):>9s}  "
                f"{r['mass_gate_verdict']}"
            )
        add("")
        base = next(
            (r for r in out_rows
             if r["cmpd_id"] == cid and r["dilate_kernel"] == "1x1"),
            None,
        )
        best = min(
            (r for r in out_rows if r["cmpd_id"] == cid),
            key=lambda r: (r["edit_distance"] is None,
                           r["edit_distance"] or 10**9),
        )
        if base:
            add(
                f"best kernel {best['dilate_kernel']} "
                f"(CER {best['cer']}, edits {best['edit_distance']}) "
                f"vs no dilation (CER {base['cer']}, "
                f"edits {base['edit_distance']})"
            )
            if best["dilate_kernel"] == "1x1":
                add(
                    "dilation did NOT help: the residual errors are not "
                    "fixed by thickening strokes"
                )
        add("")
        add(f"--- residual differences ({best['dilate_kernel']}) ---")
        ops = json.loads(
            (DILATE_OUT / best["diff_file"]).read_text(encoding="utf-8")
        )
        if not ops:
            add("  (none - exact character match)")
        for o in ops:
            flag = "  <-- in C(=O)" if o["in_carbonyl"] else ""
            add(
                f"  {o['op']:7s} @{o['ref_pos']:>3d} "
                f"ref={o['ref_text']!r} -> ocr={o['ocr_text']!r}{flag}"
            )
            add(f"           context ...{o['context']}...")
        add("")

    (DILATE_OUT / "dilate_report.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {DILATE_OUT/'dilate_scan.csv'}")
    print(f"wrote {DILATE_OUT/'dilate_report.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmpds", default="1036")
    args = ap.parse_args()
    run([c.strip() for c in args.cmpds.split(",") if c.strip()])


if __name__ == "__main__":
    main()
