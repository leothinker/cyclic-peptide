"""DECIMER single-image demo on cmpd 1036.

Runs ``predict_SMILES(confidence=True)`` on the cropped structure PNG
of cmpd 1036 (page I100297) and dumps the full SMILES plus the
per-token confidence list. We then highlight:

  * min_conf / mean_conf / low_conf_count across the SMILES
  * which token positions fall below threshold (signature of phantom
    fragments -- the canonical DECIMER failure mode)
  * what truncating the SMILES at the first low-conf position looks
    like (the simplest decoder-modification-equivalent we can do in
    post-processing without touching the SavedModel)

This is the per-cmpd reportable artifact for the morning meeting.

Run::

    .venv\\Scripts\\python.exe -u _demo_decimer_one.py

Writes to ``demo_decimer_one/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from DECIMER import predict_SMILES  # type: ignore[import-not-found]

IMG = Path("data/processed/cells/PCTCN2025075389-ftappb-I100297_1036_struct.png")
OUT = Path("demo_decimer_one")
THRESHOLD = 0.5


def _render_full(tokens: list[tuple[str, float]], threshold: float) -> str:
    out = [
        "pos | token     | conf  | bar                     | notes",
        "----+-----------+-------+-------------------------+--------------------------",
    ]
    bar_width = 20
    for i, (tok, conf) in enumerate(tokens):
        bar = "#" * max(0, int(conf * bar_width))
        marker = "" if conf >= threshold else "  <- below threshold"
        out.append(f"{i:>3} | {tok!r:<9s} | {conf:.3f} | {bar:<{bar_width}s} |{marker}")
    return "\n".join(out)


def _canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def main() -> int:
    if not IMG.exists():
        print(f"[!] image missing: {IMG}")
        return 1
    OUT.mkdir(exist_ok=True)

    print(f"[decimer] running predict_SMILES(confidence=True) on {IMG.name} ...")
    raw = predict_SMILES(str(IMG), confidence=True)
    if raw is None:
        print("[!] no prediction")
        return 1
    if isinstance(raw, str):
        print(f"[!] returned plain string, not (smi, tokens): {raw[:80]}")
        return 1
    smi, tokens_raw = raw
    tokens = [(t, float(c)) for t, c in tokens_raw]

    n = len(tokens)
    confs = [c for _, c in tokens]
    min_conf = min(confs) if confs else 0.0
    mean_conf = (sum(confs) / n) if n else 0.0
    n_low = sum(1 for c in confs if c < THRESHOLD)
    first_low = next((i for i, c in enumerate(confs) if c < THRESHOLD), -1)

    print(f"[decimer] SMILES ({len(smi)} chars): {smi}")
    print(f"[decimer] tokens: {n}, min_conf={min_conf:.3f}, mean_conf={mean_conf:.3f}, low_conf={n_low}, first_low_pos={first_low}")

    raw_canon = _canon(smi)
    print(f"[decimer] RDKit-canonical of raw   : {raw_canon if raw_canon else '(invalid)'}")

    demo_text = [
        f"# DECIMER per-token confidence on {IMG.name}",
        f"# source image  : {IMG}",
        f"# tokens        : {n}",
        f"# min_conf      : {min_conf:.4f}",
        f"# mean_conf     : {mean_conf:.4f}",
        f"# low_conf (<{THRESHOLD}) : {n_low}",
        f"# first_low_pos : {first_low}",
        f"# raw_canon     : {raw_canon}",
        "",
        _render_full(tokens, THRESHOLD),
    ]

    if first_low >= 0:
        prefix_tokens = tokens[:first_low]
        prefix_text = "".join(t for t, _ in prefix_tokens)
        truncated = prefix_text
        if "." in truncated:
            truncated = truncated.rsplit(".", 1)[0]

        demo_text += [
            "",
            "## decoder modification: truncate at first low-conf position",
            "",
            f"SMILES prefix (tokens 0..{first_low-1}) : {prefix_text!r}",
            f"After snapping back to previous '.':   : {truncated!r}",
            f"RDKit-canonical of truncated prefix    : {_canon(truncated) or '(invalid)'}",
        ]

    demo_text.append("")
    (OUT / "decimer_one.txt").write_text("\n".join(demo_text), encoding="utf-8")

    payload = {
        "image": str(IMG),
        "smiles": smi,
        "tokens": [{"pos": i, "token": t, "conf": round(c, 4)} for i, (t, c) in enumerate(tokens)],
        "min_conf": round(min_conf, 4),
        "mean_conf": round(mean_conf, 4),
        "low_conf_count": n_low,
        "first_low_pos": first_low,
        "raw_canon": raw_canon,
    }
    if first_low >= 0:
        prefix_tokens = tokens[:first_low]
        prefix_text = "".join(t for t, _ in prefix_tokens)
        truncated = prefix_text
        if "." in truncated:
            truncated = truncated.rsplit(".", 1)[0]
        payload["truncate_at_low_conf"] = {
            "prefix_text": prefix_text,
            "snapped_text": truncated,
            "snapped_canon": _canon(truncated),
        }

    (OUT / "decimer_one.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[decimer] wrote {OUT / 'decimer_one.txt'} + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
