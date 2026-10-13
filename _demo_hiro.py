"""Hiro-OCSR classify_smiles sweep on the DECIMER cache.

``hiro_ocsr.classify.classify_smiles`` runs in pure RDKit+regex (no LLM,
no GPU). It labels each SMILES as ``molecule`` / ``fragment`` / ``markush``
/ ``irregular`` / ``invalid`` plus a feature bag (``isotope``, ``h2o``,
``salt``, ``metal``, ``charge``). We sweep every DECIMER entry so the
morning report can show how Hiro's post-processing categorises our
predicted SMILES, even before its 8B-VLM inference is on the table.

Run::

    .venv\\Scripts\\python.exe -u _demo_hiro.py

Writes to ``demo_hiro_classify/``.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

from rdkit import RDLogger  # type: ignore[import-not-found]

RDLogger.DisableLog("rdApp.*")
logging.getLogger().setLevel(logging.CRITICAL)

from hiro_ocsr.classify import classify_smiles  # type: ignore[import-not-found]

CACHE = Path("data/processed/decimer_cache.json")
OUT = Path("demo_hiro_classify")


def _entry_smiles(entry: object) -> str:
    if not isinstance(entry, dict):
        return ""
    for k in ("smiles", "raw", "cleaned", "canon"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    if not CACHE.exists():
        print(f"[!] cache missing: {CACHE}")
        return 1
    OUT.mkdir(exist_ok=True)

    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    cache = raw["entries"] if isinstance(raw, dict) and "entries" in raw else raw
    n = len(cache)
    print(f"[hiro] classify_smiles sweep on {n} DECIMER SMILES ...")

    type_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    invalid_smiles: list[tuple[str, str]] = []
    type_samples: dict[str, list[str]] = {k: [] for k in ("molecule", "fragment", "markush", "irregular", "invalid", "empty")}
    rows: list[dict] = []

    for cmpd_id, entry in cache.items():
        smi = _entry_smiles(entry)
        if not smi:
            rows.append(
                {
                    "cmpd_id": cmpd_id,
                    "image_type": "empty",
                    "isotope": "", "h2o": "", "salt": "", "metal": "", "charge": "",
                }
            )
            type_counts["empty"] += 1
            type_samples["empty"].append(cmpd_id)
            continue
        try:
            image_type, feats = classify_smiles(smi)
        except Exception as exc:
            image_type, feats = "invalid", {
                "isotope": False, "h2o": False, "salt": False,
                "metal": False, "charge": False,
            }
            invalid_smiles.append((cmpd_id, f"classify exception: {exc}"))
        type_counts[image_type] += 1
        for k in ("isotope", "h2o", "salt", "metal", "charge"):
            if feats.get(k):
                feature_counts[k] += 1
        type_samples.setdefault(image_type, []).append(cmpd_id)
        rows.append(
            {
                "cmpd_id": cmpd_id,
                "image_type": image_type,
                "isotope": int(bool(feats.get("isotope"))),
                "h2o": int(bool(feats.get("h2o"))),
                "salt": int(bool(feats.get("salt"))),
                "metal": int(bool(feats.get("metal"))),
                "charge": int(bool(feats.get("charge"))),
            }
        )

    if rows:
        with open(OUT / "hiro_classify.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary_lines = [
        "# Hiro-OCSR classify_smiles sweep",
        f"# source: {CACHE}  ({n} entries)",
        "",
        "## image_type distribution",
    ]
    for k in ("molecule", "fragment", "markush", "irregular", "invalid", "empty"):
        c = type_counts.get(k, 0)
        pct = (100 * c / n) if n else 0
        summary_lines.append(f"  - {k:<10s}: {c:>4d}  ({pct:5.1f}%)")

    summary_lines += ["", "## feature distribution", "  (counted over all entries)"]
    for k in ("isotope", "h2o", "salt", "metal", "charge"):
        summary_lines.append(f"  - {k:<10s}: {feature_counts.get(k, 0)}")

    summary_lines += ["", "## sample cmpds per image_type (first 5 each)"]
    for k in ("molecule", "fragment", "markush", "irregular", "invalid", "empty"):
        sample = type_samples.get(k, [])[:5]
        summary_lines.append(f"  - {k:<10s}: {', '.join(sample) if sample else '(none)'}")

    summary_lines += ["", "## invalid SMILES (first 10)"]
    summary_lines += [f"  - {c}: {msg}" for c, msg in invalid_smiles[:10]]
    summary_lines.append("")
    (OUT / "hiro_classify.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("[hiro] image_type distribution:")
    for k in ("molecule", "fragment", "markush", "irregular", "invalid", "empty"):
        c = type_counts.get(k, 0)
        pct = (100 * c / n) if n else 0
        print(f"  {k:<10s}: {c:>4d}  ({pct:5.1f}%)")
    print(f"[hiro] wrote {OUT / 'hiro_classify.csv'} + .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
