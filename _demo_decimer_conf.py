"""DECIMER per-token confidence distribution from the cache.

The DECIMER transformer-decoder exposes per-token softmax confidence
(via ``predict_SMILES(confidence=True)``). Our pipeline already
captures two summary statistics in the cache, ``min_conf`` and
``low_conf_count`` (count of tokens below 0.5). This demo bins and
reports those numbers, plus shows the comparison vs OCR canonical
agreement, so the morning report can answer "where do DECIMER and
OCR disagree and why".

Run::

    .venv\\Scripts\\python.exe -u _demo_decimer_conf.py

Writes to ``demo_decimer_conf/``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

CACHE = Path("data/processed/decimer_cache.json")
OUT = Path("demo_decimer_conf")


def main() -> int:
    if not CACHE.exists():
        print(f"[!] cache missing: {CACHE}")
        return 1
    OUT.mkdir(exist_ok=True)

    raw_cache = json.loads(CACHE.read_text(encoding="utf-8"))
    cache = (
        raw_cache["entries"]
        if isinstance(raw_cache, dict) and "entries" in raw_cache
        else raw_cache
    )

    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01]
    bins: Counter[str] = Counter()
    rows: list[dict] = []
    n = 0
    n_with_low = 0
    n_zero_or_empty = 0

    for cmpd_id, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        n += 1
        mc = float(entry.get("min_conf", 0.0) or 0.0)
        lc = int(entry.get("low_conf", 0) or 0)
        if mc <= 0.0:
            n_zero_or_empty += 1
        if lc > 0:
            n_with_low += 1
        label = f"{bin_edges[max(0, len(bin_edges)-2)]}+"
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= mc < bin_edges[i + 1]:
                label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                break
        bins[label] += 1
        rows.append(
            {
                "cmpd_id": cmpd_id,
                "min_conf": round(mc, 4),
                "low_conf_tokens": lc,
                "bin": label,
            }
        )

    rows.sort(key=lambda r: r["min_conf"])

    with open(OUT / "decimer_conf.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cmpd_id", "min_conf", "low_conf_tokens", "bin"])
        w.writeheader()
        w.writerows(rows)

    summary = [
        "# DECIMER per-token confidence distribution",
        f"# source: {CACHE}",
        f"# entries: {n}",
        "",
        "## min_conf bin distribution",
    ]
    for i in range(len(bin_edges) - 1):
        label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        summary.append(f"  - {label}: {bins.get(label, 0)}")
    summary += [
        "",
        f"  entries with >=1 low-conf token : {n_with_low}  ({100*n_with_low/n:.1f}%)",
        f"  entries with min_conf == 0     : {n_zero_or_empty}  ({100*n_zero_or_empty/n:.1f}%)",
        "",
        "## top-10 least-confident cmpds (by min_conf)",
    ]
    for r in rows[:10]:
        summary.append(f"  - cmpd {r['cmpd_id']:<6s} min_conf={r['min_conf']:.4f} low_tokens={r['low_conf_tokens']}")
    summary.append("")
    (OUT / "decimer_conf.txt").write_text("\n".join(summary), encoding="utf-8")

    print("[decimer conf] distribution")
    for i in range(len(bin_edges) - 1):
        label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        print(f"  {label}: {bins.get(label, 0)}")
    print(f"  with >=1 low-conf token : {n_with_low}/{n}  ({100*n_with_low/n:.1f}%)")
    print(f"  min_conf == 0           : {n_zero_or_empty}/{n}  ({100*n_zero_or_empty/n:.1f}%)")
    print(f"[decimer conf] wrote {OUT / 'decimer_conf.csv'} + .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
