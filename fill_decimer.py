"""Fill the ``smiles_decimer`` column of compounds_combined.csv.

Standalone companion to ``main.py``: it does NOT touch RapidOCR, just
walks every ``data/processed/cells/*_struct.png`` and runs DECIMER on it.

* Caches results in ``data/processed/decimer_cache.json`` so a Ctrl+C or
  crash does not lose progress. Restarting the script picks up where it
  left off.
* Writes a fresh ``compounds_combined.csv`` joined on ``cmpd_id`` at
  the end (no DECIMER re-write mid-run, so partial progress is safe).
* If the user does not have DECIMER installed, the script aborts with
  a clear message instead of silently writing empty columns.

Usage::

    $env:PYTHONIOENCODING = "utf-8"
    .venv\\Scripts\\python.exe -u fill_decimer.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

# UTF-8 console so the progress bar / emojis do not mangle under cp1252
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from cpd.chem import validate_and_enrich
from cpd.decimer import is_available, predict_smiles


CACHE = Path("data/processed/decimer_cache.json")
CELLS_DIR = Path("data/processed/cells")
COMPOUNDS_CSV = Path("data/processed/compounds.csv")
ASSAYS_CSV = Path("data/processed/assays.csv")
COMBINED_CSV = Path("data/processed/compounds_combined.csv")

CMPD_RE = re.compile(r"^(?P<src>.+)_(?P<cid>\d{4})_struct\.png$")


def _load_cache() -> dict[str, dict[str, str]]:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"   [!] {CACHE} unreadable, starting fresh")
    return {}


def _save_cache(cache: dict[str, dict[str, str]]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _struct_files() -> list[Path]:
    return sorted(CELLS_DIR.glob("*_struct.png"))


def _all_struct_ids(files: list[Path]) -> list[tuple[Path, str]]:
    """Extract (path, cmpd_id) for every crop. Files that don't match the
    naming convention are skipped with a warning."""
    out: list[tuple[Path, str]] = []
    for p in files:
        m = CMPD_RE.match(p.name)
        if not m:
            continue
        out.append((p, m.group("cid")))
    return out


def _run_decimer(struct_ids: list[tuple[Path, str]],
                  cache: dict[str, dict[str, str]]) -> None:
    if not is_available():
        sys.exit(
            "DECIMER is not installed. Run:\n"
            "  uv pip install --python .venv\\Scripts\\python.exe DECIMER\n"
            "and try again."
        )

    pending = [(p, cid) for p, cid in struct_ids if cid not in cache]
    print(f"   {len(struct_ids)} structure crops, {len(pending)} pending "
          f"({len(struct_ids) - len(pending)} cached)")

    if not pending:
        print("   nothing to do -- cache is up to date")
        return

    t0 = time.monotonic()
    for i, (path, cid) in enumerate(pending, start=1):
        smi = predict_smiles(path)
        if smi:
            chem = validate_and_enrich(smi)
            canon = chem.canonical_smiles if chem else ""
            cache[cid] = {"raw": smi, "canon": canon}
        else:
            cache[cid] = {"raw": "", "canon": ""}
        # Save every 10 records so a Ctrl+C does not lose much.
        if i % 10 == 0 or i == len(pending):
            _save_cache(cache)
            elapsed = time.monotonic() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(pending) - i) / max(rate, 1e-6)
            print(f"   [{i}/{len(pending)}]  "
                  f"elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min   ",
                  end="\r")
    print()  # newline after the carriage-return progress line
    _save_cache(cache)
    print(f"   done in {(time.monotonic() - t0)/60:.1f} min")


def _read_csv_dicts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_combined(compounds: list[dict], assays: dict[str, dict],
                     decimer: dict[str, dict[str, str]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        "cmpd_id", "smiles", "canonical_smiles",
        "smiles_decimer", "canonical_smiles_decimer", "smiles_agree",
        "molecular_weight", "heavy_atom_count", "is_valid",
        "structure_image", "source_image",
        "kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity",
        "assay_source_image",
    )
    seen: set[str] = set()
    rows: list[dict] = []
    for c in compounds:
        cid = c["cmpd_id"]
        a = assays.get(cid, {})
        d = decimer.get(cid, {"raw": "", "canon": ""})
        c_canon = c.get("canonical_smiles") or ""
        d_canon = d.get("canon") or ""
        agree = bool(c_canon and d_canon and c_canon == d_canon)
        rows.append({
            "cmpd_id": cid,
            "smiles": c.get("smiles") or "",
            "canonical_smiles": c_canon,
            "smiles_decimer": d.get("raw") or "",
            "canonical_smiles_decimer": d_canon,
            "smiles_agree": "True" if agree else "False",
            "molecular_weight": c.get("molecular_weight") or "",
            "heavy_atom_count": c.get("heavy_atom_count") or "",
            "is_valid": c.get("is_valid") or "False",
            "structure_image": c.get("structure_image") or "",
            "source_image": c.get("source_image") or "",
            "kd_nm": a.get("kd_nm") or "",
            "rt_min": a.get("rt_min") or "",
            "ms_mz": a.get("ms_mz") or "",
            "lcms_method": a.get("lcms_method") or "",
            "ms_polarity": a.get("ms_polarity") or "",
            "assay_source_image": a.get("source_image") or "",
        })
        seen.add(cid)
    # Cmpds that only have an assay (no structure) still get a row.
    for cid, a in assays.items():
        if cid in seen:
            continue
        rows.append({
            "cmpd_id": cid,
            "smiles": "",
            "canonical_smiles": "",
            "smiles_decimer": "",
            "canonical_smiles_decimer": "",
            "smiles_agree": "False",
            "molecular_weight": "",
            "heavy_atom_count": "",
            "is_valid": "False",
            "structure_image": "",
            "source_image": "",
            "kd_nm": a.get("kd_nm") or "",
            "rt_min": a.get("rt_min") or "",
            "ms_mz": a.get("ms_mz") or "",
            "lcms_method": a.get("lcms_method") or "",
            "ms_polarity": a.get("ms_polarity") or "",
            "assay_source_image": a.get("source_image") or "",
        })
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main_() -> None:
    if not CELLS_DIR.exists():
        sys.exit(f"no {CELLS_DIR} -- run main.py first to crop the structures")
    if not COMPOUNDS_CSV.exists() or not ASSAYS_CSV.exists():
        sys.exit(f"need {COMPOUNDS_CSV} and {ASSAYS_CSV} -- run main.py first")

    print(f"[1/3] loading cache + structure crops ...")
    cache = _load_cache()
    struct_ids = _all_struct_ids(_struct_files())
    print(f"      cache has {len(cache)} entries")

    print(f"[2/3] running DECIMER on {len(struct_ids)} structures ...")
    _run_decimer(struct_ids, cache)

    print(f"[3/3] writing {COMBINED_CSV} ...")
    compounds = _read_csv_dicts(COMPOUNDS_CSV)
    assays_list = _read_csv_dicts(ASSAYS_CSV)
    assays = {a["cmpd_id"]: a for a in assays_list}
    n = _write_combined(compounds, assays, cache, COMBINED_CSV)
    n_dec = sum(1 for v in cache.values() if v.get("raw"))
    n_agree = sum(
        1 for c in compounds
        if cache.get(c["cmpd_id"], {}).get("canon")
        and c.get("canonical_smiles")
        and c["canonical_smiles"] == cache[c["cmpd_id"]]["canon"]
    )
    print("=" * 55)
    print(f"combined: {n} rows  |  DECIMER filled: {n_dec}  |  agree: {n_agree}")
    print(f"cache: {CACHE}  ({len(cache)} entries)")
    print("=" * 55)


if __name__ == "__main__":
    main_()
