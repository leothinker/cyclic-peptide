"""End-to-end patent table pipeline.

Phases:
  1. Scan golden tables directory
  2. Extract compounds + assays (with optional top-k SMILES repair)
  3. Run DECIMER on cropped structures (cache-aware, confidence-enabled)
  4. Merge everything into a single combined CSV
"""

from __future__ import annotations

import argparse
import csv as _csv_mod
import json
import statistics
import sys
import time
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

from cpd.decimer import (  # type: ignore[import-not-found]
    is_available,
    low_confidence_tokens,
    predict,
)
from cpd.chem import (
    close_unbalanced_brackets,
    largest_fragment_canonical,
    validate_and_enrich,
)  # type: ignore[import-not-found]
from cpd.merge import (  # type: ignore[import-not-found]
    dedupe_assays,
    dedupe_compounds,
    normalise_cmpd_id,
)
from cpd.parsers.table_extractor import (  # type: ignore[import-not-found]
    AssayRecord,
    Column,
    CompoundRecord,
    RapidOcrTableExtractor,
)


COMBINED_COLUMNS: tuple[str, ...] = (
    "cmpd_id",
    "smiles",
    "canonical_smiles",
    "smiles_decimer",
    "canonical_smiles_decimer",
    "smiles_best",
    "smiles_best_source",
    "smiles_agree",
    "molecular_weight",
    "heavy_atom_count",
    "is_valid",
    "structure_image",
    "source_image",
    "kd_nm",
    "rt_min",
    "ms_mz",
    "lcms_method",
    "ms_polarity",
    "assay_source_image",
    "decimer_min_conf",
    "decimer_low_conf_count",
    "decimer_render_path",
    "ocr_avg_entropy",
    "ocr_min_top1",
    "ocr_low_conf_count",
)

# OCR entropy gate: when the average per-char entropy of the SMILES
# cell exceeds this AND DECIMER has a valid canonical SMILES,
# prefer DECIMER. Higher entropy = the recogniser spread probability
# across many candidates, which empirically correlates with SMILES
# that fail RDKit validation. 0.5 bits/char is roughly the
# transition between "argmax picked a clear winner" (~0.0) and
# "the recogniser was guessing" (~1.0+).
OCR_TRUST_ENTROPY_THRESHOLD = 0.5

# As a coarser safety net, a SMILES with this many low-confidence
# characters (top-1 < 0.7) is treated as untrustworthy regardless
# of the average entropy.
OCR_TRUST_LOW_CONF_LIMIT = 5

DECIMER_CACHE_FILENAME = "decimer_cache.json"
DECIMER_CACHE_VERSION = 2  # bumped when the schema changes (e.g. adding confidence)


def _iter_jpgs(img_dir: Path) -> Iterable[Path]:
    return sorted(img_dir.glob("*.jpg"))


def _load_decimer_cache(path: Path) -> dict[str, dict]:
    """Read a previously-saved DECIMER cache. Returns ``{}`` if missing or stale."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("__version__") != DECIMER_CACHE_VERSION:
        return {}
    body = data.get("entries")
    return body if isinstance(body, dict) else {}


def _save_decimer_cache(path: Path, entries: dict[str, dict]) -> None:
    """Persist ``entries`` to ``path`` with the current schema version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"__version__": DECIMER_CACHE_VERSION, "entries": entries}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _safe_extract(
    extractor: RapidOcrTableExtractor,
    image_path: Path,
    cells_dir: Path,
    prev_columns: list[Column],
) -> tuple[list[CompoundRecord], list[AssayRecord], list[Column]]:
    try:
        return extractor.extract_records(
            image_path, cells_dir=cells_dir, prev_columns=prev_columns
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"   [warn] skipping {image_path.name}: extractor crashed: {exc}")
        return [], [], prev_columns


def _extract_all(
    img_dir: Path,
    cells_dir: Path,
    rec_top_k: int = 0,
    smiles_route: str = "production",
) -> tuple[list[CompoundRecord], dict[str, AssayRecord], dict[str, str]]:
    """Run the extractor over every image, deduping across pages.

    Returns ``(compounds, assays, failures)``:
      * ``compounds`` is a list of unique CompoundRecord (one per cmpd_id).
      * ``assays``    is a dict keyed by cmpd_id for fast lookup in
        ``_build_combined_rows``.
    """
    extractor = RapidOcrTableExtractor(
        rec_top_k=rec_top_k, smiles_route=smiles_route
    )
    compounds: dict[str, CompoundRecord] = {}
    assays: dict[str, AssayRecord] = {}
    failures: dict[str, str] = {}
    prev_columns: list[Column] = []

    jpgs = list(_iter_jpgs(img_dir))
    for idx, image_path in enumerate(jpgs, start=1):
        page_compounds, page_assays, columns = _safe_extract(
            extractor, image_path, cells_dir, prev_columns
        )
        if columns:
            prev_columns = columns

        if not (page_compounds or page_assays):
            failures[image_path.name] = "no rows"

        page_compounds = [c for c in page_compounds if normalise_cmpd_id(c.cmpd_id)]
        page_assays = [a for a in page_assays if normalise_cmpd_id(a.cmpd_id)]

        for cid, rec in dedupe_compounds(page_compounds).items():
            compounds.setdefault(cid, rec)
        for cid, rec in dedupe_assays(page_assays).items():
            assays.setdefault(cid, rec)

        print(
            f"   [{idx}/{len(jpgs)}] {image_path.name} -> "
            f"+{len(page_compounds)} cmpd, +{len(page_assays)} assay "
            f"| total: {len(compounds)} cmpd, {len(assays)} assay",
            end="\r",
        )
    print()
    return list(compounds.values()), assays, failures



def _load_from_csv(
    compounds_csv: Path,
    assays_csv: Path,
    cells_dir: Path,
    img_dir: Path,
) -> tuple[list, dict, dict]:
    """Load OCR results from legacy compounds.csv + assays.csv.

    Used by --skip-ocr so a DECIMER-only refresh does not re-OCR.
    Verifies each compound structure PNG actually exists on disk and
    silently drops rows where the crop is missing.
    """
    _csv = _csv_mod

    compounds: dict[str, CompoundRecord] = {}
    assays: dict[str, AssayRecord] = {}
    failures: dict[str, str] = {}

    def _opt_float(s):
        try:
            return float(s) if s not in ("", "None") else None
        except ValueError:
            return None

    if compounds_csv.exists():
        with open(compounds_csv, encoding="utf-8-sig", newline="") as fh:
            for r in _csv.DictReader(fh):
                cid = (r.get("cmpd_id") or "").strip()
                if not cid:
                    continue
                struct = (r.get("structure_image") or "").strip()
                if struct:
                    abs_path = cells_dir.parent / struct
                    if not abs_path.exists():
                        struct = ""
                compounds[cid] = CompoundRecord(
                    cmpd_id=cid,
                    smiles=(r.get("smiles") or "").strip() or None,
                    canonical_smiles=(r.get("canonical_smiles") or "").strip() or None,
                    molecular_weight=_opt_float(r.get("molecular_weight")),
                    heavy_atom_count=(int(r["heavy_atom_count"])
                                       if r.get("heavy_atom_count") not in ("", "None")
                                       else None),
                    is_valid=(r.get("is_valid") == "True"),
                    structure_image=struct or None,
                    source_image=(r.get("source_image") or "").strip() or None,
                )

    if assays_csv.exists():
        with open(assays_csv, encoding="utf-8-sig", newline="") as fh:
            for r in _csv.DictReader(fh):
                cid = (r.get("cmpd_id") or "").strip()
                if not cid:
                    continue
                assays[cid] = AssayRecord(
                    cmpd_id=cid,
                    kd_nm=_opt_float(r.get("kd_nm")),
                    rt_min=_opt_float(r.get("rt_min")),
                    ms_mz=_opt_float(r.get("ms_mz")),
                    lcms_method=(r.get("lcms_method") or "").strip() or None,
                    ms_polarity=(r.get("ms_polarity") or "").strip() or None,
                    source_image=(r.get("source_image") or "").strip() or None,
                )

    print(f"   loaded {len(compounds)} compounds and {len(assays)} assays from legacy CSVs")
    return list(compounds.values()), assays, failures


def _run_decimer(
    compounds: list[CompoundRecord],
    out_dir: Path,
    cache_path: Path,
    low_conf_threshold: float = 0.5,
    limit: int = 0,
    offset: int = 0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, int]]:
    """Run DECIMER on every cropped structure PNG, with on-disk caching.

    Returns ``(decimer_raw, decimer_canon, decimer_min_conf,
    decimer_low_conf_count)`` keyed by ``cmpd_id``.

    Results are merged with the on-disk cache so a second ``run_pipeline``
    call only re-runs DECIMER for cmpds not seen yet. The cache is rewritten
    every 10 cells so a crash mid-run still leaves useful state behind.

    When DECIMER is not installed the function logs once and returns four
    empty dicts so the rest of the pipeline still runs.
    """
    raw: dict[str, str] = {}
    canon: dict[str, str] = {}
    min_conf: dict[str, float] = {}
    low_conf: dict[str, int] = {}

    cache = _load_decimer_cache(cache_path)
    for cid, entry in cache.items():
        if entry.get("raw"):
            raw[cid] = entry["raw"]
        if entry.get("canon"):
            canon[cid] = entry["canon"]
        if entry.get("min_conf") is not None:
            min_conf[cid] = float(entry["min_conf"])
        if entry.get("low_conf") is not None:
            low_conf[cid] = int(entry["low_conf"])

    if not is_available():
        print("   [DECIMER] not installed -> smiles_decimer column will be empty")
        print("              install with:  uv pip install --python .venv/Scripts/python.exe DECIMER")
        return raw, canon, min_conf, low_conf

    targets = [c for c in compounds if c.structure_image]
    if limit and limit > 0 and len(targets) > limit:
        targets = sorted(targets, key=lambda c: c.cmpd_id)
        targets = targets[offset:offset + limit]

    n_cache_hit = sum(1 for c in targets if c.cmpd_id in raw)
    n_to_run = len(targets) - n_cache_hit
    print(
        f"   [DECIMER] {len(targets)} cropped structures "
        f"(cache hit: {n_cache_hit}, to-run: {n_to_run}, "
        f"per-token confidence enabled, low-conf threshold={low_conf_threshold}) ..."
    )

    cells_root = out_dir  # structure_image paths are relative to out_dir.
    t_run = time.monotonic()
    n_new = 0
    for i, c in enumerate(targets, start=1):
        cid = c.cmpd_id
        if cid in raw:
            continue
        img_path = cells_root / c.structure_image  # type: ignore[operator]
        pred = predict(img_path)
        if pred is not None and pred.smiles:
            raw[cid] = pred.smiles
            if pred.tokens:
                min_conf[cid] = round(pred.min_conf, 4)
                low_conf[cid] = len(low_confidence_tokens(pred, low_conf_threshold))
            # Apply phantom-fragment cleanup first so a trailing
            # ``CC.CC`` does not turn into a canonical disagreement. If the
            # cleaned SMILES still has dangling ``)`` / ``]`` (DECIMER
            # truncates at ~300 tokens), fall back to a naive
            # ``close_unbalanced_brackets`` so we recover a larger fraction
            # of the long cyclic-peptide SMILES.
            cleaned = (
                largest_fragment_canonical(pred.smiles)
                or close_unbalanced_brackets(pred.smiles)
                or pred.smiles
            )
            chem = validate_and_enrich(cleaned)
            if chem is not None:
                canon[cid] = chem.canonical_smiles
            cache[cid] = {
                "raw": pred.smiles,
                "cleaned": cleaned,
                "canon": canon.get(cid, ""),
                "min_conf": min_conf.get(cid),
                "low_conf": low_conf.get(cid),
            }
        n_new += 1
        if n_new % 10 == 0 or i == len(targets):
            elapsed = time.monotonic() - t_run
            print(
                f"      [{i}/{len(targets)}] parsed   "
                f"(cache {n_cache_hit} + new {n_new}, "
                f"{elapsed/max(1, n_new):.1f}s/new cell)",
                end="\r",
            )
            _save_decimer_cache(cache_path, cache)
    print()

    print(f"   [DECIMER] got SMILES for {len(raw)} / {len(targets)} structures")
    if low_conf:
        worst = sorted(low_conf.items(), key=lambda kv: -kv[1])[:5]
        print(f"   [DECIMER] top-5 low-confidence compounds: {worst}")
    _save_decimer_cache(cache_path, cache)
    return raw, canon, min_conf, low_conf





def _render_decimer(
    decimer_canon: dict[str, str],
    out_dir: Path,
    linearize: bool = True,
) -> dict[str, str]:
    """Render every DECIMER canonical SMILES to a 2D structure PNG.

    Returns ``{cmpd_id: relative_png_path}`` for the entries that
    rendered successfully. Failed renders (RDKit parse error, blank
    output) are silently skipped and do not appear in the dict, so the
    CSV column will be empty for them.

    Output files are written under ``<out_dir>/rendered/<cid>.png`` and
    the returned path is relative to ``out_dir`` so it slots straight
    into the combined CSV without further processing.
    """
    from cpd.render import render_mol_pil

    rendered_dir = out_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for cid, smi in decimer_canon.items():
        if not smi:
            continue
        img = render_mol_pil(
            smi, size=(520, 360), linearize_macrocycle=linearize
        )
        if img is None:
            continue
        out = rendered_dir / f"{cid}.png"
        try:
            img.save(out)
        except Exception:
            continue
        paths[cid] = str(out.relative_to(out_dir))
    return paths


def _build_combined_rows(
    compounds: list[CompoundRecord],
    assays: dict[str, AssayRecord],
    decimer_raw: dict[str, str],
    decimer_canon: dict[str, str],
    decimer_min_conf: dict[str, float] | None = None,
    decimer_low_conf: dict[str, int] | None = None,
    decimer_render: dict[str, str] | None = None,
) -> list[dict]:
    """Flatten compounds + assays + DECIMER into COMBINED_COLUMNS-shaped dicts."""
    rows: list[dict] = []
    seen_assay_ids: set[str] = set()
    for c in compounds:
        a = assays.get(c.cmpd_id)
        if a is not None:
            seen_assay_ids.add(c.cmpd_id)
        d_raw = decimer_raw.get(c.cmpd_id, "")
        d_can = decimer_canon.get(c.cmpd_id, "")
        d_min = (decimer_min_conf or {}).get(c.cmpd_id)
        d_low = (decimer_low_conf or {}).get(c.cmpd_id)
        d_render = (decimer_render or {}).get(c.cmpd_id, "")
        agree = bool(
            c.canonical_smiles and d_can and c.canonical_smiles == d_can
        )

        # Pick the best SMILES. Source precedence:
        #   * "ocr+decimer" - both canonical SMILES match (highest confidence)
        #   * "ocr"         - OCR canonical SMILES parses AND the cell's
        #                     top-k distribution looks trustworthy
        #   * "decimer_fallback" - OCR canonical parses but the cell entropy
        #                     is high / too many low-conf chars; DECIMER is
        #                     RDKit-valid so we prefer it
        #   * "ocr_only_invalid" - OCR canonical parses but DECIMER is empty
        #                     (we still keep the OCR string; flag it for review)
        #   * "decimer"     - DECIMER canonical SMILES parses (OCR was empty
        #                     or invalid)
        #   * ""            - both empty (rare; cmpd only had a structure crop)
        ocr_trustworthy = (
            c.ocr_avg_entropy is None  # no top-k data -> trust OCR as before
            or c.ocr_avg_entropy <= OCR_TRUST_ENTROPY_THRESHOLD
        ) and (
            c.ocr_low_conf_count is None
            or c.ocr_low_conf_count <= OCR_TRUST_LOW_CONF_LIMIT
        )
        if c.canonical_smiles and c.canonical_smiles == d_can:
            best = c.canonical_smiles
            best_src = "ocr+decimer"
        elif c.canonical_smiles and ocr_trustworthy:
            best = c.canonical_smiles
            best_src = "ocr"
        elif c.canonical_smiles and d_can:
            best = d_can
            best_src = "decimer_fallback"
        elif c.canonical_smiles:
            best = c.canonical_smiles
            best_src = "ocr_only_invalid"
        elif d_can:
            best = d_can
            best_src = "decimer"
        else:
            best = ""
            best_src = ""

        rows.append({
            "cmpd_id": c.cmpd_id,
            "smiles": c.smiles or "",
            "canonical_smiles": c.canonical_smiles or "",
            "smiles_decimer": d_raw,
            "canonical_smiles_decimer": d_can,
            "smiles_best": best,
            "smiles_best_source": best_src,
            "smiles_agree": "True" if agree else "False",
            "molecular_weight": c.molecular_weight if c.molecular_weight is not None else "",
            "heavy_atom_count": c.heavy_atom_count if c.heavy_atom_count is not None else "",
            "is_valid": "True" if c.is_valid else "False",
            "structure_image": c.structure_image or "",
            "source_image": c.source_image or "",
            "kd_nm": a.kd_nm if a is not None and a.kd_nm is not None else "",
            "rt_min": a.rt_min if a is not None and a.rt_min is not None else "",
            "ms_mz": a.ms_mz if a is not None and a.ms_mz is not None else "",
            "lcms_method": (a.lcms_method or "") if a is not None else "",
            "ms_polarity": (a.ms_polarity or "") if a is not None else "",
            "assay_source_image": (a.source_image or "") if a is not None else "",
            "decimer_min_conf": d_min if d_min is not None else "",
            "decimer_low_conf_count": d_low if d_low is not None else "",
            "decimer_render_path": d_render,
            "ocr_avg_entropy": (
                c.ocr_avg_entropy if c.ocr_avg_entropy is not None else ""
            ),
            "ocr_min_top1": c.ocr_min_top1 if c.ocr_min_top1 is not None else "",
            "ocr_low_conf_count": (
                c.ocr_low_conf_count if c.ocr_low_conf_count is not None else ""
            ),
        })
    for cid, a in assays.items():
        if cid in seen_assay_ids:
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
            "kd_nm": a.kd_nm if a is not None and a.kd_nm is not None else "",
            "rt_min": a.rt_min if a is not None and a.rt_min is not None else "",
            "ms_mz": a.ms_mz if a is not None and a.ms_mz is not None else "",
            "lcms_method": a.lcms_method or "",
            "ms_polarity": a.ms_polarity or "",
            "assay_source_image": a.source_image or "",
            "decimer_min_conf": "",
            "decimer_low_conf_count": "",
            "decimer_render_path": "",
            "ocr_avg_entropy": "",
            "ocr_min_top1": "",
            "ocr_low_conf_count": "",
        })
    return rows


def _write_csv(
    rows: Iterable[dict],
    path: Path,
    columns: tuple[str, ...],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = _csv_mod.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def _l1_breakdown(combined_rows: list[dict], threshold: float = 0.5) -> dict:
    """L1 phantom-fragment summary used in run summaries and reports.

    ``combined_rows`` is the list-of-dicts produced by :func:`_build_combined_rows`
    (or loaded from a previously-written combined CSV).
    """
    both = [
        r for r in combined_rows
        if r.get("smiles", "").strip()
        and r.get("smiles_decimer", "").strip()
        and r.get("decimer_min_conf", "") != ""
    ]
    n_both = len(both)
    n_agree = sum(1 for r in both if r.get("smiles_agree") == "True")
    disagree = [r for r in both if r.get("smiles_agree") != "True"]
    n_phantom = sum(1 for r in disagree if float(r["decimer_min_conf"]) < threshold)
    n_real_ocr_err = n_both - n_agree - n_phantom
    median_conf = (
        statistics.median(float(r["decimer_min_conf"]) for r in both) if both else 0.0
    )
    return {
        "n_both": n_both,
        "n_agree": n_agree,
        "n_phantom": n_phantom,
        "n_real_ocr_err": n_real_ocr_err,
        "median_conf": median_conf,
        "threshold": threshold,
    }


def _format_l1_block(stats: dict) -> str:
    n_both = stats["n_both"]
    if not n_both:
        return (
            "   L1 phantom-fragment breakdown: no rows with both OCR+DECIMER SMILES yet\n"
            "      (re-run with --rec-top-k 5 to populate the OCR canonical column)."
        )
    pct = lambda n: f"{n*100/n_both:.0f}%"
    return (
        f"   L1 phantom-fragment breakdown (where both OCR+DECIMER have SMILES, n={n_both}):\n"
        f"     agree                          : {stats['n_agree']:>4}  ({pct(stats['n_agree'])})\n"
        f"     disagree, DECIMER phantom frg.: {stats['n_phantom']:>4}  ({pct(stats['n_phantom'])})  conf < {stats['threshold']}\n"
        f"     disagree, real OCR error       : {stats['n_real_ocr_err']:>4}  ({pct(stats['n_real_ocr_err'])})  conf >= {stats['threshold']}\n"
        f"     median DECIMER min_conf        : {stats['median_conf']:.3f}"
    )


def run_pipeline(
    img_dir: str = "data/raw/WO2025162428/golden_tables",
    out_dir: str = "data/processed",
    skip_decimer: bool = False,
    decimer_limit: int = 0,
    decimer_offset: int = 0,
    rec_top_k: int = 0,
    smiles_route: str = "production",
    skip_ocr: bool = False,
) -> None:
    """Run extraction -> DECIMER -> write the combined CSV in order."""
    img_path = Path(img_dir)
    out_path = Path(out_dir)
    cells_dir = out_path / "cells"
    cache_path = out_path / DECIMER_CACHE_FILENAME

    print(f"[1/4] scanning {img_path} ...")
    jpgs = list(_iter_jpgs(img_path))
    print(f"   -> {len(jpgs)} JPGs")

    print(
        f"\n[2/4] extracting SMILES + structure + KD/LCMS "
        f"(rec_top_k={rec_top_k}, smiles_route={smiles_route}) ..."
    )
    if skip_ocr:
        compounds, assays, failures = _load_from_csv(
            out_path / "compounds.csv", out_path / "assays.csv",
            cells_dir, img_path,
        )
    else:
        compounds, assays, failures = _extract_all(
            img_path, cells_dir,
            rec_top_k=rec_top_k, smiles_route=smiles_route,
        )

    if skip_decimer:
        print("\n[3/4] skipping DECIMER (--skip-decimer); combined CSV will have empty decimer columns")
        decimer_raw, decimer_canon, decimer_min_conf, decimer_low_conf = {}, {}, {}, {}
        decimer_render = {}
    else:
        print(f"\n[3/4] running DECIMER (OCSR) on cropped structures "
              f"(limit={decimer_limit}, offset={decimer_offset}) ...")
        decimer_raw, decimer_canon, decimer_min_conf, decimer_low_conf = _run_decimer(
            compounds, out_path, cache_path=cache_path,
            limit=decimer_limit, offset=decimer_offset,
        )
        print(f"\n[3b/4] rendering DECIMER structures to 2D PNGs (linearize=True) ...")
        decimer_render = _render_decimer(decimer_canon, out_path, linearize=True)
        print(f"   rendered {len(decimer_render)} / {len(decimer_canon)} structures")

    print(f"\n[4/4] writing combined dataset to {out_path} ...")
    combined_rows = _build_combined_rows(
        compounds, assays, decimer_raw, decimer_canon,
        decimer_min_conf, decimer_low_conf, decimer_render,
    )
    # VSCode holds an exclusive write lock on compounds_combined.csv while
    # it is open in the Source Control diff view; writing to the canonical
    # filename raises PermissionError on Windows. Auto-detect and fall back
    # to a sibling name; the user can swap once the editor releases the file.
    canonical_csv = out_path / "compounds_combined.csv"
    swap_csv = out_path / "compounds_combined.new.csv"
    try:
        with open(canonical_csv, "r+b"):
            target_csv = canonical_csv
    except PermissionError:
        target_csv = swap_csv
        print(f"   [!] {canonical_csv.name} is locked; writing to {swap_csv.name} instead")

    n_combined = _write_csv(
        combined_rows,
        target_csv,
        COMBINED_COLUMNS,
    )

    n_decimer = len(decimer_raw)
    n_agree = sum(1 for r in combined_rows if r["smiles_agree"] == "True")

    print("\n" + "=" * 60)
    print(f"compounds_combined.csv : {n_combined} rows")
    print(f"DECIMER SMILES         : {n_decimer} rows; agreement: {n_agree}")
    if failures:
        print(f"[!] images without rows ({len(failures)}): {', '.join(list(failures)[:5])}"
              + ("..." if len(failures) > 5 else ""))

    stats = _l1_breakdown(combined_rows)
    print(_format_l1_block(stats))
    print("=" * 60)


def report_l1(csv_path: str = "data/processed/compounds_combined.csv",
              threshold: float = 0.5) -> None:
    """Print the L1 breakdown from an existing combined CSV (no re-run needed)."""
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader = _csv_mod.DictReader(fh)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    stats = _l1_breakdown(rows, threshold=threshold)
    print(_format_l1_block(stats))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract compounds + assays + DECIMER SMILES from patent tables.",
    )
    p.add_argument("--img-dir", default="data/raw/WO2025162428/golden_tables",
                   help="Directory of source table JPGs (default: %(default)s).")
    p.add_argument("--out-dir", default="data/processed",
                   help="Output directory for CSVs and cropped structures (default: %(default)s).")
    p.add_argument("--skip-decimer", action="store_true",
                   help="Skip the multi-hour DECIMER pass; the combined CSV will have empty decimer columns.")
    p.add_argument("--decimer-limit", type=int, default=0,
                   help="Process at most N cropped structures through DECIMER (0 = all).")
    p.add_argument("--decimer-offset", type=int, default=0,
                   help="Skip the first N cropped structures (use with --decimer-limit to split work across nights).")
    p.add_argument("--rec-top-k", type=int, default=5,
                   help="Per-character top-k returned by the SMILES-cell recogniser (default: 5). "
                        "Set to 0 to disable the RDKit-validated top-k repair.")
    p.add_argument("--smiles-route", choices=("production", "validated"),
                   default="production",
                   help="How the printed SMILES cell is read. 'production' is the "
                        "historical behaviour (English PP-OCRv4; drops the left "
                        "bracket of C(=O)). 'validated' is the route validated on "
                        "patent page I100298 (21/21 printed rows character-exact). "
                        "Default: %(default)s.")
    p.add_argument("--report-l1", action="store_true",
                   help="Skip the pipeline; print the L1 breakdown from the existing combined CSV instead.")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Skip OCR; load OCR text from data/processed/compounds.csv (when available).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.report_l1:
        report_l1()
    else:
        run_pipeline(
            img_dir=args.img_dir,
            out_dir=args.out_dir,
            skip_decimer=args.skip_decimer,
            decimer_limit=args.decimer_limit,
            decimer_offset=args.decimer_offset,
            rec_top_k=args.rec_top_k,
            smiles_route=args.smiles_route,
            skip_ocr=args.skip_ocr,
        )

