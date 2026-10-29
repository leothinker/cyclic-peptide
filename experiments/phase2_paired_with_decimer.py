from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from cpd.ms_validation import validate_against_ms

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/processed/compounds_combined.csv"
OUT_DIR = ROOT / "data/audit/phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "paired_alignment_with_decimer.csv"

def _to_float(x):
    if x in (None, "", "nan"): return None
    try: return float(x)
    except Exception: return None

def _row_for(df, i, smi):
    pol = str(df.iloc[i]["ms_polarity"] or "").strip()
    mz = _to_float(df.iloc[i]["ms_mz"])
    if not smi or smi.lower() == "nan":
        return dict(formula="", observed_mass="", expected_mass="",
                    delta_da="", abs_error_ppm="", tolerance_ppm="",
                    verdict="no_candidate", parsed_by_rdkit="")
    res = validate_against_ms(smi, mz, pol)
    obs = res.get("observed_mass"); exp = res.get("expected_mass")
    delta = (float(obs) - float(exp)) if (obs is not None and exp is not None) else None
    abs_ppm = abs(res["error_ppm"]) if res.get("error_ppm") is not None else None
    tol = res.get("tolerance_ppm")
    parsed = "yes" if Chem.MolFromSmiles(smi) else "no"
    return dict(formula=(res.get("formula") or ""),
                observed_mass=(obs if obs is not None else ""),
                expected_mass=(exp if exp is not None else ""),
                delta_da=(round(delta, 4) if delta is not None else ""),
                abs_error_ppm=(round(abs_ppm, 2) if abs_ppm is not None else ""),
                tolerance_ppm=(round(tol, 2) if tol is not None else ""),
                verdict=(res.get("verdict") or "unknown"),
                parsed_by_rdkit=parsed)

def _norm(s):
    if s is None: return ""
    s = str(s).strip()
    return "" if s.lower() == "nan" else s

def main():
    df = pd.read_csv(CSV_PATH)
    rows = []
    for i in range(len(df)):
        cid = int(df.iloc[i]["cmpd_id"])
        decimer = _norm(df.iloc[i]["smiles_decimer"])
        best = _norm(df.iloc[i]["smiles_best"])
        repaired = _norm(df.iloc[i]["repaired_smiles"])
        method = _norm(df.iloc[i]["repair_method"])
        case = "case1_intact_decimer" if best else ("case3_repair_attempted" if repaired else "no_candidate")
        bd = _row_for(df, i, best)
        rd = _row_for(df, i, repaired)
        dd = _row_for(df, i, decimer)
        shift = ""
        if rd["delta_da"] != "" and dd["delta_da"] != "":
            try: shift = round(float(rd["delta_da"]) - float(dd["delta_da"]), 4)
            except Exception: pass
        rows.append(dict(
            cmpd_id=cid, pipeline_case=case,
            ms_polarity=_norm(df.iloc[i]["ms_polarity"]),
            ms_mz=_norm(df.iloc[i]["ms_mz"]),
            repair_method=method,
            smiles_decimer=decimer, decimer_formula=dd["formula"],
            decimer_observed_mass=dd["observed_mass"],
            decimer_expected_mass=dd["expected_mass"],
            decimer_delta_da=dd["delta_da"],
            decimer_abs_error_ppm=dd["abs_error_ppm"],
            decimer_tolerance_ppm=dd["tolerance_ppm"],
            decimer_verdict=dd["verdict"],
            decimer_parsed_by_rdkit=dd["parsed_by_rdkit"],
            smiles_best=best, best_formula=bd["formula"],
            best_observed_mass=bd["observed_mass"],
            best_expected_mass=bd["expected_mass"],
            best_delta_da=bd["delta_da"],
            best_abs_error_ppm=bd["abs_error_ppm"],
            best_tolerance_ppm=bd["tolerance_ppm"],
            best_verdict=bd["verdict"],
            best_parsed_by_rdkit=bd["parsed_by_rdkit"],
            repaired_smiles=repaired, repaired_formula=rd["formula"],
            repaired_observed_mass=rd["observed_mass"],
            repaired_expected_mass=rd["expected_mass"],
            repaired_delta_da=rd["delta_da"],
            repaired_abs_error_ppm=rd["abs_error_ppm"],
            repaired_tolerance_ppm=rd["tolerance_ppm"],
            repaired_verdict=rd["verdict"],
            repaired_parsed_by_rdkit=rd["parsed_by_rdkit"],
            repaired_minus_decimer_delta_da=shift))
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote", OUT_CSV, "(", len(rows), "rows)")
    import statistics
    print("case1:", sum(1 for r in rows if r["pipeline_case"]=="case1_intact_decimer"))
    print("case3:", sum(1 for r in rows if r["pipeline_case"]=="case3_repair_attempted"))
    print("no_candidate:", sum(1 for r in rows if r["pipeline_case"]=="no_candidate"))
    print("decimer in tol:", sum(1 for r in rows if r["decimer_verdict"]=="match"))
    print("best in tol:", sum(1 for r in rows if r["best_verdict"]=="match"))
    print("repaired in tol:", sum(1 for r in rows if r["repaired_verdict"]=="match"))
    shifts = [r["repaired_minus_decimer_delta_da"] for r in rows if isinstance(r["repaired_minus_decimer_delta_da"], (int, float))]
    if shifts:
        print("repair shifts (repaired - decimer, n=" + str(len(shifts)) + "): median", round(statistics.median(shifts), 3), "mean", round(sum(shifts)/len(shifts), 3))

if __name__ == "__main__":
    main()
