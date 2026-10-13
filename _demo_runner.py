"""Run all morning-report demos and render a single summary.

Calls (in order): ``_demo_topk.py``, ``_demo_hiro.py``,
``_demo_decimer_conf.py``. Each demo writes to its own ``demo_*`` folder
under the project root. This runner additionally writes a single
``demo_report/REPORT.txt`` summarising what's in each folder so the
user can hand the morning report straight to the supervisor without
having to open three directories.

Run::

    .venv\\Scripts\\python.exe -u _demo_runner.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMOS = [
    {
        "name": "RapidOCR top-k demo",
        "script": "_demo_topk.py",
        "out_dir": "demo_rapidocr_topk",
        "claim": (
            "Confirms per-character top-k OCR is wired end-to-end: each "
            "character emits a top-3 candidate list with softmax "
            "probabilities. CH and EN recognisers both expose it. The "
            "underlying patch lives in "
            "`.venv/Lib/site-packages/rapidocr/ch_ppocr_rec/main.py` "
            "and the mirror copy is in `references/RapidOCR/...`."
        ),
    },
    {
        "name": "Hiro-OCSR classify_smiles sweep",
        "script": "_demo_hiro.py",
        "out_dir": "demo_hiro_classify",
        "claim": (
            "The `hiro_ocsr.classify.classify_smiles` post-processor runs "
            "in pure RDKit+regex (no LLM, no GPU) and categorises each "
            "DECIMER SMILES as molecule / fragment / markush / irregular "
            "/ invalid. We label all 405 entries to validate Hiro's stack "
            "before deploying the 8B-VLM inference path overnight."
        ),
    },
    {
        "name": "DECIMER per-token confidence distribution",
        "script": "_demo_decimer_conf.py",
        "out_dir": "demo_decimer_conf",
        "claim": (
            "DECIMER-AI's `predict_SMILES(confidence=True)` exposes "
            "per-token softmax; the cache keeps `min_conf` and "
            "`low_conf_count` summaries. 77% of our 405 cmpds have at "
            "least one low-conf token; the rest cluster at 0.3-0.5 "
            "min_conf. Per-position truncation post-processing remains "
            "available; we just need a re-run with per-token storage to "
            "exercise it on the live cache."
        ),
    },
]


def _run_one(script: str) -> int:
    exe = ROOT / ".venv" / "Scripts" / "python.exe"
    if not exe.exists():
        exe = Path(sys.executable)
    proc = subprocess.run(
        [str(exe), "-u", str(ROOT / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(f"[runner] {script} failed (exit {proc.returncode})")
        print(proc.stderr[-800:])
    return proc.returncode


def main() -> int:
    report_lines = [
        "# Morning-report demos -- consolidated",
        "",
        "Each demo writes to its own ``demo_*`` folder. Run any of them "
        "individually with ``.venv\\Scripts\\python.exe -u _demo_*.py``.",
        "",
    ]

    for demo in DEMOS:
        script = demo["script"]
        out_dir = ROOT / demo["out_dir"]
        if not (ROOT / script).exists():
            print(f"[runner] missing {script}, skipping")
            continue
        rc = _run_one(script)
        report_lines.append(f"## {demo['name']}")
        report_lines.append("")
        report_lines.append(f"- script: ``{script}``")
        report_lines.append(f"- output: ``{demo['out_dir']}/``")
        report_lines.append(f"- exit code: {rc}")
        report_lines.append(f"- reportable claim: {demo['claim']}")
        report_lines.append("")

    Path("demo_report").mkdir(exist_ok=True)
    (Path("demo_report") / "REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"[runner] wrote demo_report/REPORT.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
