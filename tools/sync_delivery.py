"""Copy the curated deliverable from the working repo into the delivery repo.

The two repositories have different jobs
---------------------------------------
``cyclic-peptide`` (this repository)
    The working repository.  It holds the code, every experiment, every
    audit artefact and the research log.  It is where a new patent is
    adapted and where an experiment is run.

``drug_ocr`` (the delivery repository)
    The reviewable deliverable.  It holds the same code, the patent
    inputs, the combined CSV, and the evidence documents that back the
    headline claims -- and nothing else.  Everything a reviewer needs
    in order to re-run the pipeline and check the numbers.

Keeping them in step by hand does not work: the two drift within a
day.  This script copies the curated set from the first to the second
and verifies every file by SHA256, so "the delivery repo matches the
working repo" is a checkable statement rather than a hope.

Identical, with one deliberate exception
----------------------------------------
The two repositories are *not* meant to be byte-identical everywhere,
and pretending otherwise would be a lie in both directions.  The
curated **code, data and evidence** are copied byte for byte, because
the point of the delivery repository is that a reviewer runs the very
code that produced the CSV.  The **narrative** is different: a reader
who opens ``drug_ocr/README.md`` should be told that this is the
delivery copy and where the working one is, and a reader who opens
``cyclic-peptide/README.md`` should be told the opposite.  One file
copied to both places cannot say both.

So the documents that describe the relationship carry a *role banner*:
a marked block that the working repository fills with its own wording
and that the sync rewrites to the delivery wording.  Everything
outside the banner is identical, so the difference between the two
repositories is explicit, small and auditable -- two documents, one
paragraph each -- rather than accidental.

The banner is an *anchor*: if the working wording is not found in the
source file the sync fails loudly.  The wording cannot therefore
drift away from the copy without the sync noticing.

Usage::

    .venv\\Scripts\\python.exe tools/sync_delivery.py
    .venv\\Scripts\\python.exe tools/sync_delivery.py --apply
    .venv\\Scripts\\python.exe tools/sync_delivery.py --apply --prune

Without arguments the script reports what would change.  ``--apply``
copies the files.  ``--prune`` additionally removes anything in the
delivery repository that is not part of the curated set, which is how
the delivery repository is kept free of the working repository's
one-off experiment files.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO.parent / "drug_ocr"

# --- the role banner ----------------------------------------------------
#
# Each entry is (working wording, delivery wording).  The working
# wording is what the working repository's own file contains, and it
# is the anchor the sync looks for.

WORKING_README_BANNER = """<!-- role-banner:start -->
> **Working repository.**  This is `cyclic-peptide`: the code, every
> experiment, every audit artefact and the research log.  The
> reviewable deliverable is a separate repository, `drug_ocr`, which
> `tools/sync_delivery.py` fills from here and verifies file by file.
<!-- role-banner:end -->"""

DELIVERY_README_BANNER = """<!-- role-banner:start -->
> **Delivery repository.**  This is `drug_ocr`: the code, the patent
> inputs, the deliverable CSV and the evidence behind the headline
> claims, and nothing else.  It is filled from the working repository
> `cyclic-peptide` by `tools/sync_delivery.py` and verified file by
> file, so the code here is the code that produced the CSV.
<!-- role-banner:end -->"""

WORKING_PROGRESS_BANNER = """<!-- role-banner:start -->
> **Working repository.**  This log is written from `cyclic-peptide`,
> the working repository, where the experiments are run.  The
> reviewable deliverable is `drug_ocr`, kept in step with this one by
> `tools/sync_delivery.py`.
<!-- role-banner:end -->"""

DELIVERY_PROGRESS_BANNER = """<!-- role-banner:start -->
> **Delivery repository.**  This is `drug_ocr`, the reviewable
> deliverable.  The log it carries was written from the working
> repository `cyclic-peptide`, where the experiments were run, and is
> kept in step with it by `tools/sync_delivery.py`.  The code, the data
> and the evidence are byte-identical to the ones there.
<!-- role-banner:end -->"""

# The documents that describe the relationship between the two
# repositories, and the wording each one uses.  The key is the path of
# the file *in the target repository*.
ROLE_VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "README.md": ((WORKING_README_BANNER, DELIVERY_README_BANNER),),
    "PROGRESS.md": ((WORKING_PROGRESS_BANNER, DELIVERY_PROGRESS_BANNER),),
}

# --- the curated set ------------------------------------------------------

# (source glob relative to the working repo, destination relative to the
# delivery repo).  A destination ending in "/" is a directory: the
# matched file keeps its name inside it.  Any other destination names
# the file to write.  A glob that matches nothing is reported as an
# error, because a missing input should never pass silently.
MANIFEST: tuple[tuple[str, str], ...] = (
    ("main.py", "main.py"),
    ("conftest.py", "conftest.py"),
    ("pyproject.toml", "pyproject.toml"),
    ("uv.lock", "uv.lock"),
    ("LICENSE", "LICENSE"),
    (".gitignore", ".gitignore"),
    ("README.md", "README.md"),
    ("PROGRESS.md", "PROGRESS.md"),
    ("src/cpd/*.py", "src/cpd/"),
    ("src/cpd/ocr/*.py", "src/cpd/ocr/"),
    ("src/cpd/parsers/*.py", "src/cpd/parsers/"),
    ("tests/*.py", "tests/"),
    ("tools/*.py", "tools/"),
    ("docs/*.md", "docs/"),
    ("data/raw/WO2025162428/golden_tables/*.jpg", "data/raw/WO2025162428/golden_tables/"),
    ("data/raw/WO2022251372/golden_tables/*", "data/raw/WO2022251372/golden_tables/"),
    ("data/processed/compounds_combined.csv", "data/processed/"),
    # The second patent's own deliverable.  It has its own activity
    # columns, so it is a separate file.
    ("data/processed/WO2022251372/compounds_combined.csv", "data/processed/WO2022251372/"),
    ("data/processed/archive/*.csv", "data/processed/archive/"),
    # The evidence that backs the headline numbers.
    ("data/audit/phase3/PHASE3_SUMMARY.md", "data/audit/phase3/"),
    ("data/audit/phase3/reference_status.md", "data/audit/phase3/"),
    ("data/audit/phase3/validated_corpus/*.csv", "data/audit/phase3/validated_corpus/"),
    ("data/audit/phase3/validated_corpus/*.json", "data/audit/phase3/validated_corpus/"),
    ("data/audit/phase5/SUMMARY.md", "data/audit/phase5/"),
    # The row-crop audit: the finding that a leading ``=`` is excluded
    # from the crop before the recogniser ever sees it, and the
    # measurement of how many cells are affected.
    ("data/audit/phase2/row_crop_audit/summary.txt", "data/audit/phase2/row_crop_audit/"),
    ("data/audit/phase2/row_crop_audit/manifest.csv", "data/audit/phase2/row_crop_audit/"),
    ("data/audit/phase2/row_crop_audit/coordinate_reconciliation.txt", "data/audit/phase2/row_crop_audit/"),
    ("data/audit/phase2/row_crop_audit/flagged_montage.png", "data/audit/phase2/row_crop_audit/"),
    ("data/audit/phase2/pre_report_validation/pre_report_summary.md", "data/audit/phase2/pre_report_validation/"),
    ("data/audit/phase2/pre_report_validation/ab_results.csv", "data/audit/phase2/pre_report_validation/"),
    ("data/audit/phase2/ocr_compare_summary.txt", "data/audit/phase2/"),
    ("data/audit/phase2/box_expansion/*.txt", "data/audit/phase2/box_expansion/"),
    ("data/audit/phase2/box_expansion/*.csv", "data/audit/phase2/box_expansion/"),
    ("data/audit/phase2/box_expansion/README.md", "data/audit/phase2/box_expansion/"),
    ("data/audit/phase2/threshold_scan/*.txt", "data/audit/phase2/threshold_scan/"),
    ("data/audit/phase2/threshold_scan/*.csv", "data/audit/phase2/threshold_scan/"),
    ("data/audit/phase2/threshold_scan/README.md", "data/audit/phase2/threshold_scan/"),
    ("data/audit/phase2/recognizer_candidates/*.txt", "data/audit/phase2/recognizer_candidates/"),
    ("data/audit/phase2/recognizer_candidates/*.csv", "data/audit/phase2/recognizer_candidates/"),
    ("data/audit/phase3/external/*.md", "data/audit/phase3/external/"),
    ("data/audit/phase3/external/*.csv", "data/audit/phase3/external/"),
    ("data/audit/phase3/external/*.json", "data/audit/phase3/external/"),
    ("data/audit/phase3/external/figure_*.png", "data/audit/phase3/external/"),
    ("data/audit/phase3/i100298_final/*.txt", "data/audit/phase3/i100298_final/"),
    ("data/audit/phase3/i100298_final/*.csv", "data/audit/phase3/i100298_final/"),
    ("data/audit/phase3/i100298_final/figure_*.png", "data/audit/phase3/i100298_final/"),
    ("data/audit/demos/topk/rapidocr_topk/*.json", "data/audit/demos/topk/rapidocr_topk/"),
    ("data/audit/demos/topk/rapidocr_topk/*.txt", "data/audit/demos/topk/rapidocr_topk/"),
    ("data/audit/demos/topk/rapidocr_topk_backup/*", "data/audit/demos/topk/rapidocr_topk_backup/"),
    # The report figures.  PROGRESS.md links to them, so a delivery
    # repository without them renders a broken document.
    ("data/audit/demos/figures/*.png", "data/audit/demos/figures/"),
    ("data/audit/demos/render/pipeline_render/*.png", "data/audit/demos/render/pipeline_render/"),
    ("data/audit/demos/render/roundtrip/*.png", "data/audit/demos/render/roundtrip/"),
    ("data/audit/phase6/SUMMARY.md", "data/audit/phase6/"),
    ("data/audit/phase6/*.csv", "data/audit/phase6/"),
    ("data/audit/phase6/*.json", "data/audit/phase6/"),
    ("data/audit/phase7/*.csv", "data/audit/phase7/"),
    ("data/audit/phase7/*.txt", "data/audit/phase7/"),
    ("data/audit/phase7/*.md", "data/audit/phase7/"),
    ("data/audit/phase8/*.csv", "data/audit/phase8/"),
    ("data/audit/phase8/*.json", "data/audit/phase8/"),
    ("data/audit/phase8/*.txt", "data/audit/phase8/"),
    ("data/audit/phase8/*.md", "data/audit/phase8/"),
    ("data/audit/phase9/*.json", "data/audit/phase9/"),
    ("data/audit/phase9/*.md", "data/audit/phase9/"),
    ("data/audit/phase9/*.csv", "data/audit/phase9/"),
    ("data/audit/phase9/*.txt", "data/audit/phase9/"),
    # The 28 printed-SMILES cells of Table 1a and the multi-line views
    # of the transcribed compounds: the pixels a reviewer checks the
    # reading against.
    ("data/audit/phase4/wo2022251372/table1a_cells/*.png", "data/audit/phase4/wo2022251372/table1a_cells/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd1_lines_1_6.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd1_lines_7_10.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd4_lines_1_7.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd4_lines_8_10.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd10_lines_1_7.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd10_lines_8_10.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd28_lines_1_6.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    ("data/audit/phase4/wo2022251372/visual_checks/cmpd28_lines_7_10.png", "data/audit/phase4/wo2022251372/visual_checks/"),
    # The second patent's delivery run and its numbers.
    ("data/audit/phase10/*.json", "data/audit/phase10/"),
    ("data/audit/phase10/*.csv", "data/audit/phase10/"),
    ("data/audit/phase10/figure_patent2_delivery.png", "data/audit/phase10/"),
    # The patent-2 transcription audit.  Only the three pixels that
    # back the three corrections travel -- the other sixty bands are
    # regenerated by ``phase9_patent2_reference_audit.py``.
    ("data/audit/phase9/patent2_reference_audit/reference_audit.csv", "data/audit/phase9/patent2_reference_audit/"),
    ("data/audit/phase9/patent2_reference_audit/cmpd10_line07.png", "data/audit/phase9/patent2_reference_audit/"),
    ("data/audit/phase9/patent2_zoom/cmpd28_band2_line03.png", "data/audit/phase9/patent2_zoom/"),
    ("data/audit/phase9/patent2_zoom/cmpd28_band2_line05.png", "data/audit/phase9/patent2_zoom/"),
    # The scripts that reproduce the headline numbers.  A reviewer
    # must be able to re-run the audit that produced them, so these
    # travel with the evidence instead of staying in the working
    # repository.
    ("analysis/verify_deliverable.py", "analysis/"),
    ("analysis/make_1036_progress_figure.py", "analysis/"),
    ("analysis/make_report_figure.py", "analysis/"),
    ("analysis/render_cmpd_1036_styles.py", "analysis/"),
    ("analysis/run_roundtrip_check.py", "analysis/"),
    ("experiments/rebuild_csv.py", "experiments/"),
    ("experiments/phase6_wo2022251372_extract.py", "experiments/"),
    ("experiments/phase6_verify_accuracy.py", "experiments/"),
    ("experiments/phase6_boundary_measure.py", "experiments/"),
    ("experiments/phase6_diff_detail.py", "experiments/"),
    ("experiments/phase6_profile_snapshot.py", "experiments/"),
    ("experiments/phase7_activity_validate.py", "experiments/"),
    ("experiments/phase7_activity_dump.py", "experiments/"),
    ("experiments/phase7_activity_policy.py", "experiments/"),
    ("experiments/phase7_activity_diff.py", "experiments/"),
    ("experiments/phase7_cell_probe.py", "experiments/"),
    ("experiments/phase7_lcms_probe.py", "experiments/"),
    ("experiments/phase7_numeric_strategies.py", "experiments/"),
    ("experiments/phase7_line_probe.py", "experiments/"),
    ("experiments/phase7_route_verify.py", "experiments/"),
    ("experiments/phase7_route_diff.py", "experiments/"),
    ("experiments/phase7_cell_variants.py", "experiments/"),
    ("experiments/phase7_1037_probe.py", "experiments/"),
    ("experiments/phase7_sync_diff.py", "experiments/"),
    ("experiments/phase8_wo2022251372_line_vote.py", "experiments/"),
    ("experiments/phase8_vote_rule.py", "experiments/"),
    ("experiments/phase8_vote_probe.py", "experiments/"),
    ("experiments/phase8_patent2_route_compare.py", "experiments/"),
    ("experiments/phase8_patent2_route_diff.py", "experiments/"),
    ("experiments/phase8_pipeline_cells_corpus.py", "experiments/"),
    ("experiments/phase8_vote_evidence.py", "experiments/"),
    ("experiments/phase9_delivery_check.py", "experiments/"),
    ("experiments/phase9_csv_diff.py", "experiments/"),
    ("experiments/phase9_csv_stats.py", "experiments/"),
    ("experiments/phase9_zero_o_scan.py", "experiments/"),
    ("experiments/phase9_bond_guard_evidence.py", "experiments/"),
    ("experiments/phase9_missing_bracket_scan.py", "experiments/"),
    ("experiments/phase9_bracket_rule_effect.py", "experiments/"),
    ("experiments/phase9_vote_unit_check.py", "experiments/"),
    ("experiments/phase9_reference_check.py", "experiments/"),
    ("experiments/phase9_patent2_corrected_references.py", "experiments/"),
    ("experiments/phase9_patent2_reference_audit.py", "experiments/"),
    ("experiments/phase9_patent2_diff.py", "experiments/"),
    ("experiments/phase9_patent2_lines.py", "experiments/"),
    ("experiments/phase9_patent2_trace.py", "experiments/"),
    ("experiments/phase9_patent2_zoom.py", "experiments/"),
    ("experiments/phase9_patent2_zoom_region.py", "experiments/"),
    ("experiments/phase9_patent2_zoom_rows.py", "experiments/"),
    ("experiments/phase9_mirrored_bracket_rule.py", "experiments/"),
    ("experiments/phase9_bracket_flip_scan.py", "experiments/"),
    ("experiments/phase9_bracket_flip_patent2.py", "experiments/"),
    ("experiments/phase10_patent2_delivery.py", "experiments/"),
    ("experiments/phase10_patent2_inspect.py", "experiments/"),
    ("experiments/phase10_patent2_figure.py", "experiments/"),
    ("experiments/demo_csv_vs_golden.py", "experiments/"),
    ("experiments/verify_csv.py", "experiments/"),
    ("experiments/wo2022251372_references.py", "experiments/"),
)

# Files that are **per repository** and are therefore not copied.
#
# The delivery check records the path of the repository it ran in, so
# a copy of the working repository's report would name the working
# repository inside the delivery repository -- which is exactly the
# mistake the check exists to catch.  Each repository keeps its own
# copy, written by running the check inside it.
PER_REPOSITORY: tuple[str, ...] = (
    "data/audit/phase9/delivery_check.json",
)

# Files in the delivery repository that are never removed by
# ``--prune``: the raw patent inputs are the ground truth of every
# measurement, and the repository's own metadata is not ours to touch.
PRUNE_KEEP: tuple[str, ...] = (
    ".git",
    ".venv",
    ".gitignore",
    ".pytest_cache",
    ".pytesttmp",
    "__pycache__",
    "data/raw",
    "data/processed/cells",
    "data/processed/rendered",
    # Read-only upstream clones (RapidOCR, DECIMER, Hiro-OCSR).  They
    # are ignored by git and are not ours to delete.
    "references",
    # The delivery check records the repository it ran in, so each
    # repository keeps its own copy.  It is written by running
    # ``phase9_delivery_check.py`` inside the repository.
    "data/audit/phase9/delivery_check.json",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(rel: str, text: str) -> str:
    """Return the text the target repository should hold for ``rel``.

    Only the role banner differs between the two repositories; the
    rest of the curated set is copied verbatim.  The working wording
    is the anchor: if it is missing, the file was edited without the
    banner being kept in step, and the sync refuses to guess.

    The comparison is done with the line endings normalised, because
    the curated files are not consistent about them: ``README.md`` is
    CRLF and ``PROGRESS.md`` is LF, and the banner is written with LF.
    """
    crlf = "\r\n" in text
    normalised = text.replace("\r\n", "\n")
    for working, delivery in ROLE_VARIANTS.get(rel, ()):
        if working not in normalised:
            # Already the delivery wording: the file has been rendered
            # before.  Rendering it again is a no-op rather than an
            # error, so the tool is safe to run from inside either
            # repository.
            if delivery in normalised:
                continue
            raise SystemExit(
                f"ERROR: {rel} does not contain the working-repository "
                f"role banner, so the delivery wording cannot be "
                f"substituted.\n"
                f"   expected to find:\n"
                f"   {working.splitlines()[0]}\n"
                f"   ... and the matching 'role-banner:end' marker."
            )
        normalised = normalised.replace(working, delivery)
    if crlf:
        normalised = normalised.replace("\n", "\r\n")
    return normalised


def _read_text(path: Path) -> str:
    """Read a curated file as text, keeping its line endings."""
    return path.read_bytes().decode("utf-8")


def collect(source: Path) -> list[tuple[Path, Path]]:
    """Return the ``(source, destination)`` pairs of the curated set."""
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for pattern, destination in MANIFEST:
        matches = sorted(source.glob(pattern))
        if not matches:
            # A pattern that names a *generated* evidence directory
            # may legitimately be empty on a fresh checkout.  Only the
            # inputs of the pipeline -- the code, the patent images,
            # the schema -- are hard errors.
            if not pattern.startswith("data/audit/"):
                missing.append(pattern)
            continue
        for match in matches:
            if match.is_dir():
                continue
            relative = match.relative_to(source)
            if relative.as_posix() in PER_REPOSITORY:
                continue
            # ``destination`` is a directory for the patterns that end in
            # a slash ("src/cpd/*.py" -> "src/cpd/"), and a *file* for the
            # ones that name a file directly ("main.py" -> "main.py").
            # Appending the name to a file path would produce
            # "main.py/main.py".
            destination_path = Path(destination)
            if destination.endswith("/"):
                pairs.append((match, destination_path / match.name))
            else:
                pairs.append((match, destination_path))
    if missing:
        print("ERROR: these manifest patterns matched nothing:", file=sys.stderr)
        for pattern in missing:
            print(f"   {pattern}", file=sys.stderr)
        raise SystemExit(2)
    return pairs


def compare(source: Path, target: Path) -> tuple[int, int, int]:
    """Count identical, differing and missing files."""
    same = differ = missing = 0
    for src, dst_rel in collect(source):
        dst = target / dst_rel
        if not dst.exists():
            missing += 1
        elif _expected_hash(src, dst_rel) == _sha256(dst):
            same += 1
        else:
            differ += 1
    return same, differ, missing


def _expected_hash(src: Path, dst_rel: Path) -> str:
    """Hash the bytes the target should hold for this curated file."""
    if dst_rel.as_posix() in ROLE_VARIANTS:
        rendered = render(dst_rel.as_posix(), _read_text(src))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return _sha256(src)


def list_differences(source: Path, target: Path) -> list[str]:
    """Name the files that are not identical, and how they differ.

    The counts alone do not say *what* is out of step, and a sync
    that cannot be inspected is a sync nobody can trust.  This is the
    detail behind the numbers ``compare`` reports.
    """
    problems: list[str] = []
    for src, dst_rel in collect(source):
        dst = target / dst_rel
        if not dst.exists():
            problems.append(f"missing   {dst_rel}")
        elif _expected_hash(src, dst_rel) != _sha256(dst):
            problems.append(f"different {dst_rel}")
    return problems


def check_links(source: Path, target: Path) -> list[str]:
    """Report the links of the curated documents that would be broken.

    The delivery repository has to stand on its own: a progress report
    that links to a figure which is not in the repository renders as a
    broken image for the reader.  Every relative link of the curated
    markdown files is checked against the curated set.
    """
    curated = {(target / dst_rel).resolve() for _, dst_rel in collect(source)}
    broken: list[str] = []
    pattern = re.compile(r"\]\(([^)#][^)]*)\)")
    # A markdown link points at a file.  A chemical string such as
    # ``[C@@H](C`` has the same shape but is not a path, so only the
    # targets that look like a file are checked.
    looks_like_a_path = re.compile(
        r"^[A-Za-z0-9_./\\ -]+\.(?:png|jpg|jpeg|md|csv|json|txt|py)$",
        re.IGNORECASE,
    )
    for _, dst_rel in collect(source):
        if dst_rel.suffix.lower() != ".md":
            continue
        document = target / dst_rel
        if not document.exists():
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            link = match.group(1).strip()
            if not link or link.startswith(("http://", "https://", "mailto:")):
                continue
            if not looks_like_a_path.match(link):
                continue
            # A relative link is resolved against the directory of the
            # document that holds it, and against the root of the
            # repository.
            beside = (document.parent / link).resolve()
            if beside in curated or (target / link).resolve() in curated:
                continue
            broken.append(f"{dst_rel.as_posix()} -> {link}")
    return broken


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a file, falling back to a plain read and write.

    ``shutil.copy2`` goes through ``CopyFile2``, which fails with a
    sharing violation when anything else on the machine happens to
    hold the file open -- an image preview or a virus scanner is
    enough.  The comparison downstream only looks at the bytes, so a
    plain read and write is an acceptable substitute.  A failed
    ``CopyFile2`` can leave the destination locked for a moment, so
    the fallback retries briefly before giving up.
    """
    try:
        shutil.copy2(src, dst)
        return
    except OSError:
        pass

    payload = src.read_bytes()
    for attempt in range(5):
        try:
            dst.write_bytes(payload)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass


def apply(source: Path, target: Path, prune: bool) -> None:
    pairs = collect(source)
    for src, dst_rel in pairs:
        dst = target / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst_rel.as_posix() in ROLE_VARIANTS:
            # The role banner is the one thing that differs between the
            # two repositories; the rest of the file is copied as it
            # is, so the code and the numbers stay byte-identical.
            rendered = render(dst_rel.as_posix(), _read_text(src))
            dst.write_bytes(rendered.encode("utf-8"))
            shutil.copystat(src, dst)
        else:
            _copy_file(src, dst)
    print(f"copied {len(pairs)} files into {target}")

    if not prune:
        return

    curated = {(target / dst_rel).resolve() for _, dst_rel in pairs}
    keep_roots = {(target / name).resolve() for name in PRUNE_KEEP}

    # Walk the delivery repository and delete every file that the
    # curated set does not mention.  The exception is the raw patent
    # input and the caches, which are ground truth and build output
    # rather than curated content.
    removed = 0
    for path in sorted(target.rglob("*"), reverse=True):
        resolved = path.resolve()
        if any(
            resolved == root or root in resolved.parents
            for root in keep_roots
        ):
            continue
        if path.is_file():
            if resolved in curated:
                continue
            path.unlink()
            removed += 1
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # Not empty: a curated file still lives in it.
                pass
    print(f"pruned {removed} files that are not part of the curated set")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET,
        help="The delivery repository (default: %(default)s).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Copy the files.  Without it the script only reports.",
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="With --apply, also remove everything that is not curated.",
    )
    args = parser.parse_args(argv)

    source = REPO
    target = args.target.resolve()
    if not target.is_dir():
        print(f"delivery repository not found: {target}", file=sys.stderr)
        return 2

    same, differ, missing = compare(source, target)
    total = same + differ + missing
    print(f"curated files      : {total}")
    print(f"identical          : {same}")
    print(f"different          : {differ}")
    print(f"missing in target  : {missing}")

    problems = list_differences(source, target)
    if problems:
        print("\nthe files that are out of step:")
        for problem in problems:
            print(f"    {problem}")

    broken = check_links(source, target)
    print(f"broken links       : {len(broken)}")
    for link in broken[:20]:
        print(f"    {link}")

    if not args.apply:
        print("\ndry run; pass --apply to copy")
        return 0 if (differ == 0 and missing == 0 and not broken) else 1

    apply(source, target, args.prune)
    same, differ, missing = compare(source, target)
    print(f"after the copy     : identical {same}, different {differ}, missing {missing}")
    broken = check_links(source, target)
    print(f"broken links       : {len(broken)}")
    for link in broken[:20]:
        print(f"    {link}")
    return 0 if (differ == 0 and missing == 0 and not broken) else 1


if __name__ == "__main__":
    raise SystemExit(main())

