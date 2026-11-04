"""Tests for the two-repository sync.

The delivery repository is a copy of the working one, and the point
of the copy is that a reviewer runs the code that produced the CSV.
So the code, the data and the evidence are byte-identical -- but the
narrative is not, and cannot be: a document that says "this is the
working repository" is wrong in the delivery repository, and one that
says "this is the delivery repository" is wrong in the working one.

These tests pin that contract down so it cannot rot.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_sync_module():
    """Load ``tools/sync_delivery.py`` as a module.

    It lives in ``tools/`` rather than in the package, so it is loaded
    from its path instead of being imported.
    """
    path = REPO / "tools" / "sync_delivery.py"
    spec = importlib.util.spec_from_file_location("sync_delivery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_banner_is_substituted() -> None:
    """The working wording is replaced by the delivery wording."""
    sync = _load_sync_module()
    text = (
        "# Title\n\n"
        f"{sync.WORKING_README_BANNER}\n"
        "\nrest of the document\n"
    )
    rendered = sync.render("README.md", text)
    assert sync.WORKING_README_BANNER not in rendered
    assert sync.DELIVERY_README_BANNER in rendered
    # Everything outside the banner is untouched.
    assert "rest of the document" in rendered


def test_role_banner_keeps_line_endings() -> None:
    """A CRLF document stays CRLF, an LF document stays LF."""
    sync = _load_sync_module()
    lf = "# Title\n\n" + sync.WORKING_README_BANNER + "\n"
    crlf = lf.replace("\n", "\r\n")
    assert "\r\n" in sync.render("README.md", crlf)
    assert "\r\n" not in sync.render("README.md", lf)


def test_missing_banner_is_an_error() -> None:
    """A document that lost its banner is refused, not guessed at."""
    sync = _load_sync_module()
    try:
        sync.render("README.md", "# Title\n\nno banner here\n")
    except SystemExit as exc:
        assert "role banner" in str(exc)
    else:
        raise AssertionError("a missing role banner must be refused")


def test_working_repo_documents_carry_the_working_banner() -> None:
    """The working repository's own documents carry the working wording.

    Skipped when the suite runs inside the delivery repository, where
    the documents carry the delivery wording instead.
    """
    sync = _load_sync_module()
    if _is_delivery_copy(REPO):
        return
    for name, working in (
        ("README.md", sync.WORKING_README_BANNER),
        ("PROGRESS.md", sync.WORKING_PROGRESS_BANNER),
    ):
        text = (REPO / name).read_text(encoding="utf-8")
        assert working in text, f"{name} has lost its working role banner"


def _is_delivery_copy(repo: Path) -> bool:
    """True when ``repo`` is the delivery copy rather than the working one.

    The delivery repository carries the same tool and the same tests,
    so they have to work from either side.  The repository is
    recognised by the wording its documents carry.
    """
    sync = _load_sync_module()
    text = (repo / "README.md").read_text(encoding="utf-8")
    return sync.DELIVERY_README_BANNER in text


def test_code_and_data_are_not_role_variants() -> None:
    """Only the narrative is role-specific; code and data are not.

    If a code or data file ever became a role variant, the delivery
    repository would no longer be running the code that produced the
    CSV -- the guarantee the whole copy exists for.
    """
    sync = _load_sync_module()
    for rel in sync.ROLE_VARIANTS:
        assert rel.endswith(".md"), f"{rel} must be a document, not code or data"
    for rel in ("main.py", "src/cpd/chem.py", "data/processed/compounds_combined.csv"):
        assert rel not in sync.ROLE_VARIANTS


def test_curated_set_has_no_broken_links() -> None:
    """Every relative link of the curated documents resolves."""
    sync = _load_sync_module()
    target = REPO.parent / "drug_ocr"
    if not target.is_dir():
        return  # the delivery repository is not checked out here
    broken = sync.check_links(REPO, target)
    assert not broken, f"broken links in the delivery repository: {broken}"


def test_delivery_repository_is_in_step() -> None:
    """The delivery repository matches the working one, file for file.

    ``compare`` compares the *rendered* text, so the two role banners
    do not show up as a difference: they are the one intended
    difference.
    """
    sync = _load_sync_module()
    target = REPO.parent / "drug_ocr"
    if not target.is_dir():
        return  # the delivery repository is not checked out here
    same, differ, missing = sync.compare(REPO, target)
    assert differ == 0, f"{differ} curated files are out of step"
    assert missing == 0, f"{missing} curated files are missing"
    assert same > 300


def test_delivery_documents_say_which_repository_they_are() -> None:
    """The delivery copy says "delivery", the working copy says "working".

    This is the difference the two repositories are *supposed* to
    have.  A supervisor opening the delivery repository must not be
    told that it is the working one.
    """
    sync = _load_sync_module()
    target = REPO.parent / "drug_ocr"
    if not target.is_dir():
        return  # the delivery repository is not checked out here
    if _is_delivery_copy(REPO):
        # Running inside the delivery repository: the pair of documents
        # to compare is this one and the working repository, and the
        # working one is not checked out next to it.
        return
    for name, working, delivery in (
        ("README.md", sync.WORKING_README_BANNER, sync.DELIVERY_README_BANNER),
        ("PROGRESS.md", sync.WORKING_PROGRESS_BANNER, sync.DELIVERY_PROGRESS_BANNER),
    ):
        in_working = (REPO / name).read_text(encoding="utf-8")
        in_delivery = (target / name).read_text(encoding="utf-8")
        assert working in in_working
        assert delivery in in_delivery
        assert working not in in_delivery
        assert delivery not in in_working


def test_curated_files_are_not_git_ignored() -> None:
    """Every curated file can actually be committed.

    The sync copies the curated set into the delivery repository, but
    a file that ``.gitignore`` excludes is a file that a reviewer who
    clones the repository never receives.  This happened for real: the
    patent-2 golden tables were curated and copied, but the ignore
    rule for ``data/raw/*`` excluded them, so the ground truth for the
    second patent was silently missing from the deliverable.

    The check is run in the delivery repository as well, because that
    is the copy a reviewer gets.
    """
    sync = _load_sync_module()
    relative = [destination.as_posix() for _, destination in sync.collect(REPO)]
    assert relative

    for repository in (REPO, REPO.parent / "drug_ocr"):
        if not repository.is_dir():
            continue
        # Bytes, not text: on Windows a text pipe rewrites "\n" as
        # "\r\n" and git then looks up paths that do not exist, which
        # reports a file as ignored when it is not.
        payload = ("\n".join(relative) + "\n").encode("utf-8")
        result = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "--stdin"],
            input=payload,
            capture_output=True,
        )
        ignored = [
            line for line in result.stdout.decode("utf-8").splitlines() if line.strip()
        ]
        assert not ignored, (
            f"{len(ignored)} curated files are git-ignored in {repository.name}: "
            f"{ignored[:5]}"
        )
