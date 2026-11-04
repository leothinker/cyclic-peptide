"""Check the README project tree against the two repositories.

The README shows the layout of the working repository.  The delivery
repository holds a *subset* of it, so a path listed in the tree may
exist in one repository and not in the other.  This script reports
which, so the tree can be written to be honest about both.

Usage::

    .venv\Scripts\python.exe tools/check_readme_tree.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DELIVERY = REPO.parent / "drug_ocr"


def tree_paths(readme: Path) -> list[str]:
    """Return the paths listed in the README's ```` ```text ```` tree."""
    text = readme.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)```", text, re.S)
    if not match:
        return []
    block = match.group(1)
    # The tree is drawn with ``+-`` for the last child and ``\-`` for
    # the others.  The depth of an entry is the column its *name*
    # starts at, and a directory is closed when a later entry of the
    # same or a shallower depth appears.
    stack: list[tuple[int, str]] = []
    paths: list[str] = []
    for line in block.splitlines():
        marker = re.search(r"[+\\]- ", line)
        if marker is None:
            continue
        column = marker.start() + 3
        name = line[column:].split("#")[0].strip()
        if name.endswith("/"):
            while stack and stack[-1][0] >= column:
                stack.pop()
            stack.append((column, name))
        else:
            parts = [d for c, d in stack if c < column]
            paths.append("".join(parts) + name)
    return paths


def actual_paths(repo: Path) -> list[str]:
    """Return the files the curated directories actually hold.

    The README tree is compared against this, so the two can be
    checked against each other rather than against a hand-written
    list that drifts.
    """
    found: list[str] = []
    for top in ("src/cpd", "tests", "tools"):
        directory = repo / top
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix == ".py":
                found.append(path.relative_to(repo).as_posix())
    return found


def main() -> int:
    readme = REPO / "README.md"
    if not readme.exists():
        print(f"no README at {readme}", file=sys.stderr)
        return 2
    if "--write" in sys.argv:
        text = readme.read_text(encoding="utf-8")
        block = render_tree(REPO)
        new_text, count = re.subn(
            r"```text\n.*?```",
            "```text\n" + block + "\n```",
            text,
            count=1,
            flags=re.S,
        )
        if count != 1:
            print("no ````text`` tree block to rewrite", file=sys.stderr)
            return 2
        readme.write_text(new_text, encoding="utf-8", newline="")
        print(f"rewrote the project tree in {readme}")
        return 0
    paths = tree_paths(readme)
    if not paths:
        print("no tree block found in README.md")
        return 2
    only_working = only_delivery = neither = 0
    for path in paths:
        in_working = (REPO / path).exists()
        in_delivery = (DELIVERY / path).exists()
        if in_working and in_delivery:
            continue
        if in_working:
            only_working += 1
            print(f"working only    {path}")
        elif in_delivery:
            only_delivery += 1
            print(f"delivery only   {path}")
        else:
            neither += 1
            print(f"in neither      {path}")
    print()
    print(f"tree entries      : {len(paths)}")
    print(f"working only      : {only_working}")
    print(f"delivery only     : {only_delivery}")
    print(f"in neither       : {neither}")
    listed = {p for p in paths if "/" in p and p.startswith(("src/", "tests/", "tools/"))}
    actual = set(actual_paths(REPO))
    missing = sorted(actual - listed)
    extra = sorted(listed - actual)
    if missing:
        print("\nfiles that exist but are not in the tree:")
        for path in missing:
            print(f"    {path}")
    if extra:
        print("\nfiles in the tree that do not exist:")
        for path in extra:
            print(f"    {path}")
    return 0


def render_tree(repo: Path) -> str:
    """Render the project tree from the files that are really there."""
    lines = ["cyclic-peptide/"]
    lines.append(" +- main.py                       # Pipeline entry point (OCR + DECIMER -> one CSV)")
    lines.append(" +- pyproject.toml                # Project metadata + dependency pins")
    lines.append(" +- uv.lock                       # Locked dependency graph")
    lines.append(" +- README.md                     # This file")
    lines.append(" +- PROGRESS.md                   # Progress report + audit results")
    lines.append(" +- LICENSE")
    lines.append(" +- src/cpd/")
    for path in sorted(p for p in (repo / "src" / "cpd").glob("*.py")):
        lines.append(f" |   +- {path.name}")
    lines.append(" |   +- ocr/")
    for path in sorted(p for p in (repo / "src" / "cpd" / "ocr").glob("*.py")):
        lines.append(f" |       +- {path.name}")
    lines.append(" |   +- parsers/")
    for path in sorted(p for p in (repo / "src" / "cpd" / "parsers").glob("*.py")):
        lines.append(f" |       +- {path.name}")
    lines.append(" +- tests/")
    for path in sorted(p for p in (repo / "tests").glob("*.py")):
        lines.append(f" |   +- {path.name}")
    lines.append(" +- tools/")
    for path in sorted(p for p in (repo / "tools").glob("*.py")):
        lines.append(f" |   +- {path.name}")
    lines.append(" +- experiments/                  # One-off analysis scripts (not a package)")
    lines.append(" +- analysis/                     # Exploratory analysis scripts")
    lines.append(" +- docs/")
    lines.append(" |   +- ARCHITECTURE.md           # module map, the routes, adding a patent")
    lines.append(" +- data/")
    lines.append(" |   +- raw/WO2025162428/")
    lines.append(" |   |   +- golden_tables/        # 103 source pages used by the pipeline")
    lines.append(" |   +- raw/WO2022251372/")
    lines.append(" |   |   +- golden_tables/        # 20 full-page TIFF scans")
    lines.append(" |   +- processed/")
    lines.append(" |   |   +- compounds_combined.csv")
    lines.append(" |   +- audit/                    # evidence: phase2 / phase3 / demos")
    lines.append(" +- references/                   # Read-only upstream OCSR/OCR source clones")
    lines.append(" +- .gitignore")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
