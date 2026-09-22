"""Parse WIPO body XML to map every <img> figure to a compound/formula.

Why this exists
---------------
WIPO's "Initial Publication" package (and the FullText XML it ships) tags every
figure with an `id` and a `file=` attribute. The figure's *purpose* (Markush
formula vs. specific example vs. reaction scheme) is encoded in the surrounding
paragraphs as plain text -- e.g. ``compound of Formula (A):`` introduces a
Markush, ``compound 10`` introduces a specific cyclic-peptide example.

This module walks the body XML, finds every ``<img>``, scans 2-3 preceding
``<p>`` siblings (plus 1-2 following ones) for known identifier patterns, and
emits an ``ImageRecord`` per figure with a best-guess ``category`` plus the
matched identifier. Downstream the OCSR pipeline uses ``category`` to skip
reaction schemes / tables / cover artwork.

Notes on case sensitivity
-------------------------
WIPO body XML uses both ``Compound N`` (Title case, in figure legends) and
``compound N`` (lowercase, in the running text of synthesis paragraphs). All
identifier regexes below are case-insensitive so both spellings match.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# Identifier patterns. Case-insensitive (re.IGNORECASE) so we catch both
# "Compound 10" in figure legends and "compound 10" in running text.
# Order matters: we keep the FIRST hit in priority order.
_FORMULA_RE = re.compile(r"\bFormula\s*\(([A-Za-z0-9]+)\)", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"\bExample\s+(\d+)\b", re.IGNORECASE)
_COMPOUND_RE = re.compile(r"\bCompound\s+(\d+)\b", re.IGNORECASE)
_SCHEME_RE = re.compile(r"\bScheme\s+(\d+)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\bFigure\s+(\d+)\b", re.IGNORECASE)

# Priority: Formula trumps everything else (we don't want to OCSR a Markush).
# Then Scheme (reaction mechanism), then specific examples.
_CATEGORY_PRIORITY = ("formula", "scheme", "example", "compound", "figure")

# How many surrounding <p> siblings to harvest on each side.
_BEFORE_LIMIT = 3
_AFTER_LIMIT = 2


@dataclass
class ImageRecord:
    """One ``<img>`` figure and its best-guess compound / formula association.

    Attributes
    ----------
    img_id
        The ``id=`` attribute on the ``<img>`` (e.g. ``idf0001``).
    img_file
        The ``file=`` attribute (e.g. ``PCTCN2025075389-ftappb-I100001.jpg``).
    img_path
        Absolute path on disk (filled in when the caller knows the img dir).
    category
        One of ``formula``, ``scheme``, ``example``, ``compound``, ``figure``,
        ``unknown``.
    compound_id
        Normalised identifier like ``"Example 42"`` / ``"Compound 1041"`` /
        ``"Scheme 3"`` -- or ``None`` when the figure is a generic Formula
        or didn't match anything.
    formula_label
        Only set when ``category == "formula"``: e.g. ``"(A)"``.
    example_number
        Only set when ``category == "example"``: the raw integer.
    context_before, context_after
        Up to ``_BEFORE_LIMIT`` / ``_AFTER_LIMIT`` preceding / following
        ``<p>`` texts (in document order), for downstream debugging.
    """

    img_id: str | None
    img_file: str
    img_path: str | None
    category: str
    compound_id: str | None = None
    formula_label: str | None = None
    example_number: int | None = None
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


# ---------- public API ----------

def parse_body_xml(xml_path: Path) -> list[ImageRecord]:
    """Walk body XML and emit one ImageRecord per ``<img>`` figure.

    ``img_path`` is set to the absolute path under ``xml_path.parent`` so the
    caller can hand the records straight to an OCSR reader.
    """
    raw = xml_path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")
    base_dir = xml_path.parent
    records: list[ImageRecord] = []
    for img in soup.find_all("img"):
        file_attr = img.get("file")
        if not file_attr:
            continue  # skip <img> without a filename (very rare)
        records.append(_build_record(img, soup, base_dir=base_dir))
    return records


def write_map_json(
    records: list[ImageRecord], out_path: Path, *,
    patent_id: str, source_xml: Path,
) -> None:
    """Persist the mapping as JSON for downstream consumption."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.category] = counts.get(r.category, 0) + 1
    payload = {
        "patent_id": patent_id,
        "source_xml": str(source_xml),
        "total_records": len(records),
        "category_counts": counts,
        "records": [asdict(r) for r in records],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_map_json(path: Path) -> list[ImageRecord]:
    """Inverse of :func:`write_map_json`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        ImageRecord(
            img_id=row.get("img_id"),
            img_file=row["img_file"],
            img_path=row.get("img_path"),
            category=row["category"],
            compound_id=row.get("compound_id"),
            formula_label=row.get("formula_label"),
            example_number=row.get("example_number"),
            context_before=row.get("context_before") or [],
            context_after=row.get("context_after") or [],
        )
        for row in data["records"]
    ]


# ---------- internals ----------

def _build_record(
    img: Tag, soup: BeautifulSoup, *, base_dir: Path,
) -> ImageRecord:
    parent_p = img.find_parent("p")

    before = _collect_sibling_texts(parent_p, direction="previous", limit=_BEFORE_LIMIT)
    after = _collect_sibling_texts(parent_p, direction="next", limit=_AFTER_LIMIT)

    # The search window: before + (the parent <p> text, if any) + after.
    # The parent <p> often holds the identifier text (e.g. "compound of
    # Formula (A):" + <br/> + <img/>).
    window_parts = list(before)
    if parent_p is not None:
        own_text = parent_p.get_text(" ", strip=True)
        if own_text:
            window_parts.append(own_text)
    window_parts.extend(after)
    window = " ".join(window_parts)

    category, compound_id, formula_label, example_number = _classify(window)

    file_attr = img.get("file", "")
    img_path = str((base_dir / file_attr).resolve()) if file_attr else None

    return ImageRecord(
        img_id=img.get("id"),
        img_file=file_attr,
        img_path=img_path,
        category=category,
        compound_id=compound_id,
        formula_label=formula_label,
        example_number=example_number,
        context_before=before,
        context_after=after,
    )


def _collect_sibling_texts(
    parent_p: Tag | None, *, direction: str, limit: int,
) -> list[str]:
    if parent_p is None:
        return []
    method = (
        parent_p.find_previous_sibling
        if direction == "previous"
        else parent_p.find_next_sibling
    )
    out: list[str] = []
    sib = method("p")
    while sib is not None and len(out) < limit:
        text = sib.get_text(" ", strip=True)
        if text:
            if direction == "previous":
                out.insert(0, text)
            else:
                out.append(text)
        sib = method("p")
    return out


def _classify(window: str) -> tuple[str, str | None, str | None, int | None]:
    """Return (category, compound_id, formula_label, example_number)."""
    formula_m = _FORMULA_RE.search(window)
    if formula_m:
        return "formula", None, f"({formula_m.group(1)})", None
    scheme_m = _SCHEME_RE.search(window)
    if scheme_m:
        return "scheme", f"Scheme {scheme_m.group(1)}", None, None
    example_m = _EXAMPLE_RE.search(window)
    if example_m:
        n = int(example_m.group(1))
        return "example", f"Example {n}", None, n
    compound_m = _COMPOUND_RE.search(window)
    if compound_m:
        return "compound", f"Compound {compound_m.group(1)}", None, None
    figure_m = _FIGURE_RE.search(window)
    if figure_m:
        return "figure", f"Figure {figure_m.group(1)}", None, None
    return "unknown", None, None, None


__all__ = [
    "ImageRecord",
    "parse_body_xml",
    "write_map_json",
    "load_map_json",
]
