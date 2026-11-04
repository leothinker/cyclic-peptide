"""Patent-specific table profiles.

The extraction and recognition code is shared.  A profile describes only
the parts that legitimately change between patents:

* accepted input-image patterns;
* whether an input is a cropped table or a full page;
* table layouts and the fields they contain;
* the compound-number syntax;
* how activity values are represented; and
* the measured printed-SMILES route.

Keeping this information as data makes a new patent an explicit,
reviewable configuration change instead of a collection of conditionals
inside the OCR code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnTemplate:
    """One logical table column.

    ``x1`` and ``x2`` are fractions of the table width.  They are used by
    the original patent, whose images are already cropped to one table.
    Full-page profiles leave them unset because their column boundaries
    are measured from the ruled table in the source page.
    """

    name: str
    x1: float | None = None
    x2: float | None = None
    is_image: bool = False


@dataclass(frozen=True)
class LayoutSpec:
    """A logical table layout and its ordered columns.

    ``header_bands`` is the number of leading row bands that carry the
    column headings rather than data.  It is only consulted for
    full-page inputs, where the row bands are measured from the ruled
    table itself.

    ``continuation`` marks a layout whose printed row may run over the
    bottom of a page and resume on the next one.  The resumed band
    carries no compound number, so the pipeline appends its SMILES to
    the compound that is still open instead of starting a new record.
    """

    name: str
    columns: tuple[ColumnTemplate, ...]
    header_bands: int = 0
    continuation: bool = False


@dataclass(frozen=True)
class AssayFieldSpec:
    """One assay field and the representation used by its parser.

    ``kind`` selects the parser:

    * ``"text"`` -- the generic cell reader.
    * ``"number"`` -- the generic cell reader, parsed as a float.
    * ``"grade"`` -- a value from a closed set such as A/B/C/D.  The
      reader is restricted to that set, so a misread glyph becomes an
      empty cell rather than a wrong value.

    ``allowed`` lists the accepted values for a constrained field.
    """

    name: str
    kind: str = "text"
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldLayoutRule:
    """Header-field rule used when a visual column count is ambiguous."""

    layout: str
    required: frozenset[str]
    excluded: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TableProfile:
    """All patent-specific choices for one patent's characterisation table."""

    name: str
    image_patterns: tuple[str, ...]
    source_mode: str
    layouts: dict[str, LayoutSpec]
    layout_by_column_count: dict[int, str]
    default_layout: str
    field_layout_rules: tuple[FieldLayoutRule, ...]
    cmpd_id_pattern: str
    cmpd_id_top_px: int | None
    cmpd_id_inset_px: int
    assay_fields: tuple[AssayFieldSpec, ...]
    assay_trigger_fields: tuple[str, ...]
    smiles_route: str
    merge_compounds: str = "first_valid"

    def layout(self, name: str) -> LayoutSpec:
        try:
            return self.layouts[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown layout {name!r} in profile {self.name!r}"
            ) from exc

    def layout_for_column_count(self, count: int) -> str | None:
        return self.layout_by_column_count.get(count)

    def assay_field(self, name: str) -> AssayFieldSpec | None:
        """Return the assay-field spec for ``name``, if the profile has one."""
        for field in self.assay_fields:
            if field.name == name:
                return field
        return None

    def layout_for_fields(self, fields: set[str]) -> str | None:
        """Resolve a layout from canonical header fields."""
        if not fields:
            return None
        for rule in self.field_layout_rules:
            if (
                rule.required.issubset(fields)
                and not (rule.excluded & fields)
            ):
                return rule.layout
        return None

    def scaled_columns(self, layout_name: str, width: int) -> list[tuple]:
        """Return ``(name, x1, x2, is_image)`` tuples for a layout."""
        result: list[tuple] = []
        for column in self.layout(layout_name).columns:
            if column.x1 is None or column.x2 is None:
                raise ValueError(
                    f"layout {layout_name!r} in profile {self.name!r} "
                    "has no fractional geometry"
                )
            result.append(
                (
                    column.name,
                    int(column.x1 * width),
                    int(column.x2 * width),
                    column.is_image,
                )
            )
        return result


def _original_patent_profile() -> TableProfile:
    """Preserve the WO2025162428 behaviour exactly."""
    activity = LayoutSpec(
        "activity",
        (
            ColumnTemplate("cmpd_id", 0.00, 0.09),
            ColumnTemplate("structure", 0.10, 0.48, True),
            ColumnTemplate("rt_min", 0.48, 0.57),
            ColumnTemplate("ms_mz", 0.58, 0.67),
            ColumnTemplate("lcms_method", 0.67, 0.78),
            ColumnTemplate("ms_polarity", 0.79, 0.88),
            ColumnTemplate("kd_nm", 0.89, 1.00),
        ),
    )
    dense = LayoutSpec(
        "dense",
        (
            ColumnTemplate("cmpd_id", 0.00, 0.17),
            ColumnTemplate("rt_min", 0.19, 0.34),
            ColumnTemplate("ms_mz", 0.35, 0.50),
            ColumnTemplate("lcms_method", 0.51, 0.67),
            ColumnTemplate("ms_polarity", 0.68, 0.79),
            ColumnTemplate("kd_nm", 0.80, 0.97),
        ),
    )
    smiles = LayoutSpec(
        "smiles",
        (
            ColumnTemplate("cmpd_id", 0.00, 0.10),
            ColumnTemplate("structure", 0.11, 0.56, True),
            ColumnTemplate("smiles", 0.58, 0.99),
        ),
    )
    return TableProfile(
        name="WO2025162428",
        image_patterns=("*.jpg",),
        source_mode="table_image",
        layouts={
            "activity": activity,
            "dense": dense,
            "smiles": smiles,
        },
        layout_by_column_count={3: "smiles", 6: "dense", 7: "activity"},
        default_layout="activity",
        field_layout_rules=(
            FieldLayoutRule("smiles", frozenset({"smiles"})),
            FieldLayoutRule(
                "dense", frozenset({"kd_nm"}), frozenset({"structure"})
            ),
            FieldLayoutRule("activity", frozenset({"kd_nm"})),
        ),
        cmpd_id_pattern=r"\d{4}",
        cmpd_id_top_px=None,
        cmpd_id_inset_px=0,
        assay_fields=(
            AssayFieldSpec("kd_nm", "number"),
            AssayFieldSpec("rt_min", "number"),
            AssayFieldSpec("ms_mz", "number"),
            AssayFieldSpec("lcms_method", "text"),
            AssayFieldSpec("ms_polarity", "text"),
        ),
        assay_trigger_fields=("kd_nm", "rt_min", "ms_mz"),
        # The route that reaches character-level accuracy on this
        # patent: the upstream default recogniser on our own ink-band
        # segmentation, with the table rules trimmed away, and a
        # second recogniser that settles the characters the first one
        # lost.  The historical ``production`` route is still
        # selectable with ``--smiles-route production`` and still
        # reproduces the first-generation CSV byte for byte; it is
        # simply no longer the default, because the English PP-OCRv4
        # recogniser it uses drops the left bracket of every
        # ``C(=O)``.
        #
        # The vote was measured on the pipeline's own cells (335
        # compounds of the pages that carry a printed-SMILES column):
        # 169 of them parse with ``validated`` and 171 with
        # ``validated_vote``.  The vote broke none of the readings
        # that parsed before and lost no double bond in any of them.
        # On the three hand-verified references the two routes are
        # identical and character exact: 1036, 1037 and 1038 all
        # read with an edit distance of 0.
        smiles_route="validated_vote",
    )


def _second_patent_profile() -> TableProfile:
    """Configure WO2022251372 for full-page raster scans."""
    activity = LayoutSpec(
        "activity8",
        (
            ColumnTemplate("cmpd_id"),
            ColumnTemplate("smiles"),
            ColumnTemplate("mdm2"),
            ColumnTemplate("mdm4"),
            ColumnTemplate("molm13"),
            ColumnTemplate("mv4_11"),
            ColumnTemplate("waga"),
            ColumnTemplate("mkl1"),
        ),
        # Every table on every page starts with one heading band.  The
        # band is 201 px tall for Table 1 and 68 px for Table 1a, and
        # it is the only band whose number cell reads "No.".
        header_bands=1,
        # A compound's printed row can run past the bottom of a page
        # and resume at the top of the next one.  The resumed band has
        # no compound number, which is how the pipeline recognises it.
        continuation=True,
    )
    smiles = LayoutSpec(
        "smiles3",
        (
            ColumnTemplate("cmpd_id"),
            ColumnTemplate("smiles"),
            ColumnTemplate("structure", is_image=True),
        ),
        header_bands=1,
        continuation=True,
    )
    return TableProfile(
        name="WO2022251372",
        image_patterns=("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"),
        source_mode="full_page",
        layouts={"activity8": activity, "smiles3": smiles},
        layout_by_column_count={8: "activity8", 3: "smiles3"},
        default_layout="activity8",
        field_layout_rules=(),
        cmpd_id_pattern=r"\d{1,2}",
        cmpd_id_top_px=90,
        cmpd_id_inset_px=4,
        assay_fields=(
            # Every activity value in this patent is a letter grade,
            # not a number.  The reader is restricted to A-D, so a
            # misread glyph leaves the cell empty instead of writing
            # a wrong value into the output.
            AssayFieldSpec("mdm2", "grade", ("A", "B", "C", "D")),
            AssayFieldSpec("mdm4", "grade", ("A", "B", "C", "D")),
            AssayFieldSpec("molm13", "grade", ("A", "B", "C", "D")),
            AssayFieldSpec("mv4_11", "grade", ("A", "B", "C", "D")),
            AssayFieldSpec("waga", "grade", ("A", "B", "C", "D")),
            AssayFieldSpec("mkl1", "grade", ("A", "B", "C", "D")),
        ),
        assay_trigger_fields=(
            "mdm2",
            "mdm4",
            "molm13",
            "mv4_11",
            "waga",
            "mkl1",
        ),
        # ``padded_vote`` is ``padded`` plus the second recogniser.
        # Measured on the 28 compounds of Table 1a: 24 of them parse
        # with ``padded`` and 27 with ``padded_vote``.  All six hand
        # transcriptions are character exact (0 edits, CER 0.000%),
        # measured against the corrected transcriptions -- three of
        # the six original transcriptions had been mis-read from the
        # page, and the corrections are recorded with their pixel
        # evidence in ``experiments/wo2022251372_references.py``.
        # The vote changed three strings; all three were unreadable
        # before and all three are a single missing bracket.
        smiles_route="padded_vote",
        merge_compounds="prefer_structure",
    )


_PROFILES = {
    "WO2025162428": _original_patent_profile(),
    "WO2022251372": _second_patent_profile(),
}


def get_profile(name: str = "WO2025162428") -> TableProfile:
    """Return a registered patent profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILES))
        raise KeyError(
            f"unknown patent profile {name!r}; available: {available}"
        ) from exc


def available_profiles() -> tuple[str, ...]:
    """Return all registered profile names."""
    return tuple(sorted(_PROFILES))


__all__ = [
    "AssayFieldSpec",
    "ColumnTemplate",
    "FieldLayoutRule",
    "LayoutSpec",
    "TableProfile",
    "available_profiles",
    "get_profile",
]
