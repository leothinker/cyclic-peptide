"""Hand transcriptions of WO2022251372 Table 1a printed SMILES.

The source PDF (``WO2022251372-PAMPH-20221201-7554.pdf``) is a pure
rasterised document: 152 image streams and zero fonts, so there is no
text layer to read the SMILES out of.  The only ground truth available
is a human reading of the 300 dpi page images.

Each entry below was transcribed line by line from
``data/raw/WO2022251372/golden_tables`` at high magnification and
cross-checked against the rendered crops.  The line breaks are the ones
actually printed, because the OCR pipeline reads the table line by line.

Three of the transcriptions were wrong and have been corrected against
the magnified pixels; see :data:`CORRECTIONS` below for the evidence.
The correction matters because the patent prints its chiral marker as
a *circled italic "a"*, which is exactly the glyph that makes
``[C@@H]`` easy to mis-transcribe as ``[C@H]`` or ``[C@@@H]``.

Status: text verified by hand.  This says nothing about whether the
printed molecule is the intended compound, and it is not a statement
about any structure validation.
"""

from __future__ import annotations

# Compounds 1, 2, 3 and 4 share their first nine printed lines.
PEPTIDE_HEAD = (
    "COCC[C@@H]1NC(=O)[C",
    "@H]2CCCN2C(=O)[C@H](",
    "Cc2ccccc2)N(C)C(=O)[C@",
    "H](Cc2ccc(Cl)cc2)NC(=O)[",
    "C@@H](NC(=O)[C@H](Cc",
    "2ccc(F)cc2)NC(=O)CN(CCO",
    "C)C(=O)[C@@H](C)N(C)C(",
    "=O)[C@H](Cc2cscn2)NC(=",
    "O)[C@H](C)N(C)C1=O)[C",
)

REFERENCE_LINES: dict[int, tuple[str, ...]] = {
    1: PEPTIDE_HEAD + ("@@H](C)OC(C)=O",),
    2: PEPTIDE_HEAD + ("@@H](C)OC(=O)c1ccccc1",),
    3: PEPTIDE_HEAD + ("@@H](C)OC(=O)c1cccnc1",),
    4: PEPTIDE_HEAD + ("@@H](C)OC(=O)COC",),
    10: (
        "COCC[C@@H]1NC(=O)[C",
        "@H]2CCCN2C(=O)[C@H](",
        "Cc2ccccc2)N(C)C(=O)[C@",
        "H](Cc2ccc(Cl)cc2)NC(=O)[",
        "C@H](CC2CCCC2)N(C)C(=",
        "O)[C@H](Cc2ccc(F)cc2)NC",
        "(=O)CN(CCOC)C(=O)[C@",
        "@H](C)N(C)C(=O)[C@H](C",
        "c2cscn2)NC(=O)[C@H](C)N",
        "(C)C1=O",
    ),
    28: (
        "CCC[C@@H]1N(C)C(=O)[",
        "C@@H](C)N(C)C(=O)[C@",
        "H](Cc2cncs2)NC(=O)[C@H]",
        "(C)N(C)C(=O)[C@H](CCO",
        "C)NC(=O)[C@H]2CCCN2C",
        "(=O)[C@H](Cc2ccccc2)N(C",
        ")C(=O)[C@H](Cc2ccc(Cl)cc",
        "2)NC(=O)[C@H](C(C)C)N(",
        "C)C(=O)[C@H](Cc2ccc(F)c",
        "c2)NC1=O",
    ),
}

# The corrections applied to :data:`REFERENCE_LINES`, with the pixel
# evidence for each.  The "was" values are the ones transcribed first,
# so that the correction stays auditable rather than silent.
CORRECTIONS: tuple[dict[str, object], ...] = (
    {
        "compound": 10,
        "line": 7,
        "was": "@@H](C)N(C)C(=O)[C@H](C",
        "now": "@H](C)N(C)C(=O)[C@H](C",
        "reason": (
            "the page prints two circled 'a' glyphs here, so the "
            "chiral marker is '@@'; the transcription had three"
        ),
        "evidence": (
            "data/audit/phase9/patent2_reference_audit/cmpd10_line07.png"
        ),
    },
    {
        "compound": 28,
        "line": 3,
        "was": "(C)N(C)C(=O)[C@@H](CCO",
        "now": "(C)N(C)C(=O)[C@H](CCO",
        "reason": (
            "the page prints one circled 'a' glyph here, so the "
            "chiral marker is '@'"
        ),
        "evidence": (
            "data/audit/phase9/patent2_zoom/cmpd28_band2_line03.png"
        ),
    },
    {
        "compound": 28,
        "line": 5,
        "was": "(=O)[C@H](Cc2ccccc2)NC(C",
        "now": "(=O)[C@H](Cc2ccccc2)N(C",
        "reason": "the page prints 'N(C', not 'NC(C'",
        "evidence": (
            "data/audit/phase9/patent2_zoom/cmpd28_band2_line05.png"
        ),
    },
)

# Which source page each compound is printed on.
REFERENCE_PAGES = {
    1: "000116",
    2: "000116",
    3: "000117",
    4: "000117",
    10: "000120",
    28: "000126",
}

CELL_FILES = {
    1: "000116_cmpd1.png",
    2: "000116_cmpd2.png",
    3: "000117_cmpd3.png",
    4: "000117_cmpd4.png",
    10: "000120_cmpd10.png",
    28: "000126_cmpd28.png",
}


def reference_text(compound_id: int) -> str:
    """The full SMILES string as printed, for one compound."""
    return "".join(REFERENCE_LINES[compound_id])


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance between two strings."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of the hypothesis against the reference."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


__all__ = [
    "CELL_FILES",
    "PEPTIDE_HEAD",
    "REFERENCE_LINES",
    "REFERENCE_PAGES",
    "cer",
    "edit_distance",
    "reference_text",
]
