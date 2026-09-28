"""Header-text normalisation for patent table OCR.

Patent applicants use varying wording across pages (``Cmpd #`` vs
``Compound #`` vs ``Compd #``, ``G12V GDP KD nM`` vs
``KRAS G12V GDP KD (nM)``, ...). This module collapses whatever OCR
returns on a header cell to a small fixed set of canonical field names
so the rest of the pipeline can rely on them.

The mapping is intentionally literal: each alias is exactly the cleaned
header text (whitespace collapsed, lowercased, no punctuation). If a
patent uses new wording, add a row to :data:`HEADER_ALIASES`.
"""

from __future__ import annotations

import re


HEADER_ALIASES: dict[str, str] = {
    # ----- compound identifier -----
    "cmpd #": "cmpd_id",
    "cmpd#": "cmpd_id",
    "cmpd": "cmpd_id",
    "compound #": "cmpd_id",
    "compound#": "cmpd_id",
    "compound": "cmpd_id",
    "compd #": "cmpd_id",
    "compd": "cmpd_id",
    # ----- canonical SMILES -----
    "smiles": "smiles",
    # ----- 2D structure drawing (image column) -----
    "structure": "structure",
    # ----- bioactivity -----
    "g12v gdp kd nm": "kd_nm",
    "g12v gdp kd (nm)": "kd_nm",
    "kras g12v gdp kd nm": "kd_nm",
    "kras g12v gdp kd (nm)": "kd_nm",
    "kd nm": "kd_nm",
    "kd (nm)": "kd_nm",
    # ----- LCMS -----
    "lcms rt (min)": "rt_min",
    "lcms rt min": "rt_min",
    "rt (min)": "rt_min",
    "rt min": "rt_min",
    "ms (m/z)": "ms_mz",
    "ms m/z": "ms_mz",
    "lcms method": "lcms_method",
    "ms polarity": "ms_polarity",
}


def _clean(text: str) -> str:
    """Collapse whitespace and lowercase for alias lookup."""
    return re.sub(r"\s+", " ", text.strip().lower())


def canonical_field(text: str | None) -> str | None:
    """Map an OCR'd header cell to a canonical field name, or None."""
    if not text:
        return None
    return HEADER_ALIASES.get(_clean(text))


def is_known_header(text: str | None) -> bool:
    """Return True when ``text`` matches any alias in :data:`HEADER_ALIASES`."""
    return canonical_field(text) is not None


__all__ = ["HEADER_ALIASES", "canonical_field", "is_known_header"]
