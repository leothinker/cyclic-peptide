"""Extract structured rows from activity / SMILES table images.

Why this exists
---------------
After :mod:`cpd.parsers.table_finder` tells us *which* FullText JPGs are
the activity / SMILES table pages, this module extracts the rows inside
them. The pipeline is built around a :class:`TableExtractor` protocol so
the actual backend (pytesseract + regex, easyocr, GPT-4o Vision, Claude
3.5 Sonnet, ... ) can be swapped without touching the call site.

Row schemas
-----------
The data we want is small and fixed:

  Activity table row (image ``I100280.jpg`` etc.):
    - ``cmpd_id``     -> "1001"
    - ``kd_nm``       -> 0.062           (G12V GDP KD nM)
    - ``rt_min``      -> 1.70            (LCMS RT min)
    - ``ms_mz``       -> 1512.8          (MS m/z)
    - ``lcms_method`` -> "10-80-2min"
    - ``ms_polarity`` -> "[M+H]+"

  SMILES table row (image ``I100298.jpg`` etc.):
    - ``cmpd_id``     -> "1038"
    - ``smiles``      -> "CC[C@H](C)[C@H]1C(=O)N(C)..."

Both tables share a Cmpd # column, so :mod:`cpd.merge` can inner-join on it.

Backends
--------
* :class:`StubTableExtractor`     -- returns ``[]`` for everything; lets the
                                       pipeline run end-to-end on machines
                                       with no OCR / VLM available.
* :class:`RegexTableExtractor`    -- expects pre-OCR'd text + uses regex
                                       to find rows. Useful in tests and as
                                       the deterministic fallback when an
                                       OCR backend is wired but produces
                                       noisy output.
* :class:`VisionLlmTableExtractor`-- skeleton that wraps an OpenAI-style
                                       vision chat completion. The user
                                       fills in the API call body; the
                                       parser and dataclasses here make
                                       sure the response is validated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from PIL import Image


# ---------- row schemas ----------


@dataclass(frozen=True)
class ActivityRow:
    """One row of an activity table image."""

    cmpd_id: str
    kd_nm: float | None = None
    rt_min: float | None = None
    ms_mz: float | None = None
    lcms_method: str | None = None
    ms_polarity: str | None = None
    source_image: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "cmpd_id": self.cmpd_id,
            "kd_nm": self.kd_nm,
            "rt_min": self.rt_min,
            "ms_mz": self.ms_mz,
            "lcms_method": self.lcms_method,
            "ms_polarity": self.ms_polarity,
            "source_image": self.source_image,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SmilesRow:
    """One row of a SMILES table image."""

    cmpd_id: str
    smiles: str | None = None
    canonical_smiles: str | None = None  # populated after RDKit validation
    is_valid: bool | None = None
    source_image: str | None = None

    def to_dict(self) -> dict:
        return {
            "cmpd_id": self.cmpd_id,
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "is_valid": self.is_valid,
            "source_image": self.source_image,
        }


# ---------- the protocol every backend implements ----------


class TableExtractor(Protocol):
    """Pluggable image -> rows backend.

    A single backend may produce both ActivityRows and SmilesRows; if it
    only knows how to read one type, return ``[]`` for the other.
    """

    name: str

    def extract_activity(self, image_path: Path) -> list[ActivityRow]: ...

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]: ...


# ---------- stub / no-op backend ----------


@dataclass
class StubTableExtractor:
    """Returns ``[]`` for every call.

    Use this when no OCR / VLM is wired -- the rest of the pipeline
    (table_finder, merge, CLI) still runs end-to-end.
    """

    name: str = "stub"

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:
        return []

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:
        return []


# ---------- regex backend (for tests / OCR-text fallback) ----------


_CMPD_ID_RE = re.compile(r"\b(\d{3,5})\b")
_KD_RE = re.compile(r"(\d+\.\d+)")
_RT_RE = re.compile(r"\b(\d+\.\d{1,2})\b")
_MZ_RE = re.compile(r"\b(\d{3,4}\.\d{1,2})\b")
_POLARITY_RE = re.compile(r"\[M\+(?:H|Na|K)?\]\+?")


@dataclass
class RegexTableExtractor:
    """Heuristic regex extractor over already-OCR'd text.

    The caller passes OCR text into :meth:`parse_activity_text` /
    :meth:`parse_smiles_text`. The :meth:`extract_*` convenience methods
    fall back to a stub OCR (raises) -- so this backend is meant for
    tests and for callers who do OCR out-of-band.
    """

    name: str = "regex"

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:  # pragma: no cover
        raise NotImplementedError(
            "RegexTableExtractor is text-only; use parse_activity_text()."
        )

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:  # pragma: no cover
        raise NotImplementedError(
            "RegexTableExtractor is text-only; use parse_smiles_text()."
        )

    def parse_activity_text(
        self, text: str, *, source_image: str | None = None,
    ) -> list[ActivityRow]:
        """Parse already-OCR'd text from an activity table page."""
        rows: list[ActivityRow] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cmpd_m = _CMPD_ID_RE.search(line)
            if not cmpd_m:
                continue
            cmpd_id = cmpd_m.group(1)
            # Naive: first float ~= RT, second ~= MS m/z, last ~= KD nM.
            floats = _KD_RE.findall(line)
            if len(floats) < 3:
                continue
            try:
                rt = float(floats[0])
                mz = float(floats[1])
                kd = float(floats[-1])
            except ValueError:
                continue
            pol_m = _POLARITY_RE.search(line)
            rows.append(ActivityRow(
                cmpd_id=cmpd_id,
                kd_nm=kd,
                rt_min=rt,
                ms_mz=mz,
                lcms_method=None,
                ms_polarity=pol_m.group(0) if pol_m else None,
                source_image=source_image,
            ))
        return rows

    def parse_smiles_text(
        self, text: str, *, source_image: str | None = None,
    ) -> list[SmilesRow]:
        """Parse already-OCR'd text from a SMILES table page."""
        rows: list[SmilesRow] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # SMILES lines start with an SMILES-shaped prefix; the cmpd id
            # is usually on a separate column.
            cmpd_m = _CMPD_ID_RE.search(line)
            if not cmpd_m:
                continue
            # Heuristic: anything containing `[` (brackets = stereo / ring)
            # AND `=O` or `N` or `C` in the right pattern is a SMILES line.
            if not re.search(r"[CNO\[]=", line) and "@" not in line:
                continue
            tokens = line.split()
            cmpd_id_str = cmpd_m.group(1)
            smiles = None
            for tok in tokens:
                if tok == cmpd_id_str:
                    continue
                if any(c in tok for c in '[@=['):
                    smiles = tok
                    break
            if smiles is None:
                for tok in tokens:
                    if tok != cmpd_id_str:
                        smiles = tok
                        break
            rows.append(SmilesRow(
                cmpd_id=cmpd_m.group(1),
                smiles=smiles,
                source_image=source_image,
            ))
        return rows


# ---------- vision-LLM backend skeleton ----------


@dataclass
class VisionLlmTableExtractor:
    """Skeleton wrapper around an OpenAI-style vision chat completion.

    The user supplies :param llm_call: -- a callable that takes
    ``(image_path: Path, prompt: str) -> str`` (raw model output, ideally
    JSON). This class validates and parses the output into ActivityRows /
    SmilesRows.

    No actual HTTP call is made here; that keeps the dependency optional
    and lets users plug in their own client (openai, anthropic, azure, ...).
    """

    name: str = "vision-llm"
    llm_call: object | None = None  # Callable[[Path, str], str]
    model: str = "gpt-4o-mini"

    ACTIVITY_PROMPT = (
        "You are looking at a chemistry patent activity table image. "
        "Each row has columns: Cmpd #, Structure, LCMS RT (min), MS (m/z), "
        "LCMS Method, MS Polarity, G12V GDP KD nM. "
        "Return ONLY valid JSON: an array of objects with keys "
        "cmpd_id (string), kd_nm (number, nM), rt_min (number), ms_mz "
        "(number), lcms_method (string), ms_polarity (string). "
        "Do not include any explanation or prose."
    )
    SMILES_PROMPT = (
        "You are looking at a chemistry patent SMILES table image. "
        "Each row has columns: Cmpd #, Structure, SMILES. "
        "Return ONLY valid JSON: an array of objects with keys "
        "cmpd_id (string), smiles (string). "
        "Do not include any explanation or prose."
    )

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:
        raw = self._call(image_path, self.ACTIVITY_PROMPT)
        return self._parse_activity_json(raw, source_image=image_path.name)

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:
        raw = self._call(image_path, self.SMILES_PROMPT)
        return self._parse_smiles_json(raw, source_image=image_path.name)

    # ----- internals -----

    def _call(self, image_path: Path, prompt: str) -> str:
        if self.llm_call is None:
            raise RuntimeError(
                "VisionLlmTableExtractor.llm_call is None -- wire your "
                "OpenAI/Anthropic/Azure client first."
            )
        result = self.llm_call(image_path, prompt)
        if not isinstance(result, str):
            raise TypeError(f"llm_call must return str, got {type(result)}")
        return result

    @staticmethod
    def _parse_activity_json(raw: str, *, source_image: str | None) -> list[ActivityRow]:
        rows = _safe_json_loads(raw)
        if not isinstance(rows, list):
            return []
        out: list[ActivityRow] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            cmpd_id = str(r.get("cmpd_id", "")).strip()
            if not cmpd_id:
                continue
            out.append(ActivityRow(
                cmpd_id=cmpd_id,
                kd_nm=_as_float(r.get("kd_nm")),
                rt_min=_as_float(r.get("rt_min")),
                ms_mz=_as_float(r.get("ms_mz")),
                lcms_method=_as_str_or_none(r.get("lcms_method")),
                ms_polarity=_as_str_or_none(r.get("ms_polarity")),
                source_image=source_image,
            ))
        return out

    @staticmethod
    def _parse_smiles_json(raw: str, *, source_image: str | None) -> list[SmilesRow]:
        rows = _safe_json_loads(raw)
        if not isinstance(rows, list):
            return []
        out: list[SmilesRow] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            cmpd_id = str(r.get("cmpd_id", "")).strip()
            smiles = _as_str_or_none(r.get("smiles"))
            if not cmpd_id or not smiles:
                continue
            out.append(SmilesRow(
                cmpd_id=cmpd_id,
                smiles=smiles,
                source_image=source_image,
            ))
        return out


def _safe_json_loads(raw: str) -> object:
    """Best-effort JSON parsing -- tolerates fenced ```json blocks."""
    text = raw.strip()
    # Strip code fences.
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _as_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_str_or_none(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ---------- high-level helpers ----------


def enrich_smiles_with_rdkit(
    rows: list[SmilesRow], *, validate: bool = True,
) -> list[SmilesRow]:
    """Populate ``canonical_smiles`` / ``is_valid`` using RDKit.

    Silently skips RDKit when the package is unavailable -- in that case
    ``canonical_smiles`` stays ``None`` and ``is_valid`` stays ``None``.
    """
    try:
        from rdkit import Chem  # type: ignore
    except ImportError:
        return rows
    out: list[SmilesRow] = []
    for r in rows:
        canon = None
        ok: bool | None = None
        if validate and r.smiles:
            mol = Chem.MolFromSmiles(r.smiles)
            if mol is not None:
                canon = Chem.MolToSmiles(mol, canonical=True)
                ok = True
            else:
                ok = False
        out.append(SmilesRow(
            cmpd_id=r.cmpd_id,
            smiles=r.smiles,
            canonical_smiles=canon,
            is_valid=ok,
            source_image=r.source_image,
        ))
    return out


__all__ = [
    "ActivityRow",
    "SmilesRow",
    "TableExtractor",
    "StubTableExtractor",
    "RegexTableExtractor",
    "VisionLlmTableExtractor",
    "enrich_smiles_with_rdkit",
]
