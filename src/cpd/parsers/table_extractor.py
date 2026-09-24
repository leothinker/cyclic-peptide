"""Extract structured rows from activity / SMILES table images.

Supported Backends:
- RapidOcrTableExtractor: Local high-precision OpenCV line detection + RapidOCR.
- RegexTableExtractor: Fast deterministic regex parser over OCR text (used in unit tests).
- VisionLlmTableExtractor: Multi-modal LLM API integration.
- StubTableExtractor: No-op fallback for offline pipeline testing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cpd.chem import auto_repair_smiles, validate_and_enrich


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
    canonical_smiles: str | None = None
    molecular_weight: float | None = None
    heavy_atom_count: int | None = None
    is_valid: bool | None = None
    source_image: str | None = None

    def to_dict(self) -> dict:
        return {
            "cmpd_id": self.cmpd_id,
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "molecular_weight": self.molecular_weight,
            "heavy_atom_count": self.heavy_atom_count,
            "is_valid": self.is_valid,
            "source_image": self.source_image,
        }


# ---------- the protocol every backend implements ----------


class TableExtractor(Protocol):
    """Pluggable image -> rows backend."""

    name: str

    def extract_activity(self, image_path: Path) -> list[ActivityRow]: ...

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]: ...


# ---------- Local Production Backend: RapidOCR + OpenCV ----------


class RapidOcrTableExtractor:
    """Local production extractor using OpenCV grid detection and RapidOCR."""

    name: str = "rapidocr"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def _find_row_slices(self, img: np.ndarray) -> list[tuple[int, int]]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1))
        lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        row_sums = np.sum(lines_img, axis=1)
        line_ys = np.where(row_sums > 0)[0]

        clusters: list[int] = []
        if len(line_ys) > 0:
            cur = [int(line_ys[0])]
            for y in line_ys[1:]:
                if y - cur[-1] <= 10:
                    cur.append(int(y))
                else:
                    clusters.append(int(np.mean(cur)))
                    cur = [int(y)]
            clusters.append(int(np.mean(cur)))

        inner_lines = [y for y in clusters if 40 < y < h - 40]
        if len(inner_lines) == 3:
            dividers = [0] + inner_lines + [h]
        else:
            dividers = [int(i * h / 4) for i in range(5)]

        slices = []
        for i in range(len(dividers) - 1):
            y1 = dividers[i] + 1 if i > 0 else 0
            y2 = dividers[i + 1] - 1 if i < len(dividers) - 2 else h
            slices.append((y1, y2))
        return slices

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:
        """Extract compound ID and SMILES from a SMILES table image."""
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        slices = self._find_row_slices(img)

        # 1. 编号识别
        left_col = img[:, : int(w * 0.15)]
        id_res = self.engine(left_col)
        found_ids: list[tuple[float, str]] = []
        if id_res:
            txts = (
                id_res.txts if hasattr(id_res, "txts") else [item[1] for item in id_res]
            )
            boxes = (
                id_res.boxes
                if hasattr(id_res, "boxes")
                else [item[0] for item in id_res]
            )
            for box, txt in zip(boxes, txts):
                clean = txt.strip()
                if re.match(r"^\d{4}$", clean):
                    center_y = float((box[0][1] + box[2][1]) / 2.0)
                    found_ids.append((center_y, clean))

        found_ids.sort(key=lambda x: x[0])
        cmpd_names = [item[1] for item in found_ids]
        while len(cmpd_names) < len(slices):
            cmpd_names.append(f"Row_{len(cmpd_names) + 1}")

        rows: list[SmilesRow] = []

        # 2. SMILES 识别
        for i, (y_start, y_end) in enumerate(slices):
            cmpd_id = cmpd_names[i]
            smiles_crop = img[y_start:y_end, int(w * 0.55) :]

            padded = cv2.copyMakeBorder(
                smiles_crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            zoomed = cv2.resize(
                padded, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )

            smiles_res = self.engine(zoomed)
            line_items: list[tuple[float, str]] = []
            if smiles_res:
                boxes = (
                    smiles_res.boxes
                    if hasattr(smiles_res, "boxes")
                    else [item[0] for item in smiles_res]
                )
                txts = (
                    smiles_res.txts
                    if hasattr(smiles_res, "txts")
                    else [item[1] for item in smiles_res]
                )
                for box, t in zip(boxes, txts):
                    clean_t = t.strip()
                    if len(clean_t) > 2 and not re.match(r"^[().=]+$", clean_t):
                        box_y = float((box[0][1] + box[2][1]) / 2.0)
                        line_items.append((box_y, clean_t))

            line_items.sort(key=lambda x: x[0])
            raw_smiles = "".join(item[1] for item in line_items)
            fixed_smiles = auto_repair_smiles(raw_smiles)

            if any(k in fixed_smiles.upper() for k in ("MIN", "[M+", "LCMS")):
                continue

            # RDKit 校验
            chem_info = validate_and_enrich(fixed_smiles)

            rows.append(
                SmilesRow(
                    cmpd_id=cmpd_id,
                    smiles=fixed_smiles,
                    canonical_smiles=chem_info.canonical_smiles if chem_info else None,
                    molecular_weight=chem_info.molecular_weight if chem_info else None,
                    heavy_atom_count=chem_info.heavy_atom_count if chem_info else None,
                    is_valid=chem_info is not None,
                    source_image=image_path.name,
                )
            )

        return rows

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:
        """Extract compound ID and Kd, supporting both 4-row cards and 40-row dense tables."""
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        h, w = img.shape[:2]

        # 针对第二张图这种整页纯数据表：直接 OCR 提取每行文本，精度最高！
        ocr_res = self.engine(img)
        if not ocr_res:
            return []

        boxes = (
            ocr_res.boxes
            if hasattr(ocr_res, "boxes")
            else [item[0] for item in ocr_res]
        )
        txts = (
            ocr_res.txts if hasattr(ocr_res, "txts") else [item[1] for item in ocr_res]
        )

        # 收集所有的文本框，并计算中心坐标
        items = []
        for box, txt in zip(boxes, txts):
            clean_t = txt.strip()
            if not clean_t:
                continue
            center_y = (box[0][1] + box[2][1]) / 2.0
            center_x = (box[0][0] + box[1][0]) / 2.0
            items.append({"text": clean_t, "x": center_x, "y": center_y})

        # 1. 找出所有位于左半边（X < 25%）的 4 位化合物编号 (1001~2999)
        id_candidates = [
            it
            for it in items
            if it["x"] < w * 0.25 and re.match(r"^[12]\d{3}$", it["text"])
        ]
        id_candidates.sort(key=lambda it: it["y"])

        # 2. 找出所有位于右半边（X > 75%）的纯浮点数（Kd 数值）
        kd_candidates = [
            it
            for it in items
            if it["x"] > w * 0.75 and re.match(r"^\d+\.\d+$", it["text"])
        ]

        rows: list[ActivityRow] = []

        # 3. 按垂直高度（Y 坐标差 < 20 像素）将编号与 Kd 进行同一行绑定
        for cid_item in id_candidates:
            cid_y = cid_item["y"]
            # 找到同一水平行最接近的 Kd
            matched_kd = None
            min_dist = 25.0  # Y 轴公差 25 像素
            for kd_item in kd_candidates:
                dist = abs(kd_item["y"] - cid_y)
                if dist < min_dist:
                    try:
                        matched_kd = float(kd_item["text"])
                        min_dist = dist
                    except ValueError:
                        pass

            if matched_kd is not None:
                rows.append(
                    ActivityRow(
                        cmpd_id=cid_item["text"],
                        kd_nm=matched_kd,
                        source_image=image_path.name,
                    )
                )

        return rows


# ---------- regex backend (for tests / OCR-text fallback) ----------

_CMPD_ID_RE = re.compile(r"\b(\d{3,5})\b")
_KD_RE = re.compile(r"(\d+\.\d+)")
_RT_RE = re.compile(r"\b(\d+\.\d{1,2})\b")
_MZ_RE = re.compile(r"\b(\d{3,4}\.\d{1,2})\b")
_POLARITY_RE = re.compile(r"\[M\+(?:H|Na|K)?\]\+?")


@dataclass
class RegexTableExtractor:
    """Heuristic regex extractor over already-OCR'd text."""

    name: str = "regex"

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:
        raise NotImplementedError(
            "RegexTableExtractor is text-only; use parse_activity_text()."
        )

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:
        raise NotImplementedError(
            "RegexTableExtractor is text-only; use parse_smiles_text()."
        )

    def parse_activity_text(
        self,
        text: str,
        *,
        source_image: str | None = None,
    ) -> list[ActivityRow]:
        rows: list[ActivityRow] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cmpd_m = _CMPD_ID_RE.search(line)
            if not cmpd_m:
                continue
            cmpd_id = cmpd_m.group(1)
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
            rows.append(
                ActivityRow(
                    cmpd_id=cmpd_id,
                    kd_nm=kd,
                    rt_min=rt,
                    ms_mz=mz,
                    lcms_method=None,
                    ms_polarity=pol_m.group(0) if pol_m else None,
                    source_image=source_image,
                )
            )
        return rows

    def parse_smiles_text(
        self,
        text: str,
        *,
        source_image: str | None = None,
    ) -> list[SmilesRow]:
        rows: list[SmilesRow] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cmpd_m = _CMPD_ID_RE.search(line)
            if not cmpd_m:
                continue
            if not re.search(r"[CNO\[]=", line) and "@" not in line:
                continue
            tokens = line.split()
            cmpd_id_str = cmpd_m.group(1)
            smiles = None
            for tok in tokens:
                if tok == cmpd_id_str:
                    continue
                if any(c in tok for c in "[@=["):
                    smiles = tok
                    break
            if smiles is None:
                for tok in tokens:
                    if tok != cmpd_id_str:
                        smiles = tok
                        break
            rows.append(
                SmilesRow(
                    cmpd_id=cmpd_m.group(1),
                    smiles=smiles,
                    source_image=source_image,
                )
            )
        return rows


# ---------- stub / no-op backend ----------


@dataclass
class StubTableExtractor:
    name: str = "stub"

    def extract_activity(self, image_path: Path) -> list[ActivityRow]:
        return []

    def extract_smiles(self, image_path: Path) -> list[SmilesRow]:
        return []


# ---------- vision-LLM backend skeleton ----------


@dataclass
class VisionLlmTableExtractor:
    name: str = "vision-llm"
    llm_call: object | None = None
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

    def _call(self, image_path: Path, prompt: str) -> str:
        if self.llm_call is None:
            raise RuntimeError("VisionLlmTableExtractor.llm_call is None.")
        result = self.llm_call(image_path, prompt)
        if not isinstance(result, str):
            raise TypeError(f"llm_call must return str, got {type(result)}")
        return result

    @staticmethod
    def _parse_activity_json(
        raw: str, *, source_image: str | None
    ) -> list[ActivityRow]:
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
            out.append(
                ActivityRow(
                    cmpd_id=cmpd_id,
                    kd_nm=_as_float(r.get("kd_nm")),
                    rt_min=_as_float(r.get("rt_min")),
                    ms_mz=_as_float(r.get("ms_mz")),
                    lcms_method=_as_str_or_none(r.get("lcms_method")),
                    ms_polarity=_as_str_or_none(r.get("ms_polarity")),
                    source_image=source_image,
                )
            )
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
            out.append(
                SmilesRow(
                    cmpd_id=cmpd_id,
                    smiles=smiles,
                    source_image=source_image,
                )
            )
        return out


def _safe_json_loads(raw: str) -> object:
    text = raw.strip()
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
    rows: list[SmilesRow],
    *,
    validate: bool = True,
) -> list[SmilesRow]:
    """Populate canonical_smiles using cpd.chem.validate_and_enrich."""
    out: list[SmilesRow] = []
    for r in rows:
        if not r.smiles or not validate:
            out.append(r)
            continue
        chem = validate_and_enrich(r.smiles)
        out.append(
            SmilesRow(
                cmpd_id=r.cmpd_id,
                smiles=r.smiles,
                canonical_smiles=chem.canonical_smiles if chem else None,
                molecular_weight=chem.molecular_weight if chem else None,
                heavy_atom_count=chem.heavy_atom_count if chem else None,
                is_valid=chem is not None,
                source_image=r.source_image,
            )
        )
    return out


__all__ = [
    "ActivityRow",
    "SmilesRow",
    "TableExtractor",
    "RapidOcrTableExtractor",
    "RegexTableExtractor",
    "StubTableExtractor",
    "VisionLlmTableExtractor",
    "enrich_smiles_with_rdkit",
]
